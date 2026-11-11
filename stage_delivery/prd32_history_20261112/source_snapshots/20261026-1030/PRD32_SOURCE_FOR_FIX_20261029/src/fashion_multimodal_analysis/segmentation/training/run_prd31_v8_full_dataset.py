#!/usr/bin/env python3
"""3.1.1 实例分割：数据与模型版本分别记录，指标按固定匹配和计时口径解释。

八类原始训练数据：准备索引、从 V5 微调、断点续跑、固定口径评估。

放到现有项目 scripts/：
  python -u scripts/segmentation/training/run_prd31_v8_full_dataset.py --phase prepare
  python -u scripts/segmentation/training/run_prd31_v8_full_dataset.py --phase all --epochs 1
重复同一训练命令会从最近完成的 batch 自动继续；--epochs 是总轮数。

项目根目录由当前安装包定位；数据默认位于相邻的 ../fashion_data。
DF2 原始标注相对数据目录为 raw/train/train/annos。
Fashionpedia --fp-mode auto 优先使用 instances_attributes_train2020.json；
若缺少原始 train，只复用 V5 清单里的 Fashionpedia 训练图片，报告明确标为
“DF2 全量 + Fashionpedia 既有训练图片”，并保留这些图片来自 val2020 的记录。
此情形不是两套原始训练集全量；--fp-mode native-train 可强制要求两套 train。
不会把其余 val/test 或 FashionAI 属性测试图加入实例分割训练。

每轮无放回遍历全部入选图片；同图八类有效实例全部保留，实例掩码可重叠。
掩码按需解码，不生成数十万张全尺寸 PNG。索引为 SQLite，支持多进程读。
排除固定开发/Core/审核图片、精确 RGB 副本和可识别的受保护 DF2 pair_id。
这不是感知相似图去重；Core400 已被使用过，只能作为固定回归而非盲测验收。
默认微调一轮，不保证达标。结果写入独立 V8 目录，不改写 V4--V7。
all 完成后对照 V5，评估开发集及 Core400，并打包 prd31_v8_full_dataset_results.zip。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import zipfile
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from fashion_multimodal_analysis.common.paths import resolve_record_path

CLASS_TO_ID = dict(
    top=1, pants=2, skirt=3, outerwear=4, dress=5, shoe=6, bag=7, accessory=8
)
DF2_CLASS = {
    1: "top",
    2: "top",
    3: "outerwear",
    4: "outerwear",
    5: "top",
    6: "top",
    7: "pants",
    8: "pants",
    9: "skirt",
    10: "dress",
    11: "dress",
    12: "dress",
    13: "dress",
}
FP_CLASS = {
    "shirt, blouse": "top",
    "top, t-shirt, sweatshirt": "top",
    "sweater": "top",
    "vest": "top",
    "cardigan": "outerwear",
    "jacket": "outerwear",
    "coat": "outerwear",
    "cape": "outerwear",
    "pants": "pants",
    "shorts": "pants",
    "skirt": "skirt",
    "dress": "dress",
    "shoe": "shoe",
    "bag, wallet": "bag",
    "glasses": "accessory",
    "hat": "accessory",
    "headband, head covering, hair accessory": "accessory",
    "tie": "accessory",
    "glove": "accessory",
    "watch": "accessory",
    "belt": "accessory",
    "leg warmer": "accessory",
    "tights, stockings": "accessory",
    "sock": "accessory",
    "scarf": "accessory",
}
CORE = "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv"
DEV = "configs/prd_8class_val_v2_full_targets.csv"
V5_TRAIN = "configs/prd_8class_train_v5_full_targets.csv"
V5_CP = "outputs/prd_instance_segmentation/maskrcnn_8class_v5_full_targets/checkpoint_last.pth"
FROZEN_SHA = {
    CORE: "c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89",
    DEV: "39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93",
    V5_TRAIN: "36ff78831b20e492d787c66161f05747433f8e85f7a089fd5df981c2bdf476e6",
}
V5_CP_SHA = "ea5a4ddf0f002256f1ce5164fc1b4b5b4d2263698f9525c9fbe7d8cf98bc1a65"
FAST_RUNTIME_SHA = "ff85ca020695498ab287d73e8afd915b44345ecd2168b38eb24410fb8ddddfb8"
FORMAT = 1
STOP = False


def log(event: str, **values: Any) -> None:
    """打印带时间戳的流程事件与关键数值，便于从日志确认任务进度。

    Args:
        event: 写入进度日志的事件名称。
    """
    print(
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        event,
        json.dumps(values, ensure_ascii=False),
        flush=True,
    )


def digest(path: str | Path) -> str:
    """分块计算文件 SHA-256，核对文件身份而不加载整个权重到内存。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        64 位十六进制 SHA-256 摘要字符串。
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def obj_digest(obj: Any) -> str:
    """对稳定排序的 JSON 表示计算哈希，校验实验配置身份。

    Args:
        obj: 需要序列化或计算摘要的 JSON 兼容对象。

    Returns:
        64 位十六进制 SHA-256 摘要字符串。
    """
    return hashlib.sha256(
        json.dumps(
            obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    """保存结构化实验记录。

    Args:
        path: 要读取或写入的文件路径。
        obj: 需要序列化或计算摘要的 JSON 兼容对象。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def read_csv(path: str | Path) -> Iterator[Any]:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Yields:
        每行一个字段字典；值保留原始字符串，不自动改变标签。
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def resolve(value: Any, root: Path, data: Path) -> Path:
    """将历史原图/掩码引用迁移到本次使用的项目和数据目录。

    Args:
        value: 待解析或规范化的输入值。
        root: 当前项目根目录。
        data: 外部服饰数据根目录。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_record_path(str(value), root, data)


def rgb_digest(image: Any) -> str:
    """根据尺寸和 RGB 像素计算摘要，识别精确内容副本。

    Args:
        image: 本步骤处理的图像对象。

    Returns:
        64 位十六进制 SHA-256 摘要字符串。
    """
    return hashlib.sha256(
        f"{image.width}x{image.height}:".encode() + image.tobytes()
    ).hexdigest()


def protected_inventory(root: Path, data: Any) -> Any:
    """冻结本轮训练需要排除的评估和审核图片。

    Args:
        root: 当前项目根目录。
        data: 当前外部服饰数据目录。

    Returns:
        路径集合、精确 RGB 摘要集合、已知 DF2 pair_id、逐清单凭据和集合签名。

    Raises:
        ValueError: Core 或开发清单的原始 SHA-256 与固定值不符。
        FileNotFoundError: 保护图片缺失，无法完成内容级隔离检查。

    Notes:
        除路径隔离外，检查像素完全相同的副本和已知正 pair_id。
        这不是感知相似图去重，也不能把已使用过的 Core400 重新变成盲测集。
    """
    from PIL import Image

    for relative in (CORE, DEV):
        path = root / relative
        if not path.is_file() or digest(path) != FROZEN_SHA[relative]:
            raise ValueError("固定评估清单缺失或被修改，请核对：" + str(path))
    files = sorted(
        {
            root / CORE,
            root / DEV,
            *root.glob("configs/*val*.csv"),
            *root.glob("configs/*test*.csv"),
            *root.glob("benchmark/**/manifests/*.csv"),
            *root.glob("benchmark/**/review/*.csv"),
        }
    )
    paths, evidence = set(), []
    for path in files:
        refs = set()
        for row in read_csv(path):
            for key in ("source_image", "image_path", "query_image", "image"):
                value = row.get(key, "").strip()
                if value and Path(value).suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    ".bmp",
                }:
                    refs.add(str(resolve(value, root, data)))
        paths.update(refs)
        evidence.append(
            dict(
                path=path.relative_to(root).as_posix(),
                sha256=digest(path),
                images=len(refs),
            )
        )
    hashes, pairs = set(), set()
    for i, value in enumerate(sorted(paths), 1):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError("受保护图片无法读取，不能完成隔离：" + value)
        with Image.open(path) as image:
            hashes.add(rgb_digest(image.convert("RGB")))
        ann = path.parent.parent / "annos" / (path.stem + ".json")
        if path.parent.name == "image" and ann.is_file():
            pair = int(
                json.loads(ann.read_text(encoding="utf-8-sig")).get("pair_id", 0) or 0
            )
            if pair > 0:
                pairs.add(pair)
        if i % 200 == 0:
            log("PROTECTED_IMAGES", checked=i, total=len(paths))
    signature = obj_digest(
        dict(
            files=evidence, paths=sorted(paths), rgb=sorted(hashes), pairs=sorted(pairs)
        )
    )
    return dict(
        paths=paths, hashes=hashes, pairs=pairs, evidence=evidence, signature=signature
    )


def find_fp_json(data: Path, name: str) -> Any:
    """查找指定 Fashionpedia 原始标注，多个副本时要求显式选择。

    Args:
        data: 外部服饰数据根目录。
        name: 当前标签、字段或产物名称。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    found = sorted(
        {
            p.resolve()
            for base in (data / "raw/fashionpedia", data / "fashionpedia")
            if base.is_dir()
            for p in base.rglob(name)
        }
    )
    if len(found) > 1:
        raise ValueError(f"{name} 有多个副本，请用 --fp-annotations 指定。")
    return found[0] if found else None


def discover(a: Any, root: Path, data: Path) -> Any:
    """确定本地实际可用的训练来源和全量范围。

    Args:
        a: 命令行配置，包含 DF2 目录、Fashionpedia 模式和显式标注路径。
        root: 项目根目录，用于读取未修改的 V5 训练清单。
        data: 外部数据根目录。

    Returns:
        原始标注/图片目录、Fashionpedia 模式、训练白名单及实际范围说明。

    Raises:
        FileNotFoundError: 原始训练图/标注不齐，或指定模式缺少对应原始标注。
        ValueError: 来源属于 val/test、存在无法唯一选择的目录，或 V5 清单被修改。

    Notes:
        auto 优先使用 train2020；没有它时只复用 V5 已使用的 Fashionpedia 图。
        该替代模式来源于 val2020，不能称为两套原始训练集全量，也不新增其余验证图。
    """
    if a.df2_train:
        base = resolve(a.df2_train, root, data)
        annos = base if base.name == "annos" else base / "annos"
    else:
        candidates = [
            data / p
            for p in (
                "raw/train/train/annos",
                "raw/train/annos",
                "train/train/annos",
                "train/annos",
                "raw/deepfashion2/train/annos",
                "raw/DeepFashion2/train/annos",
            )
        ]
        found = {p.resolve() for p in candidates if p.is_dir()}
        if len(found) != 1:
            raise FileNotFoundError(
                "找不到唯一 DF2 train/annos；应为 raw/train/train/annos，可用 --df2-train 指定。"
            )
        annos = found.pop()
    if any(part.lower() in {"val", "validation", "test"} for part in annos.parts):
        raise ValueError("DF2 数据来源必须是 train。")
    images = annos.parent / "image"
    annotations = sorted(annos.glob("*.json"))
    if not annotations or not images.is_dir():
        raise FileNotFoundError("DF2 图片或 JSON 缺失：" + str(annos))
    img_stems = {
        p.stem
        for p in images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    if img_stems - {p.stem for p in annotations}:
        raise ValueError("DF2 有图片缺少同名 JSON，原始训练标注不齐全。")
    fp = (
        resolve(a.fp_annotations, root, data)
        if a.fp_annotations
        else find_fp_json(data, "instances_attributes_train2020.json")
    )
    mode = a.fp_mode
    if mode == "auto":
        mode = "native-train" if fp else "existing-train"
    whitelist = None
    if mode == "native-train":
        if (
            not fp
            or not fp.is_file()
            or fp.name != "instances_attributes_train2020.json"
        ):
            raise FileNotFoundError(
                "缺少 Fashionpedia 原始 instances_attributes_train2020.json。"
                + "不能用 val/test 替代；可用 --fp-mode existing-train，仅复用 V5 既有训练图片。"
            )
        scope = "全部本地 DF2 原始 train + Fashionpedia train2020，扣除明确列出的隔离/无目标/异常图片"
    else:
        manifest = root / V5_TRAIN
        if not manifest.is_file() or digest(manifest) != FROZEN_SHA[V5_TRAIN]:
            raise ValueError("复用 Fashionpedia 需要未修改的 V5 完整目标训练清单。")
        whitelist = {
            str(resolve(r["source_image"], root, data))
            for r in read_csv(manifest)
            if r["source_dataset"] == "Fashionpedia"
        }
        if not whitelist:
            raise ValueError("V5 清单没有 Fashionpedia 训练图片。")
        fp = (
            resolve(a.fp_annotations, root, data)
            if a.fp_annotations
            else find_fp_json(data, "instances_attributes_val2020.json")
        )
        if not fp or not fp.is_file() or fp.name != "instances_attributes_val2020.json":
            raise FileNotFoundError(
                "复用 V5 图片需要原始 instances_attributes_val2020.json。"
            )
        scope = "全部本地 DF2 原始 train + V5 既有 Fashionpedia 训练图片；不是两套原始训练集全量"
        log(
            "FASHIONPEDIA_SCOPE_NOTICE",
            scope=scope,
            images=len(whitelist),
            annotation_origin="val2020 中原已分配给训练的图片，不新加其余验证/测试图",
        )
    return dict(
        df2_annotations=annotations,
        df2_images=images,
        fp_annotation=fp,
        fp_mode=mode,
        whitelist=whitelist,
        scope=scope,
    )


def json_items(path: str | Path, key: Any) -> Iterator[Any]:
    """流式读取大型标注数组，避免一次把完整 JSON 加载进内存。

    Args:
        path: 要读取或写入的文件路径。
        key: 标识。

    Yields:
        按需产生的记录/任务；不一次性加载全部条目。

    Raises:
        RuntimeError: 原始 Fashionpedia JSON 较大，请先运行 python -m pip install ijson
    """
    try:
        import ijson
    except ImportError:
        if Path(path).stat().st_size > 200 * 1024 * 1024:
            raise RuntimeError(
                "原始 Fashionpedia JSON 较大，请先运行 python -m pip install ijson"
            )
        with Path(path).open(encoding="utf-8-sig") as f:
            yield from json.load(f)[key]
    else:
        with Path(path).open("rb") as f:
            yield from ijson.items(f, key + ".item", use_float=True)


def fp_roots(a: Any, data: Path, annotation: dict[str, Any]) -> list[Any]:
    """收集 Fashionpedia 的候选原图目录。

    Args:
        a: 当前函数的第一个输入，含义随运算而定。
        data: 外部服饰数据根目录。
        annotation: 原始实例标注。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if a.fp_images:
        return [Path(a.fp_images).expanduser().resolve()]
    bases = list(
        dict.fromkeys(
            [annotation.parent.parent, data / "raw/fashionpedia", data / "fashionpedia"]
        )
    )
    return [
        base / p
        for base in bases
        for p in (
            "images/train",
            "images/train2020",
            "train",
            "train/train",
            "train2020",
            "images",
            "",
        )
        if (base / p).is_dir()
    ]


def match_fp_image(name: str, roots: Any, whitelist: Any) -> Any:
    """在原图目录中定位指定文件，并限定已有训练白名单。

    Args:
        name: 当前标签、字段或产物名称。
        roots: roots。
        whitelist: whitelist。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("无效 Fashionpedia file_name：" + name)
    if whitelist is not None:
        found = [Path(p) for p in whitelist if Path(p).name == relative.name]
    else:
        found = list(
            {
                (base / v).resolve()
                for base in roots
                for v in (relative, Path(relative.name))
                if (base / v).is_file()
            }
        )
    if len(found) > 1:
        raise ValueError(
            "Fashionpedia 图片存在多个匹配，请用 --fp-images 指定：" + name
        )
    return (
        found[0]
        if found
        else (
            roots[0] / relative
            if roots
            else Path("/missing_fashionpedia_images") / relative
        )
    )


def stage_fp(sources: dict[str, Any], path: str | Path) -> tuple[Any, ...]:
    """将 Fashionpedia 可用标注整理到准备阶段的索引中。

    Args:
        sources: 记录字段，使用 fp_annotation, whitelist。
        path: 要读取或写入的文件路径。

    Returns:
        按顺序返回 db 等结果。

    Raises:
        ValueError: 原始 FP 标注未覆盖全部既有训练图片。
    """
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE images(id TEXT PRIMARY KEY,name TEXT,width INTEGER,height INTEGER);"
        + "CREATE TABLE objects(image_id TEXT,data TEXT);"
    )
    names = {
        str(c["id"]): " ".join(c["name"].lower().split())
        for c in json_items(sources["fp_annotation"], "categories")
    }
    allowed = (
        {Path(p).name for p in sources["whitelist"]}
        if sources["whitelist"] is not None
        else None
    )
    if allowed is not None and len(allowed) != len(sources["whitelist"]):
        raise ValueError("Fashionpedia 既有训练图片存在同名文件。")
    image_ids = set()
    selected_names = set()
    for image in json_items(sources["fp_annotation"], "images"):
        if allowed is not None and Path(image["file_name"]).name not in allowed:
            continue
        db.execute(
            "INSERT INTO images VALUES(?,?,?,?)",
            (
                str(image["id"]),
                image["file_name"],
                int(image["width"]),
                int(image["height"]),
            ),
        )
        image_ids.add(str(image["id"]))
        selected_names.add(Path(image["file_name"]).name)
    if allowed is not None and selected_names != allowed:
        raise ValueError("原始 FP 标注未覆盖全部既有训练图片。")
    excluded = Counter()
    count = 0
    for annotation in json_items(sources["fp_annotation"], "annotations"):
        if str(annotation["image_id"]) not in image_ids:
            continue
        fine = names[str(annotation["category_id"])]
        category = FP_CLASS.get(fine)
        if category is None:
            excluded["outside_eight_class_scope"] += 1
            continue
        if int(annotation.get("iscrowd", 0)):
            excluded["crowd"] += 1
            continue
        item = dict(
            label=CLASS_TO_ID[category],
            fine=fine,
            annotation_id=str(annotation["id"]),
            segmentation=annotation.get("segmentation"),
            source_bbox=annotation.get("bbox"),
        )
        db.execute(
            "INSERT INTO objects VALUES(?,?)",
            (str(annotation["image_id"]), json.dumps(item, separators=(",", ":"))),
        )
        count += 1
        if count % 10000 == 0:
            db.commit()
            log("FP_ANNOTATIONS", indexed=count)
    db.execute("CREATE INDEX objects_by_image ON objects(image_id)")
    db.commit()
    return db, dict(excluded)


def decode_mask(segmentation: Any, width: int, height: int) -> Any:
    """解码原始分割标注，保留同一实例的全部前景区域。

    Args:
        segmentation: 原始分割标注，可能为多边形或 COCO RLE。
        width: 图像或目标表示的宽度。
        height: 图像或目标表示的高度。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        ValueError: 掩码为空或尺寸无效
        RuntimeError: RLE 需要 python -m pip install pycocotools
    """
    import numpy as np
    from PIL import Image, ImageDraw

    if isinstance(segmentation, list):
        polygons = (
            [segmentation]
            if segmentation and isinstance(segmentation[0], (int, float))
            else segmentation
        )
        if not polygons:
            raise ValueError("缺少分割多边形")
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for polygon in polygons:
            if (
                len(polygon) < 6
                or len(polygon) % 2
                or not all(math.isfinite(float(v)) for v in polygon)
            ):
                raise ValueError("分割多边形坐标无效")
            draw.polygon(
                [
                    (float(polygon[i]), float(polygon[i + 1]))
                    for i in range(0, len(polygon), 2)
                ],
                fill=1,
            )
        mask = np.asarray(image, dtype=np.uint8)
    elif isinstance(segmentation, dict):
        try:
            from pycocotools import mask as coco
        except ImportError as e:
            raise RuntimeError("RLE 需要 python -m pip install pycocotools") from e
        rle = dict(segmentation)
        if list(rle.get("size", [])) != [height, width]:
            raise ValueError("RLE size 与原图不符")
        if isinstance(rle.get("counts"), list):
            rle = coco.frPyObjects(rle, height, width)
        elif isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        mask = coco.decode(rle)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2).astype(np.uint8)
    else:
        raise ValueError("缺少 polygon/RLE 掩码")
    if mask.shape != (height, width) or not mask.any():
        raise ValueError("掩码为空或尺寸无效")
    return (mask > 0).astype(np.uint8)


def mask_box(mask: Any) -> list[Any]:
    """从前景掩码取得边界框。

    Args:
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    import numpy as np

    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def prepare_image(job: Any, protected: dict[str, Any]) -> Any:
    """校验单张原图和其完整目标，生成可进入索引的记录。

    Args:
        job: job。
        protected: 记录字段，使用 paths, hashes, pairs。

    Returns:
        结果字段包括 error。

    Raises:
        ValueError: 原始 bbox 面积非正
    """
    from PIL import Image

    dataset, path, annotation_path, expected_size, objects = job
    identity = dict(
        dataset=dataset, path=str(path), annotation_path=str(annotation_path)
    )
    try:
        if str(path) in protected["paths"]:
            return dict(identity, excluded="protected_path")
        ann_sha = ""
        if dataset == "DeepFashion2":
            raw = Path(annotation_path).read_bytes()
            source = json.loads(raw.decode("utf-8-sig"))
            ann_sha = hashlib.sha256(raw).hexdigest()
            pair = int(source.get("pair_id", 0) or 0)
            if pair > 0 and pair in protected["pairs"]:
                return dict(identity, excluded="protected_df2_pair_id")
            objects = []
            for key, item in sorted(source.items()):
                if not re.fullmatch(r"item\d+", key) or not isinstance(item, dict):
                    continue
                cat = int(item.get("category_id", -1))
                if cat not in DF2_CLASS:
                    raise ValueError("未知 DF2 category_id：" + str(cat))
                objects.append(
                    dict(
                        label=CLASS_TO_ID[DF2_CLASS[cat]],
                        fine=str(cat),
                        annotation_id=key,
                        segmentation=item.get("segmentation"),
                        source_bbox=item.get("bounding_box"),
                    )
                )
        if not objects:
            return dict(identity, excluded="no_eight_class_targets")
        stat = Path(path).stat()
        with Image.open(path) as original:
            image = original.convert("RGB")
            width, height = image.size
            rgb = rgb_digest(image)
        if rgb in protected["hashes"]:
            return dict(identity, excluded="protected_exact_rgb_copy")
        if expected_size and (width, height) != tuple(expected_size):
            raise ValueError("图片尺寸与原始标注不符")
        ids = set()
        for item in objects:
            if item["annotation_id"] in ids:
                raise ValueError("同图实例 ID 重复")
            ids.add(item["annotation_id"])
            box = item["source_bbox"]
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(math.isfinite(float(v)) for v in box)
            ):
                raise ValueError("原始 bbox 无效")
            if (
                dataset == "DeepFashion2" and (box[2] <= box[0] or box[3] <= box[1])
            ) or (dataset == "Fashionpedia" and (box[2] <= 0 or box[3] <= 0)):
                raise ValueError("原始 bbox 面积非正")
            mask = decode_mask(item["segmentation"], width, height)
            item["box"] = mask_box(mask)
            item["area"] = int(mask.sum())
            del item["source_bbox"]
        return dict(
            identity,
            width=width,
            height=height,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            rgb_sha256=rgb,
            annotation_sha256=ann_sha,
            objects=objects,
        )
    except Exception as e:
        return dict(identity, error=f"{type(e).__name__}: {e}")


def jobs(a: Any, data: Any, source: dict[str, Any], fp: Any) -> Iterator[Any]:
    """按来源生成图片准备任务，避免提前解码全部掩码。

    Args:
        a: 当前函数的第一个输入，含义随运算而定。
        data: 外部数据目录或本函数处理的数据结构；具体用途由操作对象决定。
        source: 记录字段，使用 df2_annotations, fp_annotation, df2_images, whitelist。
        fp: Fashionpedia。

    Yields:
        按需产生的记录/任务；不一次性加载全部条目。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    for annotation in source["df2_annotations"]:
        found = [
            source["df2_images"] / (annotation.stem + ext)
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
            if (source["df2_images"] / (annotation.stem + ext)).is_file()
        ]
        if len(found) > 1:
            raise ValueError("DF2 同一标注对应多张同名图：" + str(annotation))
        image = found[0] if found else source["df2_images"] / (annotation.stem + ".jpg")
        yield ("DeepFashion2", image.resolve(), annotation, None, None)
    roots = fp_roots(a, data, source["fp_annotation"])
    for key, name, width, height in fp.execute(
        "SELECT id,name,width,height FROM images ORDER BY id"
    ):
        objects = [
            json.loads(row[0])
            for row in fp.execute("SELECT data FROM objects WHERE image_id=?", (key,))
        ]
        if objects:
            yield (
                "Fashionpedia",
                match_fp_image(name, roots, source["whitelist"]),
                source["fp_annotation"],
                (width, height),
                objects,
            )


def bounded_map(fn: Any, iterable: Any, workers: int) -> Iterator[Any]:
    """限制待处理任务数量，避免数据准备阶段积压全部图片任务。

    Args:
        fn: 处理一个任务的回调函数。
        iterable: 按需产生任务的可迭代对象。
        workers: 当前阶段的数据读取或准备进程/线程数量。

    Yields:
        按需产生的记录/任务；不一次性加载全部条目。
    """
    if workers <= 1:
        yield from map(fn, iterable)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        iterator = iter(iterable)
        for _ in range(workers * 4):
            item = next(iterator, None)
            if item is None:
                break
            pending.append(executor.submit(fn, item))
        while pending:
            yield pending.popleft().result()
            item = next(iterator, None)
            if item is not None:
                pending.append(executor.submit(fn, item))


def prepare(
    a: Any, root: Path, data: Any, output: Any, report: Any, protected: dict[str, Any]
) -> Any:
    """校验完整原图并建立只读训练索引。

    Args:
        a: 当前准备参数，包含排除坏图的显式开关和准备线程数。
        root: 项目根目录。
        data: 外部服饰数据目录。
        output: 当前 V8 版本的索引/权重输出目录。
        report: 当前 V8 版本的报告目录。
        protected: 必须从训练中排除的图片与内容摘要。

    Returns:
        完整准备凭据：图片/实例计数、类别分布、索引摘要、来源守卫和排除原因。

    Raises:
        ValueError: 原图或标注无效、八类覆盖不完整、输入身份改变或标注被修改。
        RuntimeError: 索引/凭据不完整，或已有权重却缺少原索引。

    Notes:
        同一原图的全部八类有效实例一起保存，掩码允许重叠。
        SQLite 只存原始分割表示，取样时再解码，避免生成几十万张全尺寸 PNG。
        既有索引通过 SHA 和来源守卫校验后才能复用；不同输入必须创建新 run-name。
    """
    index = output / "dataset.sqlite"
    registry = report / "data_preparation.json"
    source = discover(a, root, data)
    # 来源文件列表也参与身份检查，防止目录变化后误用旧索引。
    source_paths_sha = obj_digest(
        sorted(str(p) for p in source["df2_annotations"] + [source["fp_annotation"]])
    )
    request = obj_digest(
        dict(
            schema=FORMAT,
            data=str(data),
            protected=protected["signature"],
            df2=a.df2_train,
            fp_mode=a.fp_mode,
            fp_annotations=a.fp_annotations,
            fp_images=a.fp_images,
            allow_bad_images=a.allow_bad_images,
            actual_fp_mode=source["fp_mode"],
            source_paths_sha256=source_paths_sha,
        )
    )
    if index.exists() or registry.exists():
        if not (index.is_file() and registry.is_file()):
            raise RuntimeError(
                "索引产物不完整，请保留目录并换 --run-name v8_full_dataset_2。"
            )
        saved = json.loads(registry.read_text(encoding="utf-8"))
        if saved["request_sha256"] != request or digest(index) != saved["index_sha256"]:
            raise ValueError(
                "准备输入、隔离清单或索引改变；请使用新的 --run-name，不能覆盖已有实验。"
            )
        for guard in saved["source_guards"]:
            path = Path(guard["path"])
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (guard["size"], guard["mtime_ns"]):
                raise ValueError("原始标注改变：" + str(path))
        log(
            "REUSE_DATA_INDEX",
            images=saved["training_images"],
            instances=saved["training_instances"],
            scope=saved["scope"],
        )
        return saved
    if (output / "checkpoint_last.pth").exists():
        raise RuntimeError("checkpoint 已有但索引缺失，不能重建覆盖。")
    log(
        "PREPARE_START",
        df2_annotation_files=len(source["df2_annotations"]),
        fp_mode=source["fp_mode"],
        scope=source["scope"],
    )
    guards = []
    for path in source["df2_annotations"] + [source["fp_annotation"]]:
        stat = path.stat()
        guards.append(
            dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        )
    fp_sha = digest(source["fp_annotation"])
    stage = output / "fp_staging.tmp.sqlite"
    fp, fp_exclusions = stage_fp(source, stage)
    temp = output / "dataset.tmp.sqlite"
    temp.unlink(missing_ok=True)
    db = sqlite3.connect(temp)
    db.execute(
        "CREATE TABLE images(id INTEGER PRIMARY KEY,dataset TEXT,path TEXT UNIQUE,width INTEGER,height INTEGER,"
        + "size INTEGER,mtime_ns INTEGER,rgb_sha256 TEXT,objects TEXT)"
    )
    counts, by_dataset, excluded = Counter(), Counter(), Counter()
    scanned, selected, instances, errors = 0, 0, 0, 0
    started = time.monotonic()
    annotations_sha = hashlib.sha256()
    try:
        with (report / "excluded_images.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=["dataset", "path", "annotation_path", "reason"]
            )
            writer.writeheader()
            for item in bounded_map(
                lambda job: prepare_image(job, protected),
                jobs(a, data, source, fp),
                a.prepare_workers,
            ):
                scanned += 1
                if "error" in item or "excluded" in item:
                    reason = item.get("error") or item["excluded"]
                    writer.writerow(
                        {k: item[k] for k in ("dataset", "path", "annotation_path")}
                        | dict(reason=reason)
                    )
                    excluded[
                        "invalid_complete_image" if "error" in item else reason
                    ] += 1
                    if "error" in item:
                        errors += 1
                        if errors <= 5:
                            log("IMAGE_ERROR", **item)
                else:
                    selected += 1
                    instances += len(item["objects"])
                    by_dataset[item["dataset"]] += 1
                    counts.update(
                        next(k for k, v in CLASS_TO_ID.items() if v == obj["label"])
                        for obj in item["objects"]
                    )
                    annotations_sha.update(
                        (
                            item["annotation_path"]
                            + ":"
                            + item["annotation_sha256"]
                            + "\n"
                        ).encode()
                    )
                    db.execute(
                        "INSERT INTO images VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            selected,
                            item["dataset"],
                            item["path"],
                            item["width"],
                            item["height"],
                            item["size"],
                            item["mtime_ns"],
                            item["rgb_sha256"],
                            json.dumps(item["objects"], separators=(",", ":")),
                        ),
                    )
                if scanned % 1000 == 0:
                    db.commit()
                    f.flush()
                    log(
                        "PREPARE_PROGRESS",
                        scanned=scanned,
                        selected=selected,
                        instances=instances,
                        errors=errors,
                        elapsed_minutes=round((time.monotonic() - started) / 60, 1),
                    )
        # 坏图默认阻止训练；排除必须使用显式开关并保留逐图原因。
        if errors and not a.allow_bad_images:
            raise ValueError(
                f"{errors} 张原图/标注无效，详见 {report}/excluded_images.csv。修复后重跑，"
                + "或显式加 --allow-bad-images 排除这些整张图；目前未启动训练。"
            )
        if set(counts) != set(CLASS_TO_ID):
            raise ValueError("训练集没有覆盖八类：" + str(dict(counts)))
        for guard in guards:
            stat = Path(guard["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (guard["size"], guard["mtime_ns"]):
                raise ValueError("标注在准备期间发生改变。")
        db.commit()
        db.close()
        db = None
        temp.replace(index)
        saved = dict(
            status="PREPARED",
            schema=FORMAT,
            request_sha256=request,
            index=str(index),
            index_sha256=digest(index),
            source_paths_sha256=source_paths_sha,
            scope=source["scope"],
            fashionpedia_mode=source["fp_mode"],
            fashionpedia_annotation=str(source["fp_annotation"]),
            fashionpedia_annotation_sha256=fp_sha,
            df2_annotation_files=len(source["df2_annotations"]),
            training_images=selected,
            training_instances=instances,
            class_counts=dict(counts),
            dataset_image_counts=dict(by_dataset),
            image_exclusions=dict(excluded),
            fp_annotation_exclusions=fp_exclusions,
            source_guards=guards,
            annotation_inventory_sha256=annotations_sha.hexdigest(),
            protected_manifests=protected["evidence"],
            class_to_id=CLASS_TO_ID,
            sampling="Seeded shuffle without replacement, every selected image once, drop_last=False",
            mask_rasterization="Independent polygons via PIL as V5; RLE via pycocotools; overlapping instances retained",
            isolation="Normalized path + exact RGB + known positive DF2 pair_id; no perceptual deduplication",
            formal_blind_acceptance="NOT_CLAIMED",
            preparation_seconds=round(time.monotonic() - started, 2),
        )
        write_json(registry, saved)
        log(
            "PREPARE_DONE",
            images=selected,
            instances=instances,
            classes=dict(counts),
            scope=source["scope"],
        )
        return saved
    finally:
        if db is not None:
            db.close()
        fp.close()
        stage.unlink(missing_ok=True)


def epoch_order(length: int, seed: int, epoch: int) -> Any:
    """生成一个轮次的无放回图片顺序。

    Args:
        length: 索引中的有效图片数量。
        seed: 基础随机种子。
        epoch: 从 0 开始的轮次编号。

    Returns:
        每个图片下标恰好出现一次的随机排列。

    Notes:
        续训按 checkpoint 保存的图片偏移截取同一顺序，不从头重复整轮。
    """
    order = list(range(length))
    random.Random(seed + epoch * 1000003).shuffle(order)
    return order


class NativeDataset:
    """从 SQLite 索引逐图读取完整八类实例。

    Attributes:
        index: 准备阶段发布的 SQLite 文件。
        length: 有效训练原图数量。
        seed: 水平翻转使用的固定种子。
        hflip: 是否启用水平翻转。
        epoch: 当前轮次，用于生成可复现的翻转决策。
        db: 当前读取进程的只读数据库连接。
        pid: 连接所属进程 ID，防止多进程共用同一 SQLite 连接。

    Notes:
        每次取图检查原图大小、修改时间和 RGB 摘要；准备后改图会中止训练。
        翻转同步作用于图像、所有实例框和所有实例掩码，保持目标一致。
    """

    def __init__(self, index: int, length: int, seed: int, hflip: Any) -> None:
        """保存当前对象需要的配置、数据引用和状态。

        Args:
            index: 待访问样本的整数下标。
            length: 待生成顺序的样本数。
            seed: 控制随机过程的种子。
            hflip: hflip。
        """
        self.index = str(index)
        self.length = length
        self.seed = seed
        self.hflip = hflip
        self.epoch = 0
        self.db = None
        self.pid = None

    def __len__(self) -> Any:
        """返回当前数据集或容器中的样本数量。

        Returns:
            本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
        """
        return self.length

    def __getstate__(self) -> Any:
        """getstate。

        Returns:
            结果字段包括 db, pid。
        """
        return dict(self.__dict__, db=None, pid=None)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        """取得指定样本，并生成本数据集约定的输入和目标。

        Args:
            index: 待访问样本的整数下标。

        Returns:
            按顺序返回 target 等结果。

        Raises:
            ValueError: 输入或实验状态不符合检查条件。
        """
        import numpy as np
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        # DataLoader 子进程各自创建只读 SQLite 连接，避免跨进程复用。
        if self.pid != os.getpid():
            if self.db is not None:
                self.db.close()
            self.db = sqlite3.connect(Path(self.index).as_uri() + "?mode=ro", uri=True)
            self.pid = os.getpid()
        path, w, h, size, mtime, rgb, serialized = self.db.execute(
            "SELECT path,width,height,size,mtime_ns,rgb_sha256,objects FROM images WHERE id=?",
            (index + 1,),
        ).fetchone()
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, mtime):
            raise ValueError("原图在准备之后改变：" + path)
        with Image.open(path) as source:
            image = source.convert("RGB")
        if image.size != (w, h) or rgb_digest(image) != rgb:
            raise ValueError("图片内容与冻结索引不符：" + path)
        objects = json.loads(serialized)
        masks = np.stack([decode_mask(o["segmentation"], w, h) for o in objects])
        boxes = np.asarray([o["box"] for o in objects], dtype=np.float32)
        if (
            self.hflip
            and random.Random(self.seed + self.epoch * 1000003 + index * 7919).random()
            < 0.5
        ):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            masks = masks[:, :, ::-1].copy()
            left, right = boxes[:, 0].copy(), boxes[:, 2].copy()
            boxes[:, 0], boxes[:, 2] = w - right, w - left
        target = dict(
            boxes=torch.as_tensor(boxes, dtype=torch.float32),
            labels=torch.tensor([o["label"] for o in objects], dtype=torch.int64),
            masks=torch.as_tensor(masks, dtype=torch.uint8),
            image_id=torch.tensor([index], dtype=torch.int64),
            area=torch.tensor([o["area"] for o in objects], dtype=torch.float32),
            iscrowd=torch.zeros(len(objects), dtype=torch.int64),
        )
        return to_tensor(image), target


def collate(batch: Any) -> Any:
    """将不同实例数量的图片组成 detection 批次，不堆叠目标字典。

    Args:
        batch: 一批图像与实例目标；每张图的实例数量可以不同。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return tuple(zip(*batch))


def build_model(min_size: int, max_size: int) -> Any:
    # Match the COCO-V1 V5 trainer's FrozenBatchNorm and three trainable late stages;
    # load the existing eight-class checkpoint without downloading COCO weights.
    """重建与 V5 权重兼容的八类 Mask R-CNN。

    Args:
        min_size: 输入短边目标尺寸，既有运行使用 640。
        max_size: 输入长边限制，既有运行使用 1024。

    Returns:
        ResNet50-FPN / Mask R-CNN，包含八类前景和一个背景类别。

    Notes:
        使用 FrozenBatchNorm2d(eps=0) 并保留 V5 的早期层冻结策略。
        这里不下载 COCO 权重；训练函数加载经过类别 ID 校验的现有八类 checkpoint。
    """
    from functools import partial

    from torchvision.models import resnet50
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    from torchvision.ops.misc import FrozenBatchNorm2d

    body = resnet50(weights=None, norm_layer=partial(FrozenBatchNorm2d, eps=0.0))
    for name, p in body.named_parameters():
        if not name.startswith(("layer2.", "layer3.", "layer4.")):
            p.requires_grad_(False)
    backbone = BackboneWithFPN(
        body,
        dict(layer1="0", layer2="1", layer3="2", layer4="3"),
        [256, 512, 1024, 2048],
        256,
    )
    return MaskRCNN(backbone, num_classes=9, min_size=min_size, max_size=max_size)


def settings(a: Any) -> Any:
    """记录影响训练和续训一致性的关键配置。

    Args:
        a: 命令行配置；使用 batch_size, lr, max_size, min_size, no_amp, no_hflip, seed,
        warmup_steps 等参数。

    Returns:
        训练配方字典，供续训时逐字段核对。
    """
    return dict(
        schema=FORMAT,
        batch_size=a.batch_size,
        seed=a.seed,
        lr=a.lr,
        warmup_steps=a.warmup_steps,
        momentum=0.9,
        weight_decay=0.0005,
        min_size=a.min_size,
        max_size=a.max_size,
        hflip=not a.no_hflip,
        amp=not a.no_amp,
        grad_clip=10.0,
        sampling="shuffle_without_replacement",
        backbone="resnet50_fpn_COCO_V1_FrozenBatchNorm_eps0_early_stages_frozen",
    )


def check_resume(
    cp: dict[str, Any], prep: dict[str, Any], recipe: dict[str, Any]
) -> tuple[Any, ...]:
    """校验全量续训数据、配方及图片偏移。

    Args:
        cp: 已加载的全量 checkpoint。
        prep: 当前数据准备凭据，包含索引 SHA-256。
        recipe: 当前训练配方。

    Returns:
        已完成的轮数和下一张图片偏移。偏移为 0 表示上一轮已经完整结束。

    Raises:
        ValueError: 版本、类别 ID、索引、配方或训练进度不匹配。

    Notes:
        仅有 epoch 编号不够判断训练完整；必须同时检查样本偏移和累计图片数量。
    """
    if cp.get("full_dataset_format") != FORMAT or cp.get("class_to_id") != CLASS_TO_ID:
        raise ValueError("checkpoint 版本/八类编号不符。")
    if (
        cp.get("dataset_index_sha256") != prep["index_sha256"]
        or cp.get("settings") != recipe
    ):
        raise ValueError("续跑的数据或参数不一致，请保持原参数或换 --run-name。")
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "rng_state",
        "global_step",
        "epoch_accum",
        "epoch_log",
        "initialization",
    }
    if not required <= cp.keys():
        raise ValueError("checkpoint 缺少完整续跑状态。")
    completed, offset = cp.get("completed_epochs"), cp.get("next_image_offset")
    n = prep["training_images"]
    if (
        type(completed) is not int
        or completed < 0
        or type(offset) is not int
        or not 0 <= offset < n
        or offset % recipe["batch_size"]
    ):
        raise ValueError("checkpoint 图片位置不在合法 batch 边界。")
    if (
        cp["global_step"]
        != completed * math.ceil(n / recipe["batch_size"])
        + offset // recipe["batch_size"]
    ):
        raise ValueError("checkpoint global_step 与图片位置不一致。")
    if [row.get("epoch") for row in cp["epoch_log"]] != list(
        range(1, completed + 1)
    ) or cp["epoch_accum"]["images"] != offset:
        raise ValueError("checkpoint 轮次记录或已处理图片数不完整。")
    if not {"python", "numpy", "torch_cpu", "torch_cuda"} <= cp["rng_state"].keys():
        raise ValueError("checkpoint 缺少随机状态。")
    return completed, offset


def save_torch(torch: Any, payload: Any, path: str | Path) -> None:
    """原子保存训练状态，避免中断留下半个 checkpoint。

    Args:
        torch: 调用方提供的 PyTorch 模块，避免帮助命令触发依赖加载。
        payload: 要保存的权重、优化器和进度状态。
        path: 要读取或写入的文件路径。
    """
    temp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temp)
        with temp.open("rb") as f:
            os.fsync(f.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def request_stop(signum: Any, frame: Any) -> None:
    """记录停止请求，让训练在可保存状态的边界暂停。

    Args:
        signum: 操作系统发出的信号编号。
        frame: 信号触发时的调用帧。
    """
    global STOP
    STOP = True
    log(
        "STOP_REQUESTED",
        signal=signum,
        action="完成当前 batch 后保存，可重复同一命令继续",
    )


def train(
    a: Any, root: Path, data: Any, output: Any, report: Any, prep: dict[str, Any]
) -> Any:
    """从固定八类权重开始全量微调，或从完整训练状态续跑。

    Args:
        a: 训练参数；epochs 是本实验的目标总轮数，不是追加轮数。
        root: 项目根目录。
        data: 外部数据根目录。
        output: 当前版本的索引与权重目录。
        report: 当前版本的进度与汇总目录。
        prep: 已校验的数据准备凭据。

    Returns:
        完成目标轮数时返回 checkpoint_last.pth；响应暂停信号时返回 None。

    Raises:
        RuntimeError: CUDA 不可用，或一轮结束时没有覆盖全部入选图片。
        FileNotFoundError: 初始化权重缺失。
        ValueError: 数据、训练配方、类别或续训状态不一致。
        FloatingPointError: FP32 重试后损失/梯度仍为非有限值。

    Notes:
        每轮无放回遍历，drop_last=False。保存权重、优化器、AMP scaler、随机状态、
        图片偏移和累计统计。只有覆盖本轮全部图片才能发布完整轮 checkpoint。
        AMP 出现异常时对同一批次用 FP32 重试并记数，避免静默跳过训练图片。
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("训练需要原来的 CUDA 环境。")
    # 先核对完整训练状态，再决定续训或复用已完成的轮次。
    recipe = settings(a)
    last = output / "checkpoint_last.pth"
    cp = (
        torch.load(last, map_location="cpu", weights_only=False)
        if last.is_file()
        else None
    )
    completed, offset = check_resume(cp, prep, recipe) if cp is not None else (0, 0)
    if a.epochs < completed + int(offset > 0):
        raise ValueError("--epochs 是总轮数，不能小于已有训练进度。")
    if cp is not None and completed == a.epochs and offset == 0:
        log("REUSE_COMPLETED_TRAINING", epochs=completed)
        return last
    initial = (
        resolve(a.init_checkpoint, root, data) if a.init_checkpoint else root / V5_CP
    )
    if cp is None:
        if not initial.is_file():
            raise FileNotFoundError("缺少初始八类 checkpoint：" + str(initial))
        initial_sha = digest(initial)
        if not a.init_checkpoint and initial_sha != V5_CP_SHA:
            raise ValueError("默认 V5 checkpoint 与较优版本 SHA 不符。")
        warm = torch.load(initial, map_location="cpu", weights_only=False)
        if warm.get("class_to_id") != CLASS_TO_ID:
            raise ValueError("初始 checkpoint 类别编号不一致。")
        initialization = dict(
            path=str(initial), sha256=initial_sha, epoch=warm.get("epoch")
        )
    else:
        initialization = cp["initialization"]
        if a.init_checkpoint and (
            str(initial) != initialization["path"]
            or digest(initial) != initialization["sha256"]
        ):
            raise ValueError("已有训练的初始 checkpoint 不可更换，请新建实验。")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    model = build_model(a.min_size, a.max_size)
    # 严格加载同一标签编号与模型结构，不能把其他版本的头部混进来。
    model.load_state_dict(
        cp["model_state_dict"] if cp is not None else warm["model_state_dict"],
        strict=True,
    )
    model.to("cuda").train()
    if cp is None:
        del warm
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=a.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )
    amp = not a.no_amp
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)

        def autocast() -> Any:
            """Create an AMP context for the current batch.

            Returns:
                Torch's CUDA autocast context with the configured AMP switch.
            """
            return torch.amp.autocast("cuda", enabled=amp)

    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=1024.0)

        def autocast() -> Any:
            """Create the compatibility AMP context for older Torch versions.

            Returns:
                Torch's legacy CUDA autocast context with the same AMP switch.
            """
            return torch.cuda.amp.autocast(enabled=amp)

    step = 0
    history = []
    accum = dict(images=0, batches=0, loss_sum=0.0, fp32_retries=0)
    if cp is not None:
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        scaler.load_state_dict(cp["scaler_state_dict"])
        step = cp["global_step"]
        history = cp["epoch_log"]
        accum = cp["epoch_accum"]
        # 恢复各随机源，保留中断前后的样本顺序与随机训练状态。
        random.setstate(cp["rng_state"]["python"])
        np.random.set_state(cp["rng_state"]["numpy"])
        torch.set_rng_state(cp["rng_state"]["torch_cpu"])
        if len(cp["rng_state"]["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("续跑需要相同数量的可见 GPU。")
        torch.cuda.set_rng_state_all(cp["rng_state"]["torch_cuda"])
        del cp
    dataset = NativeDataset(
        output / "dataset.sqlite", prep["training_images"], a.seed, not a.no_hflip
    )

    def checkpoint(epoch: int, position: Any, stats: Any) -> None:
        """checkpoint。

        Args:
            epoch: 当前训练轮次。
            position: position。
            stats: stats。

        Returns:
            当前条件的校验结果；失败条件及返回形式见函数体。
        """
        payload = dict(
            full_dataset_format=FORMAT,
            epoch=epoch,
            completed_epochs=epoch,
            next_image_offset=position,
            global_step=step,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scaler_state_dict=scaler.state_dict(),
            class_to_id=CLASS_TO_ID,
            num_classes=9,
            args=vars(a),
            settings=recipe,
            dataset_index_sha256=prep["index_sha256"],
            train_manifest=str(output / "dataset.sqlite"),
            train_manifest_sha256=prep["index_sha256"],
            val_manifest=str(root / DEV),
            val_manifest_sha256=FROZEN_SHA[DEV],
            initialization=initialization,
            epoch_accum=dict(stats),
            epoch_log=list(history),
            rng_state=dict(
                python=random.getstate(),
                numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all(),
            ),
        )
        save_torch(torch, payload, last)
        write_json(
            report / "training_progress.json",
            dict(
                status="TRAINING",
                completed_epochs=epoch,
                next_image_offset=position,
                global_step=step,
                total_images=len(dataset),
                target_epochs=a.epochs,
                checkpoint=str(last),
            ),
        )

    if not last.exists():
        checkpoint(0, 0, accum)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log(
        "TRAIN_START",
        gpu=torch.cuda.get_device_name(0),
        images=len(dataset),
        epochs=a.epochs,
        completed_epochs=completed,
        next_image=offset,
        settings=recipe,
        scope=prep["scope"],
    )
    started = time.monotonic()
    session_images = 0
    for epoch in range(completed, a.epochs):
        dataset.epoch = epoch
        begin = offset if epoch == completed else 0
        position = begin
        generator = torch.Generator().manual_seed(a.seed + epoch + 917)
        loader = DataLoader(
            dataset,
            batch_size=a.batch_size,
            sampler=epoch_order(len(dataset), a.seed, epoch)[begin:],
            drop_last=False,
            num_workers=a.workers,
            collate_fn=collate,
            pin_memory=True,
            generator=generator,
            persistent_workers=False,
        )
        for images, targets in loader:
            count = len(images)
            images = [x.to("cuda", non_blocking=True) for x in images]
            targets = [
                {k: v.to("cuda", non_blocking=True) for k, v in target.items()}
                for target in targets
            ]
            factor = (
                min(1.0, (step + 1) / max(1, a.warmup_steps)) if a.warmup_steps else 1.0
            )
            for group in optimizer.param_groups:
                group["lr"] = a.lr * factor
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss = sum(model(images, targets).values())
            # 不跳过异常批次；AMP 失败时对同一批图片做 FP32 重试。
            retry = not bool(torch.isfinite(loss).item())
            if not retry:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                retry = not bool(torch.isfinite(norm).item())
                if retry and amp:
                    scaler.update()
            if retry:
                if not amp:
                    raise FloatingPointError(
                        "FP32 loss/gradient 非有限，最近正常 checkpoint 已保留。"
                    )
                optimizer.zero_grad(set_to_none=True)
                loss = sum(model(images, targets).values())
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("同一 batch FP32 重试仍出现非有限 loss。")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 10.0, error_if_nonfinite=True
                )
                optimizer.step()
                accum["fp32_retries"] += 1
            else:
                scaler.step(optimizer)
                scaler.update()
            step += 1
            position += count
            session_images += count
            accum["images"] += count
            accum["batches"] += 1
            accum["loss_sum"] += float(loss.detach().item())
            if step % a.log_every == 0 or position == len(dataset):
                elapsed = time.monotonic() - started
                remaining = (
                    (a.epochs - epoch - 1) * len(dataset) + len(dataset) - position
                )
                log(
                    "TRAIN_PROGRESS",
                    epoch=epoch + 1,
                    image=position,
                    total=len(dataset),
                    global_step=step,
                    loss=round(float(loss.detach().item()), 5),
                    lr=optimizer.param_groups[0]["lr"],
                    elapsed_minutes=round(elapsed / 60, 1),
                    eta_minutes=round(
                        remaining * elapsed / max(1, session_images) / 60, 1
                    ),
                )
            if position < len(dataset) and (step % a.save_every_steps == 0 or STOP):
                checkpoint(epoch, position, accum)
                log("CHECKPOINT_SAVED", epoch=epoch + 1, next_image=position)
            if STOP and position < len(dataset):
                log("PAUSED", checkpoint=str(last))
                return None
        # 只有覆盖整轮全部图片才发布完整轮 checkpoint。
        if position != len(dataset) or accum["images"] != len(dataset):
            raise RuntimeError("本轮未覆盖全部训练图片，不能发布完整轮 checkpoint。")
        history.append(
            dict(
                epoch=epoch + 1,
                images=accum["images"],
                batches=accum["batches"],
                mean_loss=accum["loss_sum"] / accum["batches"],
                fp32_retries=accum["fp32_retries"],
            )
        )
        accum = dict(images=0, batches=0, loss_sum=0.0, fp32_retries=0)
        checkpoint(epoch + 1, 0, accum)
        epoch_path = output / f"checkpoint_epoch_{epoch+1:02d}.pth"
        if not epoch_path.exists():
            os.link(last, epoch_path)
        log("EPOCH_DONE", **history[-1])
        offset = 0
        if STOP:
            log("PAUSED", checkpoint=str(last))
            return None
    write_json(
        report / "training_summary.json",
        dict(
            status="COMPLETED",
            epochs=a.epochs,
            checkpoint=str(last),
            checkpoint_sha256=digest(last),
            settings=recipe,
            dataset_index_sha256=prep["index_sha256"],
            scope=prep["scope"],
            initialization=initialization,
            epoch_log=history,
        ),
    )
    log("TRAIN_DONE", checkpoint=str(last))
    return last


def evaluate(
    a: Any, root: Path, data: Any, output: Any, report: Any, core: bool = False
) -> None:
    """按同一口径比较 V5 和本次全量权重。

    Args:
        a: 当前参数，包含需达到的目标总轮数。
        root: 项目根目录。
        data: 外部数据根目录。
        output: 当前全量 checkpoint 所在目录。
        report: 本次对照评估报告目录。
        core: 是否增加 Core400 固定回归；False 时只运行开发集。

    Raises:
        FileNotFoundError: 全量权重尚未生成。
        ValueError: 权重未完成目标轮数、V5/运行时身份不符，或已有结果属于其他输入。

    Notes:
        固定 score=0.4、mask=0.5、bbox IoU=0.5，预热 10 次、计时 5 遍。
        模型/清单/runtime 身份写入凭据，不覆盖不同输入的旧报告。
        Core400 已反复使用，因此结果是固定回归，正式盲测验收仍需独立数据。
    """
    import gc

    import torch

    last = output / "checkpoint_last.pth"
    if not last.is_file():
        raise FileNotFoundError("先完成训练。")
    cp = torch.load(last, map_location="cpu", weights_only=False)
    prep = json.loads((report / "data_preparation.json").read_text(encoding="utf-8"))
    completed, offset = check_resume(cp, prep, settings(a))
    if offset or completed != a.epochs:
        raise ValueError("checkpoint 尚未完成要求的全部轮数。")
    del cp
    gc.collect()
    torch.cuda.empty_cache()
    runtime = root / "src/fashion_multimodal_analysis/evaluation/runtime.py"
    if not runtime.is_file() or digest(runtime) != FAST_RUNTIME_SHA:
        raise ValueError("评估 runtime 与已验证的 V5 加速版本不同，停止以保持口径。")
    if digest(root / V5_CP) != V5_CP_SHA:
        raise ValueError("对照 V5 checkpoint SHA 不符。")
    env = dict(os.environ, FASHION_PROJECT_ROOT=str(root), FASHION_DATA_ROOT=str(data))
    env["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    splits = [("development", DEV)] + ([("core400_regression", CORE)] if core else [])
    for split, manifest in splits:
        for label, checkpoint in [("v5", root / V5_CP), ("v8", last)]:
            destination = report / f"eval_{split}_{label}_epoch_{completed:02d}"
            receipt = destination / "evaluation_receipt.json"
            signature = dict(
                checkpoint_sha256=digest(checkpoint),
                manifest_sha256=digest(root / manifest),
                runtime_sha256=FAST_RUNTIME_SHA,
                score_threshold=0.4,
                mask_threshold=0.5,
                match_bbox_iou=0.5,
                warmup_runs=10,
                timed_passes=5,
            )
            # 相同权重、清单和 runtime 才能复用结果，其他输入使用新报告目录。
            if receipt.is_file():
                if json.loads(receipt.read_text(encoding="utf-8")) != signature:
                    raise ValueError(
                        "已有评估结果对应不同输入，不能覆盖：" + str(destination)
                    )
                log("REUSE_EVALUATION", split=split, model=label)
                continue
            destination.mkdir(parents=True, exist_ok=True)
            log("EVALUATE", split=split, model=label)
            command = [
                sys.executable,
                "-u",
                "-m",
                "fashion_multimodal_analysis.segmentation.evaluation.eval_prd_8class_maskrcnn_v3_b1_dataexp",
                "--checkpoint",
                str(checkpoint),
                "--val-csv",
                str(root / manifest),
                "--report-dir",
                str(destination),
                "--device",
                "cuda",
                "--score-threshold",
                "0.4",
                "--mask-threshold",
                "0.5",
                "--match-bbox-iou",
                "0.5",
                "--warmup-runs",
                "10",
                "--timed-passes",
                "5",
            ]
            subprocess.run(command, cwd=root, env=env, check=True)
            write_json(receipt, signature)
    log("EVALUATION_DONE", note="Core400 为固定回归集；是否达标按真实结果判断。")


@contextmanager
def workflow_lock(root: Path) -> Iterator[Any]:
    """对工作副本加互斥锁，防止两个进程同时修改同一实验。

    Args:
        root: 当前项目根目录。

    Yields:
        按需产生的记录/任务；不一次性加载全部条目。

    Raises:
        RuntimeError: 已有 PRD31 任务运行，先查看原任务日志。
    """
    import fcntl

    path = root / "reports/reruns/recovery_v2/workflow.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("已有 PRD31 任务运行，先查看原任务日志。") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def package_results(root: Path, report: Any) -> Any:
    """打包已生成的结果表和状态记录，供离线复核。

    Args:
        root: 当前项目根目录。
        report: 当前版本的报告目录或报告对象。

    Returns:
        返回 archive，由函数体中同名变量的计算/收集过程得到。
    """
    archive = root / ("prd31_" + report.name + "_results.zip")
    temp = archive.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(report.rglob("*")):
            if path.is_file() and path.suffix in {".csv", ".json", ".txt"}:
                z.write(path, path.relative_to(root))
    temp.replace(archive)
    return archive


def parse_args() -> Any:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--phase",
        choices=["prepare", "all", "train", "evaluate", "core"],
        default="all",
        help="all=准备+训练+开发/Core400对照；evaluate=开发；core=开发和固定Core回归",
    )
    p.add_argument("--project", default="")
    p.add_argument(
        "--data-root", default=os.environ.get("FASHION_DATA_ROOT", "../fashion_data")
    )
    p.add_argument("--df2-train", default="")
    p.add_argument(
        "--fp-mode", choices=["auto", "native-train", "existing-train"], default="auto"
    )
    p.add_argument("--fp-annotations", default="")
    p.add_argument("--fp-images", default="")
    p.add_argument(
        "--allow-bad-images",
        action="store_true",
        help="显式记录并排除无效整张图；默认原始数据无效时不训练",
    )
    p.add_argument("--run-name", default="v8_full_dataset_recovery_20261008")
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--epochs", type=int, default=1, help="总轮数，续跑不会从头重来")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--prepare-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--min-size", type=int, default=640)
    p.add_argument("--max-size", type=int, default=1024)
    p.add_argument("--no-hflip", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--save-every-steps", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    a = p.parse_args()
    if (
        min(
            a.epochs,
            a.batch_size,
            a.prepare_workers,
            a.min_size,
            a.max_size,
            a.save_every_steps,
            a.log_every,
        )
        < 1
        or a.workers < 0
        or a.warmup_steps < 0
        or not math.isfinite(a.lr)
        or a.lr <= 0
        or a.max_size < a.min_size
    ):
        p.error("轮数/batch/尺寸/保存和日志步長/lr 必须为正；workers/warmup 不能为负。")
    if not re.fullmatch(r"v8[a-z0-9_]*", a.run_name):
        p.error("--run-name 应如 v8_full_dataset_2。")
    return a


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        FileNotFoundError: 需要的文件不存在。
        RuntimeError: 当前 Python 没有可用 CUDA，请进入原来的 py312 训练环境。
        ValueError: V5 初始 checkpoint SHA 不符。
    """
    a = parse_args()
    from fashion_multimodal_analysis.common.paths import project_root

    root = Path(a.project).expanduser().resolve() if a.project else project_root()
    if (
        not (root / "src/fashion_multimodal_analysis").is_dir()
        or not (root / "configs").is_dir()
    ):
        raise FileNotFoundError(
            "找不到项目，请先进入项目根目录，或用 --project 指定相对目录。"
        )
    data = resolve(a.data_root, root, root.parent / "fashion_data")
    a.data_root = str(data)
    output = (
        root / "outputs/prd_instance_segmentation" / ("maskrcnn_8class_" + a.run_name)
    )
    report = root / "reports/reruns/recovery_v2" / a.run_name
    os.environ["FASHION_PROJECT_ROOT"] = str(root)
    os.environ["FASHION_DATA_ROOT"] = str(data)
    sys.path.insert(0, str(root / "src"))
    with workflow_lock(root):
        output.mkdir(parents=True, exist_ok=True)
        report.mkdir(parents=True, exist_ok=True)
        status = report / "workflow_status.json"
        try:
            log("START", phase=a.phase, project=str(root), data=str(data))
            write_json(status, dict(status="RUNNING", phase=a.phase))
            if a.phase in ("all", "train"):
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "当前 Python 没有可用 CUDA，请进入原来的 py312 训练环境。"
                    )
                initial = (
                    resolve(a.init_checkpoint, root, data)
                    if a.init_checkpoint
                    else root / V5_CP
                )
                if not (output / "checkpoint_last.pth").is_file():
                    if not initial.is_file():
                        raise FileNotFoundError("缺少初始 checkpoint：" + str(initial))
                    if not a.init_checkpoint and digest(initial) != V5_CP_SHA:
                        raise ValueError("V5 初始 checkpoint SHA 不符。")
            protected = protected_inventory(root, data)
            prep = prepare(a, root, data, output, report, protected)
            if a.phase == "prepare":
                result = "PREPARED"
            elif a.phase in ("all", "train"):
                checkpoint = train(a, root, data, output, report, prep)
                if checkpoint is None:
                    result = "PAUSED"
                else:
                    result = "TRAINING_COMPLETED"
                    if a.phase == "all":
                        evaluate(a, root, data, output, report, core=True)
                        result = "TRAINING_AND_REGRESSION_COMPLETED"
            else:
                evaluate(a, root, data, output, report, core=a.phase == "core")
                result = "EVALUATION_COMPLETED"
            write_json(
                status,
                dict(
                    status=result,
                    phase=a.phase,
                    scope=prep["scope"],
                    training_images=prep["training_images"],
                    training_instances=prep["training_instances"],
                    checkpoint=str(output / "checkpoint_last.pth"),
                    formal_acceptance="NOT_CLAIMED",
                ),
            )
            log(
                result,
                results_zip=str(package_results(root, report)),
                report_dir=str(report),
            )
        except Exception as e:
            write_json(
                status,
                dict(status="FAILED", phase=a.phase, error=f"{type(e).__name__}: {e}"),
            )
            (report / "failure.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            package_results(root, report)
            raise


if __name__ == "__main__":
    main()
