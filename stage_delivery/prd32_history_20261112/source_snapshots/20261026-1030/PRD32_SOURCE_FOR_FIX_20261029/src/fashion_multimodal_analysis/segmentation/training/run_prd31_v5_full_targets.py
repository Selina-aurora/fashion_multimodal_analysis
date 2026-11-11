#!/usr/bin/env python3
"""3.1.1 实例分割：数据与模型版本分别记录，指标按固定匹配和计时口径解释。

Complete annotations on existing training images, train v5, compare on development
images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from zipfile import ZIP_DEFLATED, ZipFile

from fashion_multimodal_analysis.analysis import diagnose_prd31_v4 as diag

FIELDS = (
    "record_id",
    "source_dataset",
    "source_image",
    "garment_id",
    "garment_category",
    "fine_category",
    "source_category_id",
    "annotation_ref",
    "mask_path",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width",
    "bbox_height",
    "image_width",
    "image_height",
    "source_manifest",
)


def same_or_write(path: str | Path, content: Any) -> None:
    """新建文件或确认已有内容相同，拒绝覆盖不同的历史产物。

    Args:
        path: 要读取或写入的文件路径。
        content: 待发布的完整文件内容。

    Raises:
        FileExistsError: 目标位置已有不同内容，不能直接覆盖。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(
                "Existing v5 artifact differs; preserve it and select a new version: "
                + str(path)
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def csv_bytes(rows: list[dict[str, Any]]) -> Any:
    """将完整字段表转换为带 BOM 的 CSV 字节，用于一致性比较和发布。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + text.getvalue()).encode("utf-8")


def decode_mask(annotation: dict[str, Any], width: int, height: int) -> Any:
    """解码原始分割标注，保留同一实例的全部前景区域。

    Args:
        annotation: 记录字段，使用 segmentation。
        width: 图像或目标表示的宽度。
        height: 图像或目标表示的高度。

    Returns:
        返回 image，由函数体中同名变量的计算/收集过程得到。

    Raises:
        ValueError: Decoded mask is empty or has unexpected dimensions
    """
    from PIL import Image, ImageDraw

    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for polygon in segmentation:
            if len(polygon) < 6 or len(polygon) % 2:
                raise ValueError("Invalid segmentation polygon")
            draw.polygon(
                [
                    (float(polygon[i]), float(polygon[i + 1]))
                    for i in range(0, len(polygon), 2)
                ],
                fill=255,
            )
    elif isinstance(segmentation, dict):
        import numpy as np
        from pycocotools import mask as coco_mask

        rle = (
            coco_mask.frPyObjects(segmentation, height, width)
            if isinstance(segmentation["counts"], list)
            else dict(segmentation)
        )
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        array = coco_mask.decode(rle)
        if array.ndim == 3:
            array = np.any(array > 0, axis=2)
        image = Image.fromarray((array > 0).astype("uint8") * 255)
    else:
        raise ValueError("Unsupported segmentation format")
    if image.size != (width, height) or image.getbbox() is None:
        raise ValueError("Decoded mask is empty or has unexpected dimensions")
    return image


def load_fashionpedia(data: Path) -> tuple[Any, ...]:
    """建立 Fashionpedia 图片与完整实例标注的连接索引。

    Args:
        data: 外部服饰数据根目录。

    Returns:
        按顺序返回 index 等结果。

    Raises:
        ValueError: Duplicate Fashionpedia image basenames
    """
    path = data / "raw/fashionpedia/annotations/instances_attributes_val2020.json"
    source = json.loads(path.read_text(encoding="utf-8-sig"))
    names = {
        str(r["id"]): " ".join(r["name"].strip().lower().split())
        for r in source["categories"]
    }
    images = {str(r["id"]): r for r in source["images"]}
    index = {Path(r["file_name"]).name: (r, []) for r in source["images"]}
    if len(index) != len(images):
        raise ValueError("Duplicate Fashionpedia image basenames")
    for annotation in source["annotations"]:
        name = names[str(annotation["category_id"])]
        category = diag.FP_CLASS.get(name)
        if category and diag.usable_annotation(annotation, "Fashionpedia"):
            index[Path(images[str(annotation["image_id"])]["file_name"]).name][
                1
            ].append((str(annotation["id"]), category, name, annotation))
    return index, diag.digest(path)


def expand_split(
    rows: list[dict[str, Any]], split: Any, root: Path, data: Path, fp_index: Any
) -> Any:
    """补齐同一批原图的全部可用目标，不改变原图划分。

    Args:
        rows: 待处理的逐行记录。
        split: 本次处理的数据划分名称。
        root: 当前项目根目录。
        data: 外部服饰数据根目录。
        fp_index: Fashionpedia index。

    Returns:
        返回 output，由函数体中同名变量的计算/收集过程得到。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    from PIL import Image

    groups = defaultdict(list)
    for row in rows:
        groups[(row["source_dataset"], row["source_image"])].append(row)
    output = []
    for index, ((dataset, source_image), selected) in enumerate(
        sorted(groups.items()), 1
    ):
        image_path = diag.resolve(source_image, root, data)
        with Image.open(image_path) as image:
            width, height = image.size
        if dataset == "Fashionpedia":
            info, items = fp_index[image_path.name]
            if (int(info["width"]), int(info["height"])) != (width, height):
                raise ValueError(
                    "Image dimensions differ from Fashionpedia raw annotation"
                )
        elif dataset == "DeepFashion2":
            annotation_file = (
                image_path.parent.parent / "annos" / (image_path.stem + ".json")
            )
            source = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
            items = [
                (
                    key,
                    diag.DF2_CLASS[int(ann["category_id"])],
                    diag.DF2_FINE[int(ann["category_id"]) - 1],
                    ann,
                )
                for key, ann in source.items()
                if isinstance(ann, dict)
                and key.startswith("item")
                and int(ann.get("category_id", -1)) in diag.DF2_CLASS
                and diag.usable_annotation(ann, dataset)
            ]
        else:
            raise ValueError("Unsupported dataset: " + dataset)
        identities = {key: category for key, category, _, _ in items}
        for row in selected:
            if identities.get(diag.annotation_id(row)) != row["garment_category"]:
                raise ValueError(
                    "Original selected target not present in usable raw annotations: "
                    + source_image
                )
        if not items:
            raise ValueError("No usable targets on selected image: " + source_image)
        for key, category, fine, annotation in sorted(items):
            mask = decode_mask(annotation, width, height)
            # Train boxes follow the full decoded binary mask, including every disconnected component.
            x1, y1, x2, y2 = mask.getbbox()
            name = (
                "fp_" + key
                if dataset == "Fashionpedia"
                else "df2_" + image_path.stem + "_" + key
            ) + ".png"
            mask_relative = Path("processed/prd8_full_targets_v5") / split / name
            stream = io.BytesIO()
            mask.save(stream, format="PNG")
            same_or_write(data / mask_relative, stream.getvalue())
            output.append(
                dict(
                    record_id=f"v5_{split}_{len(output)+1:05d}",
                    source_dataset=dataset,
                    source_image=source_image,
                    garment_id=(
                        "fashionpedia_ann_" + key
                        if dataset == "Fashionpedia"
                        else key + "_" + fine
                    ),
                    garment_category=category,
                    fine_category=fine,
                    source_category_id=annotation["category_id"],
                    annotation_ref=key,
                    mask_path="../fashion_data/" + mask_relative.as_posix(),
                    bbox_x1=x1,
                    bbox_y1=y1,
                    bbox_x2=x2,
                    bbox_y2=y2,
                    bbox_width=x2 - x1,
                    bbox_height=y2 - y1,
                    image_width=width,
                    image_height=height,
                    source_manifest="raw_complete_targets_v5",
                )
            )
        if index % 50 == 0 or index == len(groups):
            print(
                f"{split}: {index}/{len(groups)} images, {len(output)} complete targets",
                flush=True,
            )
    return output


def prepare(root: Path, data: Any) -> Any:
    """准备当前实验所需数据，并执行来源/划分/有效性检查。

    Args:
        root: 当前项目根目录。
        data: 外部数据目录或本函数处理的数据结构；具体用途由操作对象决定。

    Returns:
        本版本的数据准备凭据，包含清单/索引身份、有效样本数和排除记录。

    Raises:
        ValueError: Development source images changed
    """
    train_source = root / "configs/prd_8class_train_v4_clean.csv"
    val_source = root / "configs/prd_8class_val_v1.csv"
    core_source = root / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv"
    train, val, core = map(diag.read_csv, (train_source, val_source, core_source))
    protected = {
        str(diag.resolve(r["source_image"], root, data).resolve()) for r in val + core
    }
    if protected & {
        str(diag.resolve(r["source_image"], root, data).resolve()) for r in train
    }:
        raise ValueError("Training image overlaps validation/Core")
    print("补齐原有训练图片和开发验证图片的原始八类目标……", flush=True)
    fp, raw_hash = load_fashionpedia(data)
    expanded_train = expand_split(train, "train", root, data, fp)
    expanded_val = expand_split(val, "validation", root, data, fp)
    if {r["source_image"] for r in expanded_train} != {
        r["source_image"] for r in train
    }:
        raise ValueError("Training source images changed")
    if {r["source_image"] for r in expanded_val} != {r["source_image"] for r in val}:
        raise ValueError("Development source images changed")
    train_path = root / "configs/prd_8class_train_v5_full_targets.csv"
    val_path = root / "configs/prd_8class_val_v2_full_targets.csv"
    # Both splits finish mask decoding before publishing either manifest.
    same_or_write(train_path, csv_bytes(expanded_train))
    same_or_write(val_path, csv_bytes(expanded_val))
    report = dict(
        status="FULL_TARGET_MANIFESTS_PREPARED",
        original_train_instances=len(train),
        complete_train_instances=len(expanded_train),
        train_images=len({r["source_image"] for r in train}),
        train_classes=dict(Counter(r["garment_category"] for r in expanded_train)),
        original_validation_instances=len(val),
        complete_validation_instances=len(expanded_val),
        validation_images=len({r["source_image"] for r in val}),
        input_sha256={
            "train_v4": diag.digest(train_source),
            "val_v1": diag.digest(val_source),
            "core": diag.digest(core_source),
            "fashionpedia_raw": raw_hash,
        },
        output_sha256={
            "train_v5": diag.digest(train_path),
            "val_v2_full": diag.digest(val_path),
        },
        scope="Same source images, complete usable targets in the fixed eight-class taxonomy. No Core GT changes. Not content-level deduplication.",
    )
    path = root / "reports/reruns/recovery_v2/v5_full_targets/data_preparation.json"
    same_or_write(
        path, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def command(root: Path, module: str, arguments: list[str]) -> None:
    """启动当前项目的子命令，并保留日志与失败状态。

    Args:
        root: 当前项目根目录。
        module: 要执行的 Python 模块名称。
        arguments: 转交给实验入口的命令行参数。
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    argv = [sys.executable, "-u", "-m", module, *arguments]
    print("Running: " + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=root, env=env, check=True)


def evaluate_development(root: Path, checkpoint: Any, name: str) -> None:
    """评估 开发集。

    Args:
        root: 当前项目根目录。
        checkpoint: 待检查或使用的训练权重/状态。
        name: 当前标签、字段或产物名称。

    Raises:
        ValueError: Existing development report has different provenance or is
        incomplete
        FileExistsError: 目标位置已有不同内容，不能直接覆盖。
    """
    folder = root / "reports/reruns/recovery_v2/v5_full_targets" / name
    val = root / "configs/prd_8class_val_v2_full_targets.csv"
    expected = dict(
        checkpoint_sha256=diag.digest(checkpoint),
        val_manifest_sha256=diag.digest(val),
        score_threshold=0.4,
        mask_threshold=0.5,
        match_bbox_iou=0.5,
        warmup_runs=10,
        timed_passes=5,
        split="DEVELOPMENT_FULL_TARGETS_NOT_CORE_ACCEPTANCE",
    )
    provenance = folder / "comparison_provenance.json"
    if provenance.exists():
        if (
            json.loads(provenance.read_text()) != expected
            or not (folder / "evaluation_summary.txt").is_file()
        ):
            raise ValueError(
                "Existing development report has different provenance or is incomplete"
            )
        print("Reusing development comparison: " + name, flush=True)
        return
    if (folder / "per_gt_results.csv").exists():
        raise FileExistsError("Preserve unverified development report: " + str(folder))
    command(
        root,
        "fashion_multimodal_analysis.segmentation.evaluation.eval_prd_8class_maskrcnn_v3_b1_dataexp",
        [
            "--checkpoint",
            str(checkpoint),
            "--val-csv",
            str(val),
            "--report-dir",
            str(folder),
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
        ],
    )
    same_or_write(provenance, (json.dumps(expected, indent=2) + "\n").encode())


@contextmanager
def workflow_lock(root: Path) -> Iterator[Any]:
    """对工作副本加互斥锁，防止两个进程同时修改同一实验。

    Args:
        root: 当前项目根目录。

    Yields:
        按需产生的记录/任务；不一次性加载全部条目。
    """
    path = root / "reports/reruns/recovery_v2/v5_full_targets/workflow.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run(root: Path, data: Any, phase: Any) -> None:
    """组织本模块的准备、执行和结果保存步骤。

    Args:
        root: 当前项目根目录。
        data: 外部数据目录或本函数处理的数据结构；具体用途由操作对象决定。
        phase: phase。

    Raises:
        RuntimeError: CUDA unavailable
    """
    os.environ["FASHION_DATA_ROOT"] = str(data)
    os.environ["FASHION_PROJECT_ROOT"] = str(root)
    sys.path.insert(0, str(root / "src"))
    if phase in ("prepare", "all"):
        # Detect optional RLE dependency before generating any new dataset artifacts.
        import pycocotools.mask

        prepare(root, data)
    if phase == "prepare":
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    from fashion_multimodal_analysis.segmentation.training.checkpoint_state import (
        validate_checkpoint,
    )

    checkpoint = (
        root
        / "outputs/prd_instance_segmentation/maskrcnn_8class_v5_full_targets/checkpoint_last.pth"
    )
    if phase == "all":
        arguments = [
            "--epochs",
            "15",
            "--device",
            "cuda",
            "--train-csv",
            "configs/prd_8class_train_v5_full_targets.csv",
            "--val-csv",
            "configs/prd_8class_val_v1.csv",
            "--output-dir",
            str(checkpoint.parent),
            "--report-dir",
            "reports/reruns/recovery_v2/v5_full_targets/training",
            "--experiment-name",
            "v5-full-targets",
            "--verify-isolation",
        ]
        completed = False
        if checkpoint.exists():
            cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
            resume = cp.get("epoch", 0) < 15
            validate_checkpoint(
                cp,
                diag.digest(root / "configs/prd_8class_train_v5_full_targets.csv"),
                diag.digest(root / "configs/prd_8class_val_v1.csv"),
                for_resume=resume,
            )
            del cp
            if resume:
                arguments += ["--resume", str(checkpoint)]
            else:
                completed = True
                print(
                    "REUSE_V5_TRAINING: completed matching 15-epoch checkpoint",
                    flush=True,
                )
        if not completed:
            command(
                root,
                "fashion_multimodal_analysis.segmentation.training.train_prd_8class_maskrcnn_v3_b1_dataexp",
                arguments,
            )
    v4 = (
        root
        / "outputs/prd_instance_segmentation/maskrcnn_8class_v4_clean/checkpoint_last.pth"
    )
    cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(
        cp,
        diag.digest(root / "configs/prd_8class_train_v5_full_targets.csv"),
        diag.digest(root / "configs/prd_8class_val_v1.csv"),
    )
    del cp
    cp = torch.load(v4, map_location="cpu", weights_only=False)
    validate_checkpoint(
        cp,
        diag.digest(root / "configs/prd_8class_train_v4_clean.csv"),
        diag.digest(root / "configs/prd_8class_val_v1.csv"),
    )
    del cp
    # Candidate selection uses fully annotated development images; Core400 remains the recorded regression.
    evaluate_development(root, v4, "v4_development")
    evaluate_development(root, checkpoint, "v5_development")
    report = root / "reports/reruns/recovery_v2/v5_full_targets"
    archive = root / "prd31_v5_results.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for path in sorted(report.rglob("*")):
            if path.is_file() and path.suffix in (".csv", ".txt", ".json"):
                bundle.write(path, str(path.relative_to(root)))
    print("V5_DEVELOPMENT_COMPARISON_COMPLETED: " + str(archive), flush=True)
    print(
        "比较开发集结果后再决定是否采用 v5；这一步不声明 Core400 或 PRD 验收达标。",
        flush=True,
    )


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".")
    parser.add_argument(
        "--data-root", default=os.environ.get("FASHION_DATA_ROOT", "../fashion_data")
    )
    parser.add_argument("--phase", choices=("prepare", "all", "compare"), default="all")
    args = parser.parse_args()
    root = Path(args.project).expanduser().resolve()
    data = Path(args.data_root).expanduser().resolve()
    try:
        # Share the original workflow's GPU lock so its timed evaluation cannot overlap this run.
        shared = root / "reports/reruns/recovery_v2/workflow.lock"
        shared.parent.mkdir(parents=True, exist_ok=True)
        with shared.open("a+") as handle:
            if os.name == "posix":
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with workflow_lock(root):
                run(root, data, args.phase)
    except (
        OSError,
        ValueError,
        KeyError,
        ImportError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        hint = (
            "先运行 python -m pip install --no-deps pycocotools==2.0.11\n"
            if isinstance(error, ImportError) and "pycocotools" in str(error)
            else ""
        )
        parser.exit(2, f"v5 阶段未完成：{type(error).__name__}: {error}\n" + hint)


if __name__ == "__main__":
    main()
