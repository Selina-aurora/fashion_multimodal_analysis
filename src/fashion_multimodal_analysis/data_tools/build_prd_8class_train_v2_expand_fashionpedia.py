"""数据处理：保存原图来源和实例标识，避免同一图片的不同实例跨训练与评估划分。

Expand the Fashionpedia training portion for PRD 3.1.1 baseline v2.

Goal
----
Keep the existing fixed validation split unchanged, but rebuild the TRAIN split as:

    DeepFashion2: keep the current train rows exactly as they are
    Fashionpedia:
        shoe       -> 100 training instances
        bag        -> 100 training instances
        accessory  -> 100 training instances

The current validation images are excluded from Fashionpedia candidate sampling,
so there is no source-image leakage into the fixed validation set.

Inputs
------
configs/prd_8class_train_v1.csv
configs/prd_8class_val_v1.csv

../fashion_data/raw/fashionpedia/
├── annotations/
│   └── instances_attributes_val2020.json
└── images/
    └── test/
        └── *.jpg

Outputs
-------
configs/prd_8class_train_v2.csv

../fashion_data/processed/fashionpedia_train_expanded_v2/
└── full_masks/
    └── *.png

reports/prd_instance_segmentation/prd_8class_train_expansion_v2/
├── summary.txt
├── class_counts_train_v2.csv
├── selected_fashionpedia_v2.csv
├── skipped_annotations.csv
├── validation_overlap_check.csv
└── contact_sheets/
    ├── shoe.jpg
    ├── bag.jpg
    └── accessory.jpg

Run from project root
---------------------
python scripts/data_tools/build_prd_8class_train_v2_expand_fashionpedia.py

Default:
    100 train instances per Fashionpedia PRD class

Optional:
    python scripts/data_tools/build_prd_8class_train_v2_expand_fashionpedia.py
    --per-class 100         --seed 20260921

Important
---------
- The validation manifest is NOT modified.
- All Fashionpedia validation source images are excluded.
- DeepFashion2 training rows are preserved.
- Raw Fashionpedia data are never modified.
- Newly generated Fashionpedia masks are full-image masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()
DATA_ROOT = data_root()

TRAIN_V1 = PROJECT_ROOT / "configs" / "prd_8class_train_v1.csv"
VAL_V1 = PROJECT_ROOT / "configs" / "prd_8class_val_v1.csv"

RAW_FP_ROOT = DATA_ROOT / "raw" / "fashionpedia"
FP_ANN = RAW_FP_ROOT / "annotations" / "instances_attributes_val2020.json"
FP_IMAGES = RAW_FP_ROOT / "images" / "test"

PROCESSED_ROOT = DATA_ROOT / "processed" / "fashionpedia_train_expanded_v2"
FULL_MASK_DIR = PROCESSED_ROOT / "full_masks"

TRAIN_V2 = PROJECT_ROOT / "configs" / "prd_8class_train_v2.csv"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_instance_segmentation"
    / "prd_8class_train_expansion_v2"
)
SUMMARY_PATH = REPORT_DIR / "summary.txt"
COUNTS_PATH = REPORT_DIR / "class_counts_train_v2.csv"
SELECTED_PATH = REPORT_DIR / "selected_fashionpedia_v2.csv"
SKIPPED_PATH = REPORT_DIR / "skipped_annotations.csv"
OVERLAP_PATH = REPORT_DIR / "validation_overlap_check.csv"
CONTACT_DIR = REPORT_DIR / "contact_sheets"

TARGET_FP_CLASSES = ("shoe", "bag", "accessory")

OUTPUT_FIELDS = [
    "record_id",
    "source_dataset",
    "source_image",
    "garment_id",
    "garment_category",
    "fine_category",
    "source_category_id",
    "annotation_ref",
    "crop_path",
    "mask_path",
    "masked_preview_path",
    "visualization_path",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width",
    "bbox_height",
    "image_width",
    "image_height",
    "source_manifest",
]


def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--per-class",
        type=int,
        default=100,
        help="Number of Fashionpedia TRAIN instances for each PRD class.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260921,
    )
    p.add_argument(
        "--contact-sheet-items",
        type=int,
        default=25,
        help="Representative examples per class in contact sheets.",
    )
    return p.parse_args()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        按顺序返回 rows, fields 等结果。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV:\n  {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    return rows, fields


def write_csv(
    path: Path,
    rows: list[dict],
    fields: list[str] | None = None,
) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
        fields: 输出字段或分组字段的顺序。
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fields is None:
        fields = list(rows[0].keys()) if rows else []

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        if not fields:
            return

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def normalize_name(name: str) -> str:
    """规范化名称，减少空格与大小写导致的标签匹配失败。

    Args:
        name: 当前标签、字段或产物名称。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return " ".join(str(name).strip().lower().replace("&", "and").split())


def map_source_category(name: str) -> str | None:
    """map 来源 类别。

    Args:
        name: 当前标签、字段或产物名称。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    n = normalize_name(name)

    if n == "shoe" or n.endswith(" shoe") or "shoes" in n:
        return "shoe"

    if "bag" in n or "wallet" in n:
        return "bag"

    accessory_keywords = (
        "glasses",
        "hat",
        "headband",
        "head covering",
        "hair accessory",
        "tie",
        "glove",
        "watch",
        "belt",
        "leg warmer",
        "tights",
        "stockings",
        "sock",
        "scarf",
    )

    if any(k in n for k in accessory_keywords):
        return "accessory"

    return None


def relative_to_project(path: Path) -> str:
    """relative 转换 project。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    path = path.resolve()

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # 路径不属于项目时，进入下方数据目录/外部路径分支；这是预期匹配失败。
        pass

    try:
        rel = path.relative_to(PROJECT_ROOT.parent)
        return f"../{rel.as_posix()}"
    except ValueError:
        return path.as_posix()


def polygons_to_mask(
    polygons: list,
    width: int,
    height: int,
) -> np.ndarray:
    """将多边形标注栅格化为二值掩码。

    Args:
        polygons: 多边形。
        width: 图像或目标表示的宽度。
        height: 图像或目标表示的高度。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    canvas = Image.new(
        "L",
        (width, height),
        0,
    )
    draw = ImageDraw.Draw(canvas)

    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue

        pts = []

        for i in range(
            0,
            len(polygon) - 1,
            2,
        ):
            pts.append(
                (
                    float(polygon[i]),
                    float(polygon[i + 1]),
                )
            )

        if len(pts) >= 3:
            draw.polygon(
                pts,
                fill=255,
            )

    return np.asarray(
        canvas,
        dtype=np.uint8,
    )


def rle_to_mask(
    segmentation: dict,
) -> np.ndarray:
    """解码 COCO RLE 掩码，兼容当前标注的计数表示。

    Args:
        segmentation: 原始分割标注，可能为多边形或 COCO RLE。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        RuntimeError: RLE annotation encountered but pycocotools is missing.
    """
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError(
            "RLE annotation encountered but pycocotools is missing.\n"
            + "Install with:\n"
            + "  pip install pycocotools"
        ) from exc

    decoded = mask_utils.decode(segmentation)

    if decoded.ndim == 3:
        decoded = np.any(
            decoded > 0,
            axis=2,
        )

    return (decoded > 0).astype(np.uint8) * 255


def segmentation_to_mask(
    segmentation: Any,
    width: int,
    height: int,
) -> np.ndarray:
    """根据标注格式选择多边形或 RLE 掩码解码。

    Args:
        segmentation: 原始分割标注，可能为多边形或 COCO RLE。
        width: 图像或目标表示的宽度。
        height: 图像或目标表示的高度。

    Returns:
        返回 mask，由函数体中同名变量的计算/收集过程得到。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    if isinstance(
        segmentation,
        list,
    ):
        return polygons_to_mask(
            segmentation,
            width,
            height,
        )

    if isinstance(
        segmentation,
        dict,
    ):
        mask = rle_to_mask(segmentation)

        if mask.shape != (
            height,
            width,
        ):
            raise ValueError(f"RLE shape {mask.shape} " + f"!= {(height, width)}")

        return mask

    raise ValueError(
        "Unsupported segmentation type: " + f"{type(segmentation).__name__}"
    )


def bbox_from_mask(
    mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """从二值掩码提取前景范围，避免框与目标像素不一致。

    Args:
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    ys, xs = np.nonzero(mask > 0)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def build_image_index(
    needed_names: set[str],
) -> dict[str, Path]:
    """按原图文件名建立检索索引，减少逐行扫描目录的开销。

    Args:
        needed_names: needed names。

    Returns:
        返回 index，由函数体中同名变量的计算/收集过程得到。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not FP_IMAGES.is_dir():
        raise FileNotFoundError(
            "Fashionpedia image folder not found:\n" + f"  {FP_IMAGES}"
        )

    index: dict[str, Path] = {}

    for name in needed_names:
        p = FP_IMAGES / name

        if p.is_file():
            index[name] = p.resolve()

    if len(index) < len(needed_names):
        for pattern in (
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.JPG",
            "*.JPEG",
            "*.PNG",
        ):
            for p in FP_IMAGES.rglob(pattern):
                if p.name in needed_names and p.name not in index:
                    index[p.name] = p.resolve()

    return index


def balanced_diverse_sample(
    candidates: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Prefer:
      1) unique source images
      2) balanced rotation across Fashionpedia fine categories

    Args:
        candidates: candidates。
        n: 本步骤使用的原始值，转换/筛选规则见函数体。
        rng: rng。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    by_fine: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    for item in candidates:
        by_fine[item["source_category"]].append(item)

    for pool in by_fine.values():
        rng.shuffle(pool)

    fine_names = list(by_fine.keys())
    rng.shuffle(fine_names)

    selected = []
    selected_ids = set()
    used_images = set()

    # Pass 1: round-robin fine categories + unique image.
    progress = True

    while len(selected) < n and progress:
        progress = False

        for fine in fine_names:
            pool = by_fine[fine]

            chosen_idx = None

            for i, item in enumerate(pool):
                ann_id = int(item["annotation"]["id"])
                image_id = int(item["image_meta"]["id"])

                if ann_id not in selected_ids and image_id not in used_images:
                    chosen_idx = i
                    break

            if chosen_idx is None:
                continue

            item = pool.pop(chosen_idx)

            selected.append(item)

            selected_ids.add(int(item["annotation"]["id"]))
            used_images.add(int(item["image_meta"]["id"]))

            progress = True

            if len(selected) >= n:
                break

    # Pass 2: allow repeated source image if needed.
    if len(selected) < n:
        leftovers = []

        for pool in by_fine.values():
            leftovers.extend(pool)

        rng.shuffle(leftovers)

        for item in leftovers:
            ann_id = int(item["annotation"]["id"])

            if ann_id in selected_ids:
                continue

            selected.append(item)
            selected_ids.add(ann_id)

            if len(selected) >= n:
                break

    return selected[:n]


def make_contact_sheet(
    rows: list[dict],
    class_name: str,
    max_items: int,
) -> None:
    """把样本图与说明拼成审核联系表，便于逐例比较。

    Args:
        rows: 待处理的逐行记录。
        class_name: 类别 name。
        max_items: max items。
    """
    rows = rows[:max_items]

    if not rows:
        return

    thumb_w = 280
    thumb_h = 280
    tile_w = 310
    tile_h = 350
    cols = 5
    rows_n = math.ceil(len(rows) / cols)

    sheet = Image.new(
        "RGB",
        (
            cols * tile_w,
            50 + rows_n * tile_h,
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    draw.text(
        (15, 15),
        ("Fashionpedia train v2 - " + f"{class_name}"),
        fill="black",
        font=font,
    )

    for i, row in enumerate(rows):
        r = i // cols
        c = i % cols

        x0 = c * tile_w
        y0 = 50 + r * tile_h

        image_path = resolve_manifest_path(row["source_image"])

        image = Image.open(image_path).convert("RGB")

        bbox = (
            int(row["bbox_x1"]),
            int(row["bbox_y1"]),
            int(row["bbox_x2"]),
            int(row["bbox_y2"]),
        )

        d = ImageDraw.Draw(image)
        d.rectangle(
            bbox,
            outline=(0, 255, 0),
            width=3,
        )

        image.thumbnail(
            (
                thumb_w,
                thumb_h,
            ),
            Image.Resampling.LANCZOS,
        )

        px = x0 + (tile_w - image.width) // 2

        py = y0 + 4

        sheet.paste(
            image,
            (px, py),
        )

        text_y = y0 + 290

        for text in (
            (f"{row['garment_category']} " + f"<- {row['fine_category']}"),
            (f"ann=" + f"{row['annotation_ref']}"),
        ):
            draw.text(
                (
                    x0 + 7,
                    text_y,
                ),
                text,
                fill="black",
                font=font,
            )
            text_y += 18

    CONTACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        CONTACT_DIR / f"{class_name}.jpg",
        quality=92,
    )


def resolve_manifest_path(
    value: str,
) -> Path:
    """定位清单文件，避免依赖脚本所在目录。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        当前工作副本可使用的路径。
    """
    p = Path(str(value).strip())

    if p.is_absolute():
        return p

    return (PROJECT_ROOT / p).resolve()


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        FileNotFoundError: 需要的文件不存在。
        RuntimeError: 当前运行条件不满足实验要求。
        ValueError: empty_mask
    """
    args = parse_args()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    FULL_MASK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_v1_rows, fields = read_csv(TRAIN_V1)

    val_rows, _ = read_csv(VAL_V1)

    if fields != OUTPUT_FIELDS:
        # Preserve current manifest schema even if field order differs.
        output_fields = fields
    else:
        output_fields = OUTPUT_FIELDS

    deep_train_rows = [
        row for row in train_v1_rows if row["source_dataset"].strip() == "DeepFashion2"
    ]

    old_fp_train_rows = [
        row for row in train_v1_rows if row["source_dataset"].strip() == "Fashionpedia"
    ]

    # Freeze validation source images. Use basename for Fashionpedia
    # because annotation file_name stores the image filename.
    val_fp_names = {
        Path(row["source_image"]).name
        for row in val_rows
        if row["source_dataset"].strip() == "Fashionpedia"
    }

    print(
        "DeepFashion2 train rows kept:",
        len(deep_train_rows),
    )
    print(
        "Old Fashionpedia train rows:",
        len(old_fp_train_rows),
    )
    print(
        "Frozen Fashionpedia val image names:",
        len(val_fp_names),
    )

    if not FP_ANN.is_file():
        raise FileNotFoundError("Fashionpedia annotation not found:\n" + f"  {FP_ANN}")

    data = json.loads(FP_ANN.read_text(encoding="utf-8"))

    category_by_id = {int(c["id"]): str(c["name"]) for c in data["categories"]}

    image_by_id = {int(i["id"]): i for i in data["images"]}

    candidates: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    for ann in data["annotations"]:
        source_cat = category_by_id.get(
            int(ann["category_id"]),
            "",
        )

        prd_class = map_source_category(source_cat)

        if prd_class not in TARGET_FP_CLASSES:
            continue

        image_meta = image_by_id.get(int(ann["image_id"]))

        if image_meta is None:
            continue

        file_name = str(image_meta["file_name"])

        # Critical: fixed validation image exclusion.
        if Path(file_name).name in val_fp_names:
            continue

        candidates[prd_class].append(
            {
                "annotation": ann,
                "image_meta": image_meta,
                "source_category": source_cat,
                "prd_class": prd_class,
            }
        )

    print()
    print("Available candidates after validation exclusion:")

    for cls in TARGET_FP_CLASSES:
        print(f"  {cls}: " + f"{len(candidates[cls])}")

        if len(candidates[cls]) < args.per_class:
            raise RuntimeError(
                f"Not enough candidates for {cls}: "
                + f"need {args.per_class}, "
                + f"found {len(candidates[cls])}"
            )

    rng = random.Random(args.seed)

    selected = []

    for cls in TARGET_FP_CLASSES:
        cls_selected = balanced_diverse_sample(
            candidates[cls],
            args.per_class,
            rng,
        )

        print(f"Selected {cls}: " + f"{len(cls_selected)}")

        selected.extend(cls_selected)

    needed_names = {str(item["image_meta"]["file_name"]) for item in selected}

    image_index = build_image_index(needed_names)

    print(f"\nImage files found: " + f"{len(image_index)}/" + f"{len(needed_names)}")

    fp_rows = []
    skipped = []

    for seq, item in enumerate(
        selected,
        start=1,
    ):
        ann = item["annotation"]
        image_meta = item["image_meta"]
        prd_class = item["prd_class"]
        source_cat = item["source_category"]

        file_name = str(image_meta["file_name"])

        image_path = image_index.get(file_name)

        if image_path is None:
            skipped.append(
                {
                    "annotation_id": ann["id"],
                    "image_id": image_meta["id"],
                    "file_name": file_name,
                    "reason": "image_not_found",
                }
            )
            continue

        try:
            image = Image.open(image_path).convert("RGB")

            width, height = image.size

            mask = segmentation_to_mask(
                ann.get("segmentation"),
                width,
                height,
            )

            bbox = bbox_from_mask(mask)

            if bbox is None:
                raise ValueError("empty_mask")

        except Exception as exc:
            skipped.append(
                {
                    "annotation_id": ann["id"],
                    "image_id": image_meta["id"],
                    "file_name": file_name,
                    "reason": str(exc),
                }
            )
            continue

        x1, y1, x2, y2 = bbox

        ann_id = int(ann["id"])
        image_id = int(image_meta["id"])

        stem = f"{prd_class}_" + f"ann{ann_id}_" + f"img{image_id}"

        mask_path = FULL_MASK_DIR / f"{stem}.png"

        Image.fromarray(mask).save(mask_path)

        fp_rows.append(
            {
                "record_id": (f"prd8v2_fp_" + f"{seq:04d}"),
                "source_dataset": "Fashionpedia",
                "source_image": relative_to_project(image_path),
                "garment_id": (f"fashionpedia_ann_" + f"{ann_id}"),
                "garment_category": prd_class,
                "fine_category": source_cat,
                "source_category_id": str(ann["category_id"]),
                "annotation_ref": str(ann_id),
                "crop_path": "",
                "mask_path": relative_to_project(mask_path),
                "masked_preview_path": "",
                "visualization_path": "",
                "bbox_x1": str(x1),
                "bbox_y1": str(y1),
                "bbox_x2": str(x2),
                "bbox_y2": str(y2),
                "bbox_width": str(x2 - x1),
                "bbox_height": str(y2 - y1),
                "image_width": str(width),
                "image_height": str(height),
                "source_manifest": (
                    "Fashionpedia "
                    + "instances_attributes_val2020.json "
                    + "expanded-train-v2"
                ),
            }
        )

    if skipped:
        print("\nWARNING: " + f"{len(skipped)} selected annotations skipped.")

    # Require exact target counts after decoding.
    fp_counts = Counter(row["garment_category"] for row in fp_rows)

    for cls in TARGET_FP_CLASSES:
        if fp_counts[cls] != args.per_class:
            raise RuntimeError(
                f"After mask decoding, {cls} has "
                + f"{fp_counts[cls]} rows; "
                + f"expected {args.per_class}. "
                + "Inspect skipped_annotations.csv and rerun after fixing."
            )

    train_v2 = deep_train_rows + fp_rows

    # Final overlap validation.
    val_keys = {
        (
            row["source_dataset"].strip(),
            Path(row["source_image"]).name,
        )
        for row in val_rows
    }

    overlap_rows = []

    for row in train_v2:
        key = (
            row["source_dataset"].strip(),
            Path(row["source_image"]).name,
        )

        if key in val_keys:
            overlap_rows.append(
                {
                    "source_dataset": key[0],
                    "source_image_basename": key[1],
                    "record_id": row.get(
                        "record_id",
                        "",
                    ),
                }
            )

    if overlap_rows:
        write_csv(
            OVERLAP_PATH,
            overlap_rows,
        )

        raise RuntimeError(
            "Train/validation source-image overlap detected. " + f"See {OVERLAP_PATH}"
        )

    # Re-number record ids globally for clarity, while preserving source-specific
    # provenance in garment_id/source_manifest.
    final_rows = []

    for i, row in enumerate(
        train_v2,
        start=1,
    ):
        new_row = dict(row)

        new_row["record_id"] = f"prd8v2_{i:04d}"

        final_rows.append(new_row)

    write_csv(
        TRAIN_V2,
        final_rows,
        output_fields,
    )

    write_csv(
        SELECTED_PATH,
        fp_rows,
        OUTPUT_FIELDS,
    )

    write_csv(
        SKIPPED_PATH,
        skipped,
        [
            "annotation_id",
            "image_id",
            "file_name",
            "reason",
        ],
    )

    write_csv(
        OVERLAP_PATH,
        [],
        [
            "source_dataset",
            "source_image_basename",
            "record_id",
        ],
    )

    counts = Counter(row["garment_category"] for row in final_rows)

    count_rows = [
        {
            "garment_category": cls,
            "count": counts.get(
                cls,
                0,
            ),
        }
        for cls in (
            "top",
            "pants",
            "skirt",
            "outerwear",
            "dress",
            "shoe",
            "bag",
            "accessory",
        )
    ]

    write_csv(
        COUNTS_PATH,
        count_rows,
        [
            "garment_category",
            "count",
        ],
    )

    # Representative contact sheets: one row per source image preferred.
    for cls in TARGET_FP_CLASSES:
        cls_rows = [row for row in fp_rows if row["garment_category"] == cls]

        rng.shuffle(cls_rows)

        make_contact_sheet(
            cls_rows,
            cls,
            max_items=args.contact_sheet_items,
        )

    fine_counts = Counter(
        (
            row["garment_category"],
            row["fine_category"],
        )
        for row in fp_rows
    )

    summary = [
        "PRD 3.1.1 train v2 - Fashionpedia expansion",
        "============================================",
        "",
        f"seed={args.seed}",
        f"target_fashionpedia_per_class={args.per_class}",
        "",
        "Fixed validation",
        "----------------",
        f"validation_rows={len(val_rows)}",
        f"frozen_fashionpedia_val_images={len(val_fp_names)}",
        "validation_manifest_modified=False",
        "train_val_source_image_overlap=0",
        "",
        "Training rows",
        "-------------",
        f"deepfashion2_train_rows={len(deep_train_rows)}",
        f"fashionpedia_train_rows={len(fp_rows)}",
        f"total_train_rows={len(final_rows)}",
        "",
        "Train class counts",
        "------------------",
    ]

    for cls in (
        "top",
        "pants",
        "skirt",
        "outerwear",
        "dress",
        "shoe",
        "bag",
        "accessory",
    ):
        summary.append(f"{cls}=" + f"{counts.get(cls, 0)}")

    summary += [
        "",
        "Fashionpedia fine-category counts",
        "--------------------------------",
    ]

    for (
        prd_cls,
        fine_cls,
    ), count in sorted(fine_counts.items()):
        summary.append(f"{prd_cls} <- " + f"{fine_cls}: " + f"{count}")

    summary += [
        "",
        "Notes",
        "-----",
        "- The original fixed validation manifest is unchanged.",
        "- Fashionpedia candidates from validation source images were excluded before sampling.",
        "- DeepFashion2 training rows were preserved from train_v1.",
        "- Fashionpedia train masks were regenerated as full-image masks.",
        "- train_v2 is intended for baseline-v2 training; compare against v1 on the same fixed validation set.",
        "- Because the class distribution is now intentionally changed, use a class-aware/balanced training strategy in the next training script rather than interpreting raw epoch counts as directly comparable.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== TRAIN V2 BUILD FINISHED ===")
    print(
        "Output:",
        TRAIN_V2.relative_to(PROJECT_ROOT),
    )
    print(
        "Total train rows:",
        len(final_rows),
    )

    for cls in (
        "top",
        "pants",
        "skirt",
        "outerwear",
        "dress",
        "shoe",
        "bag",
        "accessory",
    ):
        print(f"  {cls:10s}: " + f"{counts.get(cls, 0)}")

    print("Train/val source-image overlap: 0")
    print(
        "Summary:",
        SUMMARY_PATH.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
