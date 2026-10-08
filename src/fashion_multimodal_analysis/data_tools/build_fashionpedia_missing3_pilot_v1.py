"""数据处理：保存原图来源和实例标识，避免同一图片的不同实例跨训练与评估划分。

Build a small Fashionpedia pilot for the 3 PRD 3.1.1 categories
missing from the current DeepFashion2 setup:

    shoe
    bag
    accessory

Expected raw data layout
------------------------
D:/projects/
├── fashion_multimodal_analysis/
│   └── scripts/
│       └── build_fashionpedia_missing3_pilot_v1.py
└── fashion_data/
    └── raw/
        └── fashionpedia/
            ├── annotations/
            │   └── instances_attributes_val2020.json
            └── images/
                └── test/
                    └── *.jpg

The script DOES NOT modify raw data.

Derived data:
    ../fashion_data/processed/fashionpedia_missing3_pilot_v1/

Project outputs:
    configs/fashionpedia_missing3_pilot_manifest_v1.csv
    reports/prd_instance_segmentation/fashionpedia_missing3_pilot_v1/

Run from project root:
    python scripts/data_tools/build_fashionpedia_missing3_pilot_v1.py

Default pilot:
    20 instances per PRD class, seed 20260917
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

RAW_ROOT = DATA_ROOT / "raw" / "fashionpedia"
ANN_PATH = RAW_ROOT / "annotations" / "instances_attributes_val2020.json"
IMAGES_DIR = RAW_ROOT / "images" / "test"

PROCESSED_ROOT = DATA_ROOT / "processed" / "fashionpedia_missing3_pilot_v1"

FULL_MASK_DIR = PROCESSED_ROOT / "full_masks"
CROP_MASK_DIR = PROCESSED_ROOT / "crop_masks"
CROP_DIR = PROCESSED_ROOT / "crops"
MASKED_CROP_DIR = PROCESSED_ROOT / "masked_crops"
VIS_DIR = PROCESSED_ROOT / "visualizations"
CONTACT_DIR = PROCESSED_ROOT / "contact_sheets"

MANIFEST_PATH = PROJECT_ROOT / "configs" / "fashionpedia_missing3_pilot_manifest_v1.csv"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_instance_segmentation"
    / "fashionpedia_missing3_pilot_v1"
)

TARGET_CLASSES = ("shoe", "bag", "accessory")


# Fashionpedia source-category -> PRD category.
#
# We deliberately keep the broad PRD "accessory" category as a collapse of
# several Fashionpedia fine categories.
#
# Matching is normalized and also has a conservative keyword fallback below.
SOURCE_TO_PRD = {
    # shoe
    "shoe": "shoe",
    # bag
    "bag": "bag",
    "wallet": "bag",
    "bag, wallet": "bag",
    # accessory
    "glasses": "accessory",
    "hat": "accessory",
    "headband": "accessory",
    "head covering": "accessory",
    "hair accessory": "accessory",
    "headband, head covering, hair accessory": "accessory",
    "tie": "accessory",
    "glove": "accessory",
    "watch": "accessory",
    "belt": "accessory",
    "leg warmer": "accessory",
    "tights": "accessory",
    "stockings": "accessory",
    "tights, stockings": "accessory",
    "sock": "accessory",
    "scarf": "accessory",
}


def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-class",
        type=int,
        default=20,
        help="Maximum pilot instances per PRD class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260917,
    )
    parser.add_argument(
        "--annotations",
        default=str(ANN_PATH),
    )
    parser.add_argument(
        "--images-dir",
        default=str(IMAGES_DIR),
    )
    return parser.parse_args()


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

    if n in SOURCE_TO_PRD:
        return SOURCE_TO_PRD[n]

    # Conservative fallback for Fashionpedia wording variations.
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


def relative_display(path: Path) -> str:
    """relative display。

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
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue

        pts = []
        for i in range(0, len(polygon) - 1, 2):
            pts.append((float(polygon[i]), float(polygon[i + 1])))

        if len(pts) >= 3:
            draw.polygon(pts, fill=255)

    return np.asarray(canvas, dtype=np.uint8)


def rle_to_mask(segmentation: dict) -> np.ndarray:
    """解码 COCO RLE 掩码，兼容当前标注的计数表示。

    Args:
        segmentation: 原始分割标注，可能为多边形或 COCO RLE。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        RuntimeError: Encountered COCO RLE segmentation, but pycocotools is not
        installed. Run:
    """
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError(
            "Encountered COCO RLE segmentation, but pycocotools "
            + "is not installed. Run:\n"
            + "pip install pycocotools"
        ) from exc

    decoded = mask_utils.decode(segmentation)

    if decoded.ndim == 3:
        decoded = np.any(decoded > 0, axis=2)

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
    if isinstance(segmentation, list):
        return polygons_to_mask(segmentation, width, height)

    if isinstance(segmentation, dict):
        mask = rle_to_mask(segmentation)
        if mask.shape != (height, width):
            raise ValueError(f"RLE mask shape {mask.shape} != image {(height, width)}")
        return mask

    raise ValueError(f"Unsupported segmentation type: {type(segmentation).__name__}")


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


def ensure_dirs() -> None:
    """ensure dirs。"""
    for p in (
        FULL_MASK_DIR,
        CROP_MASK_DIR,
        CROP_DIR,
        MASKED_CROP_DIR,
        VIS_DIR,
        CONTACT_DIR,
        REPORT_DIR,
        MANIFEST_PATH.parent,
    ):
        p.mkdir(parents=True, exist_ok=True)


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str] | None = None,
) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
        fieldnames: fieldnames。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        if not fieldnames:
            return

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def find_image(
    file_name: str,
    images_dir: Path,
    index: dict[str, Path],
) -> Path | None:
    """查找 图像。

    Args:
        file_name: 文件 name。
        images_dir: 本步骤使用的目录。
        index: 待访问样本的整数下标。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    direct = images_dir / file_name
    if direct.is_file():
        return direct

    return index.get(Path(file_name).name)


def build_image_index(images_dir: Path) -> dict[str, Path]:
    """按原图文件名建立检索索引，减少逐行扫描目录的开销。

    Args:
        images_dir: 本步骤使用的目录。

    Returns:
        返回 index，由函数体中同名变量的计算/收集过程得到。
    """
    print(f"[1/6] Indexing images under:\n  {images_dir}")

    index: dict[str, Path] = {}

    for pattern in (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    ):
        for path in images_dir.rglob(pattern):
            index.setdefault(path.name, path)

    print(f"      indexed image files: {len(index)}")
    return index


def diverse_sample(
    items: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Prefer:
    1. different fine source categories
    2. different image IDs

    Args:
        items: items。
        n: 本步骤使用的原始值，转换/筛选规则见函数体。
        rng: rng。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    pools: dict[str, list[dict]] = defaultdict(list)

    for item in items:
        pools[item["source_category"]].append(item)

    for pool in pools.values():
        rng.shuffle(pool)

    category_names = list(pools.keys())
    rng.shuffle(category_names)

    selected: list[dict] = []
    selected_ann_ids: set[int] = set()
    used_image_ids: set[int] = set()

    # Pass 1: unique source image when possible.
    progress = True
    while len(selected) < n and progress:
        progress = False

        for source_cat in category_names:
            pool = pools[source_cat]

            candidate_index = None
            for i, item in enumerate(pool):
                img_id = int(item["image_meta"]["id"])
                ann_id = int(item["annotation"]["id"])

                if img_id not in used_image_ids and ann_id not in selected_ann_ids:
                    candidate_index = i
                    break

            if candidate_index is None:
                continue

            item = pool.pop(candidate_index)

            selected.append(item)
            selected_ann_ids.add(int(item["annotation"]["id"]))
            used_image_ids.add(int(item["image_meta"]["id"]))
            progress = True

            if len(selected) >= n:
                break

    # Pass 2: allow repeated image IDs if needed.
    if len(selected) < n:
        leftovers = []

        for pool in pools.values():
            leftovers.extend(pool)

        rng.shuffle(leftovers)

        for item in leftovers:
            ann_id = int(item["annotation"]["id"])

            if ann_id in selected_ann_ids:
                continue

            selected.append(item)
            selected_ann_ids.add(ann_id)

            if len(selected) >= n:
                break

    return selected[:n]


def save_masked_crop(
    crop: Image.Image,
    crop_mask: Image.Image,
    out_path: Path,
) -> None:
    """保存 masked 裁剪。

    Args:
        crop: 裁剪。
        crop_mask: 裁剪 掩码。
        out_path: 结果文件路径。
    """
    white = Image.new("RGB", crop.size, "white")

    result = Image.composite(
        crop.convert("RGB"),
        white,
        crop_mask.convert("L"),
    )

    result.save(out_path, quality=94)


def save_visualization(
    image: Image.Image,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    out_path: Path,
) -> None:
    """保存 visualization。

    Args:
        image: 本步骤处理的图像对象。
        mask: 掩码。
        bbox: xyxy 坐标的目标框。
        label: 标签。
        out_path: 结果文件路径。
    """
    base = image.convert("RGBA")

    alpha = Image.fromarray(np.where(mask > 0, 80, 0).astype(np.uint8))

    red_overlay = Image.new(
        "RGBA",
        base.size,
        (255, 0, 0, 80),
    )

    transparent = Image.new(
        "RGBA",
        base.size,
        (0, 0, 0, 0),
    )

    transparent.paste(
        red_overlay,
        (0, 0),
        alpha,
    )

    vis = Image.alpha_composite(
        base,
        transparent,
    )

    draw = ImageDraw.Draw(vis)

    x1, y1, x2, y2 = bbox

    draw.rectangle(
        [x1, y1, x2 - 1, y2 - 1],
        outline=(0, 255, 0, 255),
        width=3,
    )

    draw.text(
        (x1 + 3, max(2, y1 - 15)),
        label,
        fill=(0, 255, 0, 255),
    )

    vis.convert("RGB").save(out_path, quality=92)


def make_contact_sheet(
    rows: list[dict],
    cls: str,
) -> None:
    """把样本图与说明拼成审核联系表，便于逐例比较。

    Args:
        rows: 待处理的逐行记录。
    """
    if not rows:
        return

    thumb_w = 280
    thumb_h = 280
    tile_w = 310
    tile_h = 350
    cols = 4
    rows_n = math.ceil(len(rows) / cols)

    sheet = Image.new(
        "RGB",
        (tile_w * cols, 50 + tile_h * rows_n),
        "white",
    )

    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    draw.text(
        (15, 15),
        f"Fashionpedia pilot - PRD {cls}",
        fill="black",
        font=font,
    )

    for i, row in enumerate(rows):
        r = i // cols
        c = i % cols

        x0 = c * tile_w
        y0 = 50 + r * tile_h

        vis_path = Path(row["visualization_path_abs"])
        image = Image.open(vis_path).convert("RGB")
        image.thumbnail(
            (thumb_w, thumb_h),
            Image.Resampling.LANCZOS,
        )

        px = x0 + (tile_w - image.width) // 2
        py = y0 + 5

        sheet.paste(image, (px, py))

        text_y = y0 + 290

        labels = [
            f"{row['garment_category']} <- {row['source_category']}",
            f"ann={row['annotation_id']} img={row['image_id']}",
        ]

        for line in labels:
            draw.text(
                (x0 + 8, text_y),
                line,
                fill="black",
                font=font,
            )
            text_y += 18

        draw.rectangle(
            [x0 + 2, y0 + 2, x0 + tile_w - 2, y0 + tile_h - 2],
            outline="gray",
            width=1,
        )

    out = CONTACT_DIR / f"{cls}.jpg"
    sheet.save(out, quality=92)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        FileNotFoundError: 需要的文件不存在。
        ValueError: No 'annotations' in annotation JSON.
    """
    args = parse_args()

    ann_path = Path(args.annotations).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()

    if not ann_path.is_file():
        raise FileNotFoundError(
            "\nAnnotation JSON not found:\n"
            + f"  {ann_path}\n\n"
            + "Expected:\n"
            + "  ../fashion_data/raw/fashionpedia/annotations/"
            + "instances_attributes_val2020.json"
        )

    if not images_dir.is_dir():
        raise FileNotFoundError(
            "\nFashionpedia image folder not found:\n"
            + f"  {images_dir}\n\n"
            + "Expected:\n"
            + "  ../fashion_data/raw/fashionpedia/images/test/"
        )

    ensure_dirs()

    print("[0/6] Loading annotation JSON...")
    print("      annotation:", ann_path)
    print("      images    :", images_dir)

    with ann_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    categories = data.get("categories", [])
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    if not categories:
        raise ValueError("No 'categories' in annotation JSON.")

    if not images:
        raise ValueError("No 'images' in annotation JSON.")

    if not annotations:
        raise ValueError("No 'annotations' in annotation JSON.")

    category_by_id = {int(cat["id"]): str(cat["name"]) for cat in categories}

    image_by_id = {int(img["id"]): img for img in images}

    # Save exact Fashionpedia category inventory + our mapping.
    category_rows = []

    for cat in categories:
        source_name = str(cat["name"])
        mapped = map_source_category(source_name)

        category_rows.append(
            {
                "source_category_id": int(cat["id"]),
                "source_category_name": source_name,
                "mapped_prd_category": mapped or "",
            }
        )

    write_csv(
        REPORT_DIR / "category_mapping_used.csv",
        category_rows,
    )

    print("[2/6] Building missing-3 candidate pools...")

    candidates: dict[str, list[dict]] = defaultdict(list)

    for ann in annotations:
        source_cat = category_by_id.get(int(ann["category_id"]))

        if source_cat is None:
            continue

        prd_class = map_source_category(source_cat)

        if prd_class not in TARGET_CLASSES:
            continue

        image_meta = image_by_id.get(int(ann["image_id"]))

        if image_meta is None:
            continue

        candidates[prd_class].append(
            {
                "annotation": ann,
                "image_meta": image_meta,
                "source_category": source_cat,
                "prd_category": prd_class,
            }
        )

    for cls in TARGET_CLASSES:
        print(f"      {cls}: {len(candidates.get(cls, []))} candidates")

    missing = [cls for cls in TARGET_CLASSES if len(candidates.get(cls, [])) == 0]

    if missing:
        print(
            "\nWARNING: zero mapped annotations for:",
            ", ".join(missing),
        )
        print(
            "Open:\n"
            + "  reports/prd_instance_segmentation/"
            + "fashionpedia_missing3_pilot_v1/"
            + "category_mapping_used.csv\n"
            + "and send it to me."
        )

    image_index = build_image_index(images_dir)

    rng = random.Random(args.seed)

    print("[3/6] Sampling pilot instances...")

    selected: list[dict] = []

    for cls in TARGET_CLASSES:
        cls_selected = diverse_sample(
            candidates.get(cls, []),
            args.per_class,
            rng,
        )

        print(f"      {cls}: selected {len(cls_selected)}")

        selected.extend(cls_selected)

    print("[4/6] Decoding masks and saving crops...")

    manifest_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for idx, item in enumerate(selected, start=1):
        ann = item["annotation"]
        image_meta = item["image_meta"]
        source_cat = item["source_category"]
        prd_class = item["prd_category"]

        file_name = str(image_meta["file_name"])

        image_path = find_image(
            file_name,
            images_dir,
            image_index,
        )

        if image_path is None:
            skipped_rows.append(
                {
                    "annotation_id": ann.get("id", ""),
                    "image_id": image_meta.get("id", ""),
                    "file_name": file_name,
                    "reason": "image_not_found",
                }
            )
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            skipped_rows.append(
                {
                    "annotation_id": ann.get("id", ""),
                    "image_id": image_meta.get("id", ""),
                    "file_name": file_name,
                    "reason": f"image_open_failed: {exc}",
                }
            )
            continue

        width, height = image.size

        try:
            mask = segmentation_to_mask(
                ann.get("segmentation"),
                width,
                height,
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "annotation_id": ann.get("id", ""),
                    "image_id": image_meta.get("id", ""),
                    "file_name": file_name,
                    "reason": f"mask_decode_failed: {exc}",
                }
            )
            continue

        bbox = bbox_from_mask(mask)

        if bbox is None:
            skipped_rows.append(
                {
                    "annotation_id": ann.get("id", ""),
                    "image_id": image_meta.get("id", ""),
                    "file_name": file_name,
                    "reason": "empty_mask",
                }
            )
            continue

        x1, y1, x2, y2 = bbox

        annotation_id = int(ann["id"])
        image_id = int(image_meta["id"])

        stem = f"{prd_class}_" + f"ann{annotation_id}_" + f"img{image_id}"

        full_mask_path = FULL_MASK_DIR / f"{stem}.png"
        crop_mask_path = CROP_MASK_DIR / f"{stem}.png"
        crop_path = CROP_DIR / f"{stem}.jpg"
        masked_crop_path = MASKED_CROP_DIR / f"{stem}.jpg"
        vis_path = VIS_DIR / f"{stem}.jpg"

        full_mask_img = Image.fromarray(mask)
        full_mask_img.save(full_mask_path)

        crop = image.crop((x1, y1, x2, y2))
        crop.save(crop_path, quality=94)

        crop_mask = full_mask_img.crop((x1, y1, x2, y2))
        crop_mask.save(crop_mask_path)

        save_masked_crop(
            crop,
            crop_mask,
            masked_crop_path,
        )

        save_visualization(
            image,
            mask,
            bbox,
            f"{prd_class} / {source_cat}",
            vis_path,
        )

        source_bbox = ann.get("bbox", [])

        source_bbox_x = ""
        source_bbox_y = ""
        source_bbox_w = ""
        source_bbox_h = ""

        if isinstance(source_bbox, list) and len(source_bbox) == 4:
            (
                source_bbox_x,
                source_bbox_y,
                source_bbox_w,
                source_bbox_h,
            ) = source_bbox

        manifest_rows.append(
            {
                "source_dataset": "Fashionpedia",
                "source_image": relative_display(image_path),
                "image_id": image_id,
                "annotation_id": annotation_id,
                "garment_id": f"fashionpedia_ann_{annotation_id}",
                "source_category": source_cat,
                "garment_category": prd_class,
                "image_width": width,
                "image_height": height,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_width": x2 - x1,
                "bbox_height": y2 - y1,
                "source_bbox_x": source_bbox_x,
                "source_bbox_y": source_bbox_y,
                "source_bbox_width": source_bbox_w,
                "source_bbox_height": source_bbox_h,
                "mask_area_pixels": int((mask > 0).sum()),
                "source_area": ann.get("area", ""),
                "iscrowd": ann.get("iscrowd", ""),
                "crop_path": relative_display(crop_path),
                "masked_crop_path": relative_display(masked_crop_path),
                "full_mask_path": relative_display(full_mask_path),
                "mask_path": relative_display(crop_mask_path),
                "visualization_path": relative_display(vis_path),
                # Internal absolute path only for contact-sheet creation.
                "visualization_path_abs": str(vis_path.resolve()),
            }
        )

        if idx % 10 == 0:
            print(f"      processed {idx}/{len(selected)}")

    print("[5/6] Writing manifest and contact sheets...")

    # Keep internal helper column out of the CSV.
    csv_rows = []

    for row in manifest_rows:
        copy = dict(row)
        copy.pop("visualization_path_abs", None)
        csv_rows.append(copy)

    write_csv(
        MANIFEST_PATH,
        csv_rows,
    )

    write_csv(
        REPORT_DIR / "skipped_annotations.csv",
        skipped_rows,
        fieldnames=[
            "annotation_id",
            "image_id",
            "file_name",
            "reason",
        ],
    )

    for cls in TARGET_CLASSES:
        cls_rows = [row for row in manifest_rows if row["garment_category"] == cls]

        make_contact_sheet(
            cls_rows,
            cls,
        )

    counts = Counter(row["garment_category"] for row in manifest_rows)

    fine_counts = Counter(
        (
            row["garment_category"],
            row["source_category"],
        )
        for row in manifest_rows
    )

    summary_lines = [
        "Fashionpedia missing-3 pilot for PRD 3.1.1",
        "==========================================",
        "",
        f"annotation={relative_display(ann_path)}",
        f"images_dir={relative_display(images_dir)}",
        f"seed={args.seed}",
        f"requested_per_class={args.per_class}",
        f"selected_before_decode={len(selected)}",
        f"saved_instances={len(manifest_rows)}",
        f"skipped_instances={len(skipped_rows)}",
        "",
        "PRD class counts",
        "----------------",
    ]

    for cls in TARGET_CLASSES:
        summary_lines.append(f"{cls}={counts.get(cls, 0)}")

    summary_lines += [
        "",
        "Fine source-category counts",
        "---------------------------",
    ]

    for (prd_cls, source_cat), n in sorted(fine_counts.items()):
        summary_lines.append(f"{prd_cls} <- {source_cat}: {n}")

    summary_lines += [
        "",
        "Notes",
        "-----",
        "- Raw Fashionpedia files were not modified.",
        "- This is a small pilot, not the final training/evaluation set.",
        "- Fashionpedia fine categories are collapsed into the PRD shoe/bag/accessory categories.",
        "- Inspect shoe.jpg, bag.jpg and accessory.jpg before using the pilot downstream.",
        "- If a source category is mapped incorrectly, fix the mapping before scaling up.",
    ]

    summary_path = REPORT_DIR / "summary.txt"

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("[6/6] DONE")
    print()
    print("Saved counts:")
    for cls in TARGET_CLASSES:
        print(f"  {cls}: {counts.get(cls, 0)}")

    print(f"  skipped: {len(skipped_rows)}")
    print()
    print("Please send me these 4 outputs:")
    print(
        "1. reports/prd_instance_segmentation/"
        + "fashionpedia_missing3_pilot_v1/summary.txt"
    )
    print(
        "2. ../fashion_data/processed/"
        + "fashionpedia_missing3_pilot_v1/contact_sheets/shoe.jpg"
    )
    print(
        "3. ../fashion_data/processed/"
        + "fashionpedia_missing3_pilot_v1/contact_sheets/bag.jpg"
    )
    print(
        "4. ../fashion_data/processed/"
        + "fashionpedia_missing3_pilot_v1/contact_sheets/accessory.jpg"
    )


if __name__ == "__main__":
    main()
