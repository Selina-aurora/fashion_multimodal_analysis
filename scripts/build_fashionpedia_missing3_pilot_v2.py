"""
Fashionpedia -> PRD 3.1.1 missing-3-class pilot
================================================

Purpose
-------
补 PRD 3.1.1 当前缺失的 3 类：
    - shoe
    - bag
    - accessory

Expected raw data layout
------------------------
D:/projects/
├── fashion_multimodal_analysis/
│   ├── configs/
│   │   └── fashionpedia_to_prd_8class_mapping_v1.json
│   └── scripts/
│       └── build_fashionpedia_missing3_pilot_v2.py
└── fashion_data/
    └── raw/
        └── fashionpedia/
            ├── images/
            │   └── val/
            │       └── *.jpg
            └── annotations/
                └── instances_attributes_val2020.json

If the extracted images are not exactly under images/val/,
the script will recursively search under:
    ../fashion_data/raw/fashionpedia/

Outputs
-------
Derived images/masks:
    ../fashion_data/processed/fashionpedia_missing3_pilot_v1/
        crops/
        crop_masks/
        masked_crops/
        full_masks/
        visualizations/
        contact_sheets/

Project-side outputs:
    configs/fashionpedia_missing3_pilot_manifest_v1.csv

    reports/prd_instance_segmentation/fashionpedia_missing3_pilot_v1/
        summary.txt
        category_mapping_used.csv
        skipped_annotations.csv

Run
---
python scripts/build_fashionpedia_missing3_pilot_v2.py

Default:
    20 instances per class, total up to 60.

Optional:
python scripts/build_fashionpedia_missing3_pilot_v2.py --per-class 30
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


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "fashion_data"

RAW_FP_ROOT = DATA_ROOT / "raw" / "fashionpedia"

DEFAULT_ANN_PATH = (
    RAW_FP_ROOT
    / "annotations"
    / "instances_attributes_val2020.json"
)

DEFAULT_IMAGES_DIR = (
    RAW_FP_ROOT
    / "images"
    / "val"
)

DEFAULT_MAPPING_PATH = (
    PROJECT_ROOT
    / "configs"
    / "fashionpedia_to_prd_8class_mapping_v1.json"
)

PROCESSED_ROOT = (
    DATA_ROOT
    / "processed"
    / "fashionpedia_missing3_pilot_v1"
)

CROP_DIR = PROCESSED_ROOT / "crops"
CROP_MASK_DIR = PROCESSED_ROOT / "crop_masks"
MASKED_CROP_DIR = PROCESSED_ROOT / "masked_crops"
FULL_MASK_DIR = PROCESSED_ROOT / "full_masks"
VIS_DIR = PROCESSED_ROOT / "visualizations"
SHEET_DIR = PROCESSED_ROOT / "contact_sheets"

MANIFEST_PATH = (
    PROJECT_ROOT
    / "configs"
    / "fashionpedia_missing3_pilot_manifest_v1.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_instance_segmentation"
    / "fashionpedia_missing3_pilot_v1"
)

TARGET_PRD_CLASSES = ("shoe", "bag", "accessory")


# =========================================================
# Built-in fallback mapping
# =========================================================

DEFAULT_SOURCE_TO_PRD = {
    # shoe
    "shoe": "shoe",

    # bag
    "bag, wallet": "bag",
    "bag": "bag",
    "wallet": "bag",

    # accessory
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


# =========================================================
# Helpers
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotations",
        default=str(DEFAULT_ANN_PATH),
    )

    parser.add_argument(
        "--images-dir",
        default=str(DEFAULT_IMAGES_DIR),
    )

    parser.add_argument(
        "--mapping",
        default=str(DEFAULT_MAPPING_PATH),
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260917,
    )

    return parser.parse_args()


def resolve_file(raw: str) -> Path:
    p = Path(raw).expanduser()

    candidates = [
        p,
        Path.cwd() / p,
        PROJECT_ROOT / p,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"\n找不到文件：{raw}\n"
        f"请检查是否已经放到预期位置。"
    )


def normalize_category_name(name: str) -> str:
    return str(name).strip().lower()


def project_relative(path: Path) -> str:
    """
    Prefer a path relative to fashion_multimodal_analysis.
    Files under sibling fashion_data become ../fashion_data/...
    """
    path = path.resolve()

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        pass

    try:
        rel = path.relative_to(PROJECT_ROOT.parent)
        return f"../{rel.as_posix()}"
    except ValueError:
        return path.as_posix()


def load_mapping(mapping_path: Path | None) -> dict[str, str]:
    """
    Read the mapping JSON if available.
    Otherwise use the built-in fallback mapping.
    """
    mapping = {
        normalize_category_name(k): v
        for k, v in DEFAULT_SOURCE_TO_PRD.items()
    }

    if mapping_path is None or not mapping_path.is_file():
        print("[mapping] config file not found -> using built-in mapping")
        return mapping

    try:
        payload = json.loads(
            mapping_path.read_text(encoding="utf-8")
        )

        config_mapping = payload.get("mapping", payload)

        if not isinstance(config_mapping, dict):
            print("[mapping] invalid mapping JSON -> using built-in mapping")
            return mapping

        # Expected form:
        # {
        #   "shoe": ["shoe"],
        #   "bag": ["bag, wallet"],
        #   "accessory": ["hat", "glasses", ...]
        # }
        for prd_class, source_names in config_mapping.items():
            if prd_class not in TARGET_PRD_CLASSES:
                continue

            if isinstance(source_names, str):
                source_names = [source_names]

            if not isinstance(source_names, list):
                continue

            for source_name in source_names:
                mapping[
                    normalize_category_name(source_name)
                ] = prd_class

        print("[mapping] loaded:", mapping_path)

    except Exception as exc:
        print(
            "[mapping] failed to read config; "
            f"using built-in mapping. reason={exc}"
        )

    return mapping


def discover_validation_images(
    preferred_dir: Path,
    required_names: set[str],
) -> dict[str, Path]:
    """
    Try preferred images/val first.
    If not complete, recursively search RAW_FP_ROOT.
    """
    index: dict[str, Path] = {}

    roots: list[Path] = []

    if preferred_dir.is_dir():
        roots.append(preferred_dir)

    if RAW_FP_ROOT.is_dir():
        if RAW_FP_ROOT.resolve() not in {
            root.resolve() for root in roots
        }:
            roots.append(RAW_FP_ROOT)

    if not roots:
        raise FileNotFoundError(
            "\n找不到 Fashionpedia 图片目录。\n"
            "请先解压 val_test2020.zip 到：\n"
            f"  {RAW_FP_ROOT}\n"
        )

    for root in roots:
        print(f"[scan images] {root}")

        # Fast direct check first.
        for name in required_names:
            direct = root / name
            if direct.is_file():
                index[name] = direct.resolve()

        # Recursive search if necessary.
        if len(index) < len(required_names):
            for ext in (
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.JPG",
                "*.JPEG",
                "*.PNG",
            ):
                for path in root.rglob(ext):
                    if path.name in required_names:
                        index.setdefault(
                            path.name,
                            path.resolve(),
                        )

        if len(index) == len(required_names):
            break

    return index


def polygons_to_mask(
    polygons: list,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    for polygon in polygons:
        if not isinstance(polygon, list):
            continue

        if len(polygon) < 6:
            continue

        points = []

        for i in range(0, len(polygon) - 1, 2):
            points.append(
                (
                    float(polygon[i]),
                    float(polygon[i + 1]),
                )
            )

        if len(points) >= 3:
            draw.polygon(
                points,
                fill=255,
            )

    return np.asarray(canvas, dtype=np.uint8)


def rle_to_mask(
    segmentation: dict,
) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError(
            "遇到 RLE mask，但当前环境没有 pycocotools。\n"
            "请运行：pip install pycocotools"
        ) from exc

    decoded = mask_utils.decode(segmentation)

    if decoded.ndim == 3:
        decoded = np.any(
            decoded > 0,
            axis=2,
        )

    return (
        (decoded > 0)
        .astype(np.uint8)
        * 255
    )


def segmentation_to_mask(
    segmentation: Any,
    width: int,
    height: int,
) -> np.ndarray:
    if isinstance(segmentation, list):
        return polygons_to_mask(
            segmentation,
            width,
            height,
        )

    if isinstance(segmentation, dict):
        mask = rle_to_mask(segmentation)

        if mask.shape != (height, width):
            raise ValueError(
                f"RLE shape mismatch: "
                f"{mask.shape} != {(height, width)}"
            )

        return mask

    raise ValueError(
        f"unsupported segmentation type: "
        f"{type(segmentation)}"
    )


def bbox_from_mask(
    mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask > 0)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def save_masked_crop(
    crop: Image.Image,
    crop_mask: Image.Image,
    save_path: Path,
) -> None:
    white = Image.new(
        "RGB",
        crop.size,
        "white",
    )

    masked = Image.composite(
        crop.convert("RGB"),
        white,
        crop_mask.convert("L"),
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    masked.save(
        save_path,
        quality=94,
    )


def save_visualization(
    image: Image.Image,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    text: str,
    save_path: Path,
) -> None:
    base = image.convert("RGBA")

    rgba_overlay = Image.new(
        "RGBA",
        image.size,
        (255, 0, 0, 0),
    )

    alpha = Image.fromarray(
        np.where(
            mask > 0,
            75,
            0,
        ).astype(np.uint8)
    )

    red = Image.new(
        "RGBA",
        image.size,
        (255, 0, 0, 75),
    )

    rgba_overlay.paste(
        red,
        (0, 0),
        alpha,
    )

    vis = Image.alpha_composite(
        base,
        rgba_overlay,
    )

    draw = ImageDraw.Draw(vis)

    x1, y1, x2, y2 = bbox

    draw.rectangle(
        (x1, y1, x2 - 1, y2 - 1),
        outline=(0, 255, 0, 255),
        width=3,
    )

    draw.text(
        (x1 + 3, max(3, y1 - 18)),
        text,
        fill=(0, 255, 0, 255),
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vis.convert("RGB").save(
        save_path,
        quality=92,
    )


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        if fieldnames:
            with path.open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                )
                writer.writeheader()
        else:
            path.write_text(
                "",
                encoding="utf-8",
            )
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def sample_diverse_instances(
    candidates: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """
    Try to:
    1) diversify Fashionpedia fine categories
    2) avoid repeated source images when possible
    """
    by_source_category: dict[str, list[dict]] = defaultdict(list)

    for item in candidates:
        by_source_category[
            item["source_category"]
        ].append(item)

    for items in by_source_category.values():
        rng.shuffle(items)

    source_categories = list(
        by_source_category.keys()
    )
    rng.shuffle(source_categories)

    selected: list[dict] = []
    used_image_ids: set[int] = set()

    progress = True

    while len(selected) < n and progress:
        progress = False

        for source_category in source_categories:
            pool = by_source_category[source_category]

            while pool:
                item = pool.pop()
                image_id = int(
                    item["image_meta"]["id"]
                )

                if image_id in used_image_ids:
                    continue

                selected.append(item)
                used_image_ids.add(image_id)

                progress = True
                break

            if len(selected) >= n:
                break

    # Fill remaining slots even if image repeats are necessary.
    if len(selected) < n:
        leftovers = []

        for pool in by_source_category.values():
            leftovers.extend(pool)

        rng.shuffle(leftovers)

        for item in leftovers:
            selected.append(item)

            if len(selected) >= n:
                break

    return selected[:n]


def make_contact_sheet(
    rows: list[dict],
    title: str,
    output_path: Path,
) -> None:
    if not rows:
        return

    cols = 4
    tile_w = 340
    tile_h = 380
    img_w = 310
    img_h = 300

    total_rows = math.ceil(
        len(rows) / cols
    )

    sheet = Image.new(
        "RGB",
        (
            cols * tile_w,
            55 + total_rows * tile_h,
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    draw.text(
        (15, 15),
        title,
        fill="black",
        font=font,
    )

    for i, row in enumerate(rows):
        grid_r = i // cols
        grid_c = i % cols

        x0 = grid_c * tile_w
        y0 = 55 + grid_r * tile_h

        vis_path = (
            PROJECT_ROOT
            / row["visualization_path"]
        ).resolve()

        image = Image.open(
            vis_path
        ).convert("RGB")

        image.thumbnail(
            (img_w, img_h),
            Image.Resampling.LANCZOS,
        )

        px = x0 + (
            tile_w - image.width
        ) // 2

        py = y0 + 5

        sheet.paste(
            image,
            (px, py),
        )

        text_y = y0 + img_h + 12

        text_lines = [
            f"PRD: {row['garment_category']}",
            f"src: {row['source_category']}",
            f"ann: {row['annotation_id']}",
        ]

        for line in text_lines:
            draw.text(
                (x0 + 8, text_y),
                line,
                fill="black",
                font=font,
            )
            text_y += 18

        draw.rectangle(
            (
                x0 + 2,
                y0 + 2,
                x0 + tile_w - 2,
                y0 + tile_h - 2,
            ),
            outline="gray",
            width=1,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        quality=92,
    )


# =========================================================
# Main
# =========================================================

def main() -> None:
    args = parse_args()

    ann_path = resolve_file(
        args.annotations
    )

    mapping_candidate = Path(
        args.mapping
    ).expanduser()

    if not mapping_candidate.is_absolute():
        mapping_candidate = (
            PROJECT_ROOT
            / mapping_candidate
        )

    if not mapping_candidate.is_file():
        mapping_candidate = DEFAULT_MAPPING_PATH

    source_to_prd = load_mapping(
        mapping_candidate
        if mapping_candidate.is_file()
        else None
    )

    preferred_images_dir = Path(
        args.images_dir
    ).expanduser()

    if not preferred_images_dir.is_absolute():
        preferred_images_dir = (
            PROJECT_ROOT
            / preferred_images_dir
        )

    print("\n=== Fashionpedia missing-3 pilot ===")
    print("project root :", PROJECT_ROOT)
    print("raw root     :", RAW_FP_ROOT)
    print("annotation   :", ann_path)
    print("images pref  :", preferred_images_dir)

    payload = json.loads(
        ann_path.read_text(
            encoding="utf-8"
        )
    )

    categories = {
        int(cat["id"]): str(cat["name"])
        for cat in payload.get(
            "categories",
            [],
        )
    }

    images = {
        int(img["id"]): img
        for img in payload.get(
            "images",
            [],
        )
    }

    annotations = payload.get(
        "annotations",
        [],
    )

    if not categories:
        raise ValueError(
            "annotation JSON contains no categories"
        )

    if not images:
        raise ValueError(
            "annotation JSON contains no images"
        )

    if not annotations:
        raise ValueError(
            "annotation JSON contains no annotations"
        )

    print(
        f"categories={len(categories)}, "
        f"images={len(images)}, "
        f"annotations={len(annotations)}"
    )

    mapping_rows: list[dict] = []

    for category_id, category_name in sorted(
        categories.items()
    ):
        mapped = source_to_prd.get(
            normalize_category_name(
                category_name
            ),
            "",
        )

        mapping_rows.append(
            {
                "source_category_id": category_id,
                "source_category_name": category_name,
                "mapped_prd_category": mapped,
            }
        )

    candidates_by_class: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    for ann in annotations:
        source_category = categories.get(
            int(ann["category_id"]),
            "",
        )

        prd_category = source_to_prd.get(
            normalize_category_name(
                source_category
            )
        )

        if prd_category not in TARGET_PRD_CLASSES:
            continue

        image_meta = images.get(
            int(ann["image_id"])
        )

        if image_meta is None:
            continue

        candidates_by_class[
            prd_category
        ].append(
            {
                "annotation": ann,
                "image_meta": image_meta,
                "source_category": source_category,
                "prd_category": prd_category,
            }
        )

    print("\nCandidate counts:")
    for prd_class in TARGET_PRD_CLASSES:
        print(
            f"  {prd_class}: "
            f"{len(candidates_by_class[prd_class])}"
        )

    required_file_names = {
        str(meta["file_name"])
        for meta in images.values()
    }

    image_index = discover_validation_images(
        preferred_images_dir,
        required_file_names,
    )

    print(
        "\nImages found:",
        f"{len(image_index)}/{len(required_file_names)}",
    )

    rng = random.Random(
        args.seed
    )

    selected: list[dict] = []

    for prd_class in TARGET_PRD_CLASSES:
        class_items = sample_diverse_instances(
            candidates_by_class[
                prd_class
            ],
            args.per_class,
            rng,
        )

        selected.extend(
            class_items
        )

    for directory in [
        CROP_DIR,
        CROP_MASK_DIR,
        MASKED_CROP_DIR,
        FULL_MASK_DIR,
        VIS_DIR,
        SHEET_DIR,
        REPORT_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for item in selected:
        ann = item["annotation"]
        image_meta = item["image_meta"]
        source_category = item[
            "source_category"
        ]
        prd_category = item[
            "prd_category"
        ]

        image_id = int(
            image_meta["id"]
        )

        annotation_id = int(
            ann["id"]
        )

        file_name = str(
            image_meta["file_name"]
        )

        image_path = image_index.get(
            file_name
        )

        if image_path is None:
            skipped_rows.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "file_name": file_name,
                    "reason": "image_not_found",
                }
            )
            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        width, height = image.size

        try:
            mask = segmentation_to_mask(
                ann.get(
                    "segmentation"
                ),
                width,
                height,
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "file_name": file_name,
                    "reason": (
                        "mask_decode_failed: "
                        f"{exc}"
                    ),
                }
            )
            continue

        bbox = bbox_from_mask(
            mask
        )

        if bbox is None:
            skipped_rows.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "file_name": file_name,
                    "reason": "empty_mask",
                }
            )
            continue

        x1, y1, x2, y2 = bbox

        if (
            x2 <= x1
            or y2 <= y1
        ):
            skipped_rows.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "file_name": file_name,
                    "reason": "invalid_bbox",
                }
            )
            continue

        stem = (
            f"{prd_category}_"
            f"ann{annotation_id}_"
            f"img{image_id}"
        )

        full_mask_path = (
            FULL_MASK_DIR
            / f"{stem}.png"
        )

        crop_mask_path = (
            CROP_MASK_DIR
            / f"{stem}.png"
        )

        crop_path = (
            CROP_DIR
            / f"{stem}.jpg"
        )

        masked_crop_path = (
            MASKED_CROP_DIR
            / f"{stem}.jpg"
        )

        vis_path = (
            VIS_DIR
            / f"{stem}.jpg"
        )

        full_mask_img = Image.fromarray(
            mask
        )

        full_mask_img.save(
            full_mask_path
        )

        crop = image.crop(
            (x1, y1, x2, y2)
        )

        crop.save(
            crop_path,
            quality=94,
        )

        crop_mask = full_mask_img.crop(
            (x1, y1, x2, y2)
        )

        crop_mask.save(
            crop_mask_path
        )

        save_masked_crop(
            crop,
            crop_mask,
            masked_crop_path,
        )

        save_visualization(
            image,
            mask,
            bbox,
            (
                f"{prd_category} / "
                f"{source_category}"
            ),
            vis_path,
        )

        source_bbox = ann.get(
            "bbox",
            [],
        )

        if (
            isinstance(
                source_bbox,
                list,
            )
            and len(source_bbox) == 4
        ):
            sbx, sby, sbw, sbh = (
                source_bbox
            )
        else:
            sbx = sby = sbw = sbh = ""

        manifest_rows.append(
            {
                "source_dataset": "Fashionpedia",
                "source_image": project_relative(
                    image_path
                ),
                "image_id": image_id,
                "annotation_id": annotation_id,
                "garment_id": (
                    "fashionpedia_ann_"
                    f"{annotation_id}"
                ),
                "source_category": source_category,
                "garment_category": prd_category,
                "image_width": width,
                "image_height": height,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_width": x2 - x1,
                "bbox_height": y2 - y1,
                "source_bbox_x": sbx,
                "source_bbox_y": sby,
                "source_bbox_width": sbw,
                "source_bbox_height": sbh,
                "mask_area_pixels": int(
                    (mask > 0).sum()
                ),
                "source_area": ann.get(
                    "area",
                    "",
                ),
                "iscrowd": ann.get(
                    "iscrowd",
                    "",
                ),
                "crop_path": project_relative(
                    crop_path
                ),
                "mask_path": project_relative(
                    crop_mask_path
                ),
                "masked_crop_path": project_relative(
                    masked_crop_path
                ),
                "full_mask_path": project_relative(
                    full_mask_path
                ),
                "visualization_path": project_relative(
                    vis_path
                ),
            }
        )

    write_csv(
        MANIFEST_PATH,
        manifest_rows,
    )

    write_csv(
        REPORT_DIR
        / "category_mapping_used.csv",
        mapping_rows,
    )

    write_csv(
        REPORT_DIR
        / "skipped_annotations.csv",
        skipped_rows,
        fieldnames=[
            "annotation_id",
            "image_id",
            "file_name",
            "reason",
        ],
    )

    counts = Counter(
        row["garment_category"]
        for row in manifest_rows
    )

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
        f"annotation={project_relative(ann_path)}",
        f"seed={args.seed}",
        f"requested_per_class={args.per_class}",
        f"selected_before_decode={len(selected)}",
        f"manifest_rows={len(manifest_rows)}",
        f"skipped={len(skipped_rows)}",
        "",
        "PRD class counts",
        "----------------",
    ]

    for prd_class in TARGET_PRD_CLASSES:
        summary_lines.append(
            f"{prd_class}="
            f"{counts.get(prd_class, 0)}"
        )

    summary_lines += [
        "",
        "Fine source-category counts",
        "---------------------------",
    ]

    for (
        prd_class,
        source_category,
    ), count in sorted(
        fine_counts.items()
    ):
        summary_lines.append(
            f"{prd_class} <- "
            f"{source_category}: "
            f"{count}"
        )

    summary_lines += [
        "",
        "Notes",
        "-----",
        "- raw/fashionpedia is read-only in this script.",
        "- all derived crops/masks go to fashion_data/processed/.",
        "- this is only a pilot, not final training data.",
        "- inspect contact sheets before formal training.",
    ]

    (
        REPORT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(
            summary_lines
        ) + "\n",
        encoding="utf-8",
    )

    for prd_class in TARGET_PRD_CLASSES:
        class_rows = [
            row
            for row in manifest_rows
            if (
                row["garment_category"]
                == prd_class
            )
        ]

        make_contact_sheet(
            class_rows,
            (
                "Fashionpedia pilot - "
                f"PRD {prd_class}"
            ),
            (
                SHEET_DIR
                / f"{prd_class}.jpg"
            ),
        )

    print("\n=== DONE ===")
    print(
        "manifest:",
        MANIFEST_PATH.relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        "report:",
        REPORT_DIR.relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        "processed:",
        PROCESSED_ROOT,
    )
    print(
        "counts:",
        dict(counts),
    )
    print(
        "skipped:",
        len(skipped_rows),
    )

    print("\nPlease send me:")
    print(
        "1. reports/prd_instance_segmentation/"
        "fashionpedia_missing3_pilot_v1/summary.txt"
    )
    print(
        "2. ../fashion_data/processed/"
        "fashionpedia_missing3_pilot_v1/"
        "contact_sheets/shoe.jpg"
    )
    print(
        "3. ../fashion_data/processed/"
        "fashionpedia_missing3_pilot_v1/"
        "contact_sheets/bag.jpg"
    )
    print(
        "4. ../fashion_data/processed/"
        "fashionpedia_missing3_pilot_v1/"
        "contact_sheets/accessory.jpg"
    )


if __name__ == "__main__":
    main()
