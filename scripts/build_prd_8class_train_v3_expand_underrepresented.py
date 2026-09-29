"""
Build PRD 3.1.1 train_v2 with exactly 100 VALID Fashionpedia training
instances per new PRD class (shoe / bag / accessory).

This version fixes the v1 expansion issue where exactly 100 candidates were
sampled before mask decoding, so decode failures could leave fewer than 100
usable rows. Here we keep drawing reserve candidates until each class reaches
the requested number of SUCCESSFULLY decoded instances.

Inputs
------
configs/prd_8class_train_v1.csv
configs/prd_8class_val_v1.csv

../fashion_data/raw/fashionpedia/
├── annotations/
│   └── instances_attributes_val2020.json
└── images/
    └── test/

Outputs
-------
configs/prd_8class_train_v2.csv

../fashion_data/processed/fashionpedia_train_expanded_v3_underrepresented/
└── full_masks/

reports/prd_instance_segmentation/prd_8class_train_expansion_v3/
├── summary.txt
├── class_counts_train_v2.csv
├── selected_fashionpedia_v3.csv
├── skipped_annotations.csv
├── validation_overlap_check.csv
└── contact_sheets/
    ├── shoe.jpg
    ├── bag.jpg
    └── accessory.jpg

Run
---
cd /workspace/fashion_multimodal_analysis
python scripts/build_prd_8class_train_v2_expand_fashionpedia_fixed.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "fashion_data"

TRAIN_V1 = PROJECT_ROOT / "configs" / "prd_8class_train_v1.csv"
VAL_V1 = PROJECT_ROOT / "configs" / "prd_8class_val_v1.csv"

RAW_FP_ROOT = DATA_ROOT / "raw" / "fashionpedia"
FP_ANN = RAW_FP_ROOT / "annotations" / "instances_attributes_val2020.json"
FP_IMAGES = RAW_FP_ROOT / "images" / "test"

PROCESSED_ROOT = DATA_ROOT / "processed" / "fashionpedia_train_expanded_v3_underrepresented"
FULL_MASK_DIR = PROCESSED_ROOT / "full_masks"

TRAIN_V2 = PROJECT_ROOT / "configs" / "prd_8class_train_v2.csv"
TRAIN_V3 = PROJECT_ROOT / "configs" / "prd_8class_train_v3.csv"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_instance_segmentation"
    / "prd_8class_train_expansion_v3"
)
SUMMARY_PATH = REPORT_DIR / "summary.txt"
COUNTS_PATH = REPORT_DIR / "class_counts_train_v3.csv"
SELECTED_PATH = REPORT_DIR / "selected_fashionpedia_v3.csv"
SKIPPED_PATH = REPORT_DIR / "skipped_annotations.csv"
OVERLAP_PATH = REPORT_DIR / "validation_overlap_check.csv"
CONTACT_DIR = REPORT_DIR / "contact_sheets"

TARGET_FP_CLASSES = (
    "top",
    "pants",
    "skirt",
    "outerwear",
    "dress",
)

TARGET_FINAL_COUNTS = {
    "top": 60,
    "pants": 60,
    "skirt": 50,
    "outerwear": 50,
    "dress": 60,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--contact-sheet-items", type=int, default=25)
    return p.parse_args()

def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV:\n  {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    return rows, fields


def write_csv(
    path: Path,
    rows: list[dict],
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fields is None:
        fields = list(rows[0].keys()) if rows else []

    with path.open("w", encoding="utf-8-sig", newline="") as f:
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
    return " ".join(
        str(name)
        .strip()
        .lower()
        .replace("&", "and")
        .split()
    )


def map_source_category(name: str) -> str | None:
    """
    Conservative Fashionpedia -> existing PRD 8-class mapping.

    Keep the coarse taxonomy aligned with the existing training data.
    Jumpsuit and garment parts/decorations are intentionally excluded.
    """
    n = normalize_name(name)

    mapping = {
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
    }

    return mapping.get(n)

def relative_to_project(path: Path) -> str:
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


def resolve_manifest_path(value: str) -> Path:
    p = Path(str(value).strip())
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def polygons_to_mask(
    polygons: list,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue

        pts = [
            (float(polygon[i]), float(polygon[i + 1]))
            for i in range(0, len(polygon) - 1, 2)
        ]

        if len(pts) >= 3:
            draw.polygon(pts, fill=255)

    return np.asarray(canvas, dtype=np.uint8)


def rle_to_mask(segmentation: dict) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError(
            "RLE annotation encountered but pycocotools is missing. "
            "Install with: pip install pycocotools"
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
    if isinstance(segmentation, list):
        return polygons_to_mask(segmentation, width, height)

    if isinstance(segmentation, dict):
        mask = rle_to_mask(segmentation)
        if mask.shape != (height, width):
            raise ValueError(
                f"RLE shape {mask.shape} != {(height, width)}"
            )
        return mask

    raise ValueError(
        f"Unsupported segmentation type: {type(segmentation).__name__}"
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


def build_image_index() -> dict[str, Path]:
    if not FP_IMAGES.is_dir():
        raise FileNotFoundError(
            f"Fashionpedia image directory not found:\n  {FP_IMAGES}"
        )

    index: dict[str, Path] = {}

    print(f"Indexing Fashionpedia images under:\n  {FP_IMAGES}")

    for pattern in (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    ):
        for path in FP_IMAGES.rglob(pattern):
            index.setdefault(path.name, path.resolve())

    print(f"Indexed image files: {len(index)}")
    return index


def build_candidate_order(
    candidates: list[dict],
    rng: random.Random,
) -> list[dict]:
    """
    Create a deterministic-but-randomized reserve order.

    Pass 1:
      round-robin fine categories, prefer unique source images.

    Pass 2:
      all remaining annotations, allowing repeated source images.

    This gives diversity while still leaving reserve candidates available when
    mask decoding fails.
    """
    by_fine: dict[str, list[dict]] = defaultdict(list)

    for item in candidates:
        by_fine[item["source_category"]].append(item)

    for pool in by_fine.values():
        rng.shuffle(pool)

    fine_names = list(by_fine.keys())
    rng.shuffle(fine_names)

    ordered = []
    used_ann_ids = set()
    used_image_ids = set()

    progress = True
    while progress:
        progress = False

        for fine in fine_names:
            pool = by_fine[fine]

            chosen_idx = None

            for i, item in enumerate(pool):
                ann_id = int(item["annotation"]["id"])
                image_id = int(item["image_meta"]["id"])

                if (
                    ann_id not in used_ann_ids
                    and image_id not in used_image_ids
                ):
                    chosen_idx = i
                    break

            if chosen_idx is None:
                continue

            item = pool.pop(chosen_idx)

            ordered.append(item)
            used_ann_ids.add(int(item["annotation"]["id"]))
            used_image_ids.add(int(item["image_meta"]["id"]))
            progress = True

    leftovers = []

    for pool in by_fine.values():
        leftovers.extend(pool)

    rng.shuffle(leftovers)

    for item in leftovers:
        ann_id = int(item["annotation"]["id"])
        if ann_id in used_ann_ids:
            continue
        ordered.append(item)
        used_ann_ids.add(ann_id)

    return ordered


def make_contact_sheet(
    rows: list[dict],
    class_name: str,
    max_items: int,
) -> None:
    if not rows:
        return

    rows = rows[:max_items]

    thumb_w = 280
    thumb_h = 280
    tile_w = 310
    tile_h = 350
    cols = 5
    rows_n = math.ceil(len(rows) / cols)

    sheet = Image.new(
        "RGB",
        (cols * tile_w, 50 + rows_n * tile_h),
        "white",
    )

    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    draw.text(
        (15, 15),
        f"Fashionpedia train v3 expansion - {class_name}",
        fill="black",
        font=font,
    )

    for i, row in enumerate(rows):
        r = i // cols
        c = i % cols

        x0 = c * tile_w
        y0 = 50 + r * tile_h

        image_path = resolve_manifest_path(
            row["source_image"]
        )

        image = Image.open(image_path).convert("RGB")
        d = ImageDraw.Draw(image)

        bbox = (
            int(row["bbox_x1"]),
            int(row["bbox_y1"]),
            int(row["bbox_x2"]),
            int(row["bbox_y2"]),
        )

        d.rectangle(
            bbox,
            outline=(0, 255, 0),
            width=3,
        )

        image.thumbnail(
            (thumb_w, thumb_h),
            Image.Resampling.LANCZOS,
        )

        px = x0 + (tile_w - image.width) // 2
        py = y0 + 4

        sheet.paste(image, (px, py))

        text_y = y0 + 290

        labels = [
            f"{row['garment_category']} <- {row['fine_category']}",
            f"ann={row['annotation_ref']}",
        ]

        for text in labels:
            draw.text(
                (x0 + 7, text_y),
                text,
                fill="black",
                font=font,
            )
            text_y += 18

    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(
        CONTACT_DIR / f"{class_name}.jpg",
        quality=92,
    )


def main() -> None:
    args = parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean only derived V3 masks.
    if FULL_MASK_DIR.exists():
        shutil.rmtree(FULL_MASK_DIR)

    FULL_MASK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Base = frozen V2 training manifest.
    # --------------------------------------------------------
    train_v2_rows, output_fields = read_csv(TRAIN_V2)
    val_rows, _ = read_csv(VAL_V1)

    base_counts = Counter(
        str(r["garment_category"]).strip().lower()
        for r in train_v2_rows
    )

    required_additions = {
        cls: max(
            0,
            TARGET_FINAL_COUNTS[cls]
            - base_counts.get(cls, 0),
        )
        for cls in TARGET_FP_CLASSES
    }

    print("=== V3-B1 BASE COUNTS ===")

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
        print(
            f"  {cls:10s}: "
            f"{base_counts.get(cls, 0)}"
        )

    print()
    print("=== REQUIRED NEW INSTANCES ===")

    for cls in TARGET_FP_CLASSES:
        print(
            f"  {cls:10s}: "
            f"+{required_additions[cls]} "
            f"-> target "
            f"{TARGET_FINAL_COUNTS[cls]}"
        )

    # --------------------------------------------------------
    # Leakage / duplicate guards.
    # --------------------------------------------------------
    val_fp_names = {
        Path(row["source_image"]).name
        for row in val_rows
        if (
            str(row["source_dataset"]).strip()
            == "Fashionpedia"
        )
    }

    existing_fp_train_names = {
        Path(row["source_image"]).name
        for row in train_v2_rows
        if (
            str(row["source_dataset"]).strip()
            == "Fashionpedia"
        )
    }

    existing_fp_ann_ids = set()

    for row in train_v2_rows:
        if (
            str(row["source_dataset"]).strip()
            != "Fashionpedia"
        ):
            continue

        value = str(
            row.get("annotation_ref", "")
        ).strip()

        if value.isdigit():
            existing_fp_ann_ids.add(int(value))

    print()
    print(
        "Frozen Fashionpedia val images:",
        len(val_fp_names),
    )

    print(
        "Existing V2 Fashionpedia train images:",
        len(existing_fp_train_names),
    )

    print(
        "Existing V2 Fashionpedia annotation IDs:",
        len(existing_fp_ann_ids),
    )

    # --------------------------------------------------------
    # Load Fashionpedia annotations.
    # --------------------------------------------------------
    if not FP_ANN.is_file():
        raise FileNotFoundError(
            f"Fashionpedia annotation not found:\n  {FP_ANN}"
        )

    with FP_ANN.open("r", encoding="utf-8") as f:
        data = json.load(f)

    category_by_id = {
        int(c["id"]): str(c["name"])
        for c in data["categories"]
    }

    image_by_id = {
        int(i["id"]): i
        for i in data["images"]
    }

    candidates: dict[str, list[dict]] = defaultdict(list)

    excluded_val = Counter()
    excluded_existing_image = Counter()
    excluded_existing_ann = Counter()

    for ann in data["annotations"]:
        source_cat = category_by_id.get(
            int(ann["category_id"]),
            "",
        )

        prd_class = map_source_category(
            source_cat
        )

        if prd_class not in TARGET_FP_CLASSES:
            continue

        image_meta = image_by_id.get(
            int(ann["image_id"])
        )

        if image_meta is None:
            continue

        ann_id = int(ann["id"])
        file_name = str(
            image_meta["file_name"]
        )
        basename = Path(file_name).name

        # Frozen validation guard.
        if basename in val_fp_names:
            excluded_val[prd_class] += 1
            continue

        # Unique-data guard:
        # do not reuse a Fashionpedia source image
        # already present in train_v2.
        if basename in existing_fp_train_names:
            excluded_existing_image[prd_class] += 1
            continue

        # Annotation-level duplicate guard.
        if ann_id in existing_fp_ann_ids:
            excluded_existing_ann[prd_class] += 1
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
    print(
        "=== AVAILABLE NEW CANDIDATES "
        "AFTER ALL EXCLUSIONS ==="
    )

    for cls in TARGET_FP_CLASSES:
        print(
            f"  {cls:10s}: "
            f"{len(candidates[cls])} "
            f"(need {required_additions[cls]})"
        )

    # Fail early if raw candidate count is insufficient.
    insufficient = []

    for cls in TARGET_FP_CLASSES:
        if (
            len(candidates[cls])
            < required_additions[cls]
        ):
            insufficient.append(
                (
                    cls,
                    len(candidates[cls]),
                    required_additions[cls],
                )
            )

    if insufficient:
        lines = [
            "Insufficient unique candidates:"
        ]

        for cls, available, needed in insufficient:
            lines.append(
                f"  {cls}: "
                f"available={available}, "
                f"needed={needed}"
            )

        raise RuntimeError(
            "\n".join(lines)
        )

    image_index = build_image_index()
    rng = random.Random(args.seed)

    new_rows: list[dict] = []
    skipped: list[dict] = []
    attempts_by_class = Counter()

    # Prefer globally new images across target classes too.
    used_new_image_ids: set[int] = set()

    # --------------------------------------------------------
    # Build requested additions.
    # --------------------------------------------------------
    for cls in TARGET_FP_CLASSES:
        needed = required_additions[cls]

        if needed <= 0:
            print()
            print(
                f"{cls}: already at target; "
                "no new instances required"
            )
            continue

        ordered = build_candidate_order(
            candidates[cls],
            rng,
        )

        # Prefer images not already selected for another
        # V3 target class. Fall back only if necessary.
        globally_new = [
            item
            for item in ordered
            if int(
                item["image_meta"]["id"]
            ) not in used_new_image_ids
        ]

        already_selected_image = [
            item
            for item in ordered
            if int(
                item["image_meta"]["id"]
            ) in used_new_image_ids
        ]

        ordered = (
            globally_new
            + already_selected_image
        )

        successful_rows: list[dict] = []

        print()
        print(
            f"Building {cls}: "
            f"need_new={needed}, "
            f"final_target="
            f"{TARGET_FINAL_COUNTS[cls]}"
        )

        for item in ordered:
            if len(successful_rows) >= needed:
                break

            attempts_by_class[cls] += 1

            ann = item["annotation"]
            image_meta = item["image_meta"]
            source_cat = item["source_category"]

            ann_id = int(ann["id"])
            image_id = int(
                image_meta["id"]
            )

            file_name = str(
                image_meta["file_name"]
            )

            image_path = image_index.get(
                Path(file_name).name
            )

            if image_path is None:
                skipped.append(
                    {
                        "prd_class": cls,
                        "annotation_id": ann_id,
                        "image_id": image_id,
                        "file_name": file_name,
                        "reason": "image_not_found",
                    }
                )
                continue

            try:
                image = Image.open(
                    image_path
                ).convert("RGB")

                width, height = image.size

                mask = segmentation_to_mask(
                    ann.get("segmentation"),
                    width,
                    height,
                )

                bbox = bbox_from_mask(mask)

                if bbox is None:
                    raise ValueError(
                        "empty_mask"
                    )

            except Exception as exc:
                skipped.append(
                    {
                        "prd_class": cls,
                        "annotation_id": ann_id,
                        "image_id": image_id,
                        "file_name": file_name,
                        "reason": str(exc),
                    }
                )
                continue

            x1, y1, x2, y2 = bbox

            stem = (
                f"{cls}_"
                f"ann{ann_id}_"
                f"img{image_id}"
            )

            mask_path = (
                FULL_MASK_DIR
                / f"{stem}.png"
            )

            Image.fromarray(
                mask
            ).save(mask_path)

            row = {
                "record_id": "",
                "source_dataset": "Fashionpedia",
                "source_image": relative_to_project(
                    image_path
                ),
                "garment_id": (
                    f"fashionpedia_ann_{ann_id}"
                ),
                "garment_category": cls,
                "fine_category": source_cat,
                "source_category_id": str(
                    ann["category_id"]
                ),
                "annotation_ref": str(ann_id),
                "crop_path": "",
                "mask_path": relative_to_project(
                    mask_path
                ),
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
                    "instances_attributes_val2020.json "
                    "expanded-train-v3-"
                    "underrepresented"
                ),
            }

            successful_rows.append(row)
            used_new_image_ids.add(image_id)

            if (
                len(successful_rows) % 10 == 0
                or len(successful_rows) == needed
            ):
                print(
                    f"  {cls}: "
                    f"{len(successful_rows)}/"
                    f"{needed} valid "
                    f"(attempted "
                    f"{attempts_by_class[cls]})"
                )

        if len(successful_rows) < needed:
            write_csv(
                SKIPPED_PATH,
                skipped,
                [
                    "prd_class",
                    "annotation_id",
                    "image_id",
                    "file_name",
                    "reason",
                ],
            )

            raise RuntimeError(
                f"Could only build "
                f"{len(successful_rows)} valid "
                f"{cls} instances; "
                f"needed {needed}. "
                f"See {SKIPPED_PATH}"
            )

        new_rows.extend(
            successful_rows
        )

    # --------------------------------------------------------
    # Final V3 manifest.
    # --------------------------------------------------------
    combined_rows = (
        train_v2_rows
        + new_rows
    )

    final_rows = []

    for i, row in enumerate(
        combined_rows,
        start=1,
    ):
        new_row = dict(row)

        new_row["record_id"] = (
            f"prd8v3_{i:04d}"
        )

        for field in output_fields:
            new_row.setdefault(
                field,
                "",
            )

        final_rows.append(new_row)

    # --------------------------------------------------------
    # Frozen validation overlap check.
    # --------------------------------------------------------
    val_keys = {
        (
            str(
                row["source_dataset"]
            ).strip(),
            Path(
                row["source_image"]
            ).name,
        )
        for row in val_rows
    }

    overlap_rows = []

    for row in final_rows:
        key = (
            str(
                row["source_dataset"]
            ).strip(),
            Path(
                row["source_image"]
            ).name,
        )

        if key in val_keys:
            overlap_rows.append(
                {
                    "source_dataset": key[0],
                    "source_image_basename": key[1],
                    "garment_id": row.get(
                        "garment_id",
                        "",
                    ),
                }
            )

    write_csv(
        OVERLAP_PATH,
        overlap_rows,
        [
            "source_dataset",
            "source_image_basename",
            "garment_id",
        ],
    )

    if overlap_rows:
        raise RuntimeError(
            "Train/validation overlap detected: "
            f"{len(overlap_rows)} rows. "
            f"See {OVERLAP_PATH}"
        )

    # --------------------------------------------------------
    # Verify no V3-added Fashionpedia image was already
    # present in V2.
    # --------------------------------------------------------
    reused_existing_images = []

    for row in new_rows:
        basename = Path(
            row["source_image"]
        ).name

        if basename in existing_fp_train_names:
            reused_existing_images.append(
                basename
            )

    if reused_existing_images:
        raise RuntimeError(
            "V3 unique-data guard failed: "
            f"{len(reused_existing_images)} "
            "new rows reuse V2 Fashionpedia images."
        )

    # Write only after all guards pass.
    write_csv(
        TRAIN_V3,
        final_rows,
        output_fields,
    )

    write_csv(
        SELECTED_PATH,
        new_rows,
        output_fields,
    )

    write_csv(
        SKIPPED_PATH,
        skipped,
        [
            "prd_class",
            "annotation_id",
            "image_id",
            "file_name",
            "reason",
        ],
    )

    # --------------------------------------------------------
    # Counts.
    # --------------------------------------------------------
    final_counts = Counter(
        str(
            row["garment_category"]
        ).strip().lower()
        for row in final_rows
    )

    count_rows = [
        {
            "garment_category": cls,
            "before_v2": base_counts.get(
                cls,
                0,
            ),
            "added_v3": (
                final_counts.get(cls, 0)
                - base_counts.get(cls, 0)
            ),
            "final_v3": final_counts.get(
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
            "before_v2",
            "added_v3",
            "final_v3",
        ],
    )

    # --------------------------------------------------------
    # Contact sheets of NEW rows only.
    # --------------------------------------------------------
    for cls in TARGET_FP_CLASSES:
        cls_rows = [
            r
            for r in new_rows
            if (
                r["garment_category"]
                == cls
            )
        ]

        preview_rows = cls_rows[:]
        rng.shuffle(preview_rows)

        make_contact_sheet(
            preview_rows,
            cls,
            args.contact_sheet_items,
        )

    added_counts = Counter(
        r["garment_category"]
        for r in new_rows
    )

    fine_counts = Counter(
        (
            r["garment_category"],
            r["fine_category"],
        )
        for r in new_rows
    )

    skip_counts = Counter(
        r["prd_class"]
        for r in skipped
    )

    unique_added_images = {
        Path(
            r["source_image"]
        ).name
        for r in new_rows
    }

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------
    summary = [
        "PRD 3.1.1 train v3 - unique-data expansion",
        "===========================================",
        "",
        "Experiment",
        "----------",
        "name=V3-B1 unique-data expansion",
        f"seed={args.seed}",
        "base_manifest=configs/prd_8class_train_v2.csv",
        "output_manifest=configs/prd_8class_train_v3.csv",
        "validation_manifest=configs/prd_8class_val_v1.csv",
        "",
        "Data integrity",
        "--------------",
        f"base_train_v2_rows={len(train_v2_rows)}",
        f"new_rows_added={len(new_rows)}",
        f"unique_new_source_images={len(unique_added_images)}",
        f"total_train_v3_rows={len(final_rows)}",
        f"frozen_fashionpedia_val_images={len(val_fp_names)}",
        "train_val_source_image_overlap=0",
        "reused_v2_fashionpedia_source_images=0",
        "validation_manifest_modified=False",
        "",
        "Class counts",
        "------------",
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
        before = base_counts.get(
            cls,
            0,
        )

        after = final_counts.get(
            cls,
            0,
        )

        summary.append(
            f"{cls}: "
            f"before={before}, "
            f"added={after - before}, "
            f"final={after}"
        )

    summary += [
        "",
        "Candidate diagnostics",
        "---------------------",
    ]

    for cls in TARGET_FP_CLASSES:
        summary.append(
            f"{cls}: "
            f"available_after_exclusion="
            f"{len(candidates[cls])}, "
            f"needed={required_additions[cls]}, "
            f"attempted="
            f"{attempts_by_class.get(cls, 0)}, "
            f"valid_added="
            f"{added_counts.get(cls, 0)}, "
            f"decode_skipped="
            f"{skip_counts.get(cls, 0)}, "
            f"excluded_val="
            f"{excluded_val.get(cls, 0)}, "
            f"excluded_existing_image="
            f"{excluded_existing_image.get(cls, 0)}, "
            f"excluded_existing_ann="
            f"{excluded_existing_ann.get(cls, 0)}"
        )

    summary += [
        "",
        "New Fashionpedia fine-category counts",
        "------------------------------------",
    ]

    for (
        prd_cls,
        fine_cls,
    ), count in sorted(
        fine_counts.items()
    ):
        summary.append(
            f"{prd_cls} <- "
            f"{fine_cls}: {count}"
        )

    summary += [
        "",
        "Notes",
        "-----",
        "- V2 training rows were preserved and V3 only appends new Fashionpedia annotations.",
        "- Existing V2 Fashionpedia source images were excluded before sampling.",
        "- Frozen Fashionpedia validation source images were excluded before sampling.",
        "- New source images were preferred across target classes to maximize visual diversity.",
        "- Jumpsuit and garment-part/decorative categories were intentionally excluded.",
        "- V2 shoe, bag, and accessory rows were retained without targeted expansion.",
        "- Full-image masks were generated from Fashionpedia segmentation annotations.",
        "- Review the V3 contact sheets before starting training.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== TRAIN V3 BUILD FINISHED ===")
    print(
        f"Base V2 rows : "
        f"{len(train_v2_rows)}"
    )
    print(
        f"New rows     : "
        f"{len(new_rows)}"
    )
    print(
        f"Unique images: "
        f"{len(unique_added_images)}"
    )
    print(
        f"Total V3 rows: "
        f"{len(final_rows)}"
    )
    print()

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
        print(
            f"  {cls:10s}: "
            f"{final_counts.get(cls, 0)}"
        )

    print()
    print(
        "Train/val source-image overlap: 0"
    )
    print(
        "Reused V2 Fashionpedia images: 0"
    )
    print(
        f"Skipped decode attempts: "
        f"{len(skipped)}"
    )
    print(
        "Output manifest:",
        TRAIN_V3.relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        "Summary:",
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()
