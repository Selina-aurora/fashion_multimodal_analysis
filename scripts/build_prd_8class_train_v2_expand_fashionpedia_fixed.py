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

../fashion_data/processed/fashionpedia_train_expanded_v2/
└── full_masks/

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=100)
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
        f"Fashionpedia train v2 - {class_name}",
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

    # Clean only derived v2 masks from any failed previous attempt.
    if FULL_MASK_DIR.exists():
        shutil.rmtree(FULL_MASK_DIR)
    FULL_MASK_DIR.mkdir(parents=True, exist_ok=True)

    train_v1_rows, output_fields = read_csv(TRAIN_V1)
    val_rows, _ = read_csv(VAL_V1)

    deep_train_rows = [
        row
        for row in train_v1_rows
        if row["source_dataset"].strip() == "DeepFashion2"
    ]

    old_fp_train_rows = [
        row
        for row in train_v1_rows
        if row["source_dataset"].strip() == "Fashionpedia"
    ]

    val_fp_names = {
        Path(row["source_image"]).name
        for row in val_rows
        if row["source_dataset"].strip() == "Fashionpedia"
    }

    print(f"DeepFashion2 train rows kept: {len(deep_train_rows)}")
    print(f"Old Fashionpedia train rows: {len(old_fp_train_rows)}")
    print(f"Frozen Fashionpedia val image names: {len(val_fp_names)}")

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

    for ann in data["annotations"]:
        source_cat = category_by_id.get(
            int(ann["category_id"]),
            "",
        )

        prd_class = map_source_category(source_cat)

        if prd_class not in TARGET_FP_CLASSES:
            continue

        image_meta = image_by_id.get(
            int(ann["image_id"])
        )

        if image_meta is None:
            continue

        file_name = str(image_meta["file_name"])

        # Fixed-validation leakage guard.
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

    print("\nAvailable candidates after validation exclusion:")
    for cls in TARGET_FP_CLASSES:
        print(f"  {cls}: {len(candidates[cls])}")

    image_index = build_image_index()
    rng = random.Random(args.seed)

    fp_rows: list[dict] = []
    skipped: list[dict] = []
    attempts_by_class = Counter()

    for cls in TARGET_FP_CLASSES:
        ordered = build_candidate_order(
            candidates[cls],
            rng,
        )

        successful_rows = []

        print()
        print(f"Building {cls}: target={args.per_class}")

        for item in ordered:
            if len(successful_rows) >= args.per_class:
                break

            attempts_by_class[cls] += 1

            ann = item["annotation"]
            image_meta = item["image_meta"]
            source_cat = item["source_category"]

            ann_id = int(ann["id"])
            image_id = int(image_meta["id"])
            file_name = str(image_meta["file_name"])

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

            mask_path = FULL_MASK_DIR / f"{stem}.png"

            Image.fromarray(mask).save(mask_path)

            row = {
                "record_id": "",
                "source_dataset": "Fashionpedia",
                "source_image": relative_to_project(image_path),
                "garment_id": f"fashionpedia_ann_{ann_id}",
                "garment_category": cls,
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
                    "Fashionpedia instances_attributes_val2020.json "
                    "expanded-train-v2"
                ),
            }

            successful_rows.append(row)

            if (
                len(successful_rows) % 20 == 0
                or len(successful_rows) == args.per_class
            ):
                print(
                    f"  {cls}: "
                    f"{len(successful_rows)}/{args.per_class} valid "
                    f"(attempted {attempts_by_class[cls]})"
                )

        if len(successful_rows) < args.per_class:
            # Write the skip report BEFORE failing.
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
                f"Could only build {len(successful_rows)} valid {cls} "
                f"instances after trying {attempts_by_class[cls]} candidates. "
                f"See {SKIPPED_PATH}"
            )

        fp_rows.extend(successful_rows)

    # Fixed validation overlap check using dataset + source-image basename.
    val_keys = {
        (
            row["source_dataset"].strip(),
            Path(row["source_image"]).name,
        )
        for row in val_rows
    }

    overlap_rows = []

    for row in deep_train_rows + fp_rows:
        key = (
            row["source_dataset"].strip(),
            Path(row["source_image"]).name,
        )

        if key in val_keys:
            overlap_rows.append(
                {
                    "source_dataset": key[0],
                    "source_image_basename": key[1],
                    "garment_id": row.get("garment_id", ""),
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
            f"Train/validation source-image overlap detected: "
            f"{len(overlap_rows)} rows. See {OVERLAP_PATH}"
        )

    # Build final train_v2.
    train_v2 = deep_train_rows + fp_rows

    final_rows = []

    for i, row in enumerate(train_v2, start=1):
        new_row = dict(row)
        new_row["record_id"] = f"prd8v2_{i:04d}"

        # Ensure every current schema field exists.
        for field in output_fields:
            new_row.setdefault(field, "")

        final_rows.append(new_row)

    write_csv(
        TRAIN_V2,
        final_rows,
        output_fields,
    )

    write_csv(
        SELECTED_PATH,
        fp_rows,
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

    counts = Counter(
        row["garment_category"]
        for row in final_rows
    )

    count_rows = [
        {
            "garment_category": cls,
            "count": counts.get(cls, 0),
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
        ["garment_category", "count"],
    )

    # Contact sheets.
    for cls in TARGET_FP_CLASSES:
        cls_rows = [
            r
            for r in fp_rows
            if r["garment_category"] == cls
        ]

        preview_rows = cls_rows[:]
        rng.shuffle(preview_rows)

        make_contact_sheet(
            preview_rows,
            cls,
            args.contact_sheet_items,
        )

    fp_counts = Counter(
        r["garment_category"]
        for r in fp_rows
    )

    fine_counts = Counter(
        (
            r["garment_category"],
            r["fine_category"],
        )
        for r in fp_rows
    )

    skip_counts = Counter(
        r["prd_class"]
        for r in skipped
    )

    summary = [
        "PRD 3.1.1 train v2 - Fashionpedia expansion (fixed)",
        "====================================================",
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
        "Fashionpedia build diagnostics",
        "------------------------------",
    ]

    for cls in TARGET_FP_CLASSES:
        summary.append(
            f"{cls}: valid={fp_counts.get(cls, 0)}, "
            f"attempted={attempts_by_class.get(cls, 0)}, "
            f"skipped={skip_counts.get(cls, 0)}"
        )

    summary += [
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
        summary.append(
            f"{cls}={counts.get(cls, 0)}"
        )

    summary += [
        "",
        "Fashionpedia fine-category counts",
        "--------------------------------",
    ]

    for (prd_cls, fine_cls), count in sorted(
        fine_counts.items()
    ):
        summary.append(
            f"{prd_cls} <- {fine_cls}: {count}"
        )

    summary += [
        "",
        "Notes",
        "-----",
        "- Existing validation split was not changed.",
        "- Fashionpedia validation source images were excluded before sampling.",
        "- Decode failures were backfilled from reserve candidates until each target class reached the requested valid count.",
        "- DeepFashion2 train rows were preserved from train_v1.",
        "- New Fashionpedia masks are stored as full-image masks.",
        "- Use train_v2 for baseline-v2 training and compare on the same fixed val_v1.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== TRAIN V2 BUILD FINISHED ===")
    print(f"DeepFashion2 train rows: {len(deep_train_rows)}")
    print(f"Fashionpedia train rows: {len(fp_rows)}")
    print(f"Total train rows       : {len(final_rows)}")
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
            f"{counts.get(cls, 0)}"
        )

    print()
    print("Train/val source-image overlap: 0")
    print(f"Skipped decode attempts: {len(skipped)}")
    print(f"Output manifest: {TRAIN_V2.relative_to(PROJECT_ROOT)}")
    print(f"Summary: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
