"""Build manual-screening candidate sheets for PRD 3.1.2 regions.

This script does NOT pretend that DeepFashion2 provides ground-truth labels for
all PRD local regions. It only uses DeepFashion2 garment annotations to crop
clear garment instances, then creates deterministic candidate sheets for manual
verification of the PRD regions:

collar, cuff, hem, pocket, shoulder, waist, pattern, decoration.

Recommended workflow:
1. Run this script to generate candidate crops/contact sheets.
2. Manually inspect each region and mark target_present=yes/no/unclear in the CSV.
3. Keep at least 5 clear verified-positive examples per available region.
4. If a region cannot provide enough reliable positives, record it as a dataset
   coverage gap instead of fabricating labels.

The existing 88-case sleeve/collar/button/zipper benchmark remains a separate
diagnostic benchmark and is not overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "fashion_data" / "raw" / "train" / "train"
IMAGE_DIR = DATASET_ROOT / "image"
ANNO_DIR = DATASET_ROOT / "annos"

REPORT_DIR = PROJECT_ROOT / "reports" / "prd_region_coverage"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "prd_region_coverage"

DEFAULT_PER_REGION = 24
DEFAULT_SEED = 20260914

CATEGORY_NAMES = {
    1: "short sleeve top",
    2: "long sleeve top",
    3: "short sleeve outwear",
    4: "long sleeve outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress",
}

PRD_REGIONS = (
    "collar",
    "cuff",
    "hem",
    "pocket",
    "shoulder",
    "waist",
    "pattern",
    "decoration",
)

# These are candidate-pool filters only. They do not imply that a target is
# present. Human verification remains mandatory.
REGION_CATEGORY_PREFERENCES = {
    "collar": {1, 2, 3, 4, 5, 10, 11, 12},
    "cuff": {1, 2, 3, 4, 10, 11},
    "hem": {1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13},
    "pocket": {2, 3, 4, 5, 7, 8, 9, 10, 11},
    "shoulder": {1, 2, 3, 4, 5, 10, 11, 12, 13},
    "waist": {7, 8, 9, 10, 11, 12, 13},
    "pattern": set(CATEGORY_NAMES),
    "decoration": set(CATEGORY_NAMES),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build PRD 3.1.2 manual-screening candidate sheets."
    )
    parser.add_argument(
        "--per-region",
        type=int,
        default=DEFAULT_PER_REGION,
        help=f"Candidate count per region (default: {DEFAULT_PER_REGION}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--max-annotations",
        type=int,
        default=None,
        help="Optional annotation-file limit for a quick smoke test.",
    )
    args = parser.parse_args()

    if args.per_region <= 0:
        raise ValueError("--per-region must be greater than zero.")
    if args.max_annotations is not None and args.max_annotations <= 0:
        raise ValueError("--max-annotations must be greater than zero.")
    return args


def iter_items(annotation: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return DeepFashion2 item entries from one annotation dictionary."""
    items = []
    for key, value in annotation.items():
        if key.startswith("item") and isinstance(value, dict):
            items.append((key, value))
    return items


def normalize_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Clip a DeepFashion2 x1/y1/x2/y2 bounding box to image bounds."""
    if len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [float(value) for value in bbox]
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(left + 1, min(width, int(round(x2))))
    bottom = max(top + 1, min(height, int(round(y2))))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    margin_ratio: float = 0.04,
) -> tuple[int, int, int, int]:
    """Expand a bbox slightly while remaining inside the image."""
    left, top, right, bottom = bbox
    box_width = right - left
    box_height = bottom - top

    margin_x = int(round(box_width * margin_ratio))
    margin_y = int(round(box_height * margin_ratio))

    return (
        max(0, left - margin_x),
        max(0, top - margin_y),
        min(width, right + margin_x),
        min(height, bottom + margin_y),
    )


def collect_candidates(
    annotation_paths: list[Path],
) -> list[dict[str, Any]]:
    """Collect reasonably clear garment instances from DeepFashion2."""
    candidates = []

    for annotation_path in annotation_paths:
        image_name = f"{annotation_path.stem}.jpg"
        image_path = IMAGE_DIR / image_name
        if not image_path.is_file():
            continue

        try:
            with annotation_path.open("r", encoding="utf-8") as file:
                annotation = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError:
            continue

        for item_id, item in iter_items(annotation):
            category_id = item.get("category_id")
            if not isinstance(category_id, int):
                continue
            if category_id not in CATEGORY_NAMES:
                continue

            raw_bbox = item.get("bounding_box") or item.get("bbox")
            if not isinstance(raw_bbox, (list, tuple)):
                continue

            bbox = normalize_bbox(raw_bbox, width, height)
            if bbox is None:
                continue

            left, top, right, bottom = bbox
            box_area = (right - left) * (bottom - top)
            image_area = max(1, width * height)
            area_ratio = box_area / image_area

            # Remove extremely tiny garment instances. We still retain a broad
            # range because some local-region tasks involve modest-size garments.
            if area_ratio < 0.18:
                continue

            candidates.append(
                {
                    "image_name": image_name,
                    "item_id": item_id,
                    "category_id": category_id,
                    "category_name": CATEGORY_NAMES[category_id],
                    "bbox": bbox,
                    "bbox_area_ratio": area_ratio,
                }
            )

    return candidates


def region_pool(
    all_candidates: list[dict[str, Any]],
    region: str,
) -> list[dict[str, Any]]:
    """Filter the global pool using broad garment-category preferences."""
    preferred = REGION_CATEGORY_PREFERENCES[region]
    pool = [
        candidate
        for candidate in all_candidates
        if candidate["category_id"] in preferred
    ]
    return pool


def sample_unique_images(
    pool: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample candidates while preferring unique source images."""
    shuffled = list(pool)
    rng.shuffle(shuffled)

    selected = []
    used_images = set()

    for candidate in shuffled:
        if candidate["image_name"] in used_images:
            continue
        selected.append(candidate)
        used_images.add(candidate["image_name"])
        if len(selected) >= count:
            return selected

    # Fallback if unique-image filtering cannot fill the request.
    for candidate in shuffled:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= count:
            break

    return selected


def save_crop(
    candidate: dict[str, Any],
    region: str,
    candidate_index: int,
) -> Path:
    """Save one garment-focused candidate crop."""
    image_path = IMAGE_DIR / candidate["image_name"]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        crop_box = expand_bbox(
            candidate["bbox"],
            width=image.width,
            height=image.height,
        )
        crop = image.crop(crop_box)

    region_dir = OUTPUT_DIR / region / "crops"
    region_dir.mkdir(parents=True, exist_ok=True)

    output_name = (
        f"{candidate_index:03d}_"
        f"{Path(candidate['image_name']).stem}_"
        f"{candidate['item_id']}.jpg"
    )
    output_path = region_dir / output_name
    crop.save(output_path, quality=92)
    return output_path


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a readable font with a safe fallback."""
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def make_contact_sheets(
    region: str,
    rows: list[dict[str, Any]],
    columns: int = 4,
    rows_per_sheet: int = 6,
) -> None:
    """Create numbered contact sheets for one PRD region."""
    tile_width = 280
    tile_height = 340
    image_height = 285
    caption_height = tile_height - image_height

    font = load_font(14)
    sheet_size = columns * rows_per_sheet

    sheet_dir = OUTPUT_DIR / region / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)

    for sheet_index, start in enumerate(
        range(0, len(rows), sheet_size),
        start=1,
    ):
        batch = rows[start : start + sheet_size]
        needed_rows = (len(batch) + columns - 1) // columns
        sheet = Image.new(
            "RGB",
            (columns * tile_width, needed_rows * tile_height),
            "white",
        )

        for local_index, row in enumerate(batch):
            crop_path = PROJECT_ROOT / row["candidate_crop_path"]
            with Image.open(crop_path) as source:
                crop = source.convert("RGB")

            crop.thumbnail((tile_width - 12, image_height - 12))

            tile = Image.new("RGB", (tile_width, tile_height), "white")
            x = (tile_width - crop.width) // 2
            y = (image_height - crop.height) // 2
            tile.paste(crop, (x, y))

            draw = ImageDraw.Draw(tile)
            draw.rectangle(
                (0, 0, tile_width - 1, tile_height - 1),
                outline="black",
                width=1,
            )
            caption = (
                f"{row['candidate_id']} | {row['category_name']}\n"
                f"{row['image_name']} | {row['item_id']}"
            )
            draw.text((6, image_height + 4), caption, fill="black", font=font)

            col = local_index % columns
            row_index = local_index // columns
            sheet.paste(tile, (col * tile_width, row_index * tile_height))

        sheet.save(
            sheet_dir / f"{region}_contact_sheet_{sheet_index:02d}.jpg",
            quality=92,
        )


def write_region_csv(region: str, rows: list[dict[str, Any]]) -> Path:
    """Write the manual-review CSV for one region."""
    path = REPORT_DIR / f"{region}_candidates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "candidate_id",
        "region",
        "image_name",
        "item_id",
        "category_id",
        "category_name",
        "bbox_area_ratio",
        "candidate_crop_path",
        "target_present",
        "visibility",
        "usable_for_evaluation",
        "review_note",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def write_coverage_template() -> None:
    """Create a PRD coverage-status template."""
    path = REPORT_DIR / "prd_region_availability.csv"
    rows = []

    for region in PRD_REGIONS:
        if region == "collar":
            current_status = "existing benchmark available"
        elif region == "decoration":
            current_status = "button/zipper diagnostic evidence available"
        else:
            current_status = "candidate screening required"

        rows.append(
            {
                "region": region,
                "prd_required": "yes",
                "current_status": current_status,
                "verified_positive_count": "",
                "dataset_status": "to_review",
                "experiment_status": "pending",
                "note": "",
            }
        )

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build deterministic manual-screening candidate sheets."""
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotation_paths = sorted(ANNO_DIR.glob("*.json"))
    if args.max_annotations is not None:
        annotation_paths = annotation_paths[: args.max_annotations]

    print(f"Annotation files: {len(annotation_paths)}")
    all_candidates = collect_candidates(annotation_paths)
    print(f"Usable garment instances: {len(all_candidates)}")

    if not all_candidates:
        raise RuntimeError(
            "No garment candidates found. Check DATASET_ROOT and annotation format."
        )

    master_rows = []

    for region_index, region in enumerate(PRD_REGIONS):
        pool = region_pool(all_candidates, region)
        rng = random.Random(args.seed + region_index)
        selected = sample_unique_images(pool, args.per_region, rng)

        print(
            f"{region}: pool={len(pool)}, selected={len(selected)}"
        )

        rows = []
        for index, candidate in enumerate(selected, start=1):
            crop_path = save_crop(candidate, region, index)
            relative_crop = crop_path.relative_to(PROJECT_ROOT)

            row = {
                "candidate_id": f"{region}_{index:03d}",
                "region": region,
                "image_name": candidate["image_name"],
                "item_id": candidate["item_id"],
                "category_id": candidate["category_id"],
                "category_name": candidate["category_name"],
                "bbox_area_ratio": round(candidate["bbox_area_ratio"], 4),
                "candidate_crop_path": str(relative_crop),
                "target_present": "",
                "visibility": "",
                "usable_for_evaluation": "",
                "review_note": "",
            }
            rows.append(row)
            master_rows.append(row)

        write_region_csv(region, rows)
        make_contact_sheets(region, rows)

    master_path = REPORT_DIR / "all_prd_region_candidates.csv"
    with master_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(master_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(master_rows)

    write_coverage_template()

    print("\nFinished.")
    print(f"Candidate CSVs: {REPORT_DIR}")
    print(f"Contact sheets: {OUTPUT_DIR}")
    print(
        "\nManual review values:\n"
        "  target_present: yes / no / unclear\n"
        "  visibility: clear / partial / tiny / occluded\n"
        "  usable_for_evaluation: yes / no\n"
        "\nDo not treat these candidates as ground truth before manual review."
    )


if __name__ == "__main__":
    main()
