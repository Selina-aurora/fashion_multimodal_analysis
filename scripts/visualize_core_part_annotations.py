"""Generate visual QC sheets for the core part bbox annotations.

Reads:
    reports/prd_region_coverage/core_part_annotation_labels.csv

Uses:
    outputs/core_part_annotation_pilot/images/

Writes:
    reports/prd_region_coverage/core_part_annotation_qc_summary.csv
    outputs/core_part_annotation_qc/
        collar_qc_contact_sheet_01.jpg
        cuff_qc_contact_sheet_01.jpg
        hem_qc_contact_sheet_01.jpg

This script does not modify the labels.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "prd_region_coverage"
LABEL_FILE = REPORT_ROOT / "core_part_annotation_labels.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "core_part_annotation_qc"
SUMMARY_FILE = REPORT_ROOT / "core_part_annotation_qc_summary.csv"

REGIONS = ("collar", "cuff", "hem")


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing labels: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows."""
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a readable font."""
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def resolve_project_path(raw_path: str) -> Path:
    """Resolve project-relative path saved with Windows separators."""
    return PROJECT_ROOT / Path(raw_path.replace("\\", "/"))


def parse_boxes(raw: str) -> list[list[int]]:
    """Parse bbox JSON."""
    if not raw:
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("boxes_json must be a list.")
    return data


def validate_box(
    box: list[int],
    width: int,
    height: int,
) -> tuple[bool, str]:
    """Validate one bbox against image size."""
    if len(box) != 4:
        return False, "bbox does not have 4 values"

    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        return False, "non-positive bbox size"
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False, "bbox outside image bounds"
    if x2 - x1 < 5 or y2 - y1 < 5:
        return False, "bbox is too small"

    return True, ""


def draw_annotation(
    image: Image.Image,
    row: dict[str, str],
    boxes: list[list[int]],
) -> Image.Image:
    """Draw saved boxes and annotation metadata."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = load_font(15)

    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
        draw.text(
            (x1 + 3, max(32, y1 + 3)),
            f"{row['region']} {index}",
            fill="red",
            font=font,
        )

    banner = (
        f"{row['annotation_id']} | {row['source_candidate_id']} | "
        f"{row['label_status']} | boxes={len(boxes)}"
    )
    draw.rectangle((0, 0, canvas.width, 28), fill="white")
    draw.text((5, 6), banner, fill="black", font=font)

    return canvas


def fit_tile(
    image: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    """Letterbox image into a fixed tile."""
    tile = Image.new("RGB", size, "white")
    copy = image.copy()
    copy.thumbnail((size[0] - 10, size[1] - 10))
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    tile.paste(copy, (x, y))
    return tile


def make_contact_sheets(
    region: str,
    images: list[Image.Image],
    columns: int = 3,
    rows_per_sheet: int = 5,
) -> None:
    """Create annotation QC contact sheets."""
    tile_size = (360, 430)
    per_sheet = columns * rows_per_sheet

    region_dir = OUTPUT_DIR / region
    region_dir.mkdir(parents=True, exist_ok=True)

    for sheet_index, start in enumerate(
        range(0, len(images), per_sheet),
        start=1,
    ):
        batch = images[start : start + per_sheet]
        needed_rows = math.ceil(len(batch) / columns)

        sheet = Image.new(
            "RGB",
            (tile_size[0] * columns, tile_size[1] * needed_rows),
            "white",
        )

        for local_index, image in enumerate(batch):
            tile = fit_tile(image, tile_size)
            col = local_index % columns
            row_index = local_index // columns
            sheet.paste(
                tile,
                (col * tile_size[0], row_index * tile_size[1]),
            )

        output_path = (
            region_dir
            / f"{region}_qc_contact_sheet_{sheet_index:02d}.jpg"
        )
        sheet.save(output_path, quality=92)


def main() -> None:
    """Validate annotations and generate visual QC sheets."""
    rows = read_csv(LABEL_FILE)
    summary_rows: list[dict[str, Any]] = []
    visual_by_region: dict[str, list[Image.Image]] = {
        region: [] for region in REGIONS
    }

    for row in rows:
        image_path = resolve_project_path(row["annotation_image_path"])
        if not image_path.is_file():
            summary_rows.append(
                {
                    "annotation_id": row["annotation_id"],
                    "region": row["region"],
                    "label_status": row["label_status"],
                    "num_boxes": 0,
                    "image_found": "no",
                    "box_validation": "not_checked",
                    "issue": f"missing image: {image_path}",
                }
            )
            continue

        image = Image.open(image_path).convert("RGB")
        boxes = parse_boxes(row["boxes_json"])

        issues = []
        for box in boxes:
            valid, issue = validate_box(
                box,
                width=image.width,
                height=image.height,
            )
            if not valid:
                issues.append(issue)

        if row["label_status"] == "labeled" and not boxes:
            issues.append("labeled row has no boxes")
        if row["label_status"] == "unusable" and boxes:
            issues.append("unusable row unexpectedly has boxes")

        summary_rows.append(
            {
                "annotation_id": row["annotation_id"],
                "region": row["region"],
                "label_status": row["label_status"],
                "num_boxes": len(boxes),
                "image_found": "yes",
                "box_validation": "pass" if not issues else "fail",
                "issue": "; ".join(sorted(set(issues))),
            }
        )

        visual_by_region[row["region"]].append(
            draw_annotation(
                image=image,
                row=row,
                boxes=boxes,
            )
        )

    write_csv(SUMMARY_FILE, summary_rows)

    for region, images in visual_by_region.items():
        make_contact_sheets(region, images)

    labeled = sum(
        row["label_status"] == "labeled"
        for row in rows
    )
    unusable = sum(
        row["label_status"] == "unusable"
        for row in rows
    )
    total_boxes = sum(
        len(parse_boxes(row["boxes_json"]))
        for row in rows
    )
    failures = sum(
        row["box_validation"] == "fail"
        for row in summary_rows
    )

    print("Finished.")
    print(f"Rows: {len(rows)}")
    print(f"Labeled: {labeled}")
    print(f"Unusable: {unusable}")
    print(f"Total boxes: {total_boxes}")
    print(f"Validation failures: {failures}")
    print(f"Summary: {SUMMARY_FILE}")
    print(f"Visual QC: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
