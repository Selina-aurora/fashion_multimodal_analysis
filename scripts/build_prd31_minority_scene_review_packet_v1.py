"""
Build targeted human-review packet for remaining minority-scene Core cases.

Rationale
---------
The first high-confidence CLIP spot-check achieved only moderate agreement with
manual scene labels. Therefore CLIP scene suggestions are NOT reliable enough
to be treated as human-final metadata.

Instead of manually reviewing all remaining Core cases, this packet targets the
minority scene types that materially determine whether Core has broad scene
coverage:
- product_display
- complex_scene

Cases already manually reviewed in:
- scene_low_conf_human_review_v1.csv
- high_confidence_scene_spotcheck_human_v1.csv
are excluded.

Inputs
------
benchmark/prd_3_1_v1/freeze_v1/provisional_selection_v1/
    provisional_core_400_v1.csv

Files placed in:
benchmark/prd_3_1_v1/freeze_v1/provisional_selection_v1/human_scene_reviews/
    scene_low_conf_human_review_v1.csv
    high_confidence_scene_spotcheck_human_v1.csv

Outputs
-------
benchmark/prd_3_1_v1/freeze_v1/provisional_selection_v1/
minority_scene_review_v1/
├── minority_scene_review_v1.csv
├── minority_scene_review_summary.txt
└── contact_sheets/
    ├── minority_scene_01.jpg
    ├── ...
    └── minority_scene_XX.jpg

Run
---
cd /workspace/fashion_multimodal_analysis

python scripts/build_prd31_minority_scene_review_packet_v1.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOT = (
    PROJECT_ROOT
    / "benchmark"
    / "prd_3_1_v1"
    / "freeze_v1"
    / "provisional_selection_v1"
)

CORE_CSV = ROOT / "provisional_core_400_v1.csv"
REVIEW_DIR = ROOT / "human_scene_reviews"

LOW_REVIEW = REVIEW_DIR / "scene_low_conf_human_review_v1.csv"
SPOT_REVIEW = REVIEW_DIR / "high_confidence_scene_spotcheck_human_v1.csv"

OUT_DIR = ROOT / "minority_scene_review_v1"
SHEET_DIR = OUT_DIR / "contact_sheets"

TARGET_SCENES = {
    "product_display",
    "complex_scene",
}

CLASSES = [
    "top",
    "pants",
    "skirt",
    "outerwear",
    "dress",
    "shoe",
    "bag",
    "accessory",
]


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def resolve_project(raw: str) -> Path:
    p = Path(str(raw).strip())
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def build_sheets(rows):
    SHEET_DIR.mkdir(parents=True, exist_ok=True)

    page_size = 8
    cols = 2
    rows_per_page = 4
    tile_w = 760
    tile_h = 330

    for start in range(0, len(rows), page_size):
        page_rows = rows[start:start + page_size]
        page_no = start // page_size + 1

        canvas = Image.new(
            "RGB",
            (cols * tile_w, rows_per_page * tile_h),
            "white",
        )

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        for i, row in enumerate(page_rows):
            rr = i // cols
            cc = i % cols

            x0 = cc * tile_w
            y0 = rr * tile_h

            source_path = resolve_project(row["source_image"])
            overlay_path = resolve_project(row["gt_overlay_path"])

            try:
                full = Image.open(source_path).convert("RGB")
                overlay = Image.open(overlay_path).convert("RGB")
            except Exception:
                continue

            full.thumbnail((340, 220))
            overlay.thumbnail((340, 220))

            canvas.paste(
                full,
                (
                    x0 + 10 + (340 - full.width)//2,
                    y0 + 70 + (220 - full.height)//2,
                ),
            )

            canvas.paste(
                overlay,
                (
                    x0 + 390 + (340 - overlay.width)//2,
                    y0 + 70 + (220 - overlay.height)//2,
                ),
            )

            title = (
                f"{row['review_sample_id']} | "
                f"{row['garment_category']} | "
                f"{row['fine_or_source_category']}"
            )

            meta = (
                f"suggested={row['scene_type_suggested']} | "
                f"p={float(row['scene_confidence']):.2f} | "
                f"margin={float(row['scene_margin']):.2f} | "
                f"size={row['size_bucket']}"
            )

            draw.text(
                (x0 + 8, y0 + 8),
                title,
                fill="black",
                font=font,
            )

            draw.text(
                (x0 + 8, y0 + 28),
                meta,
                fill="black",
                font=font,
            )

            draw.text(
                (x0 + 8, y0 + 48),
                "Human scene: worn_person / product_display / partial_view / complex_scene",
                fill="black",
                font=font,
            )

            draw.rectangle(
                (
                    x0 + 2,
                    y0 + 2,
                    x0 + tile_w - 3,
                    y0 + tile_h - 3,
                ),
                outline="gray",
                width=1,
            )

        out = SHEET_DIR / f"minority_scene_{page_no:02d}.jpg"
        canvas.save(out, quality=92)


def main():
    rows = read_csv(CORE_CSV)

    already_reviewed = set()

    for p in [LOW_REVIEW, SPOT_REVIEW]:
        for r in read_csv(p):
            already_reviewed.add(r["review_sample_id"])

    target = [
        r
        for r in rows
        if r["review_sample_id"] not in already_reviewed
        and r["scene_type_suggested"] in TARGET_SCENES
    ]

    target.sort(
        key=lambda r: (
            CLASSES.index(r["garment_category"]),
            r["scene_type_suggested"],
            r["review_sample_id"],
        )
    )

    out_rows = []

    for r in target:
        o = dict(r)
        o["scene_type_human"] = ""
        o["scene_review_decision"] = ""
        o["scene_suggestion_match"] = ""
        o["scene_label_source"] = "PENDING_HUMAN_REVIEW"
        o["scene_review_note"] = ""
        out_rows.append(o)

    write_csv(
        OUT_DIR / "minority_scene_review_v1.csv",
        out_rows,
    )

    build_sheets(out_rows)

    by_scene = Counter(
        r["scene_type_suggested"]
        for r in out_rows
    )

    by_class = Counter(
        r["garment_category"]
        for r in out_rows
    )

    lines = [
        "PRD 3.1.1 Minority Scene Human Review v1",
        "========================================",
        "",
        f"cases={len(out_rows)}",
        f"product_display={by_scene['product_display']}",
        f"complex_scene={by_scene['complex_scene']}",
        "",
        "By class",
        "--------",
    ]

    for cls in CLASSES:
        lines.append(
            f"{cls}={by_class[cls]}"
        )

    lines += [
        "",
        "Review rule",
        "-----------",
        "- worn_person: ordinary person-wearing/carrying scene.",
        "- product_display: product displayed without being worn/carried by a person.",
        "- partial_view: close/cropped view where only part of wearer/item is visible.",
        "- complex_scene: runway/crowd/heavy clutter/wide difficult context.",
        "",
        "Do not freeze Core until these minority-scene cases are reviewed.",
    ]

    (
        OUT_DIR / "minority_scene_review_summary.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("FINISHED")
    print("cases=", len(out_rows))
    print("product_display=", by_scene["product_display"])
    print("complex_scene=", by_scene["complex_scene"])

    for p in [
        OUT_DIR / "minority_scene_review_v1.csv",
        OUT_DIR / "minority_scene_review_summary.txt",
    ]:
        print(p.relative_to(PROJECT_ROOT))

    print(
        "contact_sheets:",
        SHEET_DIR.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
