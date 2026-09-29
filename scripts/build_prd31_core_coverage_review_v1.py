"""
Build PRD 3.1.1 Core coverage review packet v1.

Purpose
-------
Before freezing the final 400-case Core benchmark, this script:
1) excludes already human-rejected Core cases;
2) computes proportional size-bucket quotas for the final 50/class;
3) prepares the scene/occlusion/visibility annotation table;
4) creates human-review contact sheets showing BOTH the full source image
   and GT-mask overlay for each eligible candidate.

This script does NOT freeze the final Core set.

Inputs
------
benchmark/prd_3_1_v1/freeze_v1/core_final_review_template_v1.csv

Outputs
-------
benchmark/prd_3_1_v1/freeze_v1/coverage_review_v1/
├── core_coverage_review_v1.csv
├── core_size_quota_v1.csv
├── core_fine_category_distribution_v1.csv
├── core_coverage_plan_v1.txt
└── contact_sheets/
    ├── top_01.jpg
    ├── top_02.jpg
    ├── ...
    └── accessory_03.jpg

Run
---
cd /workspace/fashion_multimodal_analysis

python scripts/build_prd31_core_coverage_review_v1.py
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCH_ROOT = PROJECT_ROOT / "benchmark" / "prd_3_1_v1"
FREEZE_ROOT = BENCH_ROOT / "freeze_v1"

INPUT_CSV = FREEZE_ROOT / "core_final_review_template_v1.csv"

OUT_DIR = FREEZE_ROOT / "coverage_review_v1"
SHEET_DIR = OUT_DIR / "contact_sheets"

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

SIZE_BUCKETS = [
    "tiny",
    "small",
    "medium",
    "large",
]

SCENE_TYPES = [
    "worn_person",
    "product_display",
    "partial_view",
    "complex_scene",
]

OCCLUSION_LEVELS = [
    "none",
    "partial",
    "heavy",
]

VISIBILITY_LEVELS = [
    "full",
    "partial",
]


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
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


def proportional_quota(
    counts: dict[str, int],
    final_n: int = 50,
) -> dict[str, int]:
    total = sum(counts.values())

    if total <= 0:
        return {
            k: 0
            for k in counts
        }

    raw = {
        k: counts[k] * final_n / total
        for k in counts
    }

    quota = {
        k: math.floor(v)
        for k, v in raw.items()
    }

    remainder = (
        final_n
        - sum(quota.values())
    )

    order = sorted(
        counts,
        key=lambda k: (
            raw[k] - quota[k]
        ),
        reverse=True,
    )

    for key in order[:remainder]:
        quota[key] += 1

    return quota


def make_page(
    rows,
    out_path: Path,
):
    """
    10 candidates per page.
    Each tile shows full image + GT overlay.
    """
    cols = 2
    rows_per_page = 5
    tile_w = 760
    tile_h = 330

    canvas = Image.new(
        "RGB",
        (
            cols * tile_w,
            rows_per_page * tile_h,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for i, row in enumerate(rows):
        rr = i // cols
        cc = i % cols

        x0 = cc * tile_w
        y0 = rr * tile_h

        source_path = resolve_project(
            row["source_image"]
        )
        overlay_path = resolve_project(
            row["gt_overlay_path"]
        )

        try:
            source = Image.open(
                source_path
            ).convert("RGB")
            overlay = Image.open(
                overlay_path
            ).convert("RGB")
        except Exception:
            continue

        source.thumbnail((340, 235))
        overlay.thumbnail((340, 235))

        sx = x0 + 10
        sy = y0 + 55

        ox = x0 + 390
        oy = y0 + 55

        canvas.paste(
            source,
            (
                sx
                + (340 - source.width) // 2,
                sy
                + (235 - source.height) // 2,
            ),
        )

        canvas.paste(
            overlay,
            (
                ox
                + (340 - overlay.width) // 2,
                oy
                + (235 - overlay.height) // 2,
            ),
        )

        title = (
            f"{row['review_sample_id']} | "
            f"{row['garment_category']} | "
            f"{row['fine_or_source_category']}"
        )

        meta = (
            f"size={row['size_bucket']} | "
            f"area={100*float(row['bbox_area_ratio']):.2f}% | "
            f"source={row['source_dataset']} | "
            f"status={row['pre_freeze_status']}"
        )

        draw.text(
            (x0 + 10, y0 + 8),
            title,
            fill="black",
            font=font,
        )

        draw.text(
            (x0 + 10, y0 + 27),
            meta,
            fill="black",
            font=font,
        )

        draw.text(
            (x0 + 10, y0 + 295),
            (
                "Fill after review: "
                "scene=[worn_person/product_display/"
                "partial_view/complex_scene] "
                "occ=[none/partial/heavy] "
                "vis=[full/partial]"
            ),
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

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        out_path,
        quality=92,
    )


def main():
    rows = read_csv(INPUT_CSV)

    # Exclude only already confirmed human exclusions.
    eligible = [
        r
        for r in rows
        if r.get(
            "pre_freeze_status",
            ""
        )
        != "EXCLUDED_BY_HUMAN_REVIEW"
    ]

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    SHEET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_rows = []

    for r in eligible:
        review_rows.append({
            "review_sample_id": r[
                "review_sample_id"
            ],
            "garment_category": r[
                "garment_category"
            ],
            "fine_or_source_category": r[
                "fine_or_source_category"
            ],
            "source_dataset": r[
                "source_dataset"
            ],
            "source_image": r[
                "source_image"
            ],
            "garment_id": r[
                "garment_id"
            ],
            "gt_mask_path": r[
                "gt_mask_path"
            ],
            "gt_overlay_path": r[
                "gt_overlay_path"
            ],
            "bbox_area_ratio": r[
                "bbox_area_ratio"
            ],
            "size_bucket": r[
                "size_bucket"
            ],
            "pre_freeze_status": r[
                "pre_freeze_status"
            ],
            "final_annotation_valid": r.get(
                "final_annotation_valid",
                "",
            ),
            "scene_type": "",
            "occlusion_level": "",
            "visibility_level": "",
            "coverage_note": "",
            "coverage_review_status": "unreviewed",
        })

    write_csv(
        OUT_DIR
        / "core_coverage_review_v1.csv",
        review_rows,
    )

    quota_rows = []
    fine_rows = []

    for cls in CLASSES:
        rr = [
            r
            for r in review_rows
            if r[
                "garment_category"
            ]
            == cls
        ]

        counts = Counter(
            r["size_bucket"]
            for r in rr
        )

        quota = proportional_quota(
            {
                k: counts.get(k, 0)
                for k in SIZE_BUCKETS
                if counts.get(k, 0) > 0
            },
            final_n=50,
        )

        for bucket in SIZE_BUCKETS:
            available = counts.get(
                bucket,
                0,
            )

            if available == 0:
                continue

            quota_rows.append({
                "garment_category": cls,
                "size_bucket": bucket,
                "eligible_count": available,
                "eligible_share": (
                    f"{available/len(rr):.4f}"
                    if rr else ""
                ),
                "recommended_final_count": quota.get(
                    bucket,
                    0,
                ),
                "recommended_final_share": (
                    f"{quota.get(bucket,0)/50:.4f}"
                ),
            })

        fine_counts = Counter(
            r[
                "fine_or_source_category"
            ]
            for r in rr
        )

        for fine, n in sorted(
            fine_counts.items()
        ):
            fine_rows.append({
                "garment_category": cls,
                "fine_or_source_category": fine,
                "eligible_count": n,
                "eligible_share_within_class": (
                    f"{n/len(rr):.4f}"
                    if rr else ""
                ),
            })

        # Build paged contact sheets.
        rr = sorted(
            rr,
            key=lambda x: (
                SIZE_BUCKETS.index(
                    x["size_bucket"]
                )
                if x[
                    "size_bucket"
                ]
                in SIZE_BUCKETS
                else 999,
                x[
                    "fine_or_source_category"
                ],
                x[
                    "review_sample_id"
                ],
            ),
        )

        page_size = 10

        for page_start in range(
            0,
            len(rr),
            page_size,
        ):
            page = (
                page_start
                // page_size
                + 1
            )

            make_page(
                rr[
                    page_start:
                    page_start
                    + page_size
                ],
                SHEET_DIR
                / f"{cls}_{page:02d}.jpg",
            )

    write_csv(
        OUT_DIR
        / "core_size_quota_v1.csv",
        quota_rows,
    )

    write_csv(
        OUT_DIR
        / "core_fine_category_distribution_v1.csv",
        fine_rows,
    )

    # Summary.
    by_class = defaultdict(list)

    for r in quota_rows:
        by_class[
            r["garment_category"]
        ].append(r)

    lines = [
        "PRD 3.1.1 Core Coverage Review Plan v1",
        "======================================",
        "",
        f"eligible_after_known_exclusions={len(review_rows)}",
        "final_target=400",
        "final_target_per_class=50",
        "class_share=12.5% each",
        "",
        "Recommended proportional size allocation",
        "----------------------------------------",
    ]

    for cls in CLASSES:
        parts = []

        for r in by_class[cls]:
            parts.append(
                f"{r['size_bucket']}="
                f"{r['recommended_final_count']}"
            )

        lines.append(
            f"{cls}: "
            + ", ".join(parts)
        )

    lines += [
        "",
        "Scene coverage fields to review",
        "-------------------------------",
        (
            "scene_type: "
            "worn_person | product_display | "
            "partial_view | complex_scene"
        ),
        (
            "occlusion_level: "
            "none | partial | heavy"
        ),
        (
            "visibility_level: "
            "full | partial"
        ),
        "",
        "Freeze principle",
        "----------------",
        "- Exactly 50 final cases per class.",
        (
            "- Keep final size distribution close to "
            "the recommended proportional allocation."
        ),
        (
            "- Avoid letting one scene type dominate "
            "a class when alternative reviewed cases exist."
        ),
        (
            "- Preserve meaningful occlusion/partial-view "
            "coverage, but do not move Stress-like ambiguous "
            "cases into Core."
        ),
        (
            "- Final membership must be decided without "
            "looking at candidate-model performance."
        ),
        "",
        "Next",
        "----",
        (
            "Complete scene/occlusion/visibility labels in "
            "core_coverage_review_v1.csv, then run the "
            "final freeze selector."
        ),
    ]

    (
        OUT_DIR
        / "core_coverage_plan_v1.txt"
    ).write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    print("FINISHED")

    for p in [
        OUT_DIR
        / "core_coverage_plan_v1.txt",
        OUT_DIR
        / "core_size_quota_v1.csv",
        OUT_DIR
        / "core_fine_category_distribution_v1.csv",
        OUT_DIR
        / "core_coverage_review_v1.csv",
    ]:
        print(
            p.relative_to(
                PROJECT_ROOT
            )
        )

    print(
        "contact sheets:",
        SHEET_DIR.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()
