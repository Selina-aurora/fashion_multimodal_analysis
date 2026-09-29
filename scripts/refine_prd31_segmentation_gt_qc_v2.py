"""
Refine PRD 3.1.1 automated GT QC after diagnosing an overly strict v1 rule.

Why v2
------
QC v1 treated mask-derived-bbox IoU < 0.90 as a failure flag. That metric
mainly measures how tightly the source bbox hugs the segmentation mask.
For DeepFashion2, source bboxes are often valid but not equal to the tight
mask bounding rectangle, so the 0.90 rule over-flags many correct annotations.

QC v2 therefore:
- keeps structural/file failures as FAIL;
- uses mask-outside-bbox and very-low-mask-fill as true geometry review flags;
- treats mask-bbox IoU only as a REVIEW PRIORITY signal when it is very low
  (< 0.60), not as an automatic annotation defect;
- never marks category semantics or image quality as automatically reviewed.

Input
-----
benchmark/prd_3_1_v1/candidates/segmentation_core_stress_v1/qc/
    automated_gt_qc_v1.csv

Outputs
-------
same qc folder:
├── automated_gt_qc_v2.csv
├── automated_gt_qc_v2_summary.csv
├── human_review_priority_v2.csv
└── automated_gt_qc_v2_summary.txt

Run
---
cd /workspace/fashion_multimodal_analysis

python scripts/refine_prd31_segmentation_gt_qc_v2.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QC_DIR = (
    PROJECT_ROOT
    / "benchmark"
    / "prd_3_1_v1"
    / "candidates"
    / "segmentation_core_stress_v1"
    / "qc"
)

INPUT_CSV = QC_DIR / "automated_gt_qc_v1.csv"

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mask-inside-bbox-ratio-min",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--mask-fill-ratio-min",
        type=float,
        default=0.03,
    )
    p.add_argument(
        "--loose-bbox-review-iou",
        type=float,
        default=0.60,
    )
    return p.parse_args()


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


def fv(row, field):
    try:
        return float(row.get(field, ""))
    except Exception:
        return float("nan")


def classify(row, args):
    reasons = []
    advisory = []

    v1_status = str(
        row.get("qc_status", "")
    ).strip()

    if v1_status == "FAIL":
        reasons.append(
            "STRUCTURAL_OR_FILE_FAILURE"
        )

    inside = fv(
        row,
        "mask_inside_bbox_ratio",
    )

    fill = fv(
        row,
        "mask_fill_ratio_in_bbox",
    )

    mb_iou = fv(
        row,
        "mask_bbox_iou_with_gt_bbox",
    )

    if inside == inside and (
        inside
        < args.mask_inside_bbox_ratio_min
    ):
        reasons.append(
            "MASK_OUTSIDE_BBOX_SEVERE"
        )

    if fill == fill and (
        fill
        < args.mask_fill_ratio_min
    ):
        reasons.append(
            "VERY_LOW_MASK_FILL"
        )

    # Advisory only: source bbox can legitimately be looser than tight mask bbox.
    if mb_iou == mb_iou and (
        mb_iou
        < args.loose_bbox_review_iou
    ):
        advisory.append(
            "LOOSE_SOURCE_BBOX_REVIEW"
        )

    if reasons:
        status = "REVIEW_REQUIRED"
    elif advisory:
        status = "REVIEW_PRIORITY"
    else:
        status = "GEOMETRY_PASS"

    priority = 0

    if status == "REVIEW_REQUIRED":
        priority += 200
    elif status == "REVIEW_PRIORITY":
        priority += 100

    if (
        str(
            row.get(
                "review_subset",
                "",
            )
        )
        == "core"
    ):
        priority += 20

    if str(
        row.get(
            "absolute_small_object",
            "",
        )
    ) in {
        "1",
        "True",
        "true",
    }:
        priority += 10

    return (
        status,
        reasons,
        advisory,
        priority,
    )


def main():
    args = parse_args()

    rows = read_csv(INPUT_CSV)

    out = []

    for row in rows:
        (
            status,
            reasons,
            advisory,
            priority,
        ) = classify(
            row,
            args,
        )

        r = dict(row)

        r[
            "qc_v2_status"
        ] = status

        r[
            "qc_v2_reasons"
        ] = "|".join(
            reasons
        )

        r[
            "qc_v2_advisory"
        ] = "|".join(
            advisory
        )

        r[
            "human_review_priority_v2"
        ] = priority

        # These are explicitly NOT auto-approved.
        r[
            "category_human_review"
        ] = ""

        r[
            "bbox_human_review"
        ] = ""

        r[
            "mask_human_review"
        ] = ""

        r[
            "image_quality_human_review"
        ] = ""

        r[
            "final_human_review_status"
        ] = "unreviewed"

        out.append(r)

    write_csv(
        QC_DIR
        / "automated_gt_qc_v2.csv",
        out,
    )

    priority_rows = sorted(
        out,
        key=lambda r: (
            -int(
                r.get(
                    "human_review_priority_v2",
                    0,
                )
                or 0
            ),
            0
            if r.get(
                "review_subset"
            )
            == "core"
            else 1,
            (
                CLASSES.index(
                    r[
                        "garment_category"
                    ]
                )
                if r.get(
                    "garment_category"
                )
                in CLASSES
                else 999
            ),
            r.get(
                "review_sample_id",
                "",
            ),
        ),
    )

    write_csv(
        QC_DIR
        / "human_review_priority_v2.csv",
        priority_rows,
    )

    summary_rows = []

    for subset in [
        "core",
        "stress",
    ]:
        for cls in CLASSES:
            rr = [
                r
                for r in out
                if r.get(
                    "review_subset"
                )
                == subset
                and r.get(
                    "garment_category"
                )
                == cls
            ]

            counts = Counter(
                r[
                    "qc_v2_status"
                ]
                for r in rr
            )

            required_reasons = Counter()
            advisories = Counter()

            for r in rr:
                for reason in str(
                    r.get(
                        "qc_v2_reasons",
                        "",
                    )
                ).split("|"):
                    if reason:
                        required_reasons[
                            reason
                        ] += 1

                for reason in str(
                    r.get(
                        "qc_v2_advisory",
                        "",
                    )
                ).split("|"):
                    if reason:
                        advisories[
                            reason
                        ] += 1

            summary_rows.append(
                {
                    "review_subset": subset,
                    "garment_category": cls,
                    "n": len(rr),
                    "geometry_pass": counts[
                        "GEOMETRY_PASS"
                    ],
                    "review_priority": counts[
                        "REVIEW_PRIORITY"
                    ],
                    "review_required": counts[
                        "REVIEW_REQUIRED"
                    ],
                    "required_reasons": "; ".join(
                        f"{k}={v}"
                        for k, v in sorted(
                            required_reasons.items()
                        )
                    ),
                    "advisories": "; ".join(
                        f"{k}={v}"
                        for k, v in sorted(
                            advisories.items()
                        )
                    ),
                }
            )

    write_csv(
        QC_DIR
        / "automated_gt_qc_v2_summary.csv",
        summary_rows,
    )

    total = Counter(
        r[
            "qc_v2_status"
        ]
        for r in out
    )

    core = [
        r
        for r in out
        if r.get(
            "review_subset"
        )
        == "core"
    ]

    core_count = Counter(
        r[
            "qc_v2_status"
        ]
        for r in core
    )

    lines = [
        (
            "PRD 3.1.1 Automated "
            "GT Annotation QC v2"
        ),
        (
            "======================================"
        ),
        "",
        f"cases={len(out)}",
        (
            "GEOMETRY_PASS="
            f"{total['GEOMETRY_PASS']}"
        ),
        (
            "REVIEW_PRIORITY="
            f"{total['REVIEW_PRIORITY']}"
        ),
        (
            "REVIEW_REQUIRED="
            f"{total['REVIEW_REQUIRED']}"
        ),
        "",
        "Core only",
        "---------",
        (
            "core_cases="
            f"{len(core)}"
        ),
        (
            "core_GEOMETRY_PASS="
            f"{core_count['GEOMETRY_PASS']}"
        ),
        (
            "core_REVIEW_PRIORITY="
            f"{core_count['REVIEW_PRIORITY']}"
        ),
        (
            "core_REVIEW_REQUIRED="
            f"{core_count['REVIEW_REQUIRED']}"
        ),
        "",
        "v2 logic",
        "--------",
        (
            "mask_inside_bbox_ratio_min="
            f"{args.mask_inside_bbox_ratio_min}"
        ),
        (
            "mask_fill_ratio_min="
            f"{args.mask_fill_ratio_min}"
        ),
        (
            "loose_bbox_review_iou="
            f"{args.loose_bbox_review_iou}"
        ),
        "",
        "Important interpretation",
        "------------------------",
        (
            "- mask-derived-bbox IoU is no longer "
            "an automatic defect threshold."
        ),
        (
            "- It measures bbox tightness as well as "
            "mask/bbox consistency."
        ),
        (
            "- GEOMETRY_PASS means file/geometry "
            "checks are acceptable; it does NOT "
            "complete human review."
        ),
        (
            "- REVIEW_PRIORITY means inspect first, "
            "but do not automatically exclude."
        ),
        (
            "- REVIEW_REQUIRED means a stronger "
            "geometry issue exists and must be "
            "resolved before freeze."
        ),
        (
            "- Category semantics, bbox semantics, "
            "mask visual quality and image quality "
            "still require human confirmation."
        ),
        "",
        "By subset/class",
        "---------------",
    ]

    for r in summary_rows:
        lines.append(
            f"{r['review_subset']} "
            f"{r['garment_category']}: "
            f"n={r['n']} "
            f"pass={r['geometry_pass']} "
            f"priority={r['review_priority']} "
            f"required={r['review_required']} "
            f"required_reasons="
            f"({r['required_reasons']}) "
            f"advisories="
            f"({r['advisories']})"
        )

    (
        QC_DIR
        / "automated_gt_qc_v2_summary.txt"
    ).write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    print("FINISHED")

    for p in [
        QC_DIR
        / "automated_gt_qc_v2_summary.txt",
        QC_DIR
        / "automated_gt_qc_v2_summary.csv",
        QC_DIR
        / "human_review_priority_v2.csv",
        QC_DIR
        / "automated_gt_qc_v2.csv",
    ]:
        print(
            p.relative_to(
                PROJECT_ROOT
            )
        )


if __name__ == "__main__":
    main()
