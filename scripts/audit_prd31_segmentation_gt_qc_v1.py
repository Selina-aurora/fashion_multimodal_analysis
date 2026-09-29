"""
Automated GT annotation QC for PRD 3.1.1 Core + Stress review pools.

Purpose
-------
Run machine-checkable integrity tests BEFORE human review. This does not
replace human review of category semantics or image quality.

Checks per case
---------------
- source image exists and opens
- GT mask exists and opens
- image/mask dimensions match
- bbox coordinates are valid and inside image
- mask is non-empty
- mask-derived bbox is consistent with annotation bbox
- almost all mask pixels lie inside the annotation bbox
- mask fill ratio inside bbox is plausible
- overlay exists
- connected-component count is reported when OpenCV is available

Default flag thresholds
-----------------------
- mask_bbox_iou < 0.90              -> FLAG
- mask_inside_bbox_ratio < 0.98     -> FLAG
- mask_fill_ratio < 0.03            -> FLAG
- bbox out of image / empty mask    -> FAIL

Outputs
-------
benchmark/prd_3_1_v1/candidates/segmentation_core_stress_v1/qc/
├── automated_gt_qc_v1.csv
├── automated_gt_qc_summary.csv
├── human_review_priority_v1.csv
└── automated_gt_qc_summary.txt

Run
---
cd /workspace/fashion_multimodal_analysis

python scripts/audit_prd31_segmentation_gt_qc_v1.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

POOL_ROOT = (
    PROJECT_ROOT
    / "benchmark"
    / "prd_3_1_v1"
    / "candidates"
    / "segmentation_core_stress_v1"
)

CORE_CSV = POOL_ROOT / "core_review_pool_v1.csv"
STRESS_CSV = POOL_ROOT / "stress_review_pool_v1.csv"
OUT_DIR = POOL_ROOT / "qc"

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
        "--mask-bbox-iou-min",
        type=float,
        default=0.90,
    )
    p.add_argument(
        "--mask-inside-bbox-ratio-min",
        type=float,
        default=0.98,
    )
    p.add_argument(
        "--mask-fill-ratio-min",
        type=float,
        default=0.03,
    )
    return p.parse_args()


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
        for k in row:
            if k not in fields:
                fields.append(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def resolve_project_path(raw: str) -> Path:
    p = Path(str(raw).strip())
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def f(v):
    return float(v)


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter

    return inter / union if union > 0 else 0.0


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    # +1 because mask pixel index is inclusive, bbox convention here is xyxy
    # with x2/y2 interpreted as exclusive for area math.
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def connected_components(mask: np.ndarray):
    try:
        import cv2
        n, _ = cv2.connectedComponents(mask.astype(np.uint8))
        return max(0, int(n) - 1)
    except Exception:
        return ""


def audit_row(row, args):
    reasons = []
    hard_fail = False

    image_path = resolve_project_path(row["source_image"])
    mask_path = resolve_project_path(row["gt_mask_path"])
    overlay_path = resolve_project_path(row["gt_overlay_path"])

    image_exists = image_path.is_file()
    mask_exists = mask_path.is_file()
    overlay_exists = overlay_path.is_file()

    if not image_exists:
        reasons.append("MISSING_IMAGE")
        hard_fail = True

    if not mask_exists:
        reasons.append("MISSING_MASK")
        hard_fail = True

    if not overlay_exists:
        reasons.append("MISSING_OVERLAY")

    if hard_fail:
        return {
            **row,
            "qc_status": "FAIL",
            "qc_reasons": "|".join(reasons),
            "image_exists": int(image_exists),
            "mask_exists": int(mask_exists),
            "overlay_exists": int(overlay_exists),
        }

    try:
        with Image.open(image_path) as im:
            iw, ih = im.size
    except Exception:
        return {
            **row,
            "qc_status": "FAIL",
            "qc_reasons": "IMAGE_OPEN_ERROR",
            "image_exists": 1,
            "mask_exists": 1,
            "overlay_exists": int(overlay_exists),
        }

    try:
        mask_img = Image.open(mask_path).convert("L")
        mw, mh = mask_img.size
        mask = np.asarray(mask_img) > 0
    except Exception:
        return {
            **row,
            "qc_status": "FAIL",
            "qc_reasons": "MASK_OPEN_ERROR",
            "image_exists": 1,
            "mask_exists": 1,
            "overlay_exists": int(overlay_exists),
        }

    dims_match = (iw == mw and ih == mh)

    if not dims_match:
        reasons.append("IMAGE_MASK_DIM_MISMATCH")
        hard_fail = True

    try:
        x1 = f(row["gt_bbox_x1"])
        y1 = f(row["gt_bbox_y1"])
        x2 = f(row["gt_bbox_x2"])
        y2 = f(row["gt_bbox_y2"])
    except Exception:
        x1 = y1 = x2 = y2 = math.nan
        reasons.append("INVALID_BBOX_NUMERIC")
        hard_fail = True

    bbox_valid = (
        not any(math.isnan(v) for v in [x1, y1, x2, y2])
        and x2 > x1
        and y2 > y1
    )

    if not bbox_valid:
        reasons.append("INVALID_BBOX_GEOMETRY")
        hard_fail = True

    bbox_inside_image = (
        bbox_valid
        and x1 >= 0
        and y1 >= 0
        and x2 <= iw
        and y2 <= ih
    )

    if not bbox_inside_image:
        reasons.append("BBOX_OUTSIDE_IMAGE")
        hard_fail = True

    mask_pixels = int(mask.sum())

    if mask_pixels == 0:
        reasons.append("EMPTY_MASK")
        hard_fail = True

    if hard_fail:
        return {
            **row,
            "qc_status": "FAIL",
            "qc_reasons": "|".join(sorted(set(reasons))),
            "image_exists": 1,
            "mask_exists": 1,
            "overlay_exists": int(overlay_exists),
            "image_width_actual": iw,
            "image_height_actual": ih,
            "mask_width_actual": mw,
            "mask_height_actual": mh,
            "dimensions_match": int(dims_match),
            "bbox_inside_image": int(bbox_inside_image),
            "mask_pixels": mask_pixels,
        }

    ann_bbox = (x1, y1, x2, y2)
    mb = mask_bbox(mask)

    mb_iou = bbox_iou(
        ann_bbox,
        tuple(map(float, mb)),
    )

    ix1 = max(0, int(math.floor(x1)))
    iy1 = max(0, int(math.floor(y1)))
    ix2 = min(iw, int(math.ceil(x2)))
    iy2 = min(ih, int(math.ceil(y2)))

    inside_pixels = int(mask[iy1:iy2, ix1:ix2].sum())
    inside_ratio = (
        inside_pixels / mask_pixels
        if mask_pixels
        else 0.0
    )

    bbox_area_px = max(
        1.0,
        (x2 - x1) * (y2 - y1),
    )
    fill_ratio = mask_pixels / bbox_area_px

    components = connected_components(mask)

    if mb_iou < args.mask_bbox_iou_min:
        reasons.append("LOW_MASK_BBOX_IOU")

    if inside_ratio < args.mask_inside_bbox_ratio_min:
        reasons.append("MASK_OUTSIDE_BBOX")

    if fill_ratio < args.mask_fill_ratio_min:
        reasons.append("VERY_LOW_MASK_FILL")

    if not overlay_exists:
        reasons.append("MISSING_OVERLAY")

    qc_status = "PASS" if not reasons else "FLAG"

    # Priority: hard flag reasons + core first + smaller/more fragmented cases.
    priority = 0
    if qc_status == "FLAG":
        priority += 100
    if row.get("review_subset") == "core":
        priority += 20
    if row.get("absolute_small_object") in {"1", 1, True}:
        priority += 10
    if isinstance(components, int) and components > 3:
        priority += min(10, components)

    return {
        **row,
        "qc_status": qc_status,
        "qc_reasons": "|".join(sorted(set(reasons))),
        "image_exists": 1,
        "mask_exists": 1,
        "overlay_exists": int(overlay_exists),
        "image_width_actual": iw,
        "image_height_actual": ih,
        "mask_width_actual": mw,
        "mask_height_actual": mh,
        "dimensions_match": int(dims_match),
        "bbox_inside_image": int(bbox_inside_image),
        "mask_pixels": mask_pixels,
        "mask_bbox_x1": mb[0],
        "mask_bbox_y1": mb[1],
        "mask_bbox_x2": mb[2],
        "mask_bbox_y2": mb[3],
        "mask_bbox_iou_with_gt_bbox": f"{mb_iou:.6f}",
        "mask_inside_bbox_ratio": f"{inside_ratio:.6f}",
        "mask_fill_ratio_in_bbox": f"{fill_ratio:.6f}",
        "connected_components": components,
        "human_review_priority": priority,
    }


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_csv(CORE_CSV) + read_csv(STRESS_CSV)

    audited = []
    for i, row in enumerate(rows, start=1):
        audited.append(audit_row(row, args))
        if i % 100 == 0:
            print(f"audited {i}/{len(rows)}")

    write_csv(
        OUT_DIR / "automated_gt_qc_v1.csv",
        audited,
    )

    # Human review priority: all non-PASS first, then core before stress.
    priority_rows = sorted(
        audited,
        key=lambda r: (
            -int(r.get("human_review_priority", 0) or 0),
            0 if r.get("review_subset") == "core" else 1,
            CLASSES.index(r["garment_category"])
            if r["garment_category"] in CLASSES
            else 999,
            r["review_sample_id"],
        ),
    )

    write_csv(
        OUT_DIR / "human_review_priority_v1.csv",
        priority_rows,
    )

    summary_rows = []

    for subset in ["core", "stress"]:
        for cls in CLASSES:
            rr = [
                r
                for r in audited
                if r.get("review_subset") == subset
                and r.get("garment_category") == cls
            ]

            counts = Counter(
                r.get("qc_status", "")
                for r in rr
            )

            reason_counts = Counter()

            for r in rr:
                for reason in str(
                    r.get("qc_reasons", "")
                ).split("|"):
                    if reason:
                        reason_counts[reason] += 1

            summary_rows.append(
                {
                    "review_subset": subset,
                    "garment_category": cls,
                    "n": len(rr),
                    "qc_pass": counts["PASS"],
                    "qc_flag": counts["FLAG"],
                    "qc_fail": counts["FAIL"],
                    "flag_reasons": "; ".join(
                        f"{k}={v}"
                        for k, v in sorted(
                            reason_counts.items()
                        )
                    ),
                }
            )

    write_csv(
        OUT_DIR / "automated_gt_qc_summary.csv",
        summary_rows,
    )

    overall = Counter(
        r["qc_status"]
        for r in audited
    )

    lines = [
        "PRD 3.1.1 Automated GT Annotation QC v1",
        "======================================",
        "",
        f"cases={len(audited)}",
        f"PASS={overall['PASS']}",
        f"FLAG={overall['FLAG']}",
        f"FAIL={overall['FAIL']}",
        "",
        "Thresholds",
        "----------",
        f"mask_bbox_iou_min={args.mask_bbox_iou_min}",
        (
            "mask_inside_bbox_ratio_min="
            f"{args.mask_inside_bbox_ratio_min}"
        ),
        f"mask_fill_ratio_min={args.mask_fill_ratio_min}",
        "",
        "Important",
        "---------",
        "- PASS means machine geometry/integrity checks passed.",
        "- PASS does NOT mean human annotation review is complete.",
        "- Category semantics and image quality still require human review.",
        "- FLAG cases should be reviewed first.",
        "- FAIL cases contain structural annotation/file issues and must not be frozen until resolved.",
        "",
        "By subset/class",
        "---------------",
    ]

    for r in summary_rows:
        lines.append(
            f"{r['review_subset']} {r['garment_category']}: "
            f"n={r['n']} pass={r['qc_pass']} "
            f"flag={r['qc_flag']} fail={r['qc_fail']} "
            f"reasons=({r['flag_reasons']})"
        )

    (OUT_DIR / "automated_gt_qc_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("FINISHED")
    for p in [
        OUT_DIR / "automated_gt_qc_summary.txt",
        OUT_DIR / "automated_gt_qc_summary.csv",
        OUT_DIR / "human_review_priority_v1.csv",
        OUT_DIR / "automated_gt_qc_v1.csv",
    ]:
        print(p.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
