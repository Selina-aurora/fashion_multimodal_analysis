"""
PRD 3.1 propagation diagnosis v2:
Does ROI/mask quality explain 3.1.3 attribute instability?

Inputs
------
configs/prd_3_1_predicted_roi_fullval_matched_v1.csv
configs/prd_8class_val_v1.csv
reports/prd_integration/predicted_roi_attributes_fullval_v1/
    per_instance_attribute_comparison.csv

Outputs
-------
reports/prd_integration/predicted_roi_attributes_fullval_v1/
├── roi_quality_attribute_analysis.csv
├── agreement_by_bbox_iou_bin.csv
├── agreement_by_mask_iou_bin.csv
└── roi_quality_diagnosis.txt

Notes
-----
- GT is used only for diagnostic quality measurement.
- Attribute agreement is still consistency vs the frozen GT-ROI outputs,
  not human-ground-truth attribute accuracy.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MATCHED_CSV = (
    PROJECT_ROOT
    / "configs"
    / "prd_3_1_predicted_roi_fullval_matched_v1.csv"
)

GT_CSV = (
    PROJECT_ROOT
    / "configs"
    / "prd_8class_val_v1.csv"
)

ATTR_CSV = (
    PROJECT_ROOT
    / "reports"
    / "prd_integration"
    / "predicted_roi_attributes_fullval_v1"
    / "per_instance_attribute_comparison.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_integration"
    / "predicted_roi_attributes_fullval_v1"
)

ATTR_FIELDS = [
    ("primary_color", "color_agree"),
    ("pattern", "pattern_agree"),
    ("sleeve_length", "sleeve_length_agree"),
    ("neckline", "neckline_agree"),
    ("silhouette_fit", "silhouette_fit_agree"),
    ("fashion_style", "fashion_style_agree"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def norm_source(v: str) -> str:
    return str(v).replace("\\", "/").strip()


def key_of(row: dict[str, str]) -> tuple[str, str]:
    return (
        norm_source(row.get("source_image", "")),
        str(row.get("garment_id", "")).strip(),
    )


def resolve_path(raw: str) -> Path:
    p = Path(str(raw).strip())
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def int_box(row: dict[str, str]) -> tuple[int, int, int, int]:
    return tuple(
        int(round(float(row[k])))
        for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    )


def load_binary(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 0


def full_mask_from_row(
    row: dict[str, str],
    image_size: tuple[int, int],
) -> np.ndarray:
    """
    Supports both full-image masks and bbox/crop masks.
    """
    w, h = image_size
    mask_path = resolve_path(row["mask_path"])
    m = load_binary(mask_path)

    if m.shape == (h, w):
        return m

    x1, y1, x2, y2 = int_box(row)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    target_w = x2 - x1
    target_h = y2 - y1

    if m.shape != (target_h, target_w):
        pil = Image.fromarray((m.astype(np.uint8) * 255), mode="L")
        pil = pil.resize((target_w, target_h), Image.Resampling.NEAREST)
        m = np.asarray(pil) > 0

    canvas = np.zeros((h, w), dtype=bool)
    canvas[y1:y2, x1:x2] = m[:target_h, :target_w]
    return canvas


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def f(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return math.nan


def iou_bin(x: float) -> str:
    if math.isnan(x):
        return "NA"
    if x < 0.70:
        return "0.50-0.69"
    if x < 0.85:
        return "0.70-0.84"
    return ">=0.85"


def summarize_by_bin(
    rows: list[dict[str, Any]],
    metric: str,
    output_name: str,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[iou_bin(f(row[metric]))].append(row)

    out = []

    order = ["0.50-0.69", "0.70-0.84", ">=0.85", "NA"]

    for bin_name in order:
        rr = grouped.get(bin_name, [])
        if not rr:
            continue

        for attr, field in ATTR_FIELDS:
            vals = []
            for row in rr:
                raw = str(row.get(field, "")).strip()
                if raw in {"0", "1"}:
                    vals.append(int(raw))

            out.append(
                {
                    "quality_metric": metric,
                    "quality_bin": bin_name,
                    "attribute": attr,
                    "comparable_pairs": len(vals),
                    "same_label_pairs": sum(vals),
                    "agreement": (
                        f"{sum(vals)/len(vals):.4f}"
                        if vals
                        else ""
                    ),
                }
            )

    write_csv(
        OUT_DIR / output_name,
        out,
    )


def main() -> None:
    matched = read_csv(MATCHED_CSV)
    gt_rows = read_csv(GT_CSV)
    attr_rows = read_csv(ATTR_CSV)

    gt_idx = {key_of(r): r for r in gt_rows}
    attr_idx = {key_of(r): r for r in attr_rows}

    out = []
    missing = []

    for m in matched:
        key = key_of(m)
        gt = gt_idx.get(key)
        ar = attr_idx.get(key)

        if gt is None or ar is None:
            missing.append(key)
            continue

        image_path = resolve_path(gt["source_image"])
        with Image.open(image_path) as im:
            image_size = im.size

        pred_full = full_mask_from_row(m, image_size)
        gt_full = full_mask_from_row(gt, image_size)

        miou = mask_iou(pred_full, gt_full)

        row = {
            "source_image": key[0],
            "garment_id": key[1],
            "garment_category": m.get("garment_category", ""),
            "bbox_iou": m.get("matched_gt_bbox_iou", ""),
            "mask_iou": f"{miou:.6f}",
            "prediction_score": m.get("prediction_score", ""),
        }

        for attr, field in ATTR_FIELDS:
            row[field] = ar.get(field, "")

        out.append(row)

    write_csv(
        OUT_DIR / "roi_quality_attribute_analysis.csv",
        out,
    )

    summarize_by_bin(
        out,
        "bbox_iou",
        "agreement_by_bbox_iou_bin.csv",
    )

    summarize_by_bin(
        out,
        "mask_iou",
        "agreement_by_mask_iou_bin.csv",
    )

    # Simple descriptive correlations:
    # per instance = fraction of comparable attributes that stayed the same.
    xs_bbox, xs_mask, ys = [], [], []

    for row in out:
        vals = []
        for _, field in ATTR_FIELDS:
            raw = str(row.get(field, "")).strip()
            if raw in {"0", "1"}:
                vals.append(int(raw))

        if not vals:
            continue

        stability = sum(vals) / len(vals)

        biou = f(row["bbox_iou"])
        miou = f(row["mask_iou"])

        if not math.isnan(biou):
            xs_bbox.append(biou)
            ys.append(stability)

        if not math.isnan(miou):
            xs_mask.append(miou)

    # Need matched y for mask separately.
    mask_pairs = []
    bbox_pairs = []

    for row in out:
        vals = []
        for _, field in ATTR_FIELDS:
            raw = str(row.get(field, "")).strip()
            if raw in {"0", "1"}:
                vals.append(int(raw))
        if not vals:
            continue
        stability = sum(vals) / len(vals)
        biou = f(row["bbox_iou"])
        miou = f(row["mask_iou"])
        if not math.isnan(biou):
            bbox_pairs.append((biou, stability))
        if not math.isnan(miou):
            mask_pairs.append((miou, stability))

    def pearson(pairs):
        if len(pairs) < 3:
            return math.nan
        a = np.asarray([p[0] for p in pairs], dtype=float)
        b = np.asarray([p[1] for p in pairs], dtype=float)
        if np.std(a) == 0 or np.std(b) == 0:
            return math.nan
        return float(np.corrcoef(a, b)[0, 1])

    bbox_r = pearson(bbox_pairs)
    mask_r = pearson(mask_pairs)

    lines = [
        "PRD 3.1 ROI Quality -> Attribute Stability Diagnostic v2",
        "=======================================================",
        "",
        f"matched_pairs={len(out)}",
        f"missing_join_pairs={len(missing)}",
        f"bbox_iou_vs_attribute_stability_pearson_r={bbox_r:.4f}" if not math.isnan(bbox_r) else "bbox_iou_vs_attribute_stability_pearson_r=NA",
        f"mask_iou_vs_attribute_stability_pearson_r={mask_r:.4f}" if not math.isnan(mask_r) else "mask_iou_vs_attribute_stability_pearson_r=NA",
        "",
        "Interpretation",
        "--------------",
        "- This is descriptive only; n is small.",
        "- A positive correlation suggests better ROI/mask overlap tends to preserve more attribute labels.",
        "- Weak/near-zero correlation suggests some attribute instability is intrinsic to the attribute classifier, not only localization quality.",
        "- Compare local attributes (sleeve_length, neckline) against global attributes (color, silhouette_fit) across IoU bins.",
        "- Attribute agreement is consistency vs frozen GT-ROI outputs, not human-GT accuracy.",
    ]

    (OUT_DIR / "roi_quality_diagnosis.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("FINISHED")
    print(
        (OUT_DIR / "roi_quality_attribute_analysis.csv").relative_to(PROJECT_ROOT)
    )
    print(
        (OUT_DIR / "agreement_by_bbox_iou_bin.csv").relative_to(PROJECT_ROOT)
    )
    print(
        (OUT_DIR / "agreement_by_mask_iou_bin.csv").relative_to(PROJECT_ROOT)
    )
    print(
        (OUT_DIR / "roi_quality_diagnosis.txt").relative_to(PROJECT_ROOT)
    )


if __name__ == "__main__":
    main()
