"""3.1 接口联调：区分预测 ROI 传递和使用 GT 的受控诊断，保留对象 ID 与坐标对应关系。

Build a paired predicted-ROI manifest for PRD 3.1 propagation analysis.

Why
---
The raw 20-image integration pilot intentionally keeps every prediction above
threshold. That is useful for end-to-end inspection, but it is too noisy for
measuring how 3.1.1 ROI errors propagate into 3.1.3 attributes.

This script creates a ONE-TO-ONE, CLASS-CORRECT matched subset:
- same source image
- same garment category
- bbox IoU >= threshold (default 0.50)
- greedy matching by highest IoU
- each GT and each prediction can be used at most once

Important
---------
For this diagnostic paired manifest:
- `source_image` is copied from the frozen GT validation row.
- `garment_id` is also copied from the matched GT row ONLY so existing
  3.1.3 outputs can be joined directly by (source_image, garment_id).
- crop_path/mask_path/masked_preview_path are still the PREDICTED ROI outputs.
- `predicted_garment_id` preserves the original prediction ID.

Therefore this is a propagation-analysis manifest, NOT a deployment manifest.

Inputs
------
configs/prd_3_1_predicted_roi_pilot20_v2.csv
configs/prd_8class_val_v1.csv

Outputs
-------
configs/prd_3_1_predicted_roi_matched_v2.csv

reports/prd_integration/predicted_roi_matched_v2/
├── matched_pairs.csv
├── unmatched_predictions.csv
├── unmatched_gt.csv
└── summary.txt

Run
---
cd fashion_multimodal_analysis

python scripts/integration/build_prd31_matched_roi_manifest_v2.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()

DEFAULT_PREDICTED = PROJECT_ROOT / "configs" / "prd_3_1_predicted_roi_pilot20_v2.csv"

DEFAULT_GT = PROJECT_ROOT / "configs" / "prd_8class_val_v1.csv"

DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "prd_3_1_predicted_roi_matched_v2.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "prd_integration" / "predicted_roi_matched_v2"

PRD_CLASSES = (
    "top",
    "pants",
    "skirt",
    "outerwear",
    "dress",
    "shoe",
    "bag",
    "accessory",
)


def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--predicted",
        default=str(DEFAULT_PREDICTED),
    )
    p.add_argument(
        "--gt",
        default=str(DEFAULT_GT),
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
    )
    return p.parse_args()


def resolve_path(raw: Any) -> Path:
    """将清单或配置中的相对路径解析到当前项目/数据目录。

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(raw)


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
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
    fields: list[str] | None = None,
) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
        fields: 输出字段或分组字段的顺序。
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    if fields is None:
        fields = list(rows[0].keys())

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


def norm_path(value: str) -> str:
    """规范化路径分隔符，兼容 Windows 与 Linux 清单。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return str(value).replace("\\", "/").strip()


def image_key(value: str) -> str:
    """Match absolute/project-relative/../fashion_data paths robustly
    using the final filename. The frozen pilot contains unique source
    filenames, so basename is sufficient here.

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return Path(norm_path(value)).name


def get_box(row: dict[str, str]) -> tuple[float, float, float, float]:
    """取得 边界框。

    Args:
        row: 记录字段，使用 bbox_x1, bbox_y1, bbox_x2, bbox_y2。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return (
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"]),
    )


def box_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """计算两个 xyxy 边界框的交并比，空并集按 0 处理。

    Args:
        a: 当前函数的第一个输入，含义随运算而定。
        b: 当前函数的第二个输入，含义随运算而定。

    Returns:
        范围为 0 到 1 的边界框交并比；不相交或空并集为 0。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    args = parse_args()

    pred_path = resolve_path(args.predicted)
    gt_path = resolve_path(args.gt)
    output_path = resolve_path(args.output)

    pred_rows = read_csv(pred_path)
    gt_rows = read_csv(gt_path)

    gt_by_image: dict[
        str,
        list[tuple[int, dict[str, str]]],
    ] = defaultdict(list)

    pred_by_image: dict[
        str,
        list[tuple[int, dict[str, str]]],
    ] = defaultdict(list)

    for gi, row in enumerate(gt_rows):
        gt_by_image[image_key(row["source_image"])].append((gi, row))

    for pi, row in enumerate(pred_rows):
        pred_by_image[image_key(row["source_image"])].append((pi, row))

    matched_rows: list[dict[str, Any]] = []
    matched_pair_rows: list[dict[str, Any]] = []

    used_pred: set[int] = set()
    used_gt: set[int] = set()

    candidate_pairs: list[
        tuple[
            float,
            int,
            int,
            dict[str, str],
            dict[str, str],
        ]
    ] = []

    # Build only same-class candidates.
    for key in sorted(set(gt_by_image) & set(pred_by_image)):
        for gi, gt in gt_by_image[key]:
            gt_cls = gt["garment_category"].strip().lower()

            gt_box = get_box(gt)

            for pi, pred in pred_by_image[key]:
                pred_cls = pred["garment_category"].strip().lower()

                if pred_cls != gt_cls:
                    continue

                iou = box_iou(
                    gt_box,
                    get_box(pred),
                )

                if iou >= args.iou_threshold:
                    candidate_pairs.append(
                        (
                            iou,
                            gi,
                            pi,
                            gt,
                            pred,
                        )
                    )

    # Greedy one-to-one by highest IoU.
    candidate_pairs.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    for (
        iou,
        gi,
        pi,
        gt,
        pred,
    ) in candidate_pairs:

        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)

        matched_id = f"matched_{len(matched_rows)+1:03d}"

        out = dict(pred)

        # Critical diagnostic join policy:
        # use GT source_image + GT garment_id as keys,
        # while preserving predicted ROI file paths.
        out["source_image"] = gt["source_image"]
        out["garment_id"] = gt["garment_id"]
        out["garment_category"] = gt["garment_category"]

        out["predicted_garment_id"] = pred.get(
            "garment_id",
            "",
        )
        out["matched_pair_id"] = matched_id
        out["matched_gt_bbox_iou"] = f"{iou:.6f}"
        out["matched_gt_record_id"] = gt.get(
            "record_id",
            "",
        )
        out["matched_gt_fine_category"] = gt.get(
            "fine_category",
            "",
        )
        out["paired_manifest_role"] = "predicted_roi_with_gt_join_key"

        matched_rows.append(out)

        matched_pair_rows.append(
            {
                "matched_pair_id": matched_id,
                "source_dataset": gt.get(
                    "source_dataset",
                    "",
                ),
                "source_image": gt["source_image"],
                "garment_category": gt["garment_category"],
                "gt_garment_id": gt["garment_id"],
                "predicted_garment_id": pred.get(
                    "garment_id",
                    "",
                ),
                "prediction_score": pred.get(
                    "prediction_score",
                    "",
                ),
                "bbox_iou": f"{iou:.6f}",
                "predicted_crop_path": pred.get(
                    "crop_path",
                    "",
                ),
                "predicted_mask_path": pred.get(
                    "mask_path",
                    "",
                ),
            }
        )

    unmatched_predictions = []

    for pi, pred in enumerate(pred_rows):
        if pi in used_pred:
            continue

        unmatched_predictions.append(
            {
                "source_dataset": pred.get(
                    "source_dataset",
                    "",
                ),
                "source_image": pred.get(
                    "source_image",
                    "",
                ),
                "predicted_garment_id": pred.get(
                    "garment_id",
                    "",
                ),
                "predicted_category": pred.get(
                    "garment_category",
                    "",
                ),
                "prediction_score": pred.get(
                    "prediction_score",
                    "",
                ),
                "reason": ("no_one_to_one_same_class_bbox50_match"),
            }
        )

    # Only GT from images actually included in the predicted 20-image pilot.
    pilot_image_keys = set(pred_by_image.keys())

    unmatched_gt = []

    for gi, gt in enumerate(gt_rows):
        if image_key(gt["source_image"]) not in pilot_image_keys:
            continue

        if gi in used_gt:
            continue

        unmatched_gt.append(
            {
                "source_dataset": gt.get(
                    "source_dataset",
                    "",
                ),
                "source_image": gt.get(
                    "source_image",
                    "",
                ),
                "gt_garment_id": gt.get(
                    "garment_id",
                    "",
                ),
                "gt_category": gt.get(
                    "garment_category",
                    "",
                ),
                "reason": ("no_one_to_one_same_class_bbox50_prediction"),
            }
        )

    preferred_fields = [
        "source_dataset",
        "source_image",
        "garment_id",
        "garment_category",
        "fine_category",
        "crop_path",
        "mask_path",
        "masked_preview_path",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "prediction_score",
        "prediction_model",
        "score_threshold",
        "mask_threshold",
        "predicted_garment_id",
        "matched_pair_id",
        "matched_gt_bbox_iou",
        "matched_gt_record_id",
        "matched_gt_fine_category",
        "paired_manifest_role",
    ]

    write_csv(
        output_path,
        matched_rows,
        preferred_fields,
    )

    write_csv(
        REPORT_DIR / "matched_pairs.csv",
        matched_pair_rows,
    )

    write_csv(
        REPORT_DIR / "unmatched_predictions.csv",
        unmatched_predictions,
    )

    write_csv(
        REPORT_DIR / "unmatched_gt.csv",
        unmatched_gt,
    )

    class_counts = Counter(row["garment_category"] for row in matched_rows)

    source_counts = Counter(
        row.get(
            "source_dataset",
            "",
        )
        for row in matched_rows
    )

    ious = [float(row["matched_gt_bbox_iou"]) for row in matched_rows]

    summary = [
        "PRD 3.1 Predicted ROI Matched Pair Manifest v1",
        "=============================================",
        "",
        f"predicted_input={pred_path.relative_to(PROJECT_ROOT)}",
        f"gt_input={gt_path.relative_to(PROJECT_ROOT)}",
        f"iou_threshold={args.iou_threshold}",
        f"raw_predictions={len(pred_rows)}",
        f"matched_one_to_one_pairs={len(matched_rows)}",
        f"unmatched_predictions={len(unmatched_predictions)}",
        f"unmatched_gt_in_pilot_images={len(unmatched_gt)}",
        (
            "mean_matched_bbox_iou=" + f"{sum(ious)/len(ious):.4f}"
            if ious
            else "mean_matched_bbox_iou=NA"
        ),
        "",
        "Matched category counts",
        "-----------------------",
    ]

    for cls in PRD_CLASSES:
        summary.append(f"{cls}={class_counts.get(cls, 0)}")

    summary += [
        "",
        "Matched source counts",
        "---------------------",
    ]

    for source, count in sorted(source_counts.items()):
        summary.append(f"{source}={count}")

    summary += [
        "",
        "Important",
        "---------",
        "- Each prediction and each GT instance appears at most once.",
        "- Matching is class-correct and requires bbox IoU >= threshold.",
        "- The output uses GT source_image + GT garment_id only as comparison join keys.",
        "- crop_path/mask_path remain predicted ROI artifacts from v2-balanced.",
        "- Use this paired manifest for GT-vs-predicted ROI attribute propagation analysis.",
        "- Do not call this a deployment manifest.",
    ]

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (REPORT_DIR / "summary.txt").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== MATCHED ROI MANIFEST FINISHED ===")
    print(
        "Matched pairs:",
        len(matched_rows),
    )
    print(
        "Output manifest:",
        output_path.relative_to(PROJECT_ROOT),
    )
    print(
        "Summary:",
        (REPORT_DIR / "summary.txt").relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
