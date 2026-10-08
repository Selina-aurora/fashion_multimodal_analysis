"""3.1.1 实例分割：数据与模型版本分别记录，指标按固定匹配和计时口径解释。

Run a score-threshold sweep on the already-trained PRD 3.1.1 8-class
Mask R-CNN checkpoint.

This does NOT retrain the model.

It repeatedly calls:
    scripts/segmentation/evaluation/eval_prd_8class_maskrcnn_baseline_v1.py

Thresholds:
    0.20, 0.30, 0.40, 0.50

For each threshold it preserves the evaluation outputs in a separate folder and
builds one comparison CSV.

Run from project root:
    python scripts/segmentation/evaluation/sweep_prd_8class_maskrcnn_thresholds_v1.py
    --device cuda

Output:
    reports/prd_instance_segmentation/maskrcnn_8class_threshold_sweep_v1/
        threshold_sweep_summary.csv
        threshold_sweep_per_source.csv
        threshold_sweep_per_class_recall.csv
        thr_0p20/
        thr_0p30/
        thr_0p40/
        thr_0p50/
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()

EVAL_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "segmentation/evaluation"
    / "eval_prd_8class_maskrcnn_baseline_v1.py"
)

SOURCE_REPORT_DIR = (
    PROJECT_ROOT / "reports" / "prd_instance_segmentation" / "maskrcnn_8class_eval_v1"
)

SWEEP_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_instance_segmentation"
    / "maskrcnn_8class_threshold_sweep_v1"
)

DEFAULT_THRESHOLDS = [0.20, 0.30, 0.40, 0.50]


def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
    )
    p.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint path. Empty = evaluator default checkpoint_last.pth.",
    )
    return p.parse_args()


def threshold_tag(value: float) -> str:
    """阈值 tag。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return f"thr_{value:.2f}".replace(".", "p")


def parse_summary(path: Path) -> dict[str, str]:
    """解析 汇总。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        返回 result，由函数体中同名变量的计算/收集过程得到。
    """
    result: dict[str, str] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()

    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。
    """
    if not path.is_file():
        return []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path,
    rows: list[dict],
    fields: list[str],
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

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    args = parse_args()

    if not EVAL_SCRIPT.is_file():
        raise FileNotFoundError(f"Evaluator not found:\n  {EVAL_SCRIPT}")

    SWEEP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []
    source_rows = []
    class_rows = []

    for threshold in args.thresholds:
        tag = threshold_tag(threshold)

        print()
        print("=" * 70)
        print(f"Evaluating score threshold = {threshold:.2f}")
        print("=" * 70)

        cmd = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--device",
            args.device,
            "--score-threshold",
            str(threshold),
            "--max-visualizations",
            "0",
        ]

        if args.checkpoint:
            cmd += [
                "--checkpoint",
                args.checkpoint,
            ]

        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
        )

        summary_path = SOURCE_REPORT_DIR / "evaluation_summary.txt"

        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Evaluation summary not found after threshold {threshold}"
            )

        parsed = parse_summary(summary_path)

        summary_rows.append(
            {
                "score_threshold": threshold,
                "predictions_above_threshold": parsed.get(
                    "predictions_above_threshold", ""
                ),
                "bbox50_recall": parsed.get("bbox50_recall", ""),
                "bbox50_precision": parsed.get("bbox50_precision", ""),
                "category_accuracy_on_localized": parsed.get(
                    "category_accuracy_on_localized", ""
                ),
                "end_to_end_bbox50_class_recall": parsed.get(
                    "end_to_end_bbox50_class_recall", ""
                ),
                "mean_bbox_iou_localized": parsed.get("mean_bbox_iou_localized", ""),
                "mean_mask_iou_correct_class": parsed.get(
                    "mean_mask_iou_correct_class", ""
                ),
                "mask_iou_ge_0_50_rate": parsed.get("mask_iou_ge_0_50_rate", ""),
                "mask_iou_ge_0_85_rate": parsed.get("mask_iou_ge_0_85_rate", ""),
                "mean_inference_ms_per_image": parsed.get(
                    "mean_inference_ms_per_image", ""
                ),
                "median_inference_ms_per_image": parsed.get(
                    "median_inference_ms_per_image", ""
                ),
                "p95_inference_ms_per_image": parsed.get(
                    "p95_inference_ms_per_image", ""
                ),
                "correct": parsed.get("correct", ""),
                "low_mask_iou": parsed.get("low_mask_iou", ""),
                "wrong_class": parsed.get("wrong_class", ""),
                "missed": parsed.get("missed", ""),
            }
        )

        # Per-source metrics are valid for comparing source-domain behavior.
        for row in read_csv(SOURCE_REPORT_DIR / "per_source_metrics.csv"):
            source_rows.append(
                {
                    "score_threshold": threshold,
                    "source_dataset": row.get("source_dataset", ""),
                    "gt_count": row.get("gt_count", ""),
                    "prediction_count": row.get("prediction_count", ""),
                    "bbox50_recall": row.get("bbox50_recall", ""),
                    "bbox50_precision": row.get("bbox50_precision", ""),
                    "category_accuracy_on_localized": row.get(
                        "category_accuracy_on_localized", ""
                    ),
                    "end_to_end_bbox50_class_recall": row.get(
                        "end_to_end_bbox50_class_recall", ""
                    ),
                    "mean_mask_iou_correct_class": row.get(
                        "mean_mask_iou_correct_class", ""
                    ),
                    "mask_iou_ge_0_85_rate": row.get("mask_iou_ge_0_85_rate", ""),
                }
            )

        # For per-class comparison, use GT-centric metrics only.
        # Do NOT use the v1 evaluator's per-class bbox precision / FP columns:
        # their numerator/denominator class conditioning is inconsistent.
        for row in read_csv(SOURCE_REPORT_DIR / "per_class_metrics.csv"):
            class_rows.append(
                {
                    "score_threshold": threshold,
                    "garment_category": row.get("garment_category", ""),
                    "gt_count": row.get("gt_count", ""),
                    "bbox50_recall": row.get("bbox50_recall", ""),
                    "category_accuracy_on_localized": row.get(
                        "category_accuracy_on_localized", ""
                    ),
                    "end_to_end_bbox50_class_recall": row.get(
                        "end_to_end_bbox50_class_recall", ""
                    ),
                    "mean_mask_iou_correct_class": row.get(
                        "mean_mask_iou_correct_class", ""
                    ),
                    "mask_iou_ge_0_85_rate": row.get("mask_iou_ge_0_85_rate", ""),
                    "missed_gt_count": row.get("missed_gt_count", ""),
                    "wrong_class_localized_count": row.get(
                        "wrong_class_localized_count", ""
                    ),
                }
            )

        dest = SWEEP_DIR / tag

        if dest.exists():
            shutil.rmtree(dest)

        shutil.copytree(
            SOURCE_REPORT_DIR,
            dest,
        )

        print(f"Saved threshold report to: " + f"{dest.relative_to(PROJECT_ROOT)}")

    write_csv(
        SWEEP_DIR / "threshold_sweep_summary.csv",
        summary_rows,
        [
            "score_threshold",
            "predictions_above_threshold",
            "bbox50_recall",
            "bbox50_precision",
            "category_accuracy_on_localized",
            "end_to_end_bbox50_class_recall",
            "mean_bbox_iou_localized",
            "mean_mask_iou_correct_class",
            "mask_iou_ge_0_50_rate",
            "mask_iou_ge_0_85_rate",
            "mean_inference_ms_per_image",
            "median_inference_ms_per_image",
            "p95_inference_ms_per_image",
            "correct",
            "low_mask_iou",
            "wrong_class",
            "missed",
        ],
    )

    write_csv(
        SWEEP_DIR / "threshold_sweep_per_source.csv",
        source_rows,
        [
            "score_threshold",
            "source_dataset",
            "gt_count",
            "prediction_count",
            "bbox50_recall",
            "bbox50_precision",
            "category_accuracy_on_localized",
            "end_to_end_bbox50_class_recall",
            "mean_mask_iou_correct_class",
            "mask_iou_ge_0_85_rate",
        ],
    )

    write_csv(
        SWEEP_DIR / "threshold_sweep_per_class_recall.csv",
        class_rows,
        [
            "score_threshold",
            "garment_category",
            "gt_count",
            "bbox50_recall",
            "category_accuracy_on_localized",
            "end_to_end_bbox50_class_recall",
            "mean_mask_iou_correct_class",
            "mask_iou_ge_0_85_rate",
            "missed_gt_count",
            "wrong_class_localized_count",
        ],
    )

    print()
    print("THRESHOLD SWEEP FINISHED.")
    print(
        "Summary:",
        (SWEEP_DIR / "threshold_sweep_summary.csv").relative_to(PROJECT_ROOT),
    )
    print(
        "Per-source:",
        (SWEEP_DIR / "threshold_sweep_per_source.csv").relative_to(PROJECT_ROOT),
    )
    print(
        "Per-class:",
        (SWEEP_DIR / "threshold_sweep_per_class_recall.csv").relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
