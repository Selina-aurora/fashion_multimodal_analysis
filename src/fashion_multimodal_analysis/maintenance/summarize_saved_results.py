"""从已保存的评估 JSON/TXT 建立结果索引，不重新运行模型。

每行保留报告目录和原始字段，便于回到逐例 CSV 检查指标分母。
没有统一协议 JSON 的旧实验标为 legacy_summary，不补造指标。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fashion_multimodal_analysis.common.io import write_csv
from fashion_multimodal_analysis.common.paths import project_root


def summary_values(path: Path) -> dict[str, str]:
    """读取 key=value 格式的实验说明，保留原始字符串。

    Args:
        path: 已保存的文本报告。

    Returns:
        文本中包含等号的字段；解释性行不转换成指标。

    Raises:
        OSError: 报告无法读取。
    """
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def collect_evaluations(root: Path) -> list[dict[str, Any]]:
    """汇总主报告树的统一协议评估和较早的文本评估。

    Args:
        root: 包含 reports 的项目目录。

    Returns:
        每个评估一行，字段包括报告路径、计数、召回、精度、F1、掩码及时间。
        缺少的历史字段留空；条件平均掩码 IoU 不转换成全部 GT 的准确率。

    Raises:
        OSError: 报告无法读取。
        ValueError: 协议 JSON 内容无效。
    """
    rows = []
    covered = set()
    for path in sorted((root / "reports").rglob("protocol_metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        micro = metrics.get("micro", {})
        text = path.with_name("evaluation_summary.txt")
        values = summary_values(text) if text.is_file() else {}
        row = {"report": path.parent.relative_to(root).as_posix(), "format": "protocol"}
        for key in (
            "gt_count",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "bbox50_recall",
            "mean_mask_iou_correct_class",
            "mask_iou_ge_0_85_count",
            "mask_iou_ge_0_85_rate_all_gt",
        ):
            row[key] = micro.get(key)
        for key in (
            "checkpoint_epoch",
            "score_threshold",
            "match_bbox_iou",
            "mask_threshold",
            "validation_images",
            "mean_inference_ms_per_image",
            "median_inference_ms_per_image",
            "p95_inference_ms_per_image",
        ):
            row[key] = values.get(key, "")
        rows.append(row)
        covered.add(path.parent)
    for path in sorted((root / "reports").rglob("evaluation_summary*.txt")):
        if path.parent in covered:
            continue
        values = summary_values(path)
        row = {"report": path.relative_to(root).as_posix(), "format": "legacy_summary"}
        for key in (
            "checkpoint_epoch",
            "score_threshold",
            "match_bbox_iou",
            "mask_threshold",
            "validation_images",
            "mean_inference_ms_per_image",
            "median_inference_ms_per_image",
            "p95_inference_ms_per_image",
        ):
            row[key] = values.get(key, "")
        for new, old in (
            ("gt_count", "gt_instances"),
            ("recall", "end_to_end_bbox50_class_recall"),
            ("bbox50_recall", "bbox50_recall"),
            ("mean_mask_iou_correct_class", "mean_mask_iou_correct_class"),
            ("mask_iou_ge_0_85_rate_all_gt", "mask_iou_ge_0_85_rate"),
        ):
            row[new] = values.get(old, "")
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["report"]))


def main() -> None:
    """解析路径并写入独立汇总表，原报告保持不变。

    Raises:
        SystemExit: 命令行参数非法。
        OSError: 结果表无法写入。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None, help="项目目录，默认自动定位")
    parser.add_argument(
        "--output",
        default="reports/repository_summary/saved_evaluations.csv",
        help="独立汇总表的相对路径",
    )
    args = parser.parse_args()
    root = Path(args.project).resolve() if args.project else project_root()
    output = Path(args.output)
    destination = output if output.is_absolute() else root / output
    rows = collect_evaluations(root)
    write_csv(destination, rows)
    print(
        f"已索引 {len(rows)} 个评估报告：{destination.relative_to(root) if destination.is_relative_to(root) else destination}"
    )


if __name__ == "__main__":
    main()
