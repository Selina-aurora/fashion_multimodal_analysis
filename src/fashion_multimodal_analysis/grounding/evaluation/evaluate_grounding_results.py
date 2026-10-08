"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

# ==========================
# Load Results
# ==========================


def load_results(folder: Any) -> Any:
    """加载 结果。

    Args:
        folder: folder。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。
    """
    json_results = []

    for file in os.listdir(folder):

        if file.endswith(".json"):

            path = os.path.join(folder, file)

            with open(path, "r", encoding="utf-8") as f:

                data = json.load(f)

            json_results.append(data)

    return json_results


# ==========================
# Evaluation
# ==========================


def evaluate(results: Any) -> dict[str, Any]:
    """按固定评估设置计算结果，保留逐例记录和运行凭据。

    Args:
        results: 结果。

    Returns:
        结果字典，主要字段为 total_images, detected_images, detection_rate, average_confidence。
    """
    total_images = 0
    detected_images = 0

    confidence_scores = []

    for result in results:

        for image_name, detections in result.items():

            total_images += 1

            # detection exists
            if len(detections) > 0:

                detected_images += 1

                for detection in detections:

                    confidence_scores.append(detection["score"])

    detection_rate = detected_images / total_images if total_images > 0 else 0

    average_confidence = mean(confidence_scores) if len(confidence_scores) > 0 else 0

    return {
        "total_images": total_images,
        "detected_images": detected_images,
        "detection_rate": detection_rate,
        "average_confidence": average_confidence,
    }


# ==========================
# Save Markdown Report
# ==========================


def save_report(metrics: dict[str, Any], output_path: str | Path) -> None:
    """保存 报告。

    Args:
        metrics: 记录字段，使用 total_images, detected_images, detection_rate,
        average_confidence。
        output_path: 对应文件的相对路径或当前解析后的路径。
    """
    output_dir = os.path.dirname(output_path)

    if output_dir:

        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("# Grounding DINO Evaluation Summary\n\n")

        f.write("| Metric | Value |\n")

        f.write("|---|---|\n")

        f.write(f"| Total Images | {metrics['total_images']} |\n")

        f.write(f"| Detected Images | {metrics['detected_images']} |\n")

        f.write(f"| Detection Rate | {metrics['detection_rate']:.2%} |\n")

        f.write(f"| Average Confidence | {metrics['average_confidence']:.4f} |\n")


# ==========================
# Main
# ==========================


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    parser = argparse.ArgumentParser(
        description="Evaluate Grounding DINO detection results"
    )

    parser.add_argument(
        "--input", required=True, help="Folder containing json result files"
    )

    parser.add_argument(
        "--output",
        default="reports/grounding_evaluation_summary.md",
        help="Output markdown report path",
    )

    args = parser.parse_args()

    results = load_results(args.input)

    metrics = evaluate(results)

    print("\nEvaluation Results")
    print("----------------------")

    for key, value in metrics.items():

        print(f"{key}: {value}")

    save_report(metrics, args.output)

    print("\nSaved report:", args.output)


if __name__ == "__main__":

    main()
