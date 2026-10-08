"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。"""

from __future__ import annotations

import csv
import json
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
# Configuration
# ==========================

INPUT_FILE = Path("outputs/prompt_optimization/prompt_variant_results.json")


OUTPUT_FILE = Path("reports/prompt_variant_summary.csv")


# ==========================
# Attribute Matching
# ==========================


def check_label_match(label: str, category: str) -> bool:
    """Check whether detected label
    contains target attribute.

    Args:
        label: 标签。
        category: 本次处理的服饰或区域类别。

    Returns:
        当前条件的校验结果；失败条件及返回形式见函数体。
    """

    label = label.lower()

    category = category.lower()

    keywords = {
        "sleeve": ["sleeve"],
        "collar": ["collar"],
        "button": ["button"],
        "zipper": ["zipper"],
    }

    for keyword in keywords.get(category, []):

        if keyword in label:
            return True

    return False


# ==========================
# Evaluation Function
# ==========================


def evaluate_prompt(detections: Any, category: str) -> dict[str, Any]:
    """评估 提示文本。

    Args:
        detections: detections。
        category: 本次处理的服饰或区域类别。

    Returns:
        结果字典，主要字段为 Total Images, Detected Images, Detection Rate, Matched Images,
        Matched Label Rate, Average Confidence, Best Confidence, Total Detections。
    """
    total_images = 0

    detected_images = 0

    matched_images = 0

    confidence_scores = []

    detection_count = 0

    for image_name, image_results in detections.items():

        total_images += 1

        if len(image_results) > 0:

            detected_images += 1

            detection_count += len(image_results)

            for result in image_results:

                score = result.get("score", 0)

                label = result.get("label", "")

                confidence_scores.append(score)

                if check_label_match(label, category):

                    matched_images += 1

                    break

    detection_rate = detected_images / total_images if total_images else 0

    match_rate = matched_images / total_images if total_images else 0

    avg_confidence = mean(confidence_scores) if confidence_scores else 0

    best_score = max(confidence_scores) if confidence_scores else 0

    return {
        "Total Images": total_images,
        "Detected Images": detected_images,
        "Detection Rate": round(detection_rate, 4),
        "Matched Images": matched_images,
        "Matched Label Rate": round(match_rate, 4),
        "Average Confidence": round(avg_confidence, 4),
        "Best Confidence": round(best_score, 4),
        "Total Detections": detection_count,
    }


# ==========================
# Main
# ==========================


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    print("Loading:", INPUT_FILE)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        data = json.load(f)

    records = []

    for category, prompts in data.items():

        for prompt, results in prompts.items():

            metrics = evaluate_prompt(results, category)

            record = {"Category": category, "Prompt": prompt, **metrics}

            records.append(record)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=records[0].keys())

        writer.writeheader()

        writer.writerows(records)

    print("\nFinished!")

    print("Saved:", OUTPUT_FILE)

    print("\nSummary:")

    for r in records:

        print(r)


if __name__ == "__main__":

    main()
