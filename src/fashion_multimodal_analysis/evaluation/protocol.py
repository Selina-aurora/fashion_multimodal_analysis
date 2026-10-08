"""评估工具：逐例表是指标回算依据，计时只含预处理、前向和后处理。

Frozen confidence-first matching and honest acceptance status helpers.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from fashion_multimodal_analysis.evaluation.metrics import box_iou


def valid_box(box: Any) -> bool:
    """有效 边界框。

    Args:
        box: xyxy 坐标的候选框。

    Returns:
        当前条件的校验结果；失败条件及返回形式见函数体。
    """
    try:
        return (
            len(box) == 4
            and all(math.isfinite(float(v)) for v in box)
            and float(box[2]) > float(box[0])
            and float(box[3]) > float(box[1])
        )
    except (TypeError, ValueError):
        return False


def confidence_first_match(
    gt_boxes: Any, pred_boxes: Any, threshold: float = 0.5, scores: Any = None
) -> Any:
    """Return (GT index, prediction index, IoU) in descending confidence order.

    Predictions must already have been filtered by the frozen score threshold.
    Stable ties use input order. Category does not participate in matching.

    Args:
        gt_boxes: 参考目标的 xyxy 边界框。
        pred_boxes: 模型预测的 xyxy 边界框。
        threshold: 当前判定规则使用的阈值。
        scores: 置信度。

    Returns:
        返回 matches，由函数体中同名变量的计算/收集过程得到。

    Raises:
        ValueError: Invalid IoU threshold
    """
    if scores is None or len(scores) != len(pred_boxes):
        raise ValueError(
            "Prediction confidence scores are required for frozen matching"
        )
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid IoU threshold")
    available = {i for i, box in enumerate(gt_boxes) if valid_box(box)}
    matches = []
    order = sorted(
        (i for i in range(len(pred_boxes)) if math.isfinite(float(scores[i]))),
        key=lambda i: (-float(scores[i]), i),
    )
    for pi in order:
        if (
            not available
            or not valid_box(pred_boxes[pi])
            or not math.isfinite(float(scores[pi]))
        ):
            continue
        candidates = [
            (box_iou(gt_boxes[gi], pred_boxes[pi]), gi) for gi in sorted(available)
        ]
        iou, gi = max(candidates, key=lambda item: (item[0], -item[1]))
        if iou >= threshold:
            available.remove(gi)
            matches.append((gi, pi, float(iou)))
    return matches


def flag(value: Any) -> bool:
    """flag。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return value is True or str(value).strip().lower() in ("1", "true", "yes")


def detection_summary(
    rows: list[dict[str, Any]], prediction_count: int
) -> dict[str, Any]:
    """detection 汇总。

    Args:
        rows: 待处理的逐行记录。
        prediction_count: 预测 数量。

    Returns:
        结果字典，主要字段为 tp, fp, fn, precision, recall, f1, localization_matches, gt_count,
        prediction_count, bbox50_recall。

    Raises:
        ValueError: Invalid detection counts or mask IoU
    """
    tp = sum(
        flag(r.get("localized_bbox50")) and flag(r.get("class_correct")) for r in rows
    )
    localized = sum(flag(r.get("localized_bbox50")) for r in rows)
    fp, fn = max(0, prediction_count - tp), len(rows) - tp
    precision = tp / prediction_count if prediction_count else 0.0
    recall = tp / len(rows) if rows else 0.0
    bbox_values = [
        float(r["bbox_iou"])
        for r in rows
        if flag(r.get("localized_bbox50")) and "bbox_iou" in r
    ]
    masks = [
        float(r["mask_iou"])
        for r in rows
        if flag(r.get("localized_bbox50")) and flag(r.get("class_correct"))
    ]
    if prediction_count < tp or any(
        not math.isfinite(v) or not 0 <= v <= 1 for v in masks
    ):
        raise ValueError("Invalid detection counts or mask IoU")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "localization_matches": localized,
        "gt_count": len(rows),
        "prediction_count": prediction_count,
        "bbox50_recall": localized / len(rows) if rows else None,
        "localized_category_accuracy": tp / localized if localized else None,
        "mean_bbox_iou_localized": (
            sum(bbox_values) / len(bbox_values) if bbox_values else None
        ),
        "mean_mask_iou_correct_class": sum(masks) / len(masks) if masks else None,
        "median_mask_iou_correct_class": statistics.median(masks) if masks else None,
        "mask_iou_ge_0_85_count": sum(v >= 0.85 for v in masks),
        "mask_iou_ge_0_85_rate_all_gt": (
            sum(v >= 0.85 for v in masks) / len(rows) if rows else None
        ),
        "mask_iou_ge_0_85_rate_matched_class_correct": (
            sum(v >= 0.85 for v in masks) / len(masks) if masks else None
        ),
    }


def macro_summary(per_class: Any) -> dict[str, Any]:
    """macro 汇总。

    Args:
        per_class: 逐 类别。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    keys = (
        "precision",
        "recall",
        "f1",
        "bbox50_recall",
        "localized_category_accuracy",
        "mean_mask_iou_correct_class",
        "median_mask_iou_correct_class",
        "mask_iou_ge_0_85_rate_all_gt",
        "mask_iou_ge_0_85_rate_matched_class_correct",
    )
    return {
        key: (
            sum(values) / len(values)
            if values and all(v is not None for v in values)
            else None
        )
        for key in keys
        for values in [[m[key] for m in per_class.values()]]
    }


def metric_status(
    value: Any,
    target: Any,
    *,
    higher: bool = True,
    coverage: bool = True,
    evaluated: bool = True,
) -> Any:
    """metric status。

    Args:
        value: 待解析或规范化的输入值。
        target: 目标。
        higher: higher。
        coverage: 覆盖。
        evaluated: evaluated。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if not evaluated or value is None or not math.isfinite(float(value)):
        return "NOT_EVALUATED"
    if not coverage:
        return "INSUFFICIENT_COVERAGE"
    passed = value >= target if higher else value <= target
    return "PASS" if passed else "FAIL"


def grounding_success(prediction: Any, truth: Any, threshold: float = 0.5) -> bool:
    """grounding success。

    Args:
        prediction: 预测。
        truth: truth。
        threshold: 当前判定规则使用的阈值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if prediction.get("status") != "detected" or not flag(
        prediction.get("semantic_target_correct")
    ):
        return False
    p, g = prediction.get("bbox"), truth.get("bbox")
    return bool(
        p and g and valid_box(p) and valid_box(g) and box_iou(p, g) >= threshold
    )


def original_box(
    local_box: Any, crop_bbox: Any, scale_x: float = 1.0, scale_y: float = 1.0
) -> list[Any]:
    """original 边界框。

    Args:
        local_box: 局部 边界框。
        crop_bbox: 裁剪 边界框。
        scale_x: scale x。
        scale_y: scale y。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    x1, y1, x2, y2 = map(float, local_box)
    ox, oy = float(crop_bbox[0]), float(crop_bbox[1])
    return [ox + x1 * scale_x, oy + y1 * scale_y, ox + x2 * scale_x, oy + y2 * scale_y]


def per_attribute_metrics(
    truth_rows: list[dict[str, Any]], prediction_index: Any, track: Any
) -> tuple[Any, ...]:
    """All reviewed eligible GT cases remain in denominator, including missing predictions.

    Args:
        truth_rows: 待处理的 truth 记录。
        prediction_index: 预测 index。
        track: track。

    Returns:
        按顺序返回 details 等结果。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    grouped, details = {}, []
    for truth in truth_rows:
        if (
            truth.get("review_status") != "reviewed"
            or truth.get("applicability") != "eligible"
            or flag(truth.get("ambiguous"))
        ):
            continue
        key = (truth["sample_id"], truth["attribute_name"])
        prediction = prediction_index.get(key, {})
        if not str(truth.get("gt_label", "")).strip():
            raise ValueError("Reviewed eligible attribute has no GT label: " + str(key))
        label = str(prediction.get("label", ""))
        ok = (
            label == truth["gt_label"]
            and prediction.get("status") == "predicted"
            and (track == "A" or flag(prediction.get("garment_class_correct")))
        )
        item = {
            "sample_id": key[0],
            "attribute_name": key[1],
            "garment_category": truth["garment_category"],
            "gt_label": truth["gt_label"],
            "predicted_label": label,
            "prediction_status": prediction.get("status", "missing"),
            "correct": int(ok),
            "track": track,
        }
        details.append(item)
        grouped.setdefault(key[1], []).append(item)
    metrics = [
        {
            "attribute": name,
            "n": len(values),
            "correct": sum(v["correct"] for v in values),
            "accuracy": sum(v["correct"] for v in values) / len(values),
        }
        for name, values in sorted(grouped.items())
    ]
    micro = sum(v["correct"] for v in details) / len(details) if details else None
    macro = sum(v["accuracy"] for v in metrics) / len(metrics) if metrics else None
    return {
        "track": track,
        "n": len(details),
        "micro_accuracy": micro,
        "macro_accuracy": macro,
        "attributes": metrics,
    }, details
