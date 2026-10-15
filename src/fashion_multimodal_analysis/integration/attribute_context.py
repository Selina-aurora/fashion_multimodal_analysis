"""领型裁剪候选：保留领口周围的原图信息，分歧时明确要求复核。

不读取人工标签；裁剪参数是待验证的开发设置，不代表校准后准确率。
此策略仅作用于预测领口；受控 GT 路径和其他属性不使用该策略。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def neckline_context_bounds(
    collar_box: Sequence[float],
    garment_box: Sequence[float],
    image_size: tuple[int, int],
    expansion: float = 2.0,
) -> list[int]:
    """以预测领口为中心扩展裁剪，限制在当前服装框与原图内。

    保留周围皮肤、领口开口和衣料，不使用局部 SAM mask 涂白。
    最小裁剪宽高分别取服装框的 25%/15%，避免只放大小块衣料。
    这些比例尚未验证，导出实际框供逐例审核。
    """
    if not math.isfinite(expansion) or expansion < 1:
        raise ValueError("Neckline expansion must be finite and >= 1")
    width, height = image_size
    for box in (collar_box, garment_box):
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError("Invalid context box")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Invalid context box")
    gx1 = max(0, math.floor(garment_box[0]))
    gy1 = max(0, math.floor(garment_box[1]))
    gx2 = min(width, math.ceil(garment_box[2]))
    gy2 = min(height, math.ceil(garment_box[3]))
    x1, y1 = max(gx1, collar_box[0]), max(gy1, collar_box[1])
    x2, y2 = min(gx2, collar_box[2]), min(gy2, collar_box[3])
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Predicted collar does not intersect garment ROI")
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    crop_w = max((x2 - x1) * expansion, (gx2 - gx1) * 0.25)
    crop_h = max((y2 - y1) * expansion, (gy2 - gy1) * 0.15)
    return [
        max(gx1, math.floor(cx - crop_w / 2)),
        max(gy1, math.floor(cy - crop_h / 2)),
        min(gx2, math.ceil(cx + crop_w / 2)),
        min(gy2, math.ceil(cy + crop_h / 2)),
    ]


def paired_neckline_result(
    legacy: dict[str, Any],
    contextual: dict[str, Any],
    tight_bounds: Sequence[float],
    context_bounds: Sequence[float],
) -> dict[str, Any]:
    """两种裁剪分歧时不给单一正式标签，保留两个模型候选供复核。

    一致只表示本次视图一致，不表示正确；CLIP 分数是候选间相对分数。
    label 为空必须结合 status 解释，不得作为 absence 或 correct。
    """
    agreement = legacy["top1_label"] == contextual["top1_label"]
    return {
        "status": "predicted" if agreement else "requires_review",
        "label": contextual["top1_label"] if agreement else "",
        "confidence": float(contextual["top1_score"]) if agreement else None,
        "roi_provenance": "predicted_collar_with_raw_context_pair",
        "input_view_agreement": agreement,
        "semantic_review_status": "pending",
        "legacy_candidate": {
            "label": legacy["top1_label"],
            "confidence": float(legacy["top1_score"]),
        },
        "context_candidate": {
            "label": contextual["top1_label"],
            "confidence": float(contextual["top1_score"]),
        },
        "legacy_crop_bbox": list(tight_bounds),
        "context_crop_bbox": list(context_bounds),
        "coordinate_space": "original_image",
        "score_note": "CLIP prompt-relative scores, not calibrated correctness probabilities.",
        "review_note": "View agreement is not verified accuracy; disagreement is not absence.",
    }
