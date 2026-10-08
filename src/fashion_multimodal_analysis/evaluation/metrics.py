"""评估工具：逐例表是指标回算依据，计时只含预处理、前向和后处理。

Unchanged bounding-box and binary-mask IoU arithmetic.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def box_iou(a: Any, b: Any) -> float:
    """计算两个 xyxy 边界框的交并比。

    Args:
        a: 第一个框，依次为 x1、y1、x2、y2，允许浮点坐标。
        b: 第二个框，使用与 a 相同的原图坐标系。

    Returns:
        交集面积 / 并集面积，范围为 0 到 1。边界相触、反向坐标或空并集为 0。

    Notes:
        宽高按 x2-x1、y2-y1 计算，不额外加 1，保持既有评估口径。
    """
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (
        max(0, ax2 - ax1) * max(0, ay2 - ay1)
        + max(0, bx2 - bx1) * max(0, by2 - by1)
        - inter
    )
    return inter / ua if ua > 0 else 0.0


def mask_iou(a: Any, b: Any) -> float:
    """计算两个同尺寸掩码的前景交并比。

    Args:
        a: 第一个二维掩码，非零像素视为前景。
        b: 第二个二维掩码，必须与 a 使用相同尺寸和坐标系。

    Returns:
        前景交集 / 前景并集。两个掩码都没有前景时返回 0。

    Notes:
        本函数没有执行坐标映射。ROI 掩码必须先还原为整图掩码，再计算本指标。
    """
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0
