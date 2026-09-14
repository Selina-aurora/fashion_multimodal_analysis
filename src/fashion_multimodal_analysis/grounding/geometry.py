"""Geometry helpers shared by grounding diagnostics."""

from __future__ import annotations

from typing import Sequence

BBox = tuple[float, float, float, float]
RelativeWindow = tuple[float, float, float, float]


def box_area_ratio(
    box: Sequence[float],
    image_width: float,
    image_height: float,
) -> float:
    """Return bounding-box area divided by full image area.

    Args:
        box: Bounding box in ``xyxy`` order.
        image_width: Full image width in pixels.
        image_height: Full image height in pixels.

    Returns:
        Bounding-box area divided by image area.

    Raises:
        ValueError: If either image dimension is non-positive.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    x1, y1, x2, y2 = map(float, box)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return (width * height) / (float(image_width) * float(image_height))


def intersection_over_union(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """Compute intersection over union for two ``xyxy`` boxes.

    Args:
        box_a: First bounding box in ``xyxy`` order.
        box_b: Second bounding box in ``xyxy`` order.

    Returns:
        Intersection-over-union value in the range ``[0, 1]``.
    """
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def relative_window_to_bbox(
    roi: Sequence[float],
    window: RelativeWindow,
) -> BBox:
    """Map a normalized local window inside an ROI to absolute coordinates.

    Args:
        roi: Garment ROI in ``xyxy`` order.
        window: Relative coordinates ``(x1, y1, x2, y2)`` in ``[0, 1]``.

    Returns:
        Absolute bounding box in ``xyxy`` order.

    Raises:
        ValueError: If the window or ROI ordering is invalid.
    """
    x1, y1, x2, y2 = map(float, roi)
    rx1, ry1, rx2, ry2 = window

    if not (0 <= rx1 <= rx2 <= 1 and 0 <= ry1 <= ry2 <= 1):
        raise ValueError("relative window values must satisfy 0 <= min <= max <= 1")

    width, height = x2 - x1, y2 - y1
    if width < 0 or height < 0:
        raise ValueError("ROI must use xyxy ordering")

    return (
        x1 + rx1 * width,
        y1 + ry1 * height,
        x1 + rx2 * width,
        y1 + ry2 * height,
    )
