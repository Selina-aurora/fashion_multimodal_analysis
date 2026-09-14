"""Unit tests for grounding geometry helpers."""

from fashion_multimodal_analysis.grounding.geometry import (
    box_area_ratio,
    intersection_over_union,
    relative_window_to_bbox,
)


def test_box_area_ratio() -> None:
    """Area ratio should use the full image area as denominator."""
    assert box_area_ratio((0, 0, 50, 50), 100, 100) == 0.25


def test_iou_identity() -> None:
    """A box should have IoU 1.0 with itself."""
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_relative_window_to_bbox() -> None:
    """Relative windows should map correctly inside an absolute ROI."""
    result = relative_window_to_bbox(
        (10, 20, 110, 220),
        (0.2, 0.1, 0.8, 0.5),
    )
    assert result == (30.0, 40.0, 90.0, 120.0)
