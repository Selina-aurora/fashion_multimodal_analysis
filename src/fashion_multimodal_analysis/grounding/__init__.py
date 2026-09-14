"""Reusable utilities for language-guided fashion-region localization."""

from fashion_multimodal_analysis.grounding.config import (
    CONDITIONS,
    DEFAULT_LOCAL_NMS_IOU,
    DEFAULT_ROI_MARGIN_RATIO,
    DEFAULT_THRESHOLD,
    LOCAL_WINDOW_RULES,
    MODEL_ID,
    SCALE_GROUPS,
    TARGET_ORDER,
)
from fashion_multimodal_analysis.grounding.geometry import (
    box_area_ratio,
    intersection_over_union,
    relative_window_to_bbox,
)
from fashion_multimodal_analysis.grounding.quality import (
    LocalizationQuality,
    QualitySummary,
    summarize_quality_labels,
)

__all__ = [
    "CONDITIONS",
    "DEFAULT_LOCAL_NMS_IOU",
    "DEFAULT_ROI_MARGIN_RATIO",
    "DEFAULT_THRESHOLD",
    "LOCAL_WINDOW_RULES",
    "MODEL_ID",
    "SCALE_GROUPS",
    "TARGET_ORDER",
    "LocalizationQuality",
    "QualitySummary",
    "box_area_ratio",
    "intersection_over_union",
    "relative_window_to_bbox",
    "summarize_quality_labels",
]
