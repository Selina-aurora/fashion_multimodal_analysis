"""Unit tests for manual localization-quality metrics."""

from fashion_multimodal_analysis.grounding.quality import summarize_quality_labels


def test_quality_summary() -> None:
    """Summary metrics should follow the project's reporting definitions."""
    summary = summarize_quality_labels(["correct", "coarse", "wrong", "missed"])
    assert summary["strict_accuracy"] == 0.25
    assert summary["usable_rate"] == 0.5
    assert summary["missed_rate"] == 0.25
