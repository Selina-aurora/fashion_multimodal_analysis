"""Manual localization-quality labels and summary helpers."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal, TypedDict

LocalizationQuality = Literal["correct", "coarse", "wrong", "missed"]


class QualitySummary(TypedDict):
    """Aggregated manual localization metrics for one evaluated group."""

    n: int
    correct: int
    coarse: int
    wrong: int
    missed: int
    strict_accuracy: float
    usable_rate: float
    missed_rate: float


def summarize_quality_labels(
    labels: Iterable[LocalizationQuality],
) -> QualitySummary:
    """Summarize manual labels using project reporting definitions.

    Args:
        labels: Localization labels from the controlled manual audit.

    Returns:
        Counts and derived strict-accuracy, usable-rate, and missed-rate metrics.

    Raises:
        ValueError: If an unsupported label is present.
    """
    values = list(labels)
    counts = Counter(values)
    allowed = {"correct", "coarse", "wrong", "missed"}
    unknown = set(counts) - allowed
    if unknown:
        raise ValueError(f"unsupported localization labels: {sorted(unknown)}")

    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "correct": 0,
            "coarse": 0,
            "wrong": 0,
            "missed": 0,
            "strict_accuracy": 0.0,
            "usable_rate": 0.0,
            "missed_rate": 0.0,
        }

    correct = counts["correct"]
    coarse = counts["coarse"]
    wrong = counts["wrong"]
    missed = counts["missed"]
    return {
        "n": n,
        "correct": correct,
        "coarse": coarse,
        "wrong": wrong,
        "missed": missed,
        "strict_accuracy": correct / n,
        "usable_rate": (correct + coarse) / n,
        "missed_rate": missed / n,
    }
