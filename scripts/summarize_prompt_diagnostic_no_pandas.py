"""Summarize prompt diagnostic results without pandas."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "prompt_diagnostic_2026_09_14"
CASE_RESULTS = REPORT_DIR / "case_results.csv"
SUMMARY = REPORT_DIR / "summary.csv"


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean, or None for an empty list."""
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    """Read case-level results and write grouped summary.csv."""
    if not CASE_RESULTS.is_file():
        raise FileNotFoundError(f"Missing case results: {CASE_RESULTS}")

    with CASE_RESULTS.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["target"],
            row["prompt_variant"],
            row["prompt"],
        )
        groups[key].append(row)

    output_rows = []
    for (target, prompt_variant, prompt), group in sorted(groups.items()):
        detected_count = sum(
            row["detected"].strip().lower() == "yes" for row in group
        )

        top_scores = [
            float(row["top_score"])
            for row in group
            if row["top_score"].strip()
        ]
        area_ratios = [
            float(row["top_box_area_ratio"])
            for row in group
            if row["top_box_area_ratio"].strip()
        ]
        inference_times = [
            float(row["inference_seconds"])
            for row in group
            if row["inference_seconds"].strip()
        ]

        mean_top_score = _mean(top_scores)
        mean_area_ratio = _mean(area_ratios)
        mean_inference_seconds = _mean(inference_times)

        output_rows.append(
            {
                "target": target,
                "prompt_variant": prompt_variant,
                "prompt": prompt,
                "n": len(group),
                "detection_rate": round(
                    100.0 * detected_count / len(group),
                    1,
                ),
                "mean_top_score": (
                    round(mean_top_score, 4)
                    if mean_top_score is not None
                    else ""
                ),
                "mean_top_box_area_ratio": (
                    round(mean_area_ratio, 4)
                    if mean_area_ratio is not None
                    else ""
                ),
                "mean_inference_seconds": (
                    round(mean_inference_seconds, 4)
                    if mean_inference_seconds is not None
                    else ""
                ),
            }
        )

    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "target",
                "prompt_variant",
                "prompt",
                "n",
                "detection_rate",
                "mean_top_score",
                "mean_top_box_area_ratio",
                "mean_inference_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Loaded {len(rows)} case-level rows.")
    print(f"Wrote {len(output_rows)} summary rows.")
    print(f"Summary: {SUMMARY}")


if __name__ == "__main__":
    main()
