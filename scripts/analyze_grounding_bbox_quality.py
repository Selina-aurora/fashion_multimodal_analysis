"""Analyze Grounding DINO bounding-box size quality."""

import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "fashion_data"

IMAGE_DIR = (
    DATASET_ROOT
    / "raw"
    / "train"
    / "train"
    / "image"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "grounding_group_evaluation"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "grounding_group_evaluation"
)

GROUP_IDS = [1, 2, 3]

LARGE_BOX_THRESHOLD = 0.50
VERY_LARGE_BOX_THRESHOLD = 0.80


def load_predictions(
    prediction_file: Path,
) -> Dict[str, Any]:
    """Load one Grounding DINO prediction file.

    Args:
        prediction_file: Prediction JSON path.

    Returns:
        Parsed prediction data.

    Raises:
        FileNotFoundError: If the prediction file does not exist.
    """
    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_file}"
        )

    with prediction_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def is_target_related(
    label: str,
    target: str,
) -> bool:
    """Check whether a predicted label contains the target keyword.

    Args:
        label: Predicted Grounding DINO label.
        target: Expected clothing component.

    Returns:
        True when the target keyword appears in the label.
    """
    return target.lower() in label.lower()


def select_primary_detection(
    detections: List[Dict[str, Any]],
    target: str,
) -> Optional[Dict[str, Any]]:
    """Select the highest-confidence target-related detection.

    Args:
        detections: Predicted detections for one image.
        target: Expected clothing component.

    Returns:
        Highest-confidence target-related detection, or None.
    """
    target_detections = [
        detection
        for detection in detections
        if is_target_related(
            label=str(detection["label"]),
            target=target,
        )
    ]

    if not target_detections:
        return None

    return max(
        target_detections,
        key=lambda detection: float(
            detection["score"]
        ),
    )


def get_image_size(
    image_name: str,
) -> tuple[int, int]:
    """Read an image width and height.

    Args:
        image_name: Dataset image filename.

    Returns:
        Image width and height.

    Raises:
        FileNotFoundError: If the image is unavailable.
    """
    image_path = IMAGE_DIR / image_name

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as image:
        return image.size


def calculate_area_ratio(
    box: List[float],
    image_width: int,
    image_height: int,
) -> float:
    """Calculate bounding-box area relative to full image area.

    Args:
        box: Bounding box in xyxy format.
        image_width: Full image width.
        image_height: Full image height.

    Returns:
        Bounding-box area ratio between 0 and 1.
    """
    x_min, y_min, x_max, y_max = box

    x_min = max(0.0, min(x_min, image_width))
    x_max = max(0.0, min(x_max, image_width))
    y_min = max(0.0, min(y_min, image_height))
    y_max = max(0.0, min(y_max, image_height))

    box_width = max(0.0, x_max - x_min)
    box_height = max(0.0, y_max - y_min)

    image_area = image_width * image_height

    if image_area <= 0:
        return 0.0

    return (
        box_width * box_height
    ) / image_area


def percentile(
    values: List[float],
    probability: float,
) -> float:
    """Calculate a simple interpolated percentile.

    Args:
        values: Numeric observations.
        probability: Percentile expressed from 0 to 1.

    Returns:
        Interpolated percentile value.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = probability * (
        len(sorted_values) - 1
    )

    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1,
    )

    fraction = position - lower_index

    return (
        sorted_values[lower_index]
        * (1 - fraction)
        + sorted_values[upper_index]
        * fraction
    )


def summarize_ratios(
    ratios: List[float],
) -> Dict[str, float]:
    """Summarize bounding-box area ratios.

    Args:
        ratios: Primary bbox area ratios.

    Returns:
        Summary statistics.
    """
    if not ratios:
        return {
            "num_predictions": 0,
            "mean_bbox_area_ratio": 0.0,
            "median_bbox_area_ratio": 0.0,
            "std_bbox_area_ratio": 0.0,
            "p90_bbox_area_ratio": 0.0,
            "large_box_rate": 0.0,
            "very_large_box_rate": 0.0,
        }

    large_count = sum(
        ratio >= LARGE_BOX_THRESHOLD
        for ratio in ratios
    )

    very_large_count = sum(
        ratio >= VERY_LARGE_BOX_THRESHOLD
        for ratio in ratios
    )

    return {
        "num_predictions": len(ratios),
        "mean_bbox_area_ratio": mean(ratios),
        "median_bbox_area_ratio": median(ratios),
        "std_bbox_area_ratio": (
            pstdev(ratios)
            if len(ratios) > 1
            else 0.0
        ),
        "p90_bbox_area_ratio": percentile(
            ratios,
            0.90,
        ),
        "large_box_rate": (
            large_count / len(ratios)
        ),
        "very_large_box_rate": (
            very_large_count / len(ratios)
        ),
    }


def collect_group_rows(
    group_id: int,
    predictions: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Collect bbox statistics for one evaluation group.

    Args:
        group_id: Evaluation group identifier.
        predictions: Grounding DINO prediction data.

    Returns:
        Per-image bbox diagnostic rows.
    """
    rows = []

    prediction_data = predictions[
        "predictions"
    ]

    for target, target_data in (
        prediction_data.items()
    ):
        for prompt_type, prompt_data in (
            target_data.items()
        ):
            prompt = prompt_data["prompt"]

            for image_name, detections in (
                prompt_data["results"].items()
            ):
                primary = select_primary_detection(
                    detections=detections,
                    target=target,
                )

                if primary is None:
                    continue

                width, height = get_image_size(
                    image_name
                )

                area_ratio = calculate_area_ratio(
                    box=primary["box"],
                    image_width=width,
                    image_height=height,
                )

                rows.append(
                    {
                        "group_id": group_id,
                        "target": target,
                        "prompt_type": prompt_type,
                        "prompt": prompt,
                        "image_name": image_name,
                        "label": primary["label"],
                        "score": float(
                            primary["score"]
                        ),
                        "bbox_area_ratio": area_ratio,
                    }
                )

    return rows


def build_summary(
    detail_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build group-level and overall bbox summaries.

    Args:
        detail_rows: Per-image bbox diagnostic rows.

    Returns:
        Aggregated bbox quality statistics.
    """
    summary_rows = []

    targets = sorted(
        {
            str(row["target"])
            for row in detail_rows
        }
    )

    prompt_types = sorted(
        {
            str(row["prompt_type"])
            for row in detail_rows
        }
    )

    group_labels: List[Any] = (
        GROUP_IDS + ["overall"]
    )

    for group_label in group_labels:
        for target in targets:
            for prompt_type in prompt_types:
                selected_rows = [
                    row
                    for row in detail_rows
                    if row["target"] == target
                    and row["prompt_type"]
                    == prompt_type
                    and (
                        group_label == "overall"
                        or row["group_id"]
                        == group_label
                    )
                ]

                if not selected_rows:
                    continue

                ratios = [
                    float(
                        row["bbox_area_ratio"]
                    )
                    for row in selected_rows
                ]

                metrics = summarize_ratios(
                    ratios
                )

                summary_rows.append(
                    {
                        "group_id": group_label,
                        "target": target,
                        "prompt_type": prompt_type,
                        "prompt": (
                            selected_rows[0][
                                "prompt"
                            ]
                        ),
                        **metrics,
                    }
                )

    return summary_rows


def save_csv(
    rows: List[Dict[str, Any]],
    output_file: Path,
) -> None:
    """Save dictionary rows as CSV.

    Args:
        rows: Data rows to save.
        output_file: Destination CSV path.
    """
    if not rows:
        return

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def print_overall_summary(
    summary_rows: List[Dict[str, Any]],
) -> None:
    """Print overall bbox quality results.

    Args:
        summary_rows: Aggregated bbox statistics.
    """
    print()
    print("Overall bbox quality")
    print("-" * 90)

    overall_rows = [
        row
        for row in summary_rows
        if row["group_id"] == "overall"
    ]

    for row in overall_rows:
        print(
            f'{row["target"]:8s} '
            f'{row["prompt_type"]:9s} | '
            f'N={row["num_predictions"]:3d} | '
            f'Mean={row["mean_bbox_area_ratio"]:.3f} | '
            f'Median={row["median_bbox_area_ratio"]:.3f} | '
            f'P90={row["p90_bbox_area_ratio"]:.3f} | '
            f'Large={row["large_box_rate"]:.3f} | '
            f'VeryLarge={row["very_large_box_rate"]:.3f}'
        )


def main() -> None:
    """Analyze bbox size quality across all evaluation groups."""
    detail_rows = []

    for group_id in GROUP_IDS:
        prediction_file = (
            PREDICTION_DIR
            / f"group_{group_id}_predictions.json"
        )

        predictions = load_predictions(
            prediction_file
        )

        group_rows = collect_group_rows(
            group_id=group_id,
            predictions=predictions,
        )

        detail_rows.extend(group_rows)

        print(
            f"Group {group_id}: "
            f"{len(group_rows)} target-related predictions"
        )

    summary_rows = build_summary(
        detail_rows
    )

    detail_file = (
        REPORT_DIR
        / "bbox_quality_details.csv"
    )

    summary_file = (
        REPORT_DIR
        / "bbox_quality_summary.csv"
    )

    save_csv(
        rows=detail_rows,
        output_file=detail_file,
    )

    save_csv(
        rows=summary_rows,
        output_file=summary_file,
    )

    print_overall_summary(
        summary_rows
    )

    print()
    print(f"Details: {detail_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()