"""Build a stratified manual audit set for grounding localization."""

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw


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

DETAIL_FILE = (
    PROJECT_ROOT
    / "reports"
    / "grounding_group_evaluation"
    / "bbox_quality_details.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "grounding_manual_audit"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "grounding_group_evaluation"
    / "manual_localization_audit.csv"
)

TARGETS = [
    "sleeve",
    "collar",
    "button",
    "zipper",
]

PROMPT_TYPES = [
    "baseline",
    "context",
]

SAMPLES_PER_CONDITION = 5
RANDOM_SEED = 42


def load_detail_rows() -> List[Dict[str, str]]:
    """Load bbox diagnostic rows.

    Returns:
        Detail rows from the bbox quality report.
    """
    with DETAIL_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def load_group_predictions(
    group_id: int,
) -> Dict[str, Any]:
    """Load predictions for one evaluation group.

    Args:
        group_id: Evaluation group identifier.

    Returns:
        Parsed prediction JSON.
    """
    prediction_file = (
        PREDICTION_DIR
        / f"group_{group_id}_predictions.json"
    )

    with prediction_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def select_primary_detection(
    detections: List[Dict[str, Any]],
    target: str,
) -> Dict[str, Any]:
    """Select the highest-confidence target-related detection.

    Args:
        detections: Candidate detections for one image.
        target: Expected clothing component.

    Returns:
        Highest-confidence target-related detection.
    """
    target_detections = [
        detection
        for detection in detections
        if target.lower()
        in str(detection["label"]).lower()
    ]

    return max(
        target_detections,
        key=lambda item: float(item["score"]),
    )


def sample_rows(
    rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Create a stratified reproducible audit sample.

    Args:
        rows: Bbox diagnostic rows.

    Returns:
        Selected audit rows.
    """
    rng = random.Random(RANDOM_SEED)
    selected = []

    for target in TARGETS:
        for prompt_type in PROMPT_TYPES:
            candidates = [
                row
                for row in rows
                if row["target"] == target
                and row["prompt_type"] == prompt_type
            ]

            sample_size = min(
                SAMPLES_PER_CONDITION,
                len(candidates),
            )

            sampled = rng.sample(
                candidates,
                sample_size,
            )

            selected.extend(sampled)

            print(
                f"{target:8s} "
                f"{prompt_type:8s}: "
                f"{sample_size}/{len(candidates)}"
            )

    return selected


def draw_prediction(
    image: Image.Image,
    detection: Dict[str, Any],
    title: str,
) -> Image.Image:
    """Draw one predicted bbox and description.

    Args:
        image: Source image.
        detection: Selected model detection.
        title: Text shown above the image.

    Returns:
        Annotated RGB image.
    """
    image = image.convert("RGB")

    top_margin = 50
    canvas = Image.new(
        "RGB",
        (
            image.width,
            image.height + top_margin,
        ),
        "white",
    )

    canvas.paste(
        image,
        (0, top_margin),
    )

    draw = ImageDraw.Draw(canvas)

    x_min, y_min, x_max, y_max = detection["box"]

    shifted_box = [
        x_min,
        y_min + top_margin,
        x_max,
        y_max + top_margin,
    ]

    draw.rectangle(
        shifted_box,
        outline="red",
        width=4,
    )

    draw.text(
        (10, 10),
        title,
        fill="black",
    )

    return canvas


def save_visualization(
    row: Dict[str, str],
    predictions: Dict[int, Dict[str, Any]],
    audit_id: int,
) -> str:
    """Create one visualized audit example.

    Args:
        row: Selected diagnostic row.
        predictions: Loaded prediction files by group.
        audit_id: Sequential audit identifier.

    Returns:
        Relative visualization path.
    """
    group_id = int(row["group_id"])
    target = row["target"]
    prompt_type = row["prompt_type"]
    image_name = row["image_name"]

    prompt_data = predictions[
        group_id
    ]["predictions"][target][prompt_type]

    detections = prompt_data[
        "results"
    ][image_name]

    detection = select_primary_detection(
        detections=detections,
        target=target,
    )

    image_path = IMAGE_DIR / image_name

    with Image.open(image_path) as image:
        title = (
            f"{audit_id:02d} | "
            f"{target} | "
            f"{prompt_type} | "
            f"score={float(detection['score']):.3f}"
        )

        annotated = draw_prediction(
            image=image,
            detection=detection,
            title=title,
        )

    condition_dir = (
        OUTPUT_DIR
        / target
        / prompt_type
    )

    condition_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_name = (
        f"{audit_id:02d}_"
        f"{Path(image_name).stem}.jpg"
    )

    output_path = (
        condition_dir
        / output_name
    )

    annotated.save(
        output_path,
        quality=95,
    )

    return str(
        output_path.relative_to(
            PROJECT_ROOT
        )
    )


def save_audit_csv(
    rows: List[Dict[str, str]],
) -> None:
    """Save manual review template.

    Args:
        rows: Audit rows ready for human labeling.
    """
    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "audit_id",
        "group_id",
        "target",
        "prompt_type",
        "prompt",
        "image_name",
        "label",
        "score",
        "bbox_area_ratio",
        "visualization_path",
        "localization_quality",
        "error_type",
        "notes",
    ]

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build the 40-case manual localization audit."""
    detail_rows = load_detail_rows()

    sampled_rows = sample_rows(
        detail_rows
    )

    predictions = {
        group_id: load_group_predictions(
            group_id
        )
        for group_id in [1, 2, 3]
    }

    audit_rows = []

    for audit_id, row in enumerate(
        sampled_rows,
        start=1,
    ):
        visualization_path = (
            save_visualization(
                row=row,
                predictions=predictions,
                audit_id=audit_id,
            )
        )

        audit_rows.append(
            {
                "audit_id": audit_id,
                "group_id": row["group_id"],
                "target": row["target"],
                "prompt_type": row[
                    "prompt_type"
                ],
                "prompt": row["prompt"],
                "image_name": row[
                    "image_name"
                ],
                "label": row["label"],
                "score": row["score"],
                "bbox_area_ratio": row[
                    "bbox_area_ratio"
                ],
                "visualization_path": (
                    visualization_path
                ),
                "localization_quality": "",
                "error_type": "",
                "notes": "",
            }
        )

    save_audit_csv(audit_rows)

    print()
    print(
        f"Audit cases: {len(audit_rows)}"
    )
    print(f"Visualizations: {OUTPUT_DIR}")
    print(f"Review sheet: {REPORT_FILE}")


if __name__ == "__main__":
    main()