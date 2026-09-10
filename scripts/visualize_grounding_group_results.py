"""Visualize Grounding DINO group evaluation predictions."""

import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "fashion_data"

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "grounding_group_evaluation"
    / "group_1_smoke_10.json"
)

IMAGE_DIR = (
    DATASET_ROOT
    / "raw"
    / "train"
    / "train"
    / "image"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "grounding_group_evaluation"
    / "sanity_10_visualization"
)


def load_results(input_file: Path) -> Dict[str, Any]:
    """Load Grounding DINO prediction results.

    Args:
        input_file: Path to the prediction JSON file.

    Returns:
        Parsed prediction dictionary.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    if not input_file.exists():
        raise FileNotFoundError(
            f"Prediction file does not exist: {input_file}"
        )

    with input_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def draw_detections(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    target: str,
    prompt_type: str,
    prompt: str,
) -> Image.Image:
    """Draw detection boxes and metadata on an image.

    Args:
        image: Source RGB image.
        detections: Grounding DINO detections.
        target: Target clothing component.
        prompt_type: Baseline or optimized prompt type.
        prompt: Text prompt used for grounding.

    Returns:
        Annotated image.
    """
    visualized = image.copy()
    draw = ImageDraw.Draw(visualized)

    header = (
        f"Target: {target} | "
        f"Type: {prompt_type} | "
        f"Prompt: {prompt}"
    )

    draw.rectangle(
        (0, 0, visualized.width, 28),
        fill="white",
    )
    draw.text(
        (8, 8),
        header,
        fill="black",
    )

    if not detections:
        draw.rectangle(
            (0, 28, visualized.width, 54),
            fill="white",
        )
        draw.text(
            (8, 34),
            "NO DETECTION",
            fill="black",
        )

        return visualized

    for detection in detections:
        box = detection["box"]
        label = detection["label"]
        score = detection["score"]

        x_min, y_min, x_max, y_max = box

        draw.rectangle(
            (
                x_min,
                y_min,
                x_max,
                y_max,
            ),
            outline="red",
            width=3,
        )

        label_text = (
            f"{label} | {score:.3f}"
        )

        text_y = max(
            30,
            int(y_min) - 18,
        )

        draw.rectangle(
            (
                int(x_min),
                text_y,
                int(x_min) + 180,
                text_y + 18,
            ),
            fill="white",
        )

        draw.text(
            (
                int(x_min) + 3,
                text_y + 3,
            ),
            label_text,
            fill="black",
        )

    return visualized


def save_visualizations(
    results: Dict[str, Any],
) -> None:
    """Generate visualization files for all predictions.

    Args:
        results: Complete Grounding DINO experiment results.
    """
    predictions = results["predictions"]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for target, target_results in predictions.items():
        for prompt_type, prompt_data in target_results.items():
            prompt = prompt_data["prompt"]

            for image_name, detections in (
                prompt_data["results"].items()
            ):
                image_path = IMAGE_DIR / image_name

                if not image_path.exists():
                    print(
                        f"Missing image: {image_path}"
                    )
                    continue

                with Image.open(image_path) as image:
                    rgb_image = image.convert("RGB")

                    visualized = draw_detections(
                        image=rgb_image,
                        detections=detections,
                        target=target,
                        prompt_type=prompt_type,
                        prompt=prompt,
                    )

                output_file = (
                    OUTPUT_DIR
                    / (
                        f"{Path(image_name).stem}_"
                        f"{target}_"
                        f"{prompt_type}.jpg"
                    )
                )

                visualized.save(
                    output_file,
                    quality=95,
                )

                print(
                    f"Saved: {output_file.name}"
                )


def main() -> None:
    """Visualize Grounding DINO smoke-test results."""
    results = load_results(INPUT_FILE)

    save_visualizations(results)

    print()
    print(
        f"Visualization directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()