"""Run Grounding DINO evaluation on a sampled image group."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)


MODEL_ID = "IDEA-Research/grounding-dino-tiny"
THRESHOLD = 0.3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "fashion_data"

MANIFEST_DIR = (
    PROJECT_ROOT
    / "reports"
    / "evaluation_sampling"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "grounding_group_evaluation"
)

PROMPT_CONFIG = {
    "sleeve": {
        "baseline": "sleeve",
        "context": "shirt sleeve",
    },
    "collar": {
        "baseline": "collar",
        "context": "shirt collar",
    },
    "button": {
        "baseline": "button",
        "context": "clothing button",
    },
    "zipper": {
        "baseline": "zipper",
        "context": "clothing zipper",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run Grounding DINO evaluation on one "
            "sampled evaluation group."
        )
    )

    parser.add_argument(
        "--group",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Evaluation group identifier.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of images to process. "
            "Useful for smoke tests."
        ),
    )

    return parser.parse_args()


def get_device() -> torch.device:
    """Select the available PyTorch device.

    Returns:
        CUDA device when available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(
    device: torch.device,
) -> tuple[Any, Any]:
    """Load the Grounding DINO processor and model.

    Args:
        device: PyTorch inference device.

    Returns:
        Grounding DINO processor and model.
    """
    print(f"Loading model: {MODEL_ID}")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
    )

    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(
            MODEL_ID,
        )
    )

    model.to(device)
    model.eval()

    return processor, model


def read_manifest(
    manifest_file: Path,
    limit: Optional[int],
) -> List[Dict[str, str]]:
    """Read sampled image records from a CSV manifest.

    Args:
        manifest_file: Evaluation group manifest.
        limit: Optional maximum number of rows.

    Returns:
        Image records to evaluate.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest does not exist: {manifest_file}"
        )

    with manifest_file.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    if limit is not None:
        rows = rows[:limit]

    return rows


def run_detection(
    image: Image.Image,
    prompt: str,
    processor: Any,
    model: Any,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Run one Grounding DINO prediction.

    Args:
        image: Input RGB image.
        prompt: Natural-language grounding prompt.
        processor: Grounding DINO processor.
        model: Grounding DINO model.
        device: PyTorch inference device.

    Returns:
        Detected labels, confidence scores, and boxes.
    """
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    results = (
        processor
        .post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=THRESHOLD,
            target_sizes=[image.size[::-1]],
        )
    )

    result = results[0]

    detections = []

    for score, label, box in zip(
        result["scores"],
        result["labels"],
        result["boxes"],
    ):
        detections.append(
            {
                "label": str(label),
                "score": float(score),
                "box": [
                    float(value)
                    for value in box.tolist()
                ],
            }
        )

    return detections


def evaluate_prompt(
    rows: List[Dict[str, str]],
    prompt: str,
    processor: Any,
    model: Any,
    device: torch.device,
) -> Dict[str, List[Dict[str, Any]]]:
    """Evaluate one prompt on all selected images.

    Args:
        rows: Evaluation image records.
        prompt: Grounding prompt.
        processor: Grounding DINO processor.
        model: Grounding DINO model.
        device: PyTorch inference device.

    Returns:
        Mapping from image name to detections.
    """
    prompt_results = {}

    progress = tqdm(
        rows,
        desc=f"Prompt: {prompt}",
    )

    for row in progress:
        image_path = (
            DATASET_ROOT
            / Path(row["relative_path"])
        )

        if not image_path.exists():
            print(
                f"\nMissing image: {image_path}"
            )
            prompt_results[
                row["image_name"]
            ] = []
            continue

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

            detections = run_detection(
                image=rgb_image,
                prompt=prompt,
                processor=processor,
                model=model,
                device=device,
            )

        prompt_results[
            row["image_name"]
        ] = detections

    return prompt_results


def save_results(
    results: Dict[str, Any],
    output_file: Path,
) -> None:
    """Save evaluation results as JSON.

    Args:
        results: Evaluation results.
        output_file: Destination JSON path.
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )


def run_experiment(
    rows: List[Dict[str, str]],
    group_id: int,
    processor: Any,
    model: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """Run baseline and optimized prompt evaluation.

    Args:
        rows: Evaluation image records.
        group_id: Selected evaluation group.
        processor: Grounding DINO processor.
        model: Grounding DINO model.
        device: PyTorch inference device.

    Returns:
        Complete experiment result structure.
    """
    results = {
        "metadata": {
            "model_id": MODEL_ID,
            "threshold": THRESHOLD,
            "group_id": group_id,
            "num_images": len(rows),
        },
        "predictions": {},
    }

    for target, prompts in PROMPT_CONFIG.items():
        print()
        print(f"Target: {target}")

        results["predictions"][target] = {}

        for prompt_type, prompt in prompts.items():
            print(
                f"{prompt_type}: {prompt}"
            )

            detections = evaluate_prompt(
                rows=rows,
                prompt=prompt,
                processor=processor,
                model=model,
                device=device,
            )

            results["predictions"][target][
                prompt_type
            ] = {
                "prompt": prompt,
                "results": detections,
            }

    return results


def main() -> None:
    """Run the sampled Grounding DINO evaluation."""
    args = parse_args()

    manifest_file = (
        MANIFEST_DIR
        / f"eval_group_{args.group}_100.csv"
    )

    output_file = (
        OUTPUT_DIR
        / f"group_{args.group}_predictions.json"
    )

    if args.limit is not None:
        output_file = (
            OUTPUT_DIR
            / (
                f"group_{args.group}"
                f"_smoke_{args.limit}.json"
            )
        )

    rows = read_manifest(
        manifest_file=manifest_file,
        limit=args.limit,
    )

    device = get_device()

    print(f"Using device: {device}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Images: {len(rows)}")
    print(f"Manifest: {manifest_file}")

    processor, model = load_model(
        device=device,
    )

    results = run_experiment(
        rows=rows,
        group_id=args.group,
        processor=processor,
        model=model,
        device=device,
    )

    save_results(
        results=results,
        output_file=output_file,
    )

    print()
    print("Evaluation completed.")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()