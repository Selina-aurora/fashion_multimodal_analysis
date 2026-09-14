"""Run annotation-assisted ROI-conditioned Grounding DINO evaluation.

This script uses DeepFashion2 garment annotations as an oracle ROI source
for diagnostic experiments. It compares language-guided localization after
restricting the visual search space to an annotated garment region.

The annotation-assisted ROI is used only for controlled diagnosis and does
not represent the final deployment pipeline.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)


MODEL_ID = "IDEA-Research/grounding-dino-tiny"

DEFAULT_THRESHOLD = 0.3
ROI_MARGIN_RATIO = 0.05

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT.parent
    / "fashion_data"
    / "raw"
    / "train"
    / "train"
)

IMAGE_DIR = DATASET_ROOT / "image"
ANNOTATION_DIR = DATASET_ROOT / "annos"

DEFAULT_AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "grounding_group_evaluation"
    / "manual_localization_audit.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "roi_conditioned_grounding"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "roi_conditioned_grounding"
)


PREFERRED_CATEGORIES = {
    "sleeve": {
        "short sleeve top",
        "long sleeve top",
        "short sleeve outwear",
        "long sleeve outwear",
        "short sleeve dress",
        "long sleeve dress",
    },
    "collar": {
        "short sleeve top",
        "long sleeve top",
        "vest",
        "short sleeve outwear",
        "long sleeve outwear",
        "short sleeve dress",
        "long sleeve dress",
        "vest dress",
        "sling dress",
        "sling",
    },
    "button": {
        "short sleeve top",
        "long sleeve top",
        "vest",
        "short sleeve outwear",
        "long sleeve outwear",
        "short sleeve dress",
        "long sleeve dress",
        "vest dress",
        "shorts",
        "trousers",
        "skirt",
    },
    "zipper": {
        "short sleeve top",
        "long sleeve top",
        "short sleeve outwear",
        "long sleeve outwear",
        "short sleeve dress",
        "long sleeve dress",
        "vest dress",
        "shorts",
        "trousers",
        "skirt",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run annotation-assisted garment ROI Grounding DINO "
            "on fixed manual audit cases."
        )
    )

    parser.add_argument(
        "--audit-file",
        type=Path,
        default=DEFAULT_AUDIT_FILE,
        help="Manual localization audit CSV.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of audit cases. "
            "Useful for smoke tests."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Grounding DINO detection threshold.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0.0 and 1.0."
        )

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "Limit must be greater than zero."
        )

    return args


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
    """Load Grounding DINO processor and model.

    Args:
        device: PyTorch inference device.

    Returns:
        Grounding DINO processor and model.
    """
    print(f"Loading model: {MODEL_ID}")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(
            MODEL_ID
        )
    )

    model.to(device)
    model.eval()

    return processor, model


def read_audit_cases(
    audit_file: Path,
    limit: int | None,
) -> list[dict[str, str]]:
    """Read fixed manual localization audit cases.

    Args:
        audit_file: Audit CSV file.
        limit: Optional maximum number of rows.

    Returns:
        Audit case records.

    Raises:
        FileNotFoundError: If the audit file does not exist.
    """
    if not audit_file.is_file():
        raise FileNotFoundError(
            f"Audit file not found: {audit_file}"
        )

    with audit_file.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        rows = list(
            csv.DictReader(csv_file)
        )

    if limit is not None:
        rows = rows[:limit]

    return rows


def load_annotation(
    image_name: str,
) -> dict[str, Any]:
    """Load a DeepFashion2 annotation file.

    Args:
        image_name: Input image filename.

    Returns:
        Parsed annotation dictionary.

    Raises:
        FileNotFoundError: If the annotation does not exist.
    """
    annotation_path = (
        ANNOTATION_DIR
        / f"{Path(image_name).stem}.json"
    )

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Annotation not found: {annotation_path}"
        )

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(json_file)


def get_garment_instances(
    annotation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract valid garment instances from annotation.

    Args:
        annotation: DeepFashion2 annotation.

    Returns:
        Garment instance records containing item id, category, and bbox.
    """
    garments = []

    for key, value in annotation.items():
        if not key.startswith("item"):
            continue

        if not isinstance(value, dict):
            continue

        bbox = value.get(
            "bounding_box"
        )

        category_name = value.get(
            "category_name"
        )

        if bbox is None:
            continue

        if category_name is None:
            continue

        if len(bbox) != 4:
            continue

        garments.append(
            {
                "item_id": key,
                "category_name": str(
                    category_name
                ),
                "bbox": [
                    int(
                        round(
                            float(coord)
                        )
                    )
                    for coord in bbox
                ],
            }
        )

    return garments


def bbox_area(
    bbox: list[int],
) -> int:
    """Calculate bounding-box area.

    Args:
        bbox: Bounding box in x_min, y_min, x_max, y_max format.

    Returns:
        Bounding-box area.
    """
    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = bbox

    width = max(
        0,
        x_max - x_min + 1,
    )

    height = max(
        0,
        y_max - y_min + 1,
    )

    return width * height


def select_garment_roi(
    garments: list[dict[str, Any]],
    target: str,
) -> dict[str, Any]:
    """Select the garment most relevant to the requested target.

    Args:
        garments: Available garment annotation instances.
        target: Fine-grained target name.

    Returns:
        Selected garment instance.

    Raises:
        ValueError: If no garment instances are available.
    """
    if not garments:
        raise ValueError(
            "No garment instance with a valid bbox."
        )

    preferred_categories = (
        PREFERRED_CATEGORIES.get(
            target,
            set(),
        )
    )

    preferred_garments = [
        garment
        for garment in garments
        if garment["category_name"]
        in preferred_categories
    ]

    candidates = (
        preferred_garments
        if preferred_garments
        else garments
    )

    return max(
        candidates,
        key=lambda garment: bbox_area(
            garment["bbox"]
        ),
    )


def expand_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> list[int]:
    """Expand a bbox while keeping it within image boundaries.

    Args:
        bbox: Original garment bounding box.
        image_width: Full image width.
        image_height: Full image height.
        margin_ratio: Relative margin added around the bbox.

    Returns:
        Expanded bounding box.
    """
    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = bbox

    width = (
        x_max - x_min + 1
    )

    height = (
        y_max - y_min + 1
    )

    x_margin = int(
        width * margin_ratio
    )

    y_margin = int(
        height * margin_ratio
    )

    return [
        max(
            0,
            x_min - x_margin,
        ),
        max(
            0,
            y_min - y_margin,
        ),
        min(
            image_width - 1,
            x_max + x_margin,
        ),
        min(
            image_height - 1,
            y_max + y_margin,
        ),
    ]


def run_detection(
    image: Image.Image,
    prompt: str,
    processor: Any,
    model: Any,
    device: torch.device,
    threshold: float,
) -> tuple[list[dict[str, Any]], float]:
    """Run Grounding DINO prediction.

    Args:
        image: ROI image.
        prompt: Language grounding prompt.
        processor: Grounding DINO processor.
        model: Grounding DINO model.
        device: PyTorch inference device.
        threshold: Detection confidence threshold.

    Returns:
        Detections and model inference latency in milliseconds.
    """
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model(
            **inputs
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_ms = (
        time.perf_counter()
        - start_time
    ) * 1000.0

    results = (
        processor
        .post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=threshold,
            target_sizes=[
                image.size[::-1]
            ],
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
                    for value
                    in box.tolist()
                ],
            }
        )

    return (
        detections,
        inference_ms,
    )


def map_box_to_full_image(
    box: list[float],
    roi_bbox: list[int],
) -> list[float]:
    """Map an ROI-local bbox to full-image coordinates.

    Args:
        box: Prediction bbox relative to ROI.
        roi_bbox: ROI bbox relative to full image.

    Returns:
        Prediction bbox in full-image coordinates.
    """
    (
        roi_x_min,
        roi_y_min,
        _,
        _,
    ) = roi_bbox

    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = box

    return [
        x_min + roi_x_min,
        y_min + roi_y_min,
        x_max + roi_x_min,
        y_max + roi_y_min,
    ]


def calculate_area_ratio(
    box: list[float],
    image_width: int,
    image_height: int,
) -> float:
    """Calculate bbox area relative to full image.

    Args:
        box: Bounding box.
        image_width: Full image width.
        image_height: Full image height.

    Returns:
        Bounding-box area ratio.
    """
    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = box

    width = max(
        0.0,
        x_max - x_min,
    )

    height = max(
        0.0,
        y_max - y_min,
    )

    image_area = float(
        image_width
        * image_height
    )

    return (
        width
        * height
        / image_area
    )


def save_visualization(
    image: Image.Image,
    roi_bbox: list[int],
    top_detection: dict[str, Any] | None,
    output_path: Path,
) -> None:
    """Save ROI and prediction visualization.

    Yellow box:
        DeepFashion2 garment ROI.

    Red box:
        Highest-confidence Grounding DINO prediction.

    Args:
        image: Full input image.
        roi_bbox: Selected garment ROI.
        top_detection: Highest-confidence prediction.
        output_path: Destination image path.
    """
    visualization = image.copy()

    draw = ImageDraw.Draw(
        visualization
    )

    draw.rectangle(
        roi_bbox,
        outline="yellow",
        width=4,
    )

    if top_detection is not None:
        prediction_box = [
            int(
                round(value)
            )
            for value
            in top_detection["full_box"]
        ]

        draw.rectangle(
            prediction_box,
            outline="red",
            width=4,
        )

        label_text = (
            f'{top_detection["label"]} '
            f'{top_detection["score"]:.3f}'
        )

        text_x = prediction_box[0]

        text_y = max(
            0,
            prediction_box[1] - 16,
        )

        draw.text(
            (
                text_x,
                text_y,
            ),
            label_text,
            fill="red",
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    visualization.save(
        output_path
    )


def process_case(
    row: dict[str, str],
    processor: Any,
    model: Any,
    device: torch.device,
    threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Run ROI-conditioned grounding for one audit case.

    Args:
        row: Manual audit record.
        processor: Grounding DINO processor.
        model: Grounding DINO model.
        device: PyTorch device.
        threshold: Detection confidence threshold.
        output_dir: Experiment output directory.

    Returns:
        Result dictionary for the audit case.
    """
    image_name = row[
        "image_name"
    ]

    target = row[
        "target"
    ]

    prompt = row[
        "prompt"
    ]

    image_path = (
        IMAGE_DIR
        / image_name
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(
        image_path
    ) as image_file:
        image = (
            image_file
            .convert("RGB")
        )

    annotation = load_annotation(
        image_name
    )

    garments = (
        get_garment_instances(
            annotation
        )
    )

    selected_garment = (
        select_garment_roi(
            garments,
            target,
        )
    )

    roi_bbox = expand_bbox(
        bbox=selected_garment[
            "bbox"
        ],
        image_width=image.width,
        image_height=image.height,
        margin_ratio=ROI_MARGIN_RATIO,
    )

    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = roi_bbox

    crop = image.crop(
        (
            x_min,
            y_min,
            x_max + 1,
            y_max + 1,
        )
    )

    detections, inference_ms = (
        run_detection(
            image=crop,
            prompt=prompt,
            processor=processor,
            model=model,
            device=device,
            threshold=threshold,
        )
    )

    mapped_detections = []

    for detection in detections:
        full_box = (
            map_box_to_full_image(
                detection["box"],
                roi_bbox,
            )
        )

        mapped_detection = {
            **detection,
            "full_box": full_box,
            "full_image_area_ratio": (
                calculate_area_ratio(
                    box=full_box,
                    image_width=image.width,
                    image_height=image.height,
                )
            ),
        }

        mapped_detections.append(
            mapped_detection
        )

    top_detection = None

    if mapped_detections:
        top_detection = max(
            mapped_detections,
            key=lambda item: item[
                "score"
            ],
        )

    crop_path = (
        output_dir
        / "crops"
        / (
            f'{row["audit_id"]}_'
            f'{Path(image_name).stem}.jpg'
        )
    )

    crop_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop.save(
        crop_path
    )

    visualization_path = (
        output_dir
        / "visualizations"
        / (
            f'{row["audit_id"]}_'
            f'{Path(image_name).stem}_'
            f'{target}_'
            f'{row["prompt_type"]}.jpg'
        )
    )

    save_visualization(
        image=image,
        roi_bbox=roi_bbox,
        top_detection=top_detection,
        output_path=visualization_path,
    )

    full_image_area = float(
        image.width
        * image.height
    )

    roi_area_ratio = (
        bbox_area(
            roi_bbox
        )
        / full_image_area
    )

    return {
        "audit_id": row[
            "audit_id"
        ],
        "group_id": row[
            "group_id"
        ],
        "target": target,
        "prompt_type": row[
            "prompt_type"
        ],
        "prompt": prompt,
        "image_name": image_name,
        "threshold": threshold,
        "roi_source": (
            "deepfashion2_annotation"
        ),
        "roi_item_id": (
            selected_garment[
                "item_id"
            ]
        ),
        "roi_category": (
            selected_garment[
                "category_name"
            ]
        ),
        "roi_bbox": roi_bbox,
        "roi_area_ratio": (
            roi_area_ratio
        ),
        "detection_count": len(
            mapped_detections
        ),
        "top_detection": (
            top_detection
        ),
        "detections": (
            mapped_detections
        ),
        "inference_ms": (
            inference_ms
        ),
        "baseline_quality": row.get(
            "localization_quality",
            "",
        ),
        "baseline_error_type": row.get(
            "error_type",
            "",
        ),
        "baseline_notes": row.get(
            "notes",
            "",
        ),
        "crop_path": str(
            crop_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "visualization_path": str(
            visualization_path.relative_to(
                PROJECT_ROOT
            )
        ),
    }


def save_json(
    rows: list[dict[str, Any]],
    threshold: float,
    output_dir: Path,
) -> None:
    """Save complete experiment results as JSON.

    Args:
        rows: Experiment result rows.
        threshold: Detection threshold.
        output_dir: Experiment output directory.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "roi_conditioned_predictions.json"
    )

    payload = {
        "metadata": {
            "model_id": MODEL_ID,
            "threshold": threshold,
            "roi_source": (
                "deepfashion2_annotation"
            ),
            "roi_margin_ratio": (
                ROI_MARGIN_RATIO
            ),
            "num_cases": len(rows),
        },
        "results": rows,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            payload,
            json_file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"Saved JSON: {output_path}"
    )


def save_csv(
    rows: list[dict[str, Any]],
    report_dir: Path,
) -> None:
    """Save compact experiment results for manual audit.

    Args:
        rows: Experiment result rows.
        report_dir: Report output directory.
    """
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        report_dir
        / "roi_conditioned_audit.csv"
    )

    fieldnames = [
        "audit_id",
        "group_id",
        "target",
        "prompt_type",
        "prompt",
        "image_name",
        "threshold",
        "roi_category",
        "roi_area_ratio",
        "detection_count",
        "label",
        "score",
        "bbox_area_ratio",
        "inference_ms",
        "baseline_quality",
        "baseline_error_type",
        "baseline_notes",
        "visualization_path",
        "roi_quality",
        "roi_error_type",
        "roi_notes",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            top_detection = row[
                "top_detection"
            ]

            label = ""
            score = ""
            bbox_area_ratio = ""

            if top_detection is not None:
                label = top_detection[
                    "label"
                ]

                score = round(
                    top_detection[
                        "score"
                    ],
                    6,
                )

                bbox_area_ratio = round(
                    top_detection[
                        "full_image_area_ratio"
                    ],
                    6,
                )

            writer.writerow(
                {
                    "audit_id": row[
                        "audit_id"
                    ],
                    "group_id": row[
                        "group_id"
                    ],
                    "target": row[
                        "target"
                    ],
                    "prompt_type": row[
                        "prompt_type"
                    ],
                    "prompt": row[
                        "prompt"
                    ],
                    "image_name": row[
                        "image_name"
                    ],
                    "threshold": row[
                        "threshold"
                    ],
                    "roi_category": row[
                        "roi_category"
                    ],
                    "roi_area_ratio": round(
                        row[
                            "roi_area_ratio"
                        ],
                        6,
                    ),
                    "detection_count": row[
                        "detection_count"
                    ],
                    "label": label,
                    "score": score,
                    "bbox_area_ratio": (
                        bbox_area_ratio
                    ),
                    "inference_ms": round(
                        row[
                            "inference_ms"
                        ],
                        2,
                    ),
                    "baseline_quality": row[
                        "baseline_quality"
                    ],
                    "baseline_error_type": row[
                        "baseline_error_type"
                    ],
                    "baseline_notes": row[
                        "baseline_notes"
                    ],
                    "visualization_path": row[
                        "visualization_path"
                    ],
                    "roi_quality": "",
                    "roi_error_type": "",
                    "roi_notes": "",
                }
            )

    print(
        f"Saved CSV: {output_path}"
    )


def main() -> None:
    """Run ROI-conditioned Grounding DINO evaluation."""
    args = parse_args()

    audit_rows = (
        read_audit_cases(
            audit_file=args.audit_file,
            limit=args.limit,
        )
    )

    experiment_name = (
        f"threshold_"
        f"{args.threshold:g}"
    )

    output_dir = (
        OUTPUT_ROOT
        / experiment_name
    )

    report_dir = (
        REPORT_ROOT
        / experiment_name
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    print(
        f"Threshold: "
        f"{args.threshold}"
    )

    print(
        "ROI source: "
        "DeepFashion2 annotation"
    )

    print(
        f"Cases: "
        f"{len(audit_rows)}"
    )

    print(
        f"Experiment: "
        f"{experiment_name}"
    )

    processor, model = (
        load_model(
            device=device
        )
    )

    experiment_rows = []

    for row in tqdm(
        audit_rows,
        desc=(
            "ROI-conditioned "
            "grounding"
        ),
    ):
        result = process_case(
            row=row,
            processor=processor,
            model=model,
            device=device,
            threshold=args.threshold,
            output_dir=output_dir,
        )

        experiment_rows.append(
            result
        )

    save_json(
        rows=experiment_rows,
        threshold=args.threshold,
        output_dir=output_dir,
    )

    save_csv(
        rows=experiment_rows,
        report_dir=report_dir,
    )

    print()
    print(
        "ROI-conditioned "
        "evaluation completed."
    )


if __name__ == "__main__":
    main()