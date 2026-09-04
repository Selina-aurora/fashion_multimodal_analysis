"""Evaluate adaptive person ROI extraction on DeepFashion2 samples."""

import csv
import logging
import random
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "facebook/mask2former-swin-tiny-coco-instance"

SAMPLE_COUNT = 10
RANDOM_SEED = 42

SMALL_BBOX_THRESHOLD = 0.30
LARGE_BBOX_THRESHOLD = 0.60

SMALL_BBOX_MARGIN = 0.20
MEDIUM_BBOX_MARGIN = 0.10
LARGE_BBOX_MARGIN = 0.05

MASK_ALPHA = 0.5
BOX_THICKNESS = 2
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
TEXT_VERTICAL_OFFSET = 10
TEXT_MIN_Y = 20

PERSON_LABEL = "person"

ORIGINAL_BOX_COLOR = (0, 255, 0)
EXPANDED_BOX_COLOR = (0, 0, 255)

SUMMARY_FIELDS = [
    "image_name",
    "status",
    "person_count",
    "score",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "bbox_area_ratio",
    "margin_ratio",
    "expanded_x_min",
    "expanded_y_min",
    "expanded_x_max",
    "expanded_y_max",
    "expanded_bbox_area_ratio",
    "mask_area_ratio",
]

BBox = tuple[int, int, int, int]
SummaryValue = str | int | float
SummaryRow = dict[str, SummaryValue]


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def get_label_name(
    id2label: Mapping[Any, str],
    label_id: int,
) -> str:
    """Resolve a class ID to a model label.

    Args:
        id2label: Model class ID to label mapping.
        label_id: Predicted class ID.

    Returns:
        Human-readable class label.
    """
    label_name = id2label.get(label_id)

    if label_name is None:
        label_name = id2label.get(str(label_id))

    if label_name is None:
        return f"class_{label_id}"

    return label_name


def run_inference(
    image: Image.Image,
    processor: Mask2FormerImageProcessor,
    model: Mask2FormerForUniversalSegmentation,
) -> dict[str, Any]:
    """Run Mask2Former instance segmentation on one image.

    Args:
        image: RGB input image.
        processor: Mask2Former image processor.
        model: Pretrained Mask2Former model.

    Returns:
        Post-processed instance segmentation result.
    """
    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    return processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[
            (
                image.height,
                image.width,
            )
        ],
    )[0]


def find_person_segments(
    segments_info: list[dict[str, Any]],
    id2label: Mapping[Any, str],
) -> list[dict[str, Any]]:
    """Return predicted segments classified as person.

    Args:
        segments_info: Predicted instance metadata.
        id2label: Model class ID to label mapping.

    Returns:
        Predicted person segments.
    """
    person_segments = []

    for segment in segments_info:
        label_id = int(segment["label_id"])
        label_name = get_label_name(id2label, label_id)

        if label_name == PERSON_LABEL:
            person_segments.append(segment)

    return person_segments


def select_primary_person(
    person_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the current primary-person candidate.

    The current experiment keeps the same selection rule as the
    fixed-margin baseline and chooses the highest-confidence person.

    Args:
        person_segments: Predicted person instances.

    Returns:
        Selected person segment.

    Raises:
        ValueError: If no person segments are provided.
    """
    if not person_segments:
        raise ValueError("Cannot select a person from an empty list")

    return max(
        person_segments,
        key=lambda item: float(item["score"]),
    )


def mask_to_bbox(mask: np.ndarray) -> BBox | None:
    """Calculate a bounding box from a binary mask.

    Args:
        mask: Two-dimensional binary mask.

    Returns:
        Bounding box as ``(x_min, y_min, x_max, y_max)``.
        Returns ``None`` when the mask is empty.
    """
    y_indices, x_indices = np.where(mask)

    if x_indices.size == 0:
        return None

    return (
        int(x_indices.min()),
        int(y_indices.min()),
        int(x_indices.max()),
        int(y_indices.max()),
    )


def calculate_bbox_area_ratio(
    bbox: BBox,
    image_width: int,
    image_height: int,
) -> float:
    """Calculate bounding-box area as a fraction of image area.

    Args:
        bbox: Bounding box.
        image_width: Input image width.
        image_height: Input image height.

    Returns:
        Bounding-box area divided by image area.
    """
    x_min, y_min, x_max, y_max = bbox
    bbox_area = (x_max - x_min + 1) * (y_max - y_min + 1)
    image_area = image_width * image_height

    return bbox_area / image_area


def select_margin_ratio(
    bbox_area_ratio: float,
) -> float:
    """Select an adaptive ROI margin from person scale.

    Args:
        bbox_area_ratio: Person bounding-box area divided by image area.

    Returns:
        Margin ratio used on each side of the person bounding box.
    """
    if bbox_area_ratio < SMALL_BBOX_THRESHOLD:
        return SMALL_BBOX_MARGIN

    if bbox_area_ratio < LARGE_BBOX_THRESHOLD:
        return MEDIUM_BBOX_MARGIN

    return LARGE_BBOX_MARGIN


def expand_bbox(
    bbox: BBox,
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> BBox:
    """Expand a bounding box while keeping it inside image boundaries.

    Args:
        bbox: Original person bounding box.
        image_width: Input image width.
        image_height: Input image height.
        margin_ratio: Fraction of bbox width and height added per side.

    Returns:
        Expanded and clipped bounding box.

    Raises:
        ValueError: If the margin ratio is negative.
    """
    if margin_ratio < 0:
        raise ValueError("Margin ratio must be non-negative")

    x_min, y_min, x_max, y_max = bbox

    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    x_margin = int(bbox_width * margin_ratio)
    y_margin = int(bbox_height * margin_ratio)

    return (
        max(0, x_min - x_margin),
        max(0, y_min - y_margin),
        min(image_width - 1, x_max + x_margin),
        min(image_height - 1, y_max + y_margin),
    )


def build_empty_summary_row(
    image_name: str,
    status: str,
    person_count: int,
) -> SummaryRow:
    """Build a summary row for an unsuccessful ROI case.

    Args:
        image_name: Input image filename.
        status: Processing status.
        person_count: Number of detected person instances.

    Returns:
        Summary row with empty ROI measurements.
    """
    return {
        "image_name": image_name,
        "status": status,
        "person_count": person_count,
        "score": "",
        "x_min": "",
        "y_min": "",
        "x_max": "",
        "y_max": "",
        "bbox_area_ratio": "",
        "margin_ratio": "",
        "expanded_x_min": "",
        "expanded_y_min": "",
        "expanded_x_max": "",
        "expanded_y_max": "",
        "expanded_bbox_area_ratio": "",
        "mask_area_ratio": "",
    }


def save_image(
    output_path: Path,
    image: np.ndarray,
) -> None:
    """Save an OpenCV image and validate the write operation.

    Args:
        output_path: Destination image path.
        image: Image to save.

    Raises:
        OSError: If OpenCV fails to write the image.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Failed to save image: {output_path}")


def save_summary(
    summary_path: Path,
    summary_rows: list[SummaryRow],
) -> None:
    """Save ROI evaluation rows to CSV.

    Args:
        summary_path: Destination CSV path.
        summary_rows: Per-image evaluation rows.
    """
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=SUMMARY_FIELDS,
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def evaluate_image(
    image_path: Path,
    output_dir: Path,
    processor: Mask2FormerImageProcessor,
    model: Mask2FormerForUniversalSegmentation,
) -> SummaryRow:
    """Evaluate adaptive person ROI extraction for one image.

    Args:
        image_path: Input image path.
        output_dir: Directory used for visual outputs.
        processor: Mask2Former image processor.
        model: Pretrained Mask2Former model.

    Returns:
        Summary row describing the selected person and adaptive ROI.
    """
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    result = run_inference(
        image,
        processor,
        model,
    )

    segmentation = result["segmentation"].cpu().numpy()
    segments_info = result["segments_info"]

    person_segments = find_person_segments(
        segments_info,
        model.config.id2label,
    )

    if not person_segments:
        LOGGER.warning(
            "No person detected: %s",
            image_path.name,
        )
        return build_empty_summary_row(
            image_path.name,
            "no_person",
            0,
        )

    person_segment = select_primary_person(person_segments)
    person_id = int(person_segment["id"])
    person_score = float(person_segment["score"])

    person_mask = segmentation == person_id
    bbox = mask_to_bbox(person_mask)

    if bbox is None:
        LOGGER.warning(
            "Selected person has an empty mask: %s",
            image_path.name,
        )
        return build_empty_summary_row(
            image_path.name,
            "empty_mask",
            len(person_segments),
        )

    image_width = image.width
    image_height = image.height

    bbox_area_ratio = calculate_bbox_area_ratio(
        bbox,
        image_width,
        image_height,
    )

    margin_ratio = select_margin_ratio(bbox_area_ratio)

    expanded_bbox = expand_bbox(
        bbox,
        image_width,
        image_height,
        margin_ratio,
    )

    expanded_bbox_area_ratio = calculate_bbox_area_ratio(
        expanded_bbox,
        image_width,
        image_height,
    )

    mask_area_ratio = float(person_mask.sum()) / float(image_width * image_height)

    original_rgb = np.asarray(image)
    original_bgr = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR,
    )

    overlay = original_bgr.copy()
    person_color = np.asarray(
        ORIGINAL_BOX_COLOR,
        dtype=np.float32,
    )

    overlay[person_mask] = (
        overlay[person_mask].astype(np.float32) * (1.0 - MASK_ALPHA)
        + person_color * MASK_ALPHA
    ).astype(np.uint8)

    x_min, y_min, x_max, y_max = bbox
    (
        expanded_x_min,
        expanded_y_min,
        expanded_x_max,
        expanded_y_max,
    ) = expanded_bbox

    cv2.rectangle(
        overlay,
        (x_min, y_min),
        (x_max, y_max),
        ORIGINAL_BOX_COLOR,
        BOX_THICKNESS,
    )

    cv2.rectangle(
        overlay,
        (expanded_x_min, expanded_y_min),
        (expanded_x_max, expanded_y_max),
        EXPANDED_BOX_COLOR,
        BOX_THICKNESS,
    )

    label_text = f"person: {person_score:.2f} " f"margin: {margin_ratio:.0%}"

    cv2.putText(
        overlay,
        label_text,
        (
            x_min,
            max(
                y_min - TEXT_VERTICAL_OFFSET,
                TEXT_MIN_Y,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        ORIGINAL_BOX_COLOR,
        TEXT_THICKNESS,
    )

    overlay_path = output_dir / f"{image_path.stem}_adaptive_overlay.jpg"
    save_image(
        overlay_path,
        overlay,
    )

    expanded_roi = original_bgr[
        expanded_y_min : expanded_y_max + 1,
        expanded_x_min : expanded_x_max + 1,
    ]

    roi_path = output_dir / f"{image_path.stem}_adaptive_roi.jpg"
    save_image(
        roi_path,
        expanded_roi,
    )

    binary_mask = person_mask.astype(np.uint8) * 255
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    save_image(
        mask_path,
        binary_mask,
    )

    LOGGER.info(
        "Person score=%.4f, bbox ratio=%.4f, " "margin=%.2f, expanded bbox ratio=%.4f",
        person_score,
        bbox_area_ratio,
        margin_ratio,
        expanded_bbox_area_ratio,
    )

    return {
        "image_name": image_path.name,
        "status": "success",
        "person_count": len(person_segments),
        "score": round(person_score, 4),
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "bbox_area_ratio": round(bbox_area_ratio, 4),
        "margin_ratio": round(margin_ratio, 4),
        "expanded_x_min": expanded_x_min,
        "expanded_y_min": expanded_y_min,
        "expanded_x_max": expanded_x_max,
        "expanded_y_max": expanded_y_max,
        "expanded_bbox_area_ratio": round(
            expanded_bbox_area_ratio,
            4,
        ),
        "mask_area_ratio": round(mask_area_ratio, 4),
    }


def select_sample_images(
    image_dir: Path,
    sample_count: int,
    random_seed: int,
) -> list[Path]:
    """Select a deterministic random sample of input images.

    Args:
        image_dir: Directory containing input JPEG images.
        sample_count: Number of images to sample.
        random_seed: Seed used for deterministic sampling.

    Returns:
        Selected image paths.

    Raises:
        FileNotFoundError: If the image directory does not exist.
        ValueError: If sample count is invalid or too few images exist.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if sample_count <= 0:
        raise ValueError("Sample count must be positive")

    image_paths = sorted(image_dir.glob("*.jpg"))

    if len(image_paths) < sample_count:
        raise ValueError(
            "Not enough images for ROI evaluation: "
            f"requested {sample_count}, found {len(image_paths)}"
        )

    random_generator = random.Random(random_seed)

    return random_generator.sample(
        image_paths,
        sample_count,
    )


def main() -> None:
    """Run adaptive person ROI evaluation."""
    configure_logging()

    project_root = Path(__file__).resolve().parents[1]

    image_dir = (
        project_root.parent / "fashion_data" / "raw" / "train" / "train" / "image"
    )

    output_dir = project_root / "outputs" / "person_roi_adaptive_margin_batch"
    summary_path = output_dir / "summary.csv"

    selected_images = select_sample_images(
        image_dir,
        SAMPLE_COUNT,
        RANDOM_SEED,
    )

    LOGGER.info("Loading image processor")

    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_NAME)

    LOGGER.info("Loading Mask2Former model")

    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME)
    model.eval()

    summary_rows: list[SummaryRow] = []

    for index, image_path in enumerate(
        selected_images,
        start=1,
    ):
        LOGGER.info(
            "[%d/%d] Processing %s",
            index,
            SAMPLE_COUNT,
            image_path.name,
        )

        summary_row = evaluate_image(
            image_path,
            output_dir,
            processor,
            model,
        )
        summary_rows.append(summary_row)

    save_summary(
        summary_path,
        summary_rows,
    )

    success_count = sum(row["status"] == "success" for row in summary_rows)

    LOGGER.info(
        "Adaptive ROI evaluation completed: %d/%d successful",
        success_count,
        len(summary_rows),
    )
    LOGGER.info(
        "Results saved to: %s",
        output_dir,
    )
    LOGGER.info(
        "Summary saved to: %s",
        summary_path,
    )


if __name__ == "__main__":
    main()
