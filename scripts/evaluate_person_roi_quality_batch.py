"""Evaluate person ROI quality using connected-component diagnostics."""

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
FULL_FRAME_THRESHOLD = 0.85

SMALL_BBOX_MARGIN = 0.20
MEDIUM_BBOX_MARGIN = 0.10
LARGE_BBOX_MARGIN = 0.05
FULL_FRAME_MARGIN = 0.00

ROBUST_BBOX_MIN_REDUCTION = 0.10
LARGEST_COMPONENT_MIN_SHARE = 0.85

MASK_ALPHA = 0.5
BOX_THICKNESS = 2
TEXT_SCALE = 0.55
TEXT_THICKNESS = 2
TEXT_VERTICAL_OFFSET = 10
TEXT_MIN_Y = 20

PERSON_LABEL = "person"

RAW_BOX_COLOR = (0, 255, 0)
COMPONENT_BOX_COLOR = (255, 0, 0)
FINAL_ROI_COLOR = (0, 0, 255)

SUMMARY_FIELDS = [
    "image_name",
    "status",
    "roi_strategy",
    "person_count",
    "score",
    "raw_bbox_area_ratio",
    "mask_area_ratio",
    "bbox_fill_ratio",
    "component_count",
    "largest_component_share",
    "largest_component_bbox_area_ratio",
    "bbox_reduction",
    "margin_ratio",
    "final_roi_area_ratio",
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
    """Return predicted person segments.

    Args:
        segments_info: Predicted instance metadata.
        id2label: Model class ID to label mapping.

    Returns:
        Predicted person segments.
    """
    person_segments = []

    for segment in segments_info:
        label_id = int(segment["label_id"])

        if get_label_name(id2label, label_id) == PERSON_LABEL:
            person_segments.append(segment)

    return person_segments


def select_primary_person(
    person_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the highest-confidence person.

    Args:
        person_segments: Predicted person instances.

    Returns:
        Highest-confidence person segment.

    Raises:
        ValueError: If no person segments are provided.
    """
    if not person_segments:
        raise ValueError("Person segment list must not be empty")

    return max(
        person_segments,
        key=lambda item: float(item["score"]),
    )


def mask_to_bbox(mask: np.ndarray) -> BBox | None:
    """Calculate a bounding box from a binary mask.

    Args:
        mask: Two-dimensional binary mask.

    Returns:
        Bounding box as ``(x_min, y_min, x_max, y_max)``, or ``None``.
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
    """Calculate bounding-box area relative to image area.

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


def get_largest_component(
    person_mask: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Extract the largest connected component from a person mask.

    Args:
        person_mask: Binary person mask.

    Returns:
        Largest-component mask, foreground component count, and the
        fraction of person-mask pixels belonging to the largest component.

    Raises:
        ValueError: If the person mask has no foreground component.
    """
    binary_mask = person_mask.astype(np.uint8)

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    component_count = label_count - 1

    if component_count <= 0:
        raise ValueError("Person mask has no foreground component")

    foreground_areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]
    largest_offset = int(np.argmax(foreground_areas))
    largest_label = largest_offset + 1
    largest_area = int(foreground_areas[largest_offset])

    total_area = int(binary_mask.sum())
    largest_component_share = largest_area / total_area

    largest_component_mask = labels == largest_label

    return (
        largest_component_mask,
        component_count,
        largest_component_share,
    )


def select_adaptive_margin(
    bbox_area_ratio: float,
) -> float:
    """Select an adaptive margin for a non-full-frame bbox.

    Args:
        bbox_area_ratio: Bounding-box area divided by image area.

    Returns:
        Margin ratio.
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
    """Expand a bounding box and clip it to image boundaries.

    Args:
        bbox: Original bounding box.
        image_width: Input image width.
        image_height: Input image height.
        margin_ratio: Fraction of bbox width and height added per side.

    Returns:
        Expanded bounding box.
    """
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


def choose_roi(
    raw_bbox: BBox,
    component_bbox: BBox,
    raw_bbox_area_ratio: float,
    component_bbox_area_ratio: float,
    largest_component_share: float,
    image_width: int,
    image_height: int,
) -> tuple[str, float, BBox]:
    """Choose the ROI using full-frame and component diagnostics.

    Args:
        raw_bbox: Bbox from the complete person mask.
        component_bbox: Bbox from the largest connected component.
        raw_bbox_area_ratio: Raw bbox area divided by image area.
        component_bbox_area_ratio: Component bbox area divided by image area.
        largest_component_share: Share of mask pixels in the largest component.
        image_width: Input image width.
        image_height: Input image height.

    Returns:
        ROI strategy name, margin ratio, and final ROI bbox.
    """
    bbox_reduction = raw_bbox_area_ratio - component_bbox_area_ratio

    can_use_robust_bbox = (
        raw_bbox_area_ratio > FULL_FRAME_THRESHOLD
        and bbox_reduction >= ROBUST_BBOX_MIN_REDUCTION
        and largest_component_share >= LARGEST_COMPONENT_MIN_SHARE
    )

    if can_use_robust_bbox:
        margin_ratio = select_adaptive_margin(component_bbox_area_ratio)
        final_bbox = expand_bbox(
            component_bbox,
            image_width,
            image_height,
            margin_ratio,
        )
        return (
            "largest_component_bbox",
            margin_ratio,
            final_bbox,
        )

    if raw_bbox_area_ratio > FULL_FRAME_THRESHOLD:
        return (
            "full_frame_no_expand",
            FULL_FRAME_MARGIN,
            raw_bbox,
        )

    margin_ratio = select_adaptive_margin(raw_bbox_area_ratio)
    final_bbox = expand_bbox(
        raw_bbox,
        image_width,
        image_height,
        margin_ratio,
    )

    return (
        "adaptive_margin",
        margin_ratio,
        final_bbox,
    )


def build_empty_summary_row(
    image_name: str,
    status: str,
    person_count: int,
) -> SummaryRow:
    """Build an empty summary row for a failed sample.

    Args:
        image_name: Input image filename.
        status: Processing status.
        person_count: Number of detected persons.

    Returns:
        Summary row.
    """
    row: SummaryRow = {field_name: "" for field_name in SUMMARY_FIELDS}
    row["image_name"] = image_name
    row["status"] = status
    row["person_count"] = person_count

    return row


def save_image(
    output_path: Path,
    image: np.ndarray,
) -> None:
    """Save an OpenCV image.

    Args:
        output_path: Destination image path.
        image: Image to save.

    Raises:
        OSError: If OpenCV cannot write the image.
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
    """Save experiment results to CSV.

    Args:
        summary_path: Destination CSV path.
        summary_rows: Per-image experiment rows.
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
    """Evaluate person-mask connectivity and ROI quality.

    Args:
        image_path: Input image path.
        output_dir: Experiment output directory.
        processor: Mask2Former image processor.
        model: Pretrained Mask2Former model.

    Returns:
        Summary row for one image.
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
    raw_bbox = mask_to_bbox(person_mask)

    if raw_bbox is None:
        return build_empty_summary_row(
            image_path.name,
            "empty_mask",
            len(person_segments),
        )

    (
        largest_component_mask,
        component_count,
        largest_component_share,
    ) = get_largest_component(person_mask)

    component_bbox = mask_to_bbox(largest_component_mask)

    if component_bbox is None:
        return build_empty_summary_row(
            image_path.name,
            "empty_component",
            len(person_segments),
        )

    image_width = image.width
    image_height = image.height
    image_area = image_width * image_height

    raw_bbox_area_ratio = calculate_bbox_area_ratio(
        raw_bbox,
        image_width,
        image_height,
    )
    component_bbox_area_ratio = calculate_bbox_area_ratio(
        component_bbox,
        image_width,
        image_height,
    )
    mask_area_ratio = float(person_mask.sum()) / float(image_area)
    bbox_fill_ratio = mask_area_ratio / raw_bbox_area_ratio
    bbox_reduction = raw_bbox_area_ratio - component_bbox_area_ratio

    (
        roi_strategy,
        margin_ratio,
        final_bbox,
    ) = choose_roi(
        raw_bbox,
        component_bbox,
        raw_bbox_area_ratio,
        component_bbox_area_ratio,
        largest_component_share,
        image_width,
        image_height,
    )

    final_roi_area_ratio = calculate_bbox_area_ratio(
        final_bbox,
        image_width,
        image_height,
    )

    original_rgb = np.asarray(image)
    original_bgr = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR,
    )

    overlay = original_bgr.copy()
    mask_color = np.asarray(
        RAW_BOX_COLOR,
        dtype=np.float32,
    )

    overlay[person_mask] = (
        overlay[person_mask].astype(np.float32) * (1.0 - MASK_ALPHA)
        + mask_color * MASK_ALPHA
    ).astype(np.uint8)

    raw_x_min, raw_y_min, raw_x_max, raw_y_max = raw_bbox
    cv2.rectangle(
        overlay,
        (raw_x_min, raw_y_min),
        (raw_x_max, raw_y_max),
        RAW_BOX_COLOR,
        BOX_THICKNESS,
    )

    (
        component_x_min,
        component_y_min,
        component_x_max,
        component_y_max,
    ) = component_bbox
    cv2.rectangle(
        overlay,
        (component_x_min, component_y_min),
        (component_x_max, component_y_max),
        COMPONENT_BOX_COLOR,
        BOX_THICKNESS,
    )

    final_x_min, final_y_min, final_x_max, final_y_max = final_bbox
    cv2.rectangle(
        overlay,
        (final_x_min, final_y_min),
        (final_x_max, final_y_max),
        FINAL_ROI_COLOR,
        BOX_THICKNESS,
    )

    label_text = (
        f"person {person_score:.2f} "
        f"components {component_count} "
        f"share {largest_component_share:.2f}"
    )
    cv2.putText(
        overlay,
        label_text,
        (
            raw_x_min,
            max(
                raw_y_min - TEXT_VERTICAL_OFFSET,
                TEXT_MIN_Y,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        RAW_BOX_COLOR,
        TEXT_THICKNESS,
    )

    overlay_path = output_dir / f"{image_path.stem}_quality_overlay.jpg"
    save_image(
        overlay_path,
        overlay,
    )

    final_roi = original_bgr[
        final_y_min : final_y_max + 1,
        final_x_min : final_x_max + 1,
    ]
    roi_path = output_dir / f"{image_path.stem}_quality_roi.jpg"
    save_image(
        roi_path,
        final_roi,
    )

    LOGGER.info(
        "%s: raw_bbox=%.4f, mask=%.4f, fill=%.4f, "
        "components=%d, largest_share=%.4f, component_bbox=%.4f, "
        "strategy=%s, final_roi=%.4f",
        image_path.name,
        raw_bbox_area_ratio,
        mask_area_ratio,
        bbox_fill_ratio,
        component_count,
        largest_component_share,
        component_bbox_area_ratio,
        roi_strategy,
        final_roi_area_ratio,
    )

    return {
        "image_name": image_path.name,
        "status": "success",
        "roi_strategy": roi_strategy,
        "person_count": len(person_segments),
        "score": round(person_score, 4),
        "raw_bbox_area_ratio": round(
            raw_bbox_area_ratio,
            4,
        ),
        "mask_area_ratio": round(
            mask_area_ratio,
            4,
        ),
        "bbox_fill_ratio": round(
            bbox_fill_ratio,
            4,
        ),
        "component_count": component_count,
        "largest_component_share": round(
            largest_component_share,
            4,
        ),
        "largest_component_bbox_area_ratio": round(
            component_bbox_area_ratio,
            4,
        ),
        "bbox_reduction": round(
            bbox_reduction,
            4,
        ),
        "margin_ratio": round(
            margin_ratio,
            4,
        ),
        "final_roi_area_ratio": round(
            final_roi_area_ratio,
            4,
        ),
    }


def select_sample_images(
    image_dir: Path,
    sample_count: int,
    random_seed: int,
) -> list[Path]:
    """Select a deterministic random sample of images.

    Args:
        image_dir: Directory containing JPEG images.
        sample_count: Number of images to sample.
        random_seed: Random seed.

    Returns:
        Selected image paths.

    Raises:
        FileNotFoundError: If the image directory does not exist.
        ValueError: If too few images are available.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(image_dir.glob("*.jpg"))

    if len(image_paths) < sample_count:
        raise ValueError("Not enough images for evaluation")

    random_generator = random.Random(random_seed)

    return random_generator.sample(
        image_paths,
        sample_count,
    )


def main() -> None:
    """Run connected-component ROI quality evaluation."""
    configure_logging()

    project_root = Path(__file__).resolve().parents[1]

    image_dir = (
        project_root.parent / "fashion_data" / "raw" / "train" / "train" / "image"
    )

    output_dir = project_root / "outputs" / "person_roi_quality_batch"
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

        summary_rows.append(
            evaluate_image(
                image_path,
                output_dir,
                processor,
                model,
            )
        )

    save_summary(
        summary_path,
        summary_rows,
    )

    LOGGER.info("Quality evaluation completed")
    LOGGER.info(
        "Summary saved to: %s",
        summary_path,
    )


if __name__ == "__main__":
    main()
