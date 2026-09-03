"""Run a Mask2Former pretrained baseline on one DeepFashion2 image."""

import logging
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "facebook/mask2former-swin-tiny-coco-instance"
SAMPLE_IMAGE_NAME = "000001.jpg"

MASK_ALPHA = 0.5
BOX_THICKNESS = 2
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
TEXT_VERTICAL_OFFSET = 10
TEXT_MIN_Y = 20

INSTANCE_COLORS = (
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
)


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def mask_to_bbox(
    mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Calculate a bounding box from a binary instance mask.

    Args:
        mask: Two-dimensional binary mask.

    Returns:
        Bounding box as ``(x_min, y_min, x_max, y_max)``.
        Returns ``None`` when the mask contains no foreground pixels.
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


def get_label_name(
    id2label: Mapping[Any, str],
    label_id: int,
) -> str:
    """Resolve a model class ID to its label name.

    Args:
        id2label: Model class ID to label mapping.
        label_id: Predicted class ID.

    Returns:
        Human-readable label name.
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

    result = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[
            (
                image.height,
                image.width,
            )
        ],
    )[0]

    return result


def visualize_instances(
    image_bgr: np.ndarray,
    segmentation: np.ndarray,
    segments_info: list[dict[str, Any]],
    id2label: Mapping[Any, str],
) -> np.ndarray:
    """Visualize predicted masks, bounding boxes, and class labels.

    Args:
        image_bgr: Original image in OpenCV BGR format.
        segmentation: Instance-ID segmentation map.
        segments_info: Metadata for each predicted instance.
        id2label: Model class ID to label mapping.

    Returns:
        Image containing instance masks, boxes, labels, and scores.
    """
    visualization = image_bgr.copy()

    for index, segment in enumerate(segments_info):
        segment_id = int(segment["id"])
        label_id = int(segment["label_id"])
        score = float(segment["score"])

        mask = segmentation == segment_id
        bbox = mask_to_bbox(mask)

        if bbox is None:
            LOGGER.warning(
                "Skipping empty predicted instance: id=%d",
                segment_id,
            )
            continue

        color = INSTANCE_COLORS[index % len(INSTANCE_COLORS)]

        color_array = np.asarray(
            color,
            dtype=np.float32,
        )

        visualization[mask] = (
            visualization[mask].astype(np.float32) * (1.0 - MASK_ALPHA)
            + color_array * MASK_ALPHA
        ).astype(np.uint8)

        x_min, y_min, x_max, y_max = bbox

        cv2.rectangle(
            visualization,
            (x_min, y_min),
            (x_max, y_max),
            color,
            BOX_THICKNESS,
        )

        label_name = get_label_name(
            id2label,
            label_id,
        )

        label_text = f"{label_name}: {score:.2f}"

        cv2.putText(
            visualization,
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
            color,
            TEXT_THICKNESS,
        )

        LOGGER.info(
            "Instance id=%d, label=%s, score=%.4f, " "bbox=(%d, %d, %d, %d)",
            segment_id,
            label_name,
            score,
            x_min,
            y_min,
            x_max,
            y_max,
        )

    return visualization


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

    if not cv2.imwrite(
        str(output_path),
        image,
    ):
        raise OSError(f"Failed to save image: {output_path}")


def main() -> None:
    """Run the Mask2Former baseline inference workflow."""
    configure_logging()

    project_root = Path(__file__).resolve().parents[1]

    image_path = (
        project_root.parent
        / "fashion_data"
        / "raw"
        / "train"
        / "train"
        / "image"
        / SAMPLE_IMAGE_NAME
    )

    output_dir = project_root / "outputs" / "mask2former_baseline"

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    LOGGER.info(
        "Input image: %s",
        image_path,
    )
    LOGGER.info(
        "Image size: %s",
        image.size,
    )

    LOGGER.info("Loading image processor")

    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_NAME)

    LOGGER.info("Loading Mask2Former model")

    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME)
    model.eval()

    LOGGER.info("Running baseline inference")

    result = run_inference(
        image,
        processor,
        model,
    )

    segmentation = result["segmentation"].cpu().numpy()

    segments_info = result["segments_info"]

    LOGGER.info(
        "Detected instances: %d",
        len(segments_info),
    )

    image_rgb = np.asarray(image)
    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    visualization = visualize_instances(
        image_bgr,
        segmentation,
        segments_info,
        model.config.id2label,
    )

    output_path = output_dir / f"{image_path.stem}_instances.jpg"

    save_image(
        output_path,
        visualization,
    )

    LOGGER.info(
        "Saved baseline visualization: %s",
        output_path,
    )


if __name__ == "__main__":
    main()
