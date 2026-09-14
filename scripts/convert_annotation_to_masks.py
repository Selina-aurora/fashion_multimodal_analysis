"""Convert DeepFashion2 polygon annotations to binary instance masks.

This script reads one DeepFashion2 image and its JSON annotation, converts
each clothing instance segmentation polygon into a binary mask, and saves
the masks and a visualization for manual verification.
"""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)


def load_annotation(annotation_path: Path) -> dict[str, Any]:
    """Load a DeepFashion2 JSON annotation.

    Args:
        annotation_path: Path to the JSON annotation file.

    Returns:
        Parsed annotation dictionary.

    Raises:
        FileNotFoundError: If the annotation file does not exist.
        json.JSONDecodeError: If the JSON file is invalid.
    """
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_path}"
        )

    with annotation_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_binary_mask(
    segmentation: list[list[float]],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Convert segmentation polygons into a binary mask.

    Args:
        segmentation: List of flattened polygon coordinate lists.
        image_height: Original image height.
        image_width: Original image width.

    Returns:
        Binary mask with foreground value 255 and background value 0.
    """
    mask = np.zeros(
        (image_height, image_width),
        dtype=np.uint8,
    )

    for polygon in segmentation:
        if len(polygon) < 6 or len(polygon) % 2 != 0:
            LOGGER.warning(
                "Skipping invalid polygon with %d values",
                len(polygon),
            )
            continue

        points = np.asarray(
            polygon,
            dtype=np.float32,
        ).reshape(-1, 2)

        points[:, 0] = np.clip(
            points[:, 0],
            0,
            image_width - 1,
        )
        points[:, 1] = np.clip(
            points[:, 1],
            0,
            image_height - 1,
        )

        points = points.astype(np.int32)

        cv2.fillPoly(
            mask,
            [points],
            255,
        )

    return mask


def save_instance_masks(
    annotation: dict[str, Any],
    image_height: int,
    image_width: int,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    """Create and save binary masks for all clothing instances.

    Args:
        annotation: Parsed DeepFashion2 annotation.
        image_height: Original image height.
        image_width: Original image width.
        output_dir: Directory used to save masks.

    Returns:
        Mapping from instance name to its generated binary mask.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    masks: dict[str, np.ndarray] = {}

    for item_name, item_data in annotation.items():
        if not item_name.startswith("item"):
            continue

        if not isinstance(item_data, dict):
            continue

        segmentation = item_data.get("segmentation")
        category_name = item_data.get("category_name", "unknown")

        if not isinstance(segmentation, list):
            LOGGER.warning(
                "No valid segmentation found for %s",
                item_name,
            )
            continue

        mask = create_binary_mask(
            segmentation,
            image_height,
            image_width,
        )

        mask_path = (
            output_dir
            / f"{item_name}_mask.png"
        )

        if not cv2.imwrite(
            str(mask_path),
            mask,
        ):
            raise RuntimeError(
                f"Failed to save mask: {mask_path}"
            )

        masks[item_name] = mask

        LOGGER.info(
            "Saved %s (%s): %s",
            item_name,
            category_name,
            mask_path,
        )

    return masks


def create_mask_overlay(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    """Overlay generated instance masks on the original image.

    Args:
        image: Original RGB image.
        masks: Mapping of instance names to binary masks.

    Returns:
        RGB visualization containing the instance masks.
    """
    visualization = image.copy()

    colors = [
        np.array([0, 255, 0], dtype=np.uint8),
        np.array([255, 0, 0], dtype=np.uint8),
        np.array([0, 0, 255], dtype=np.uint8),
        np.array([255, 255, 0], dtype=np.uint8),
    ]

    for index, mask in enumerate(masks.values()):
        foreground = mask > 0
        color = colors[index % len(colors)]

        visualization[foreground] = (
            visualization[foreground] * 0.5
            + color * 0.5
        ).astype(np.uint8)

    return visualization


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def main() -> None:
    """Convert one DeepFashion2 annotation into instance masks."""
    configure_logging()

    project_root = Path(__file__).resolve().parents[1]

    image_path = (
        project_root.parent
        / "fashion_data"
        / "raw"
        / "train"
        / "train"
        / "image"
        / "000001.jpg"
    )

    annotation_path = (
        project_root.parent
        / "fashion_data"
        / "raw"
        / "train"
        / "train"
        / "annos"
        / "000001.json"
    )

    output_dir = (
        project_root
        / "outputs"
        / "annotation_masks"
        / "000001"
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image)

    annotation = load_annotation(
        annotation_path
    )

    masks = save_instance_masks(
        annotation,
        image.height,
        image.width,
        output_dir,
    )

    overlay = create_mask_overlay(
        image_array,
        masks,
    )

    overlay_bgr = cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR,
    )

    overlay_path = (
        output_dir
        / "000001_mask_overlay.jpg"
    )

    if not cv2.imwrite(
        str(overlay_path),
        overlay_bgr,
    ):
        raise RuntimeError(
            f"Failed to save overlay: {overlay_path}"
        )

    LOGGER.info(
        "Generated %d instance masks",
        len(masks),
    )
    LOGGER.info(
        "Overlay saved to: %s",
        overlay_path,
    )


if __name__ == "__main__":
    main()