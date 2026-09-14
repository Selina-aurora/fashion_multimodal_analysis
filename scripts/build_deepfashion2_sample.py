"""Build one DeepFashion2 training sample.

This script converts one DeepFashion2 image and its annotation into
instance-level class labels and binary mask labels for later model training.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


LOGGER = logging.getLogger(__name__)

DEEPFASHION2_CLASS_COUNT = 13


@dataclass(frozen=True)
class DeepFashion2Sample:
    """Represent one DeepFashion2 training sample.

    Attributes:
        image: RGB image.
        class_labels: Zero-based class IDs with shape [N].
        mask_labels: Binary instance masks with shape [N, H, W].
        category_names: Human-readable category names.
    """

    image: Image.Image
    class_labels: torch.Tensor
    mask_labels: torch.Tensor
    category_names: list[str]


def load_annotation(annotation_path: Path) -> dict[str, Any]:
    """Load a DeepFashion2 JSON annotation.

    Args:
        annotation_path: Path to the JSON annotation file.

    Returns:
        Parsed annotation dictionary.

    Raises:
        FileNotFoundError: If the annotation file does not exist.
        json.JSONDecodeError: If the annotation cannot be parsed.
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
        segmentation: Flattened polygon coordinate lists.
        image_height: Original image height.
        image_width: Original image width.

    Returns:
        Binary mask containing values 0 and 1.
    """
    mask = np.zeros(
        (image_height, image_width),
        dtype=np.uint8,
    )

    for polygon in segmentation:
        if len(polygon) < 6 or len(polygon) % 2 != 0:
            LOGGER.warning(
                "Skipping invalid polygon with %d coordinates",
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

        cv2.fillPoly(
            mask,
            [points.astype(np.int32)],
            1,
        )

    return mask


def build_training_sample(
    image_path: Path,
    annotation_path: Path,
) -> DeepFashion2Sample:
    """Build one DeepFashion2 training sample.

    Args:
        image_path: Path to the RGB image.
        annotation_path: Path to the corresponding JSON annotation.

    Returns:
        DeepFashion2 sample containing image, classes, and masks.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the annotation contains an invalid category ID or
            no valid clothing instances.
    """
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")
    annotation = load_annotation(annotation_path)

    class_ids: list[int] = []
    category_names: list[str] = []
    instance_masks: list[np.ndarray] = []

    for item_name, item_data in annotation.items():
        if not item_name.startswith("item"):
            continue

        if not isinstance(item_data, dict):
            continue

        category_id = item_data.get("category_id")
        category_name = item_data.get("category_name")
        segmentation = item_data.get("segmentation")

        if not isinstance(category_id, int):
            LOGGER.warning(
                "Skipping %s because category_id is invalid",
                item_name,
            )
            continue

        if not 1 <= category_id <= DEEPFASHION2_CLASS_COUNT:
            raise ValueError(
                f"Invalid category_id {category_id} in {item_name}"
            )

        if not isinstance(category_name, str):
            LOGGER.warning(
                "Skipping %s because category_name is invalid",
                item_name,
            )
            continue

        if not isinstance(segmentation, list):
            LOGGER.warning(
                "Skipping %s because segmentation is invalid",
                item_name,
            )
            continue

        mask = create_binary_mask(
            segmentation,
            image.height,
            image.width,
        )

        if not mask.any():
            LOGGER.warning(
                "Skipping %s because its mask is empty",
                item_name,
            )
            continue

        class_ids.append(category_id - 1)
        category_names.append(category_name)
        instance_masks.append(mask)

    if not instance_masks:
        raise ValueError(
            f"No valid clothing instances found in {annotation_path}"
        )

    class_labels = torch.tensor(
        class_ids,
        dtype=torch.long,
    )

    mask_labels = torch.from_numpy(
        np.stack(instance_masks)
    ).to(torch.float32)

    return DeepFashion2Sample(
        image=image,
        class_labels=class_labels,
        mask_labels=mask_labels,
        category_names=category_names,
    )


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def main() -> None:
    """Build and inspect one DeepFashion2 training sample."""
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

    sample = build_training_sample(
        image_path,
        annotation_path,
    )

    LOGGER.info(
        "Image size: %s",
        sample.image.size,
    )
    LOGGER.info(
        "Number of instances: %d",
        len(sample.category_names),
    )
    LOGGER.info(
        "Category names: %s",
        sample.category_names,
    )
    LOGGER.info(
        "Class labels: %s",
        sample.class_labels.tolist(),
    )
    LOGGER.info(
        "Mask labels shape: %s",
        tuple(sample.mask_labels.shape),
    )
    LOGGER.info(
        "Mask values: %s",
        torch.unique(sample.mask_labels).tolist(),
    )


if __name__ == "__main__":
    main()