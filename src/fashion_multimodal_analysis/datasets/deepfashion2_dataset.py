"""DeepFashion2 dataset utilities for Mask2Former training."""

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from transformers import Mask2FormerImageProcessor

LOGGER = logging.getLogger(__name__)

DEEPFASHION2_CLASS_COUNT = 13
DEFAULT_TARGET_SIZE = (384, 384)


class Mask2FormerSample(TypedDict):
    """Training sample returned by DeepFashion2Dataset."""

    pixel_values: Tensor
    mask_labels: Tensor
    class_labels: Tensor
    category_names: list[str]
    image_name: str


def _load_annotation(annotation_path: Path) -> dict[str, Any]:
    """Load one DeepFashion2 annotation file.

    Args:
        annotation_path: Path to the annotation JSON file.

    Returns:
        Parsed annotation dictionary.

    Raises:
        FileNotFoundError: If the annotation file does not exist.
        json.JSONDecodeError: If the annotation file contains invalid JSON.
    """
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _get_item_index(item_name: str) -> int:
    """Extract the numeric index from a DeepFashion2 item key.

    Args:
        item_name: Annotation key such as ``item1`` or ``item2``.

    Returns:
        Numeric item index.

    Raises:
        ValueError: If the item name does not contain a valid index.
    """
    index_text = item_name.removeprefix("item")

    if not index_text.isdigit():
        raise ValueError(f"Invalid item name: {item_name}")

    return int(index_text)


def _create_binary_mask(
    segmentation: list[Any],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Convert segmentation polygons into one binary instance mask.

    Args:
        segmentation: Polygon annotations for one clothing instance.
        image_height: Original image height in pixels.
        image_width: Original image width in pixels.

    Returns:
        Binary mask with foreground pixels set to 1.
    """
    mask = np.zeros(
        (image_height, image_width),
        dtype=np.uint8,
    )

    for polygon in segmentation:
        if not isinstance(polygon, list):
            LOGGER.warning(
                "Skipping polygon with invalid type: %s",
                type(polygon).__name__,
            )
            continue

        if len(polygon) < 6 or len(polygon) % 2 != 0:
            LOGGER.warning(
                "Skipping invalid polygon with %d coordinates",
                len(polygon),
            )
            continue

        try:
            points = np.asarray(
                polygon,
                dtype=np.float32,
            ).reshape(-1, 2)
        except (TypeError, ValueError):
            LOGGER.warning("Skipping polygon containing invalid coordinates")
            continue

        if not np.isfinite(points).all():
            LOGGER.warning("Skipping polygon containing non-finite coordinates")
            continue

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


def _resize_instance_masks(
    masks: list[np.ndarray],
    target_height: int,
    target_width: int,
) -> Tensor:
    """Resize instance masks while preserving binary labels.

    Args:
        masks: Independent binary masks for all clothing instances.
        target_height: Target mask height in pixels.
        target_width: Target mask width in pixels.

    Returns:
        Float tensor with shape ``[N, H, W]``.
    """
    resized_masks = [
        cv2.resize(
            mask,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        for mask in masks
    ]

    return torch.from_numpy(np.stack(resized_masks)).to(torch.float32)


class DeepFashion2Dataset(Dataset[Mask2FormerSample]):
    """Load DeepFashion2 samples for Mask2Former training.

    Each clothing instance is retained as an independent binary mask so
    overlapping DeepFashion2 instance annotations are preserved.

    Args:
        image_dir: Directory containing DeepFashion2 images.
        annotation_dir: Directory containing annotation JSON files.
        processor: Mask2Former image processor.
        target_size: Training image size as ``(height, width)``.

    Raises:
        FileNotFoundError: If an input directory does not exist.
        ValueError: If target size is invalid or no annotations are found.
    """

    def __init__(
        self,
        image_dir: Path,
        annotation_dir: Path,
        processor: Mask2FormerImageProcessor,
        target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    ) -> None:
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        if not annotation_dir.is_dir():
            raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

        target_height, target_width = target_size

        if target_height <= 0 or target_width <= 0:
            raise ValueError("Target size must contain positive values")

        annotation_paths = sorted(annotation_dir.glob("*.json"))

        if not annotation_paths:
            raise ValueError(f"No annotations found in: {annotation_dir}")

        self._image_dir = image_dir
        self._annotation_paths = annotation_paths
        self._processor = processor
        self._target_size = target_size

    def __len__(self) -> int:
        """Return the number of annotation samples."""
        return len(self._annotation_paths)

    def _build_targets(
        self,
        annotation: dict[str, Any],
        image_height: int,
        image_width: int,
    ) -> tuple[list[int], list[np.ndarray], list[str]]:
        """Build class labels and independent instance masks.

        Args:
            annotation: Parsed DeepFashion2 annotation dictionary.
            image_height: Original image height in pixels.
            image_width: Original image width in pixels.

        Returns:
            Class IDs, independent binary masks, and category names.

        Raises:
            ValueError: If an invalid category ID is found or no valid
                clothing instances remain.
        """
        class_ids: list[int] = []
        masks: list[np.ndarray] = []
        category_names: list[str] = []

        items = [
            (item_name, item_data)
            for item_name, item_data in annotation.items()
            if item_name.startswith("item") and isinstance(item_data, dict)
        ]

        items.sort(key=lambda item: _get_item_index(item[0]))

        for item_name, item_data in items:
            category_id = item_data.get("category_id")
            category_name = item_data.get("category_name")
            segmentation = item_data.get("segmentation")

            if not isinstance(category_id, int):
                LOGGER.warning(
                    "Skipping %s with invalid category_id",
                    item_name,
                )
                continue

            if not 1 <= category_id <= DEEPFASHION2_CLASS_COUNT:
                raise ValueError(
                    f"Invalid category_id " f"{category_id} in {item_name}"
                )

            if not isinstance(category_name, str):
                LOGGER.warning(
                    "Skipping %s with invalid category_name",
                    item_name,
                )
                continue

            if not isinstance(segmentation, list):
                LOGGER.warning(
                    "Skipping %s with invalid segmentation",
                    item_name,
                )
                continue

            mask = _create_binary_mask(
                segmentation,
                image_height,
                image_width,
            )

            if not mask.any():
                LOGGER.warning(
                    "Skipping %s with empty mask",
                    item_name,
                )
                continue

            class_ids.append(category_id - 1)
            masks.append(mask)
            category_names.append(category_name)

        if not masks:
            raise ValueError("No valid clothing instances found")

        return class_ids, masks, category_names

    def __getitem__(
        self,
        index: int,
    ) -> Mask2FormerSample:
        """Load and preprocess one DeepFashion2 training sample.

        Args:
            index: Dataset sample index.

        Returns:
            Preprocessed Mask2Former training sample.

        Raises:
            FileNotFoundError: If the corresponding image does not exist.
            RuntimeError: If processed image and mask dimensions differ.
        """
        annotation_path = self._annotation_paths[index]

        image_path = self._image_dir / f"{annotation_path.stem}.jpg"

        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        annotation = _load_annotation(annotation_path)

        class_ids, masks, category_names = self._build_targets(
            annotation,
            image.height,
            image.width,
        )

        target_height, target_width = self._target_size

        resized_image = image.resize(
            (target_width, target_height),
            resample=Image.Resampling.BILINEAR,
        )

        mask_labels = _resize_instance_masks(
            masks,
            target_height,
            target_width,
        )

        processed_image = self._processor(
            images=resized_image,
            do_resize=False,
            return_tensors="pt",
        )

        pixel_values = processed_image["pixel_values"].squeeze(0)

        class_labels = torch.tensor(
            class_ids,
            dtype=torch.long,
        )

        expected_size = (
            target_height,
            target_width,
        )

        if tuple(pixel_values.shape[-2:]) != expected_size:
            raise RuntimeError(
                "Processed image dimensions do not match "
                f"target size: {tuple(pixel_values.shape[-2:])} "
                f"!= {expected_size}"
            )

        if tuple(mask_labels.shape[-2:]) != expected_size:
            raise RuntimeError(
                "Mask dimensions do not match "
                f"target size: {tuple(mask_labels.shape[-2:])} "
                f"!= {expected_size}"
            )

        return {
            "pixel_values": pixel_values,
            "mask_labels": mask_labels,
            "class_labels": class_labels,
            "category_names": category_names,
            "image_name": image_path.name,
        }
