"""Check DeepFashion2Dataset output with one sample."""

import logging
from pathlib import Path

import torch
from transformers import Mask2FormerImageProcessor

from fashion_multimodal_analysis.datasets.deepfashion2_dataset import (
    DeepFashion2Dataset,
)


LOGGER = logging.getLogger(__name__)

MODEL_NAME = "facebook/mask2former-swin-tiny-coco-instance"


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def main() -> None:
    """Load and inspect one DeepFashion2 dataset sample."""
    configure_logging()

    project_root = Path(__file__).resolve().parents[1]

    image_dir = (
        project_root.parent
        / "fashion_data"
        / "raw"
        / "train"
        / "train"
        / "image"
    )

    annotation_dir = (
        project_root.parent
        / "fashion_data"
        / "raw"
        / "train"
        / "train"
        / "annos"
    )

    processor = (
        Mask2FormerImageProcessor
        .from_pretrained(MODEL_NAME)
    )

    dataset = DeepFashion2Dataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        processor=processor,
    )

    sample = dataset[0]

    overlap_pixels = int(
        (
            sample["mask_labels"].sum(
                dim=0
            )
            > 1
        )
        .sum()
        .item()
    )

    LOGGER.info(
        "Dataset size: %d",
        len(dataset),
    )

    LOGGER.info(
        "Image name: %s",
        sample["image_name"],
    )

    LOGGER.info(
        "Category names: %s",
        sample["category_names"],
    )

    LOGGER.info(
        "Class labels: %s",
        sample["class_labels"].tolist(),
    )

    LOGGER.info(
        "Pixel values shape: %s",
        tuple(
            sample["pixel_values"].shape
        ),
    )

    LOGGER.info(
        "Mask labels shape: %s",
        tuple(
            sample["mask_labels"].shape
        ),
    )

    LOGGER.info(
        "Mask values: %s",
        torch.unique(
            sample["mask_labels"]
        ).tolist(),
    )

    LOGGER.info(
        "Overlapping mask pixels: %d",
        overlap_pixels,
    )


if __name__ == "__main__":
    main()