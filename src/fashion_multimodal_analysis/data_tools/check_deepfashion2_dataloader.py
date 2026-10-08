"""数据处理：保存原图来源和实例标识，避免同一图片的不同实例跨训练与评估划分。

Check batched DeepFashion2 data loading.
"""

from __future__ import annotations

import logging
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import Mask2FormerImageProcessor

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)
from fashion_multimodal_analysis.datasets.collate import (
    mask2former_collate_fn,
)
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
    """Create one DeepFashion2 batch and inspect its structure."""
    configure_logging()

    project_root = get_project_root()

    image_dir = data_root() / "raw" / "train" / "train" / "image"

    annotation_dir = data_root() / "raw" / "train" / "train" / "annos"

    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_NAME)

    dataset = DeepFashion2Dataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        processor=processor,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=mask2former_collate_fn,
    )

    batch = next(iter(data_loader))

    LOGGER.info(
        "Pixel values shape: %s",
        tuple(batch["pixel_values"].shape),
    )

    LOGGER.info(
        "Batch size: %d",
        len(batch["mask_labels"]),
    )

    for index, (
        masks,
        classes,
    ) in enumerate(
        zip(
            batch["mask_labels"],
            batch["class_labels"],
        )
    ):
        LOGGER.info(
            "Sample %d mask shape: %s",
            index,
            tuple(masks.shape),
        )

        LOGGER.info(
            "Sample %d class labels: %s",
            index,
            classes.tolist(),
        )


if __name__ == "__main__":
    main()
