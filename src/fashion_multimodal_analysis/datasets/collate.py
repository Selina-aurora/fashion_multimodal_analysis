"""Batch collation utilities for fashion segmentation datasets."""

from typing import TypedDict

import torch
from torch import Tensor

from fashion_multimodal_analysis.datasets.deepfashion2_dataset import Mask2FormerSample


class Mask2FormerBatch(TypedDict):
    """Batch used as Mask2Former training input."""

    pixel_values: Tensor
    mask_labels: list[Tensor]
    class_labels: list[Tensor]


def mask2former_collate_fn(
    samples: list[Mask2FormerSample],
) -> Mask2FormerBatch:
    """Combine samples into one Mask2Former training batch.

    Images are stacked because all samples share the same spatial size.
    Instance masks and class labels remain as lists because each image
    can contain a different number of clothing instances.

    Args:
        samples: Individual preprocessed dataset samples.

    Returns:
        Batch containing stacked image tensors and variable-length
        instance targets.

    Raises:
        ValueError: If the input sample list is empty.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    pixel_values = torch.stack([sample["pixel_values"] for sample in samples])

    mask_labels = [sample["mask_labels"] for sample in samples]

    class_labels = [sample["class_labels"] for sample in samples]

    return {
        "pixel_values": pixel_values,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }
