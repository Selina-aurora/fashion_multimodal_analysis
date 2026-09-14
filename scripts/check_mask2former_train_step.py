"""Check one Mask2Former optimization step on DeepFashion2."""

import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

from fashion_multimodal_analysis.datasets.collate import (
    mask2former_collate_fn,
)
from fashion_multimodal_analysis.datasets.deepfashion2_dataset import (
    DeepFashion2Dataset,
)


LOGGER = logging.getLogger(__name__)

MODEL_NAME = "facebook/mask2former-swin-tiny-coco-instance"
LEARNING_RATE = 1e-5

DEEPFASHION2_LABELS = {
    0: "short sleeve top",
    1: "long sleeve top",
    2: "short sleeve outwear",
    3: "long sleeve outwear",
    4: "vest",
    5: "sling",
    6: "shorts",
    7: "trousers",
    8: "skirt",
    9: "short sleeve dress",
    10: "long sleeve dress",
    11: "vest dress",
    12: "sling dress",
}


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def create_label_mappings() -> tuple[dict[int, str], dict[str, int]]:
    """Create DeepFashion2 label mappings.

    Returns:
        Mapping from class IDs to names and the reverse mapping.
    """
    id2label = DEEPFASHION2_LABELS.copy()

    label2id = {
        label_name: label_id
        for label_id, label_name in id2label.items()
    }

    return id2label, label2id


def main() -> None:
    """Run one complete Mask2Former optimization step."""
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

    LOGGER.info("Loading image processor")

    processor = Mask2FormerImageProcessor.from_pretrained(
        MODEL_NAME
    )

    dataset = DeepFashion2Dataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        processor=processor,
    )

    # 本地 CPU 只用 batch_size=1 做训练链路验证。
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=mask2former_collate_fn,
    )

    batch = next(iter(data_loader))

    id2label, label2id = create_label_mappings()

    LOGGER.info("Loading Mask2Former model")

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    parameter_before = (
        model.class_predictor.weight
        .detach()
        .clone()
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    LOGGER.info("Running forward pass")

    outputs = model(
        pixel_values=batch["pixel_values"],
        mask_labels=batch["mask_labels"],
        class_labels=batch["class_labels"],
    )

    loss = outputs.loss

    LOGGER.info(
        "Loss before optimization: %.6f",
        loss.item(),
    )

    LOGGER.info("Running backward pass")

    loss.backward()

    gradient = model.class_predictor.weight.grad

    if gradient is None:
        raise RuntimeError(
            "Class predictor gradient was not generated"
        )

    gradient_norm = gradient.norm().item()

    LOGGER.info(
        "Class predictor gradient norm: %.6f",
        gradient_norm,
    )

    LOGGER.info("Updating model parameters")

    optimizer.step()

    parameter_after = (
        model.class_predictor.weight
        .detach()
        .clone()
    )

    parameter_change = (
        parameter_after - parameter_before
    ).abs().sum().item()

    LOGGER.info(
        "Parameter change: %.10f",
        parameter_change,
    )

    if parameter_change == 0.0:
        raise RuntimeError(
            "Model parameters were not updated"
        )

    LOGGER.info(
        "Training step completed successfully"
    )


if __name__ == "__main__":
    main()