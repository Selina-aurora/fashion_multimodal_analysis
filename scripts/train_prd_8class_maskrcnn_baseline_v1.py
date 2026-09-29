"""
Train PRD 3.1.1 8-class Mask R-CNN baseline.

Inputs
------
configs/prd_8class_train_v1.csv
configs/prd_8class_val_v1.csv

Important
---------
The manifests are grouped by source image. This dataset loader reconstructs a
full-image binary mask for every instance:
- if mask_path already has the full source-image size, use it directly;
- if mask_path is a bbox-sized crop mask (e.g. Fashionpedia pilot), paste it
  back into the full image at bbox_x1/y1/x2/y2.

Thus DeepFashion2 and Fashionpedia can be trained in one full-image detector.

Outputs
-------
outputs/prd_instance_segmentation/maskrcnn_8class_baseline_v1/
    checkpoint_epoch_XX.pth
    checkpoint_last.pth

reports/prd_instance_segmentation/maskrcnn_8class_baseline_v1/
    train_log.csv
    summary.txt
    data_check.txt

Recommended first command:
    python scripts/train_prd_8class_maskrcnn_baseline_v1.py --dry-run

Then train:
    python scripts/train_prd_8class_maskrcnn_baseline_v1.py --epochs 15

On Windows, workers=0 is intentionally the default.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = PROJECT_ROOT / "configs" / "prd_8class_train_v1.csv"
VAL_CSV = PROJECT_ROOT / "configs" / "prd_8class_val_v1.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_instance_segmentation"
    / "maskrcnn_8class_baseline_v1"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_instance_segmentation"
    / "maskrcnn_8class_baseline_v1"
)

TRAIN_LOG = REPORT_DIR / "train_log.csv"
SUMMARY_PATH = REPORT_DIR / "summary.txt"
DATA_CHECK_PATH = REPORT_DIR / "data_check.txt"

CLASS_TO_ID = {
    "top": 1,
    "pants": 2,
    "skirt": 3,
    "outerwear": 4,
    "dress": 5,
    "shoe": 6,
    "bag": 7,
    "accessory": 8,
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

NUM_CLASSES = 1 + len(CLASS_TO_ID)  # + background


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    parser.add_argument("--seed", type=int, default=20260918)

    parser.add_argument(
        "--min-size",
        type=int,
        default=640,
        help="Mask R-CNN short-side resize.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Mask R-CNN long-side resize cap.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--no-hflip",
        action="store_true",
        help="Disable random horizontal flip.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data loading only; do not train.",
    )

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_path(value: str) -> Path:
    p = Path(str(value).strip())

    if p.is_absolute():
        return p

    return (PROJECT_ROOT / p).resolve()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest:\n  {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def group_manifest(
    rows: list[dict[str, str]],
) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        dataset = str(row["source_dataset"]).strip()
        source_image = str(row["source_image"]).strip()

        grouped[(dataset, source_image)].append(row)

    samples = []

    for (dataset, source_image), instances in grouped.items():
        samples.append(
            {
                "source_dataset": dataset,
                "source_image": source_image,
                "instances": instances,
            }
        )

    samples.sort(
        key=lambda s: (
            s["source_dataset"],
            s["source_image"],
        )
    )

    return samples


def open_binary_mask(path: Path) -> Image.Image:
    mask = Image.open(path).convert("L")
    # Force clean binary values.
    arr = np.asarray(mask)
    arr = (arr > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def reconstruct_full_mask(
    mask_path: Path,
    image_size: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    """
    Return a full-image binary mask.

    Supports both:
    - full-image masks
    - bbox/crop masks
    """
    image_w, image_h = image_size
    x1, y1, x2, y2 = bbox

    mask = open_binary_mask(mask_path)

    if mask.size == (image_w, image_h):
        return mask

    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)

    if mask.size != (bbox_w, bbox_h):
        mask = mask.resize(
            (bbox_w, bbox_h),
            resample=Image.Resampling.NEAREST,
        )

    canvas = Image.new("L", (image_w, image_h), 0)

    # Clip bbox to image bounds.
    cx1 = max(0, min(image_w, x1))
    cy1 = max(0, min(image_h, y1))
    cx2 = max(0, min(image_w, x2))
    cy2 = max(0, min(image_h, y2))

    if cx2 <= cx1 or cy2 <= cy1:
        return canvas

    # If clipping happened, crop corresponding area from resized bbox mask.
    mx1 = cx1 - x1
    my1 = cy1 - y1
    mx2 = mx1 + (cx2 - cx1)
    my2 = my1 + (cy2 - cy1)

    clipped_mask = mask.crop((mx1, my1, mx2, my2))

    canvas.paste(
        clipped_mask,
        (cx1, cy1),
    )

    return canvas


class PRD8MaskDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        training: bool,
        enable_hflip: bool = True,
    ) -> None:
        self.manifest_path = manifest_path
        self.training = training
        self.enable_hflip = enable_hflip

        rows = read_manifest(manifest_path)
        self.samples = group_manifest(rows)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image_path = resolve_project_path(
            sample["source_image"]
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Source image not found:\n  {image_path}"
            )

        image = Image.open(image_path).convert("RGB")
        image_w, image_h = image.size

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for row in sample["instances"]:
            category = str(
                row["garment_category"]
            ).strip().lower()

            if category not in CLASS_TO_ID:
                raise ValueError(
                    f"Unknown PRD class: {category!r}"
                )

            x1 = int(float(row["bbox_x1"]))
            y1 = int(float(row["bbox_y1"]))
            x2 = int(float(row["bbox_x2"]))
            y2 = int(float(row["bbox_y2"]))

            # Clamp boxes to image.
            x1 = max(0, min(image_w - 1, x1))
            y1 = max(0, min(image_h - 1, y1))
            x2 = max(x1 + 1, min(image_w, x2))
            y2 = max(y1 + 1, min(image_h, y2))

            mask_path = resolve_project_path(
                row["mask_path"]
            )

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Mask not found:\n  {mask_path}"
                )

            full_mask = reconstruct_full_mask(
                mask_path=mask_path,
                image_size=(image_w, image_h),
                bbox=(x1, y1, x2, y2),
            )

            mask_arr = (
                np.asarray(full_mask) > 0
            ).astype(np.uint8)

            if mask_arr.sum() == 0:
                raise ValueError(
                    "Empty reconstructed mask:\n"
                    f"  image={image_path}\n"
                    f"  mask={mask_path}"
                )

            boxes.append(
                [float(x1), float(y1), float(x2), float(y2)]
            )
            labels.append(CLASS_TO_ID[category])
            masks.append(mask_arr)
            areas.append(float(mask_arr.sum()))
            iscrowd.append(0)

        image_tensor = F.to_tensor(image)

        boxes_t = torch.as_tensor(
            boxes,
            dtype=torch.float32,
        )
        labels_t = torch.as_tensor(
            labels,
            dtype=torch.int64,
        )
        masks_t = torch.as_tensor(
            np.stack(masks, axis=0),
            dtype=torch.uint8,
        )
        area_t = torch.as_tensor(
            areas,
            dtype=torch.float32,
        )
        iscrowd_t = torch.as_tensor(
            iscrowd,
            dtype=torch.int64,
        )

        # Horizontal flip must transform image, boxes and masks together.
        if (
            self.training
            and self.enable_hflip
            and random.random() < 0.5
        ):
            image_tensor = torch.flip(
                image_tensor,
                dims=[2],
            )
            masks_t = torch.flip(
                masks_t,
                dims=[2],
            )

            old_x1 = boxes_t[:, 0].clone()
            old_x2 = boxes_t[:, 2].clone()

            boxes_t[:, 0] = image_w - old_x2
            boxes_t[:, 2] = image_w - old_x1

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor(
                [index],
                dtype=torch.int64,
            ),
            "area": area_t,
            "iscrowd": iscrowd_t,
        }

        meta = {
            "source_dataset": sample["source_dataset"],
            "source_image": sample["source_image"],
            "instance_count": len(boxes),
        }

        return image_tensor, target, meta


def collate_fn(batch):
    images = []
    targets = []
    metas = []

    for image, target, meta in batch:
        images.append(image)
        targets.append(target)
        metas.append(meta)

    return images, targets, metas


def build_model(
    min_size: int,
    max_size: int,
):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

    model = maskrcnn_resnet50_fpn(
        weights=weights,
    )

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES,
    )

    in_features_mask = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        NUM_CLASSES,
    )

    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size

    return model


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested, but CUDA is unavailable."
            )
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def inspect_dataset(
    dataset: PRD8MaskDataset,
    name: str,
) -> list[str]:
    lines = [
        f"{name}: image_groups={len(dataset)}"
    ]

    class_counts = Counter()
    source_counts = Counter()
    total_instances = 0

    for sample in dataset.samples:
        source_counts[sample["source_dataset"]] += 1

        for row in sample["instances"]:
            cls = row["garment_category"]
            class_counts[cls] += 1
            total_instances += 1

    lines.append(
        f"{name}: instances={total_instances}"
    )
    lines.append(
        f"{name}: source_image_groups={dict(source_counts)}"
    )

    for cls in CLASS_TO_ID:
        lines.append(
            f"{name}: {cls}={class_counts.get(cls, 0)}"
        )

    # Actually load a few samples, including one from each source if possible.
    checked = 0
    checked_sources = set()

    for idx, sample in enumerate(dataset.samples):
        source = sample["source_dataset"]

        if source in checked_sources and checked >= 4:
            continue

        image, target, meta = dataset[idx]

        lines.append(
            f"loaded[{name}][{idx}]: "
            f"source={source}, "
            f"shape={tuple(image.shape)}, "
            f"instances={len(target['labels'])}, "
            f"mask_shape={tuple(target['masks'].shape)}"
        )

        checked += 1
        checked_sources.add(source)

        if checked >= 6:
            break

    return lines


def move_targets_to_device(targets, device):
    moved = []

    for target in targets:
        moved.append(
            {
                key: value.to(device)
                for key, value in target.items()
            }
        )

    return moved


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "class_to_id": CLASS_TO_ID,
        "num_classes": NUM_CLASSES,
        "args": vars(args),
    }

    torch.save(payload, path)


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset = PRD8MaskDataset(
        TRAIN_CSV,
        training=True,
        enable_hflip=not args.no_hflip,
    )

    val_dataset = PRD8MaskDataset(
        VAL_CSV,
        training=False,
        enable_hflip=False,
    )

    check_lines = [
        "PRD 3.1.1 Mask R-CNN baseline data check",
        "========================================",
        "",
    ]

    check_lines += inspect_dataset(
        train_dataset,
        "train",
    )
    check_lines.append("")
    check_lines += inspect_dataset(
        val_dataset,
        "val",
    )

    DATA_CHECK_PATH.write_text(
        "\n".join(check_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(check_lines))

    if args.dry_run:
        print()
        print("DRY RUN PASSED.")
        print(
            "No training was started."
        )
        print(
            f"Data check: "
            f"{DATA_CHECK_PATH.relative_to(PROJECT_ROOT)}"
        )
        return

    device = choose_device(args.device)

    print()
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        min_size=args.min_size,
        max_size=args.max_size,
    )

    model.to(device)

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    log_rows = []

    print()
    print("Starting training...")

    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_start = time.perf_counter()

        running_total = 0.0
        running_components = Counter()
        batch_count = 0

        for batch_index, (images, targets, _) in enumerate(
            train_loader,
            start=1,
        ):
            images = [
                image.to(device)
                for image in images
            ]
            targets = move_targets_to_device(
                targets,
                device,
            )

            loss_dict = model(
                images,
                targets,
            )

            losses = sum(
                loss
                for loss in loss_dict.values()
            )

            if not torch.isfinite(losses):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch}, "
                    f"batch={batch_index}: "
                    f"{losses.item()}"
                )

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_value = float(losses.item())

            running_total += loss_value
            batch_count += 1

            for name, value in loss_dict.items():
                running_components[name] += float(
                    value.item()
                )

            if batch_index % 25 == 0:
                print(
                    f"epoch {epoch:02d}/{args.epochs} "
                    f"batch {batch_index:03d}/{len(train_loader)} "
                    f"loss={loss_value:.4f}"
                )

        scheduler.step()

        epoch_seconds = (
            time.perf_counter() - epoch_start
        )

        avg_total = (
            running_total / max(1, batch_count)
        )

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss_total": avg_total,
            "epoch_seconds": epoch_seconds,
        }

        for name in sorted(running_components):
            row[name] = (
                running_components[name]
                / max(1, batch_count)
            )

        log_rows.append(row)

        # Rewrite log every epoch for safety.
        all_fields = [
            "epoch",
            "learning_rate",
            "train_loss_total",
        ]

        component_fields = sorted(
            {
                key
                for r in log_rows
                for key in r
                if key not in {
                    "epoch",
                    "learning_rate",
                    "train_loss_total",
                    "epoch_seconds",
                }
            }
        )

        all_fields += component_fields
        all_fields += ["epoch_seconds"]

        with TRAIN_LOG.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=all_fields,
            )
            writer.writeheader()
            writer.writerows(log_rows)

        last_path = OUTPUT_DIR / "checkpoint_last.pth"

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            args,
        )

        if (
            epoch % args.save_every == 0
            or epoch == args.epochs
        ):
            epoch_path = (
                OUTPUT_DIR
                / f"checkpoint_epoch_{epoch:02d}.pth"
            )

            shutil.copy2(
                last_path,
                epoch_path,
            )

        print(
            f"epoch {epoch:02d}/{args.epochs} done | "
            f"avg_loss={avg_total:.4f} | "
            f"time={epoch_seconds / 60:.1f} min"
        )

    total_seconds = (
        time.perf_counter() - training_start
    )

    summary = [
        "PRD 3.1.1 Mask R-CNN 8-class baseline v1",
        "=========================================",
        "",
        f"device={device}",
        (
            f"gpu={torch.cuda.get_device_name(0)}"
            if device.type == "cuda"
            else "gpu=none"
        ),
        f"epochs={args.epochs}",
        f"batch_size={args.batch_size}",
        f"lr={args.lr}",
        f"momentum={args.momentum}",
        f"weight_decay={args.weight_decay}",
        f"min_size={args.min_size}",
        f"max_size={args.max_size}",
        f"hflip={not args.no_hflip}",
        f"seed={args.seed}",
        f"train_image_groups={len(train_dataset)}",
        f"val_image_groups={len(val_dataset)}",
        f"total_training_seconds={total_seconds:.2f}",
        "",
        "Final training loss",
        "-------------------",
        (
            f"train_loss_total={log_rows[-1]['train_loss_total']:.6f}"
            if log_rows
            else "train_loss_total=NA"
        ),
        "",
        "Important",
        "---------",
        "- This file reports training completion only.",
        "- Final PRD evaluation must be run separately on the fixed validation split.",
        "- Do not interpret training loss as IoU or category accuracy.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print()
    print("TRAINING FINISHED.")
    print(
        "Checkpoint:",
        (
            OUTPUT_DIR / "checkpoint_last.pth"
        ).relative_to(PROJECT_ROOT),
    )
    print(
        "Log:",
        TRAIN_LOG.relative_to(PROJECT_ROOT),
    )
    print(
        "Summary:",
        SUMMARY_PATH.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
