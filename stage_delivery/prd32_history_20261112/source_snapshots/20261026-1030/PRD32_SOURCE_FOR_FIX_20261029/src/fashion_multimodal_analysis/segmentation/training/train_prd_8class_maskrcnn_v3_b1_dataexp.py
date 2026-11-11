"""3.1.1 实例分割：数据与模型版本分别记录，指标按固定匹配和计时口径解释。

Train PRD 3.1.1 Mask R-CNN v3-b1 data expansion using the expanded train_v3 manifest.

Controlled-comparison design
----------------------------
Compared with baseline v1:
- SAME model architecture: Mask R-CNN ResNet50-FPN
- SAME pretrained initialization: COCO DEFAULT weights
- SAME fixed validation split: configs/prd_8class_val_v1.csv
- SAME optimizer / LR schedule / image resize / augmentation defaults
- CHANGED training data:
    DeepFashion2 train rows kept from v1
    Fashionpedia shoe / bag / accessory expanded to 100 valid instances each
- CHANGED sampler:
    class-aware image-group sampler with replacement

The sampler uses a damped inverse-frequency weight:
    image_weight = max((max_class_count / class_count)^alpha)
over the PRD classes present in that image.
Default alpha = 0.5 (square-root balancing), which reduces extreme oversampling.

Inputs
------
configs/prd_8class_train_v3.csv
configs/prd_8class_val_v1.csv

Outputs
-------
outputs/prd_instance_segmentation/maskrcnn_8class_v3_b1_dataexp/
    checkpoint_epoch_XX.pth
    checkpoint_last.pth

reports/prd_instance_segmentation/maskrcnn_8class_v3_b1_dataexp/
    data_check.txt
    sampler_weights.csv
    train_log.csv
    summary.txt

Recommended:
    python scripts/segmentation/training/train_prd_8class_maskrcnn_v3_b1_dataexp.py
    --dry-run

Then:
    python scripts/segmentation/training/train_prd_8class_maskrcnn_v3_b1_dataexp.py
    --epochs 15         --device cuda

Important
---------
For a fair v1-v2 comparison, evaluate v2 on the SAME val_v1 split and use the
frozen development score threshold 0.40 selected from the v1 threshold sweep.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)
from fashion_multimodal_analysis.datasets.prd8 import (
    PRD8MaskDataset,
    collate_fn,
    group_manifest,
    read_manifest,
)
from fashion_multimodal_analysis.image_processing.masks import (
    open_binary_mask,
    reconstruct_full_mask,
)
from fashion_multimodal_analysis.segmentation.modeling import (
    build_training_model as build_model,
)
from fashion_multimodal_analysis.segmentation.sampling import (
    build_sampler_weights,
    instance_class_counts,
)

PROJECT_ROOT = get_project_root()

TRAIN_CSV = PROJECT_ROOT / "configs" / "prd_8class_train_v3.csv"
VAL_CSV = PROJECT_ROOT / "configs" / "prd_8class_val_v1.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_instance_segmentation"
    / "maskrcnn_8class_v3_b1_dataexp"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_instance_segmentation"
    / "maskrcnn_8class_v3_b1_dataexp"
)

TRAIN_LOG = REPORT_DIR / "train_log.csv"
SUMMARY_PATH = REPORT_DIR / "summary.txt"
DATA_CHECK_PATH = REPORT_DIR / "data_check.txt"
SAMPLER_WEIGHTS_PATH = REPORT_DIR / "sampler_weights.csv"

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

NUM_CLASSES = 1 + len(CLASS_TO_ID)


def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    p = argparse.ArgumentParser()

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)

    p.add_argument("--lr", type=float, default=0.0025)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--step-size", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.1)

    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    p.add_argument("--seed", type=int, default=20260921)

    p.add_argument("--min-size", type=int, default=640)
    p.add_argument("--max-size", type=int, default=1024)

    p.add_argument("--save-every", type=int, default=5)

    p.add_argument(
        "--sampler-alpha",
        type=float,
        default=0.5,
        help=(
            "Damped inverse-frequency exponent. "
            + "0=no balancing, 0.5=sqrt balancing, 1=full inverse-frequency."
        ),
    )

    p.add_argument(
        "--no-balanced-sampler",
        action="store_true",
        help="Disable class-aware weighted image-group sampling.",
    )

    p.add_argument(
        "--no-hflip",
        action="store_true",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests, masks and sampler only; do not train.",
    )

    p.add_argument("--train-csv", default=str(TRAIN_CSV))
    p.add_argument("--val-csv", default=str(VAL_CSV))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--report-dir", default=str(REPORT_DIR))
    p.add_argument("--experiment-name", default="v3-b1-dataexp-historical")
    p.add_argument(
        "--verify-isolation",
        action="store_true",
        help="Reject training images used by frozen Core or validation",
    )
    p.add_argument(
        "--resume",
        default="",
        help="Resume a matching checkpoint from its last completed epoch",
    )
    return p.parse_args()


def seed_everything(seed: int) -> None:
    """固定随机种子，便于对照实验复现相同随机过程。

    Args:
        seed: 控制随机过程的种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_path(value: Any) -> Path:
    """将项目相对文件引用解析到当前工作副本。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(value)


def choose_device(requested: str) -> torch.device:
    """选择可用计算设备，并按本实验策略处理 CUDA 可用性。

    Args:
        requested: requested。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        RuntimeError: --device cuda requested, but CUDA is unavailable.
    """
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str] | None = None,
) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
        fieldnames: fieldnames。
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        if not fieldnames:
            return

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def inspect_dataset(
    dataset: PRD8MaskDataset,
    name: str,
) -> list[str]:
    """检查训练样本与类别分布，在耗时训练前发现数据问题。

    Args:
        dataset: 按原图分组后的实例数据集。
        name: 当前标签、字段或产物名称。

    Returns:
        返回 lines，由函数体中同名变量的计算/收集过程得到。
    """
    lines = [
        f"{name}: image_groups={len(dataset)}",
        f"{name}: instances={len(dataset.rows)}",
    ]

    source_counts = Counter(sample["source_dataset"] for sample in dataset.samples)

    lines.append(f"{name}: source_image_groups={dict(source_counts)}")

    counts = instance_class_counts(dataset)

    for cls in CLASS_TO_ID:
        lines.append(f"{name}: {cls}={counts.get(cls, 0)}")

    checked = 0
    checked_sources = set()

    for idx, sample in enumerate(dataset.samples):
        source = sample["source_dataset"]

        if source in checked_sources and checked >= 4:
            continue

        image, target, meta = dataset[idx]

        lines.append(
            f"loaded[{name}][{idx}]: "
            + f"source={source}, "
            + f"shape={tuple(image.shape)}, "
            + f"instances={len(target['labels'])}, "
            + f"mask_shape={tuple(target['masks'].shape)}"
        )

        checked += 1
        checked_sources.add(source)

        if checked >= 6:
            break

    return lines


def move_targets_to_device(
    targets: Any,
    device: Any,
) -> list[Any]:
    """把实例标注张量移动到与模型一致的设备。

    Args:
        targets: 当前批次的实例目标字典/张量。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return [
        {key: value.to(device) for key, value in target.items()} for target in targets
    ]


def save_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    args: argparse.Namespace,
    log_rows: list[dict[str, Any]] | None = None,
    sampler_generator: Any = None,
) -> None:
    """保存权重和训练状态，供续训或指定版本评估。

    Args:
        path: 要读取或写入的文件路径。
        model: 已构建的模型对象，由调用方负责选择权重。
        optimizer: 当前训练使用的优化器。
        scheduler: 当前训练使用的学习率调度器。
        epoch: 当前训练轮次。
        args: 已经解析的命令行配置。
        log_rows: 待处理的 log 记录。
        sampler_generator: 采样器 generator。
    """
    from fashion_multimodal_analysis.segmentation.training.checkpoint_state import (
        atomic_save,
        capture_rng_state,
    )

    atomic_save(
        {
            "checkpoint_format": 2,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_to_id": CLASS_TO_ID,
            "num_classes": NUM_CLASSES,
            "args": vars(args),
            "train_manifest": str(TRAIN_CSV),
            "val_manifest": str(VAL_CSV),
            "train_manifest_sha256": hashlib.sha256(TRAIN_CSV.read_bytes()).hexdigest(),
            "val_manifest_sha256": hashlib.sha256(VAL_CSV.read_bytes()).hexdigest(),
            "rng_state": capture_rng_state(torch, sampler_generator),
            "epoch_log": [dict(row) for row in (log_rows or [])],
        },
        path,
        torch,
    )


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
        FileExistsError: A checkpoint already exists. Use --resume or choose a new
        --output-dir and --report-d
        RuntimeError: 当前运行条件不满足实验要求。
    """
    global TRAIN_CSV, VAL_CSV, OUTPUT_DIR, REPORT_DIR, TRAIN_LOG, SUMMARY_PATH, DATA_CHECK_PATH, SAMPLER_WEIGHTS_PATH
    args = parse_args()
    TRAIN_CSV = resolve_project_path(args.train_csv)
    VAL_CSV = resolve_project_path(args.val_csv)
    OUTPUT_DIR = resolve_project_path(args.output_dir)
    REPORT_DIR = resolve_project_path(args.report_dir)
    TRAIN_LOG = REPORT_DIR / "train_log.csv"
    SUMMARY_PATH = REPORT_DIR / "summary.txt"
    DATA_CHECK_PATH = REPORT_DIR / "data_check.txt"
    SAMPLER_WEIGHTS_PATH = REPORT_DIR / "sampler_weights.csv"
    if args.verify_isolation:
        from fashion_multimodal_analysis.analysis.audit_dataset_overlap import image_key
        from fashion_multimodal_analysis.common.io import read_csv

        heldout = {
            image_key(r["source_image"])
            for p in [
                VAL_CSV,
                PROJECT_ROOT
                / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv",
            ]
            for r in read_csv(p)
        }
        overlaps = {image_key(r["source_image"]) for r in read_csv(TRAIN_CSV)} & heldout
        if overlaps:
            raise ValueError(
                f"Training isolation failed: {len(overlaps)} held-out source images"
            )

    from fashion_multimodal_analysis.segmentation.training.checkpoint_state import (
        CONTROLLED_SETTINGS,
        restore_rng_state,
        validate_checkpoint,
    )

    resume_checkpoint = None
    start_epoch = 1
    if args.resume:
        resume_path = resolve_project_path(args.resume)
        destination_checkpoint = OUTPUT_DIR / "checkpoint_last.pth"
        if (
            destination_checkpoint.is_file()
            and destination_checkpoint.resolve() != resume_path.resolve()
        ):
            if (
                hashlib.sha256(destination_checkpoint.read_bytes()).digest()
                != hashlib.sha256(resume_path.read_bytes()).digest()
            ):
                raise FileExistsError(
                    "Resume destination already contains a different checkpoint; use a new output/report directory"
                )
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        completed = validate_checkpoint(
            resume_checkpoint,
            hashlib.sha256(TRAIN_CSV.read_bytes()).hexdigest(),
            hashlib.sha256(VAL_CSV.read_bytes()).hexdigest(),
            settings={key: getattr(args, key) for key in CONTROLLED_SETTINGS},
            epochs=args.epochs,
            for_resume=True,
        )
        start_epoch = completed + 1
    elif not args.dry_run and (OUTPUT_DIR / "checkpoint_last.pth").exists():
        raise FileExistsError(
            "A checkpoint already exists. Use --resume or choose a new --output-dir and --report-dir"
        )

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

    image_weights, sampler_rows = build_sampler_weights(
        train_dataset,
        alpha=args.sampler_alpha,
    )

    write_csv(
        SAMPLER_WEIGHTS_PATH,
        sampler_rows,
        [
            "dataset_index",
            "source_dataset",
            "source_image",
            "instance_count",
            "classes",
            "sampler_weight",
        ],
    )

    check_lines = [
        f"PRD 3.1.1 Mask R-CNN {args.experiment_name} data check",
        "============================================",
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

    check_lines += [
        "",
        f"balanced_sampler={not args.no_balanced_sampler}",
        f"sampler_alpha={args.sampler_alpha}",
        f"sampler_weight_min={min(image_weights):.6f}",
        f"sampler_weight_max={max(image_weights):.6f}",
        f"fixed_validation_manifest={VAL_CSV.relative_to(PROJECT_ROOT)}",
    ]

    DATA_CHECK_PATH.write_text(
        "\n".join(check_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(check_lines))

    if args.dry_run:
        print()
        print("DRY RUN PASSED.")
        print("No training was started.")
        print(
            "Sampler report:",
            SAMPLER_WEIGHTS_PATH.relative_to(PROJECT_ROOT),
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

    sampler = None
    generator = None

    if not args.no_balanced_sampler:
        generator = torch.Generator()
        generator.manual_seed(args.seed)

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(
                image_weights,
                dtype=torch.double,
            ),
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        min_size=args.min_size,
        max_size=args.max_size,
    )

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

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
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        log_rows = [dict(row) for row in resume_checkpoint["epoch_log"]]
        restore_rng_state(resume_checkpoint["rng_state"], torch, generator)
        del resume_checkpoint
        print(
            f"Resuming from completed epoch {start_epoch - 1}; next epoch {start_epoch}/{args.epochs}",
            flush=True,
        )

    print()
    print(f"Starting {args.experiment_name} training...")

    training_start = time.perf_counter()

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        model.train()

        epoch_start = time.perf_counter()

        running_total = 0.0
        running_components = Counter()
        batch_count = 0
        sampled_source_counts = Counter()
        sampled_class_counts = Counter()

        for batch_index, (
            images,
            targets,
            metas,
        ) in enumerate(
            train_loader,
            start=1,
        ):
            for meta in metas:
                sampled_source_counts[meta["source_dataset"]] += 1

                for cls in meta["classes"]:
                    sampled_class_counts[cls] += 1

            images = [image.to(device) for image in images]

            targets = move_targets_to_device(
                targets,
                device,
            )

            loss_dict = model(
                images,
                targets,
            )

            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch}, "
                    + f"batch={batch_index}: {losses.item()}"
                )

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_value = float(losses.item())

            running_total += loss_value
            batch_count += 1

            for name, value in loss_dict.items():
                running_components[name] += float(value.item())

            if batch_index % 50 == 0:
                print(
                    f"epoch {epoch:02d}/{args.epochs} "
                    + f"batch {batch_index:03d}/{len(train_loader)} "
                    + f"loss={loss_value:.4f}"
                )

        scheduler.step()

        epoch_seconds = time.perf_counter() - epoch_start

        avg_total = running_total / max(1, batch_count)

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss_total": avg_total,
            "epoch_seconds": epoch_seconds,
            "sampled_DeepFashion2_images": sampled_source_counts.get(
                "DeepFashion2",
                0,
            ),
            "sampled_Fashionpedia_images": sampled_source_counts.get(
                "Fashionpedia",
                0,
            ),
        }

        for cls in CLASS_TO_ID:
            row[f"sampled_images_with_{cls}"] = sampled_class_counts.get(
                cls,
                0,
            )

        for name in sorted(running_components):
            row[name] = running_components[name] / max(1, batch_count)

        log_rows.append(row)

        all_fields = [
            "epoch",
            "learning_rate",
            "train_loss_total",
            "epoch_seconds",
            "sampled_DeepFashion2_images",
            "sampled_Fashionpedia_images",
        ]

        all_fields += [f"sampled_images_with_{cls}" for cls in CLASS_TO_ID]

        component_fields = sorted(
            {key for r in log_rows for key in r if key not in all_fields}
        )

        all_fields += component_fields

        write_csv(
            TRAIN_LOG,
            log_rows,
            all_fields,
        )

        last_path = OUTPUT_DIR / "checkpoint_last.pth"

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            args,
            log_rows,
            generator,
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            epoch_path = OUTPUT_DIR / f"checkpoint_epoch_{epoch:02d}.pth"

            shutil.copy2(
                last_path,
                epoch_path,
            )

        print(
            f"epoch {epoch:02d}/{args.epochs} done | "
            + f"avg_loss={avg_total:.4f} | "
            + f"time={epoch_seconds / 60:.1f} min"
        )

    total_seconds = time.perf_counter() - training_start

    train_counts = instance_class_counts(train_dataset)

    summary = [
        f"PRD 3.1.1 Mask R-CNN 8-class {args.experiment_name}",
        "=========================================",
        "",
        "[Controlled comparison]",
        "initialization=COCO DEFAULT (same as v1; no warm-start from v1)",
        f"train_manifest={TRAIN_CSV.relative_to(PROJECT_ROOT)}",
        f"fixed_val_manifest={VAL_CSV.relative_to(PROJECT_ROOT)}",
        f"balanced_sampler={not args.no_balanced_sampler}",
        f"sampler_alpha={args.sampler_alpha}",
        "",
        "[Training setup]",
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
        f"step_size={args.step_size}",
        f"gamma={args.gamma}",
        f"min_size={args.min_size}",
        f"max_size={args.max_size}",
        f"hflip={not args.no_hflip}",
        f"seed={args.seed}",
        f"resumed_from_completed_epoch={start_epoch - 1}",
        "",
        "[Dataset]",
        f"train_instances={len(train_dataset.rows)}",
        f"train_image_groups={len(train_dataset)}",
        f"val_instances={len(val_dataset.rows)}",
        f"val_image_groups={len(val_dataset)}",
    ]

    for cls in CLASS_TO_ID:
        summary.append(f"train_{cls}={train_counts.get(cls, 0)}")

    summary += [
        "",
        "[Result]",
        (
            f"final_train_loss={log_rows[-1]['train_loss_total']:.6f}"
            if log_rows
            else "final_train_loss=NA"
        ),
        f"total_training_seconds={total_seconds:.2f}",
        f"all_completed_epoch_seconds={sum(float(row['epoch_seconds']) for row in log_rows):.2f}",
        "",
        "[Next evaluation]",
        "Use the SAME val_v1 split.",
        "Use score_threshold=0.40 as the frozen v1-selected development threshold.",
        "Do not choose a new threshold from v2 validation results before the primary v1-v2 comparison.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"{args.experiment_name} TRAINING FINISHED.")
    print(
        "Checkpoint:",
        (OUTPUT_DIR / "checkpoint_last.pth").relative_to(PROJECT_ROOT),
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
