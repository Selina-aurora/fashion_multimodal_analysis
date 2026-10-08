"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。

Train a small supervised part detector for collar / cuff / hem.

Model:
    Faster R-CNN MobileNetV3-Large FPN pretrained on COCO.

Inputs:
    reports/prd_region_coverage/core_part_annotation_labels.csv
    reports/prd_region_coverage/core_part_detector_split_v1.csv

Outputs:
    models/core_part_detector_v1/best_model.pt
    reports/prd_region_coverage/core_part_detector_v1/training_history.csv
    reports/prd_region_coverage/core_part_detector_v1/best_val_metrics.csv
    reports/prd_region_coverage/core_part_detector_v1/experiment_notes.md
    outputs/core_part_detector_v1/val_predictions/
    outputs/core_part_detector_v1/val_contact_sheet.jpg

Important:
    This is a feasibility pilot, not a PRD acceptance result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms import functional as tvf

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()
REPORT_ROOT = PROJECT_ROOT / "reports" / "prd_region_coverage"

LABELS_FILE = REPORT_ROOT / "core_part_annotation_labels.csv"
SPLIT_FILE = REPORT_ROOT / "core_part_detector_split_v1.csv"

REPORT_DIR = REPORT_ROOT / "core_part_detector_v1"
MODEL_DIR = PROJECT_ROOT / "models" / "core_part_detector_v1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "core_part_detector_v1"

CLASS_TO_ID = {
    "collar": 1,
    "cuff": 2,
    "hem": 3,
}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}
NUM_CLASSES = 1 + len(CLASS_TO_ID)  # + background


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        已通过参数合法性检查的命令行配置。

    Raises:
        ValueError: --batch-size must be positive.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional smoke-test limit per epoch.",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    return args


def set_seed(seed: int) -> None:
    """Set deterministic random seeds where practical.

    Args:
        seed: 控制随机过程的种子。
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file.

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV.

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_project_path(raw_path: str | Path) -> Path:
    """将项目相对文件引用解析到当前工作副本。

    Args:
        raw_path: 对应文件的相对路径或当前解析后的路径。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(raw_path)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 for a small image file.

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        64 位十六进制 SHA-256 摘要字符串。
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_boxes(raw: str) -> list[list[float]]:
    """Parse bbox JSON as float xyxy boxes.

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        ValueError: Labeled row must contain at least one bbox.
    """
    data = json.loads(raw)
    if not isinstance(data, list) or not data:
        raise ValueError("Labeled row must contain at least one bbox.")
    return [[float(v) for v in box] for box in data]


def build_samples(split_name: str) -> list[dict[str, Any]]:
    """Build samples, merging exact duplicate images across region rows.

    Args:
        split_name: split name。

    Returns:
        返回 samples，由函数体中同名变量的计算/收集过程得到。

    Raises:
        KeyError: 所需字段不存在。
        ValueError: 输入或实验状态不符合检查条件。
        FileNotFoundError: 需要的文件不存在。
    """
    labels = {row["annotation_id"]: row for row in read_csv(LABELS_FILE)}
    split_rows = [row for row in read_csv(SPLIT_FILE) if row["split"] == split_name]

    raw_items = []
    for split_row in split_rows:
        annotation_id = split_row["annotation_id"]
        row = labels.get(annotation_id)
        if row is None:
            raise KeyError(f"Missing annotation row: {annotation_id}")
        if row["label_status"] != "labeled":
            raise ValueError(f"Split contains non-labeled row: {annotation_id}")

        image_path = resolve_project_path(row["annotation_image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        boxes = parse_boxes(row["boxes_json"])
        class_id = CLASS_TO_ID[row["region"]]

        raw_items.append(
            {
                "annotation_id": annotation_id,
                "image_name": row["image_name"],
                "image_path": image_path,
                "image_hash": sha256_file(image_path),
                "boxes": boxes,
                "labels": [class_id] * len(boxes),
            }
        )

    # Merge rows only when the underlying image bytes are exactly identical.
    grouped: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        key = item["image_hash"]
        if key not in grouped:
            grouped[key] = {
                "sample_id": item["image_name"],
                "image_name": item["image_name"],
                "image_path": item["image_path"],
                "annotation_ids": [],
                "boxes": [],
                "labels": [],
            }

        grouped[key]["annotation_ids"].append(item["annotation_id"])
        grouped[key]["boxes"].extend(item["boxes"])
        grouped[key]["labels"].extend(item["labels"])

    samples = list(grouped.values())

    print(
        f"{split_name}: {len(split_rows)} annotation rows -> "
        + f"{len(samples)} unique image samples"
    )
    return samples


class PartDetectionDataset(Dataset):
    """Small bbox detection dataset."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        train: bool,
    ) -> None:
        """保存当前对象需要的配置、数据引用和状态。

        Args:
            samples: samples。
            train: 训练。
        """
        self.samples = samples
        self.train = train

    def __len__(self) -> int:
        """返回当前数据集或容器中的样本数量。

        Returns:
            本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
        """
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """取得指定样本，并生成本数据集约定的输入和目标。

        Args:
            index: 待访问样本的整数下标。

        Returns:
            按顺序返回 image_tensor, target 等结果。
        """
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")

        boxes = torch.tensor(
            sample["boxes"],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            sample["labels"],
            dtype=torch.int64,
        )

        # Horizontal flip only: safe for collar/cuff/hem semantics.
        if self.train and random.random() < 0.5:
            width = image.width
            image = tvf.hflip(image)
            boxes = boxes.clone()
            old_x1 = boxes[:, 0].clone()
            old_x2 = boxes[:, 2].clone()
            boxes[:, 0] = width - old_x2
            boxes[:, 2] = width - old_x1

        image_tensor = tvf.pil_to_tensor(image).float() / 255.0

        area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
            boxes[:, 3] - boxes[:, 1]
        ).clamp(min=0)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(
                (len(boxes),),
                dtype=torch.int64,
            ),
        }
        return image_tensor, target


def collate_fn(batch: Any) -> Any:
    """Detection collate function.

    Args:
        batch: 一批图像与实例目标；每张图的实例数量可以不同。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return tuple(zip(*batch))


def build_model() -> torch.nn.Module:
    """Build pretrained Faster R-CNN with a 4-class predictor.

    Returns:
        返回 model，由函数体中同名变量的计算/收集过程得到。
    """
    try:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        )

        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights,
            min_size=640,
            max_size=960,
        )
    except (ImportError, TypeError):
        # Compatibility fallback for older torchvision versions.
        model = fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=True,
            min_size=640,
            max_size=960,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES,
    )
    return model


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None,
) -> float:
    """Train one epoch and return mean total loss.

    Args:
        model: 已构建的模型对象，由调用方负责选择权重。
        loader: loader。
        optimizer: 当前训练使用的优化器。
        device: 当前计算设备，与输入张量和模型设备保持一致。
        max_batches: max batches。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        RuntimeError: 当前运行条件不满足实验要求。
    """
    model.train()
    losses = []

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return sum(losses) / max(1, len(losses))


def greedy_match(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    iou_threshold: float,
) -> tuple[int, int, int, list[float]]:
    """Greedily match predictions to GT boxes for one class.

    Args:
        gt_boxes: 参考目标的 xyxy 边界框。
        pred_boxes: 模型预测的 xyxy 边界框。
        pred_scores: 模型候选的置信度。
        iou_threshold: 本步骤的 iou 判定阈值。

    Returns:
        按顺序返回 tp, fp, fn, matched_ious 等结果。
    """
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, []

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), []

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]

    ious = box_iou(gt_boxes, pred_boxes)
    used_gt = set()
    used_pred = set()
    matched_ious = []

    pairs = []
    for gt_index in range(ious.shape[0]):
        for pred_index in range(ious.shape[1]):
            pairs.append(
                (
                    float(ious[gt_index, pred_index]),
                    gt_index,
                    pred_index,
                )
            )

    for iou_value, gt_index, pred_index in sorted(
        pairs,
        reverse=True,
    ):
        if iou_value < iou_threshold:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue

        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matched_ious.append(iou_value)

    tp = len(used_gt)
    fp = len(pred_boxes) - len(used_pred)
    fn = len(gt_boxes) - len(used_gt)
    return tp, fp, fn, matched_ious


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    """Evaluate class-wise bbox matching metrics.

    Args:
        model: 已构建的模型对象，由调用方负责选择权重。
        loader: loader。
        device: 当前计算设备，与输入张量和模型设备保持一致。
        score_threshold: 保留候选的置信度阈值。
        iou_threshold: 本步骤的 iou 判定阈值。

    Returns:
        结果字典，主要字段为 rows, macro_recall, image_success_rate。
    """
    model.eval()

    class_stats = {
        class_id: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_ious": [],
        }
        for class_id in ID_TO_CLASS
    }

    total_images = 0
    successful_images = 0

    for images, targets in loader:
        images_device = [image.to(device) for image in images]
        predictions = model(images_device)

        for prediction, target in zip(predictions, targets):
            total_images += 1
            image_all_gt_matched = True

            pred_boxes = prediction["boxes"].detach().cpu()
            pred_labels = prediction["labels"].detach().cpu()
            pred_scores = prediction["scores"].detach().cpu()

            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]

            gt_boxes = target["boxes"].cpu()
            gt_labels = target["labels"].cpu()

            for class_id in ID_TO_CLASS:
                class_gt = gt_boxes[gt_labels == class_id]
                class_pred = pred_boxes[pred_labels == class_id]
                class_scores = pred_scores[pred_labels == class_id]

                tp, fp, fn, matched_ious = greedy_match(
                    class_gt,
                    class_pred,
                    class_scores,
                    iou_threshold,
                )

                stats = class_stats[class_id]
                stats["tp"] += tp
                stats["fp"] += fp
                stats["fn"] += fn
                stats["matched_ious"].extend(matched_ious)

                if len(class_gt) > 0 and fn > 0:
                    image_all_gt_matched = False

            if image_all_gt_matched:
                successful_images += 1

    metric_rows = []
    recalls = []

    for class_id, class_name in ID_TO_CLASS.items():
        stats = class_stats[class_id]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (
            2 * precision * recall / max(1e-12, precision + recall)
            if precision + recall > 0
            else 0.0
        )
        mean_iou = (
            sum(stats["matched_ious"]) / len(stats["matched_ious"])
            if stats["matched_ious"]
            else 0.0
        )

        recalls.append(recall)
        metric_rows.append(
            {
                "class": class_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_matched_iou": mean_iou,
            }
        )

    macro_recall = sum(recalls) / len(recalls)
    image_success_rate = successful_images / max(1, total_images)

    return {
        "rows": metric_rows,
        "macro_recall": macro_recall,
        "image_success_rate": image_success_rate,
    }


def draw_predictions(
    model: torch.nn.Module,
    dataset: PartDetectionDataset,
    device: torch.device,
    score_threshold: float,
) -> None:
    """Save validation prediction images and one contact sheet.

    Args:
        model: 已构建的模型对象，由调用方负责选择权重。
        dataset: 按原图分组后的实例数据集。
        device: 当前计算设备，与输入张量和模型设备保持一致。
        score_threshold: 保留候选的置信度阈值。
    """
    pred_dir = OUTPUT_DIR / "val_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    model.eval()

    for index, sample in enumerate(dataset.samples):
        image = Image.open(sample["image_path"]).convert("RGB")
        tensor, target = dataset[index]

        with torch.no_grad():
            prediction = model([tensor.to(device)])[0]

        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)

        # Ground truth: green.
        for box, label in zip(
            target["boxes"].tolist(),
            target["labels"].tolist(),
        ):
            draw.rectangle(box, outline="green", width=4)
            draw.text(
                (box[0] + 2, box[1] + 2),
                f"GT {ID_TO_CLASS[label]}",
                fill="green",
            )

        # Prediction: red.
        for box, label, score in zip(
            prediction["boxes"].detach().cpu().tolist(),
            prediction["labels"].detach().cpu().tolist(),
            prediction["scores"].detach().cpu().tolist(),
        ):
            if score < score_threshold or label not in ID_TO_CLASS:
                continue

            draw.rectangle(box, outline="red", width=3)
            draw.text(
                (box[0] + 2, max(0, box[1] - 12)),
                f"P {ID_TO_CLASS[label]} {score:.2f}",
                fill="red",
            )

        save_path = pred_dir / f"{index:02d}_{sample['image_name']}"
        canvas.save(save_path, quality=92)

        tile = Image.new("RGB", (480, 520), "white")
        preview = canvas.copy()
        preview.thumbnail((470, 500))
        x = (480 - preview.width) // 2
        y = (520 - preview.height) // 2
        tile.paste(preview, (x, y))
        panels.append(tile)

    if not panels:
        return

    columns = 2
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new("RGB", (480 * columns, 520 * rows), "white")

    for index, panel in enumerate(panels):
        x = (index % columns) * 480
        y = (index // columns) * 520
        sheet.paste(panel, (x, y))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT_DIR / "val_contact_sheet.jpg", quality=92)


def write_notes(args: argparse.Namespace, device: torch.device) -> None:
    """Write reproducibility notes.

    Args:
        args: 命令行配置；使用 batch_size, epochs, iou_threshold, lr, score_threshold, seed,
        weight_decay 等参数。
        device: 当前计算设备，与输入张量和模型设备保持一致。
    """
    text = f"""# Core part detector v1

## Purpose

Small-supervision feasibility pilot for part-level bbox localization.

## Classes

- collar
- cuff
- hem

## Model

`fasterrcnn_mobilenet_v3_large_fpn` pretrained on COCO.

## Training

- epochs: {args.epochs}
- batch size: {args.batch_size}
- learning rate: {args.lr}
- weight decay: {args.weight_decay}
- score threshold: {args.score_threshold}
- IoU threshold: {args.iou_threshold}
- seed: {args.seed}
- device: {device}

## Data policy

- Only manually labeled rows are used.
- Unusable rows are excluded.
- `hem_train_03` is conservatively excluded from v1 because its semantic
  target remains visually atypical/ambiguous.
- Validation is grouped by original image name to avoid image leakage.
- Exact duplicate image files across region annotations are merged at runtime.
- Horizontal flip is the only augmentation.

## Interpretation

This is a feasibility experiment. The frozen 15-case PRD diagnostic set is
not used for training or model selection and remains reserved for later
held-out evaluation.
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "experiment_notes.md").write_text(
        text,
        encoding="utf-8",
    )


def main() -> None:
    """Train and validate the core part detector."""
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_samples = build_samples("train")
    val_samples = build_samples("val")

    train_dataset = PartDetectionDataset(
        train_samples,
        train=True,
    )
    val_dataset = PartDetectionDataset(
        val_samples,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = build_model().to(device)

    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(5, args.epochs // 2),
        gamma=0.1,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history = []
    best_score = (-1.0, -1.0)
    best_epoch = -1
    best_train_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_train_batches,
        )

        metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
        )

        macro_recall = metrics["macro_recall"]
        image_success = metrics["image_success_rate"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            + f"loss={train_loss:.4f} | "
            + f"macro_recall@0.5={macro_recall:.3f} | "
            + f"image_success={image_success:.3f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "macro_recall_at_iou_0_5": macro_recall,
                "image_success_rate": image_success,
            }
        )

        score = (macro_recall, image_success)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_id": CLASS_TO_ID,
                    "epoch": epoch,
                    "macro_recall": macro_recall,
                    "image_success_rate": image_success,
                    "train_loss": train_loss,
                    "args": vars(args),
                },
                MODEL_DIR / "best_model.pt",
            )

        # Always save the most recent epoch, even if validation metrics do not improve.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_id": CLASS_TO_ID,
                "epoch": epoch,
                "macro_recall": macro_recall,
                "image_success_rate": image_success,
                "train_loss": train_loss,
                "args": vars(args),
            },
            MODEL_DIR / "last_model.pt",
        )

        # Diagnostic checkpoint only: preserve the epoch with the lowest training loss.
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_id": CLASS_TO_ID,
                    "epoch": epoch,
                    "macro_recall": macro_recall,
                    "image_success_rate": image_success,
                    "train_loss": train_loss,
                    "args": vars(args),
                },
                MODEL_DIR / "best_loss_model.pt",
            )

        scheduler.step()

    write_csv(REPORT_DIR / "training_history.csv", history)

    checkpoint = torch.load(
        MODEL_DIR / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    best_metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
    )

    metric_rows = []
    for row in best_metrics["rows"]:
        metric_rows.append(
            {
                "best_epoch": best_epoch,
                "macro_recall_at_iou_0_5": best_metrics["macro_recall"],
                "image_success_rate": best_metrics["image_success_rate"],
                **row,
            }
        )

    write_csv(
        REPORT_DIR / "best_val_metrics.csv",
        metric_rows,
    )

    draw_predictions(
        model=model,
        dataset=val_dataset,
        device=device,
        score_threshold=args.score_threshold,
    )
    write_notes(args, device)

    print("\nFinished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best macro recall@0.5: " + f"{best_metrics['macro_recall']:.3f}")
    print(f"Best image success rate: " + f"{best_metrics['image_success_rate']:.3f}")
    print(f"Best model: {MODEL_DIR / 'best_model.pt'}")
    print(f"Last model: {MODEL_DIR / 'last_model.pt'}")
    print(f"Best-loss model: {MODEL_DIR / 'best_loss_model.pt'}")
    print(f"Metrics: {REPORT_DIR / 'best_val_metrics.csv'}")
    print(f"Visual: {OUTPUT_DIR / 'val_contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
