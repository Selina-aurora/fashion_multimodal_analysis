"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。

Train-vs-validation diagnosis for core_part_detector_v1.

Purpose
-------
Determine whether the 5-epoch supervised detector:
1) can fit the TRAIN set but fails to generalize to VAL, or
2) also fails on TRAIN, indicating a detector/training/localization issue.

The same saved model is evaluated on TRAIN and VAL at four external
confidence thresholds: 0.25, 0.05, 0.01, 0.001.

Important
---------
The torchvision Faster R-CNN internal score threshold is forced to 0.0
during diagnosis so that low-confidence predictions are not discarded
before the external thresholds below are applied.

Expected checkpoint
-------------------
models/core_part_detector_v1/best_loss_model.pt

Inputs
------
reports/prd_region_coverage/core_part_annotation_labels.csv
reports/prd_region_coverage/core_part_detector_split_v1.csv

Outputs
-------
reports/prd_region_coverage/core_part_detector_train_vs_val_diagnostic/
    checkpoint_info.csv
    metrics_by_split_threshold.csv
    per_gt_diagnostic.csv
    experiment_notes.md

outputs/core_part_detector_train_vs_val_diagnostic/
    train/
        threshold_0p25/
        threshold_0p05/
        threshold_0p01/
        threshold_0p001/
    val/
        threshold_0p25/
        threshold_0p05/
        threshold_0p01/
        threshold_0p001/

Each threshold folder contains per-image visualizations and a contact sheet.

Visualization
-------------
green = ground-truth box
red   = prediction surviving the current threshold
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
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

MODEL_FILE = PROJECT_ROOT / "models" / "core_part_detector_v1" / "best_loss_model.pt"

REPORT_DIR = REPORT_ROOT / "core_part_detector_train_vs_val_diagnostic"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "core_part_detector_train_vs_val_diagnostic"

CLASS_TO_ID = {
    "collar": 1,
    "cuff": 2,
    "hem": 3,
}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}
NUM_CLASSES = 1 + len(CLASS_TO_ID)

THRESHOLDS = [0.25, 0.05, 0.01, 0.001]
IOU_THRESHOLD = 0.50

# Keep many low-confidence detections for diagnosis.
INTERNAL_SCORE_THRESHOLD = 0.0
DETECTIONS_PER_IMAGE = 300


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
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
    """分块计算文件 SHA-256，用于运行与备份追溯。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        64 位十六进制 SHA-256 摘要字符串。
    """
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def parse_boxes(
    raw: str,
) -> list[list[float]]:
    """解析 边界框。

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

    return [[float(value) for value in box] for box in data]


def build_samples(
    split_name: str,
) -> list[dict[str, Any]]:
    """构建 samples。

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
            raise ValueError("Split contains non-labeled row: " + f"{annotation_id}")

        image_path = resolve_project_path(row["annotation_image_path"])

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        boxes = parse_boxes(row["boxes_json"])

        region = row["region"]
        if region not in CLASS_TO_ID:
            raise ValueError(f"Unknown region: {region}")

        class_id = CLASS_TO_ID[region]

        raw_items.append(
            {
                "annotation_id": annotation_id,
                "image_name": row["image_name"],
                "image_path": image_path,
                "image_hash": sha256_file(image_path),
                "boxes": boxes,
                "labels": ([class_id] * len(boxes)),
            }
        )

    # Merge annotations only when image bytes are identical.
    grouped: dict[
        str,
        dict[str, Any],
    ] = {}

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
        f"{split_name}: "
        + f"{len(split_rows)} annotation rows -> "
        + f"{len(samples)} unique image samples"
    )

    return samples


class DiagnosticDataset(Dataset):
    """No augmentation: diagnosis must use original images."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
    ) -> None:
        """保存当前对象需要的配置、数据引用和状态。

        Args:
            samples: samples。
        """
        self.samples = samples

    def __len__(self) -> int:
        """返回当前数据集或容器中的样本数量。

        Returns:
            本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
        """
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
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

        image_tensor = tvf.pil_to_tensor(image).float() / 255.0

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image_tensor, target


def build_model() -> torch.nn.Module:
    """Build architecture only; full trained state is loaded next.

    Returns:
        返回 model，由函数体中同名变量的计算/收集过程得到。
    """
    try:
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone=None,
            min_size=640,
            max_size=960,
        )
    except TypeError:
        model = fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=False,
            pretrained_backbone=False,
            min_size=640,
            max_size=960,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES,
    )

    # Critical for low-score diagnosis.
    model.roi_heads.score_thresh = INTERNAL_SCORE_THRESHOLD
    model.roi_heads.detections_per_img = DETECTIONS_PER_IMAGE

    return model


def load_checkpoint(
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """加载 checkpoint。

    Args:
        model: 已构建的模型对象，由调用方负责选择权重。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {MODEL_FILE}")

    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint


def safe_class_name(
    class_id: int,
) -> str:
    """可靠处理 类别 name。

    Args:
        class_id: 类别 ID。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return ID_TO_CLASS.get(
        int(class_id),
        f"class_{class_id}",
    )


def threshold_tag(
    threshold: float,
) -> str:
    """阈值 tag。

    Args:
        threshold: 当前判定规则使用的阈值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    text = f"{threshold:.3f}".rstrip("0").rstrip(".")

    return text.replace(".", "p")


def greedy_match(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    iou_threshold: float,
) -> tuple[
    int,
    int,
    int,
    list[float],
]:
    """greedy 匹配。

    Args:
        gt_boxes: 参考目标的 xyxy 边界框。
        pred_boxes: 模型预测的 xyxy 边界框。
        pred_scores: 模型候选的置信度。
        iou_threshold: 本步骤的 iou 判定阈值。

    Returns:
        按顺序返回 tp, fp, fn, matched_ious 等结果。
    """
    if len(gt_boxes) == 0:
        return (
            0,
            len(pred_boxes),
            0,
            [],
        )

    if len(pred_boxes) == 0:
        return (
            0,
            0,
            len(gt_boxes),
            [],
        )

    order = torch.argsort(
        pred_scores,
        descending=True,
    )

    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    ious = box_iou(
        gt_boxes,
        pred_boxes,
    )

    pairs = []

    for gt_index in range(ious.shape[0]):
        for pred_index in range(ious.shape[1]):
            pairs.append(
                (
                    float(
                        ious[
                            gt_index,
                            pred_index,
                        ]
                    ),
                    gt_index,
                    pred_index,
                )
            )

    used_gt = set()
    used_pred = set()
    matched_ious = []

    for (
        iou_value,
        gt_index,
        pred_index,
    ) in sorted(
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

    return (
        tp,
        fp,
        fn,
        matched_ious,
    )


def best_prediction_for_gt(
    gt_box: torch.Tensor,
    gt_label: int,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
) -> dict[str, Any]:
    """best 预测 for gt。

    Args:
        gt_box: gt 边界框。
        gt_label: gt 标签。
        pred_boxes: 模型预测的 xyxy 边界框。
        pred_labels: 模型预测的类别 ID。
        pred_scores: 模型候选的置信度。

    Returns:
        结果字典包含 same_class_found, same_class_best_iou, same_class_score,
        same_class_box_json, any_class_found, any_class_best_iou, any_class_score,
        any_class_label, any_class_box_json。
    """
    result = {
        "same_class_found": "no",
        "same_class_best_iou": 0.0,
        "same_class_score": "",
        "same_class_box_json": "",
        "any_class_found": "no",
        "any_class_best_iou": 0.0,
        "any_class_score": "",
        "any_class_label": "",
        "any_class_box_json": "",
    }

    if len(pred_boxes) == 0:
        return result

    ious = box_iou(
        gt_box.unsqueeze(0),
        pred_boxes,
    )[0]

    best_any_index = int(torch.argmax(ious).item())

    result["any_class_found"] = "yes"
    result["any_class_best_iou"] = float(ious[best_any_index].item())
    result["any_class_score"] = float(pred_scores[best_any_index].item())
    result["any_class_label"] = safe_class_name(int(pred_labels[best_any_index].item()))
    result["any_class_box_json"] = json.dumps(pred_boxes[best_any_index].tolist())

    same_mask = pred_labels == gt_label

    if bool(same_mask.any()):
        same_boxes = pred_boxes[same_mask]
        same_scores = pred_scores[same_mask]

        same_ious = box_iou(
            gt_box.unsqueeze(0),
            same_boxes,
        )[0]

        best_same_index = int(torch.argmax(same_ious).item())

        result["same_class_found"] = "yes"
        result["same_class_best_iou"] = float(same_ious[best_same_index].item())
        result["same_class_score"] = float(same_scores[best_same_index].item())
        result["same_class_box_json"] = json.dumps(same_boxes[best_same_index].tolist())

    return result


def classify_gt_failure(
    diagnostic: dict[str, Any],
) -> str:
    """classify gt 失败。

    Args:
        diagnostic: 记录字段，使用 any_class_label, same_class_found, same_class_best_iou,
        any_class_best_iou, any_class_found。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if diagnostic["same_class_found"] == "no":
        if (
            diagnostic["any_class_found"] == "yes"
            and float(diagnostic["any_class_best_iou"]) >= IOU_THRESHOLD
        ):
            return "wrong_class_at_good_iou"

        return "no_same_class_prediction"

    same_iou = float(diagnostic["same_class_best_iou"])

    if same_iou >= IOU_THRESHOLD:
        return "would_count_as_tp"

    any_iou = float(diagnostic["any_class_best_iou"])

    any_label = diagnostic["any_class_label"]

    if (
        any_iou >= IOU_THRESHOLD
        and any_label != ""
        and any_label
        != diagnostic.get(
            "gt_class",
            "",
        )
    ):
        return "wrong_class_at_good_iou"

    return "iou_below_0.5"


def predict_dataset(
    model: torch.nn.Module,
    dataset: DiagnosticDataset,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """预测 数据集。

    Args:
        model: 已构建的模型对象，由调用方负责选择权重。
        dataset: 按原图分组后的实例数据集。
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        返回 predictions，由函数体中同名变量的计算/收集过程得到。
    """
    predictions = []

    model.eval()

    with torch.no_grad():
        for index in range(len(dataset)):
            tensor, _ = dataset[index]

            prediction = model([tensor.to(device)])[0]

            predictions.append(
                {
                    "boxes": (prediction["boxes"].detach().cpu()),
                    "labels": (prediction["labels"].detach().cpu()),
                    "scores": (prediction["scores"].detach().cpu()),
                }
            )

    return predictions


def filter_prediction(
    prediction: dict[str, torch.Tensor],
    score_threshold: float,
) -> dict[str, torch.Tensor]:
    """筛选 预测。

    Args:
        prediction: 记录字段，使用 scores, boxes, labels。
        score_threshold: 保留候选的置信度阈值。

    Returns:
        结果字典，主要字段为 boxes, labels, scores。
    """
    keep = prediction["scores"] >= score_threshold

    return {
        "boxes": (prediction["boxes"][keep]),
        "labels": (prediction["labels"][keep]),
        "scores": (prediction["scores"][keep]),
    }


def evaluate_split_threshold(
    split_name: str,
    dataset: DiagnosticDataset,
    raw_predictions: list[dict[str, torch.Tensor]],
    score_threshold: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """评估 split 阈值。

    Args:
        split_name: split name。
        dataset: 按原图分组后的实例数据集。
        raw_predictions: raw 预测。
        score_threshold: 保留候选的置信度阈值。

    Returns:
        按顺序返回 metric_rows, per_gt_rows 等结果。
    """
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
    per_gt_rows = []

    for index, sample in enumerate(dataset.samples):
        _, target = dataset[index]

        prediction = filter_prediction(
            raw_predictions[index],
            score_threshold,
        )

        pred_boxes = prediction["boxes"]
        pred_labels = prediction["labels"]
        pred_scores = prediction["scores"]

        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        total_images += 1
        image_all_gt_matched = True

        for class_id in ID_TO_CLASS:
            class_gt = gt_boxes[gt_labels == class_id]
            class_pred = pred_boxes[pred_labels == class_id]
            class_scores = pred_scores[pred_labels == class_id]

            (
                tp,
                fp,
                fn,
                matched_ious,
            ) = greedy_match(
                class_gt,
                class_pred,
                class_scores,
                IOU_THRESHOLD,
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

        for gt_index, (
            gt_box,
            gt_label,
        ) in enumerate(
            zip(
                gt_boxes,
                gt_labels,
            ),
            start=1,
        ):
            diagnostic = best_prediction_for_gt(
                gt_box=gt_box,
                gt_label=int(gt_label.item()),
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
            )

            diagnostic["gt_class"] = safe_class_name(int(gt_label.item()))

            failure_mode = classify_gt_failure(diagnostic)

            per_gt_rows.append(
                {
                    "split": split_name,
                    "score_threshold": (score_threshold),
                    "image_name": (sample["image_name"]),
                    "annotation_ids": ";".join(sample["annotation_ids"]),
                    "gt_index": gt_index,
                    "gt_class": (safe_class_name(int(gt_label.item()))),
                    "gt_box_json": (json.dumps(gt_box.tolist())),
                    "num_predictions": (len(pred_boxes)),
                    "same_class_found": (diagnostic["same_class_found"]),
                    "same_class_best_iou": (diagnostic["same_class_best_iou"]),
                    "same_class_score": (diagnostic["same_class_score"]),
                    "same_class_box_json": (diagnostic["same_class_box_json"]),
                    "any_class_found": (diagnostic["any_class_found"]),
                    "any_class_best_iou": (diagnostic["any_class_best_iou"]),
                    "any_class_score": (diagnostic["any_class_score"]),
                    "any_class_label": (diagnostic["any_class_label"]),
                    "any_class_box_json": (diagnostic["any_class_box_json"]),
                    "failure_mode": (failure_mode),
                }
            )

    metric_rows = []
    recalls = []

    for (
        class_id,
        class_name,
    ) in ID_TO_CLASS.items():
        stats = class_stats[class_id]

        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        precision = tp / max(
            1,
            tp + fp,
        )

        recall = tp / max(
            1,
            tp + fn,
        )

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        if stats["matched_ious"]:
            mean_iou = sum(stats["matched_ious"]) / len(stats["matched_ious"])
        else:
            mean_iou = 0.0

        recalls.append(recall)

        metric_rows.append(
            {
                "split": split_name,
                "score_threshold": (score_threshold),
                "class": class_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_matched_iou": (mean_iou),
                "macro_recall": "",
                "image_success_rate": "",
            }
        )

    macro_recall = sum(recalls) / len(recalls)

    image_success_rate = successful_images / max(
        1,
        total_images,
    )

    metric_rows.append(
        {
            "split": split_name,
            "score_threshold": (score_threshold),
            "class": "OVERALL",
            "tp": sum(class_stats[class_id]["tp"] for class_id in ID_TO_CLASS),
            "fp": sum(class_stats[class_id]["fp"] for class_id in ID_TO_CLASS),
            "fn": sum(class_stats[class_id]["fn"] for class_id in ID_TO_CLASS),
            "precision": "",
            "recall": "",
            "f1": "",
            "mean_matched_iou": "",
            "macro_recall": (macro_recall),
            "image_success_rate": (image_success_rate),
        }
    )

    return (
        metric_rows,
        per_gt_rows,
    )


def draw_threshold_visuals(
    split_name: str,
    dataset: DiagnosticDataset,
    raw_predictions: list[dict[str, torch.Tensor]],
    score_threshold: float,
) -> None:
    """绘制 阈值 visuals。

    Args:
        split_name: split name。
        dataset: 按原图分组后的实例数据集。
        raw_predictions: raw 预测。
        score_threshold: 保留候选的置信度阈值。
    """
    tag = threshold_tag(score_threshold)

    threshold_dir = OUTPUT_DIR / split_name / f"threshold_{tag}"

    per_image_dir = threshold_dir / "per_image"

    per_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    visual_paths = []

    for index, sample in enumerate(dataset.samples):
        image = Image.open(sample["image_path"]).convert("RGB")

        _, target = dataset[index]

        prediction = filter_prediction(
            raw_predictions[index],
            score_threshold,
        )

        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)

        # GT: green.
        for box, label in zip(
            target["boxes"].tolist(),
            target["labels"].tolist(),
        ):
            draw.rectangle(
                box,
                outline="green",
                width=4,
            )

            draw.text(
                (
                    box[0] + 2,
                    max(
                        0,
                        box[1] + 2,
                    ),
                ),
                ("GT " + f"{safe_class_name(label)}"),
                fill="green",
            )

        # Predictions: red.
        for (
            box,
            label,
            score,
        ) in zip(
            prediction["boxes"].tolist(),
            prediction["labels"].tolist(),
            prediction["scores"].tolist(),
        ):
            if label not in ID_TO_CLASS:
                continue

            draw.rectangle(
                box,
                outline="red",
                width=2,
            )

            draw.text(
                (
                    box[0] + 2,
                    max(
                        0,
                        box[1] - 12,
                    ),
                ),
                (f"P " + f"{safe_class_name(label)} " + f"{score:.3f}"),
                fill="red",
            )

        save_path = per_image_dir / (f"{index:02d}_" + f"{sample['image_name']}")

        canvas.save(
            save_path,
            quality=92,
        )

        visual_paths.append(save_path)

    make_contact_sheet(
        visual_paths,
        threshold_dir / "contact_sheet.jpg",
    )


def fit_tile(
    image: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    """按显示格子的尺寸缩放图片，保留可辨认的目标区域。

    Args:
        image: 本步骤处理的图像对象。
        size: size。

    Returns:
        返回 tile，由函数体中同名变量的计算/收集过程得到。
    """
    tile = Image.new(
        "RGB",
        size,
        "white",
    )

    preview = image.copy()

    preview.thumbnail(
        (
            size[0] - 10,
            size[1] - 10,
        )
    )

    x = (size[0] - preview.width) // 2

    y = (size[1] - preview.height) // 2

    tile.paste(
        preview,
        (x, y),
    )

    return tile


def make_contact_sheet(
    paths: list[Path],
    output_path: Path,
) -> None:
    """把样本图与说明拼成审核联系表，便于逐例比较。

    Args:
        paths: 路径。
        output_path: 对应文件的相对路径或当前解析后的路径。
    """
    if not paths:
        return

    tiles = []

    for path in paths:
        with Image.open(path) as source:
            tiles.append(
                fit_tile(
                    source.convert("RGB"),
                    (520, 560),
                )
            )

    columns = 2
    rows = math.ceil(len(tiles) / columns)

    sheet = Image.new(
        "RGB",
        (
            520 * columns,
            560 * rows,
        ),
        "white",
    )

    for index, tile in enumerate(tiles):
        x = (index % columns) * 520

        y = (index // columns) * 560

        sheet.paste(
            tile,
            (x, y),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        quality=92,
    )


def write_notes(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> None:
    """保存 notes。

    Args:
        checkpoint: 待检查或使用的训练权重/状态。
        device: 当前计算设备，与输入张量和模型设备保持一致。
    """
    notes = f"""# Core part detector train-vs-val diagnostic

## Purpose

Determine whether the supervised collar / cuff / hem detector can fit the
training set and whether any learned localization transfers to validation.

## Checkpoint

- file: `{MODEL_FILE.name}`
- epoch: {checkpoint.get("epoch", "unknown")}
- train loss: {checkpoint.get("train_loss", "unknown")}
- saved macro recall: {checkpoint.get("macro_recall", "unknown")}
- saved image success: {checkpoint.get("image_success_rate", "unknown")}

## Model

`fasterrcnn_mobilenet_v3_large_fpn`

## Diagnostic settings

- device: {device}
- internal torchvision score threshold: {INTERNAL_SCORE_THRESHOLD}
- detections per image: {DETECTIONS_PER_IMAGE}
- external score thresholds: {THRESHOLDS}
- IoU threshold: {IOU_THRESHOLD}

## Interpretation guide

- TRAIN good, VAL poor:
  likely small-sample overfitting / insufficient generalization.
- TRAIN poor even at low thresholds:
  detector/training/localization setup has not learned the target boxes well.
- Metrics improve strongly only when score threshold is lowered:
  confidence calibration is a major issue.
- Same-class best IoU stays low across thresholds:
  localization quality itself is the main bottleneck.
"""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (REPORT_DIR / "experiment_notes.md").write_text(
        notes,
        encoding="utf-8",
    )


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    train_samples = build_samples("train")

    val_samples = build_samples("val")

    train_dataset = DiagnosticDataset(train_samples)

    val_dataset = DiagnosticDataset(val_samples)

    model = build_model().to(device)

    checkpoint = load_checkpoint(
        model,
        device,
    )

    model.eval()

    print(
        "Loaded checkpoint: "
        + f"epoch={checkpoint.get('epoch')}, "
        + f"train_loss={checkpoint.get('train_loss')}, "
        + f"macro_recall={checkpoint.get('macro_recall')}, "
        + f"image_success={checkpoint.get('image_success_rate')}"
    )

    checkpoint_rows = [
        {
            "checkpoint_file": (MODEL_FILE.name),
            "epoch": checkpoint.get(
                "epoch",
                "",
            ),
            "train_loss": (
                checkpoint.get(
                    "train_loss",
                    "",
                )
            ),
            "macro_recall": (
                checkpoint.get(
                    "macro_recall",
                    "",
                )
            ),
            "image_success_rate": (
                checkpoint.get(
                    "image_success_rate",
                    "",
                )
            ),
            "device": str(device),
        }
    ]

    write_csv(
        REPORT_DIR / "checkpoint_info.csv",
        checkpoint_rows,
    )

    print("\nRunning model once on TRAIN...")

    train_predictions = predict_dataset(
        model,
        train_dataset,
        device,
    )

    print("Running model once on VAL...")

    val_predictions = predict_dataset(
        model,
        val_dataset,
        device,
    )

    all_metric_rows = []
    all_per_gt_rows = []

    split_data = [
        (
            "train",
            train_dataset,
            train_predictions,
        ),
        (
            "val",
            val_dataset,
            val_predictions,
        ),
    ]

    for (
        split_name,
        dataset,
        predictions,
    ) in split_data:
        for threshold in THRESHOLDS:
            print(
                f"Evaluating "
                + f"{split_name.upper()} "
                + f"at score >= "
                + f"{threshold}..."
            )

            (
                metric_rows,
                per_gt_rows,
            ) = evaluate_split_threshold(
                split_name=split_name,
                dataset=dataset,
                raw_predictions=(predictions),
                score_threshold=(threshold),
            )

            all_metric_rows.extend(metric_rows)

            all_per_gt_rows.extend(per_gt_rows)

            draw_threshold_visuals(
                split_name=split_name,
                dataset=dataset,
                raw_predictions=(predictions),
                score_threshold=(threshold),
            )

    write_csv(
        REPORT_DIR / "metrics_by_split_threshold.csv",
        all_metric_rows,
    )

    write_csv(
        REPORT_DIR / "per_gt_diagnostic.csv",
        all_per_gt_rows,
    )

    write_notes(
        checkpoint,
        device,
    )

    print("\nFinished.")

    print("Metrics: " + f"{REPORT_DIR / 'metrics_by_split_threshold.csv'}")

    print("Per-GT: " + f"{REPORT_DIR / 'per_gt_diagnostic.csv'}")

    print("Visuals: " + f"{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
