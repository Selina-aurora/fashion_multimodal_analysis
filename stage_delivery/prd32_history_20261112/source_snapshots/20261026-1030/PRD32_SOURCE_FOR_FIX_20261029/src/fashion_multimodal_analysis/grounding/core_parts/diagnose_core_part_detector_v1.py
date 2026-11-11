"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。

Diagnose validation predictions for core_part_detector_v1.

Purpose:
    Inspect whether the supervised detector is:
    1) producing no boxes,
    2) producing low-confidence boxes,
    3) predicting the wrong class, or
    4) predicting roughly correct boxes whose IoU is below 0.5.

Inputs:
    models/core_part_detector_v1/best_model.pt
    reports/prd_region_coverage/core_part_annotation_labels.csv
    reports/prd_region_coverage/core_part_detector_split_v1.csv

Outputs:
    reports/prd_region_coverage/core_part_detector_v1_diagnostic/
        per_gt_diagnostic.csv
        summary.csv
        experiment_notes.md

    outputs/core_part_detector_v1_diagnostic/
        val_diagnostic_contact_sheet.jpg
        per_image/*.jpg

Visualization:
    green = GT box
    red   = low-threshold predictions
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
MODEL_FILE = PROJECT_ROOT / "models" / "core_part_detector_v1" / "best_model.pt"

REPORT_DIR = REPORT_ROOT / "core_part_detector_v1_diagnostic"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "core_part_detector_v1_diagnostic"

CLASS_TO_ID = {"collar": 1, "cuff": 2, "hem": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
NUM_CLASSES = 4
DIAGNOSTIC_SCORE_THRESHOLD = 0.001


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
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def parse_boxes(raw: str) -> list[list[float]]:
    """解析 边界框。

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        ValueError: boxes_json must be a list.
    """
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("boxes_json must be a list.")
    return [[float(v) for v in box] for box in data]


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256，用于运行与备份追溯。

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


def build_val_samples() -> list[dict[str, Any]]:
    """构建 val samples。

    Returns:
        返回 samples，由函数体中同名变量的计算/收集过程得到。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    labels = {row["annotation_id"]: row for row in read_csv(LABELS_FILE)}
    split_rows = [row for row in read_csv(SPLIT_FILE) if row["split"] == "val"]
    grouped: dict[str, dict[str, Any]] = {}

    for split_row in split_rows:
        annotation_id = split_row["annotation_id"]
        row = labels[annotation_id]
        image_path = resolve_project_path(row["annotation_image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image_hash = sha256_file(image_path)
        boxes = parse_boxes(row["boxes_json"])
        class_id = CLASS_TO_ID[row["region"]]

        if image_hash not in grouped:
            grouped[image_hash] = {
                "image_name": row["image_name"],
                "image_path": image_path,
                "annotation_ids": [],
                "boxes": [],
                "labels": [],
            }
        grouped[image_hash]["annotation_ids"].append(annotation_id)
        grouped[image_hash]["boxes"].extend(boxes)
        grouped[image_hash]["labels"].extend([class_id] * len(boxes))

    samples = list(grouped.values())
    print(
        f"Validation: {len(split_rows)} annotation rows -> {len(samples)} unique image samples"
    )
    return samples


def build_model() -> torch.nn.Module:
    """构造与当前实验标签空间和输入尺寸一致的模型。

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
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def load_checkpoint(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
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


def safe_class_name(class_id: int) -> str:
    """可靠处理 类别 name。

    Args:
        class_id: 类别 ID。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return ID_TO_CLASS.get(int(class_id), f"class_{class_id}")


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

    ious = box_iou(gt_box.unsqueeze(0), pred_boxes)[0]
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
        same_ious = box_iou(gt_box.unsqueeze(0), same_boxes)[0]
        best_same_index = int(torch.argmax(same_ious).item())
        result["same_class_found"] = "yes"
        result["same_class_best_iou"] = float(same_ious[best_same_index].item())
        result["same_class_score"] = float(same_scores[best_same_index].item())
        result["same_class_box_json"] = json.dumps(same_boxes[best_same_index].tolist())
    return result


def draw_diagnostic(
    sample: dict[str, Any], prediction: dict[str, torch.Tensor], output_path: Path
) -> None:
    """绘制 diagnostic。

    Args:
        sample: 记录字段，使用 boxes, labels, image_path。
        prediction: 记录字段，使用 boxes, labels, scores。
        output_path: 对应文件的相对路径或当前解析后的路径。
    """
    image = Image.open(sample["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    gt_boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
    gt_labels = torch.tensor(sample["labels"], dtype=torch.int64)
    pred_boxes = prediction["boxes"].detach().cpu()
    pred_labels = prediction["labels"].detach().cpu()
    pred_scores = prediction["scores"].detach().cpu()

    keep = pred_scores >= DIAGNOSTIC_SCORE_THRESHOLD
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    for box, label in zip(gt_boxes.tolist(), gt_labels.tolist()):
        draw.rectangle(box, outline="green", width=4)
        draw.text(
            (box[0] + 2, max(0, box[1] + 2)),
            f"GT {safe_class_name(label)}",
            fill="green",
        )

    for box, label, score in zip(
        pred_boxes[:15].tolist(), pred_labels[:15].tolist(), pred_scores[:15].tolist()
    ):
        draw.rectangle(box, outline="red", width=2)
        draw.text(
            (box[0] + 2, max(0, box[1] - 12)),
            f"{safe_class_name(label)} {score:.2f}",
            fill="red",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def fit_tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """按显示格子的尺寸缩放图片，保留可辨认的目标区域。

    Args:
        image: 本步骤处理的图像对象。
        size: size。

    Returns:
        返回 tile，由函数体中同名变量的计算/收集过程得到。
    """
    tile = Image.new("RGB", size, "white")
    preview = image.copy()
    preview.thumbnail((size[0] - 10, size[1] - 10))
    x = (size[0] - preview.width) // 2
    y = (size[1] - preview.height) // 2
    tile.paste(preview, (x, y))
    return tile


def make_contact_sheet(paths: list[Path]) -> None:
    """把样本图与说明拼成审核联系表，便于逐例比较。

    Args:
        paths: 路径。
    """
    if not paths:
        return
    tiles = []
    for path in paths:
        with Image.open(path) as source:
            tiles.append(fit_tile(source.convert("RGB"), (520, 560)))
    columns = 2
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (520 * columns, 560 * rows), "white")
    for index, tile in enumerate(tiles):
        x = (index % columns) * 520
        y = (index // columns) * 560
        sheet.paste(tile, (x, y))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT_DIR / "val_diagnostic_contact_sheet.jpg", quality=92)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    samples = build_val_samples()
    model = build_model().to(device)
    checkpoint = load_checkpoint(model, device)
    # Important: expose very-low-confidence detections before
    # torchvision's normal final score filtering.
    model.roi_heads.score_thresh = 0.0
    model.roi_heads.detections_per_img = 300
    model.eval()
    print(
        "Loaded checkpoint: "
        + f"epoch={checkpoint.get('epoch', 'unknown')}, "
        + f"macro_recall={checkpoint.get('macro_recall', 'unknown')}, "
        + f"image_success={checkpoint.get('image_success_rate', 'unknown')}"
    )

    diagnostic_rows = []
    visual_paths = []

    with torch.no_grad():
        for sample_index, sample in enumerate(samples):
            image = Image.open(sample["image_path"]).convert("RGB")
            image_tensor = tvf.pil_to_tensor(image).float() / 255.0
            prediction = model([image_tensor.to(device)])[0]

            pred_boxes = prediction["boxes"].detach().cpu()
            pred_labels = prediction["labels"].detach().cpu()
            pred_scores = prediction["scores"].detach().cpu()
            keep = pred_scores >= DIAGNOSTIC_SCORE_THRESHOLD
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]

            gt_boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
            gt_labels = torch.tensor(sample["labels"], dtype=torch.int64)

            for gt_index, (gt_box, gt_label) in enumerate(
                zip(gt_boxes, gt_labels), start=1
            ):
                best = best_prediction_for_gt(
                    gt_box=gt_box,
                    gt_label=int(gt_label.item()),
                    pred_boxes=pred_boxes,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                )
                same_iou = float(best["same_class_best_iou"])
                same_score = best["same_class_score"]

                if best["same_class_found"] == "no":
                    failure_mode = "no_same_class_prediction"
                elif same_iou >= 0.5 and float(same_score) < 0.25:
                    failure_mode = "good_iou_but_score_below_0.25"
                elif same_iou < 0.5 and float(same_score) >= 0.25:
                    failure_mode = "score_ok_but_iou_below_0.5"
                elif same_iou < 0.5 and float(same_score) < 0.25:
                    failure_mode = "both_score_and_iou_low"
                else:
                    failure_mode = "would_count_as_tp"

                diagnostic_rows.append(
                    {
                        "image_name": sample["image_name"],
                        "annotation_ids": ";".join(sample["annotation_ids"]),
                        "gt_index": gt_index,
                        "gt_class": safe_class_name(int(gt_label.item())),
                        "gt_box_json": json.dumps(gt_box.tolist()),
                        "num_predictions_ge_0.01": len(pred_boxes),
                        **best,
                        "failure_mode": failure_mode,
                    }
                )

            visual_path = (
                OUTPUT_DIR / "per_image" / f"{sample_index:02d}_{sample['image_name']}"
            )
            draw_diagnostic(
                sample=sample, prediction=prediction, output_path=visual_path
            )
            visual_paths.append(visual_path)

    write_csv(REPORT_DIR / "per_gt_diagnostic.csv", diagnostic_rows)

    summary_rows = []
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    for row in diagnostic_rows:
        grouped[(row["gt_class"], row["failure_mode"])] += 1
        grouped[("OVERALL", row["failure_mode"])] += 1
    for (gt_class, failure_mode), count in sorted(grouped.items()):
        summary_rows.append(
            {"gt_class": gt_class, "failure_mode": failure_mode, "count": count}
        )
    write_csv(REPORT_DIR / "summary.csv", summary_rows)

    notes = f"""# Core part detector v1 diagnostic

## Purpose
Explain why validation macro_recall@0.5 remains zero.

## Checkpoint
- epoch: {checkpoint.get('epoch', 'unknown')}
- saved macro recall: {checkpoint.get('macro_recall', 'unknown')}
- saved image success: {checkpoint.get('image_success_rate', 'unknown')}

## Diagnostic threshold
Predictions are retained down to score >= {DIAGNOSTIC_SCORE_THRESHOLD}.

The original validation metric requires:
- correct class
- score >= 0.25
- IoU >= 0.50
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "experiment_notes.md").write_text(notes, encoding="utf-8")
    make_contact_sheet(visual_paths)

    print("\nFinished.")
    print(f"Per-GT: {REPORT_DIR / 'per_gt_diagnostic.csv'}")
    print(f"Summary: {REPORT_DIR / 'summary.csv'}")
    print(f"Visual: {OUTPUT_DIR / 'val_diagnostic_contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
