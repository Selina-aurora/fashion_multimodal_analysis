"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。

Evaluate the verified-positive benchmark with Grounding DINO.

This script performs a paired controlled diagnostic experiment:

1. Full-image Grounding DINO
2. Annotation-assisted ROI-conditioned Grounding DINO

The benchmark contains manually verified positive cases for:
- sleeve
- collar
- button
- zipper

Important:
The DeepFashion2 garment annotation is used as an oracle ROI only for
controlled diagnosis. It is not intended to represent the final deployment
pipeline.

Automatic metrics in this script measure:
- detection / no detection
- confidence
- number of detections
- predicted box size
- inference time

Fine-grained localization quality:
- correct
- coarse
- wrong
- missed

must still be manually audited because there is no manually annotated
ground-truth bbox for the fine-grained local target itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

# ============================================================
# Configuration
# ============================================================

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

DEFAULT_THRESHOLD = 0.30
DEFAULT_ROI_MARGIN_RATIO = 0.05

PROJECT_ROOT = get_project_root()

DATASET_ROOT = data_root() / "raw" / "train" / "train"

IMAGE_DIR = DATASET_ROOT / "image"
ANNOTATION_DIR = DATASET_ROOT / "annos"

DEFAULT_BENCHMARK_FILE = (
    PROJECT_ROOT
    / "reports"
    / "verified_positive_benchmark"
    / "verified_positive_benchmark.csv"
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "verified_positive_evaluation"

REPORT_ROOT = PROJECT_ROOT / "reports" / "verified_positive_evaluation"

FULL_IMAGE_OUTPUT_DIR = OUTPUT_ROOT / "full_image"

ROI_OUTPUT_DIR = OUTPUT_ROOT / "roi_conditioned"

RAW_OUTPUT_DIR = OUTPUT_ROOT / "raw_results"

PAIRED_CONTACT_DIR = OUTPUT_ROOT / "paired_contact_sheets"


TARGET_ORDER = [
    "sleeve",
    "collar",
    "button",
    "zipper",
]

SCALE_GROUPS = {
    "sleeve": "larger_part",
    "collar": "larger_part",
    "button": "small_object",
    "zipper": "small_object",
}


# ============================================================
# Contact-sheet configuration
# ============================================================

PAIRED_CASES_PER_PAGE = 4

PANEL_WIDTH = 500
PANEL_HEIGHT = 340

HEADER_HEIGHT = 45
PAIR_ROW_HEIGHT = PANEL_HEIGHT + HEADER_HEIGHT

PAIRED_SHEET_WIDTH = PANEL_WIDTH * 2


# ============================================================
# Arguments
# ============================================================


def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。

    Raises:
        ValueError: --limit must be greater than zero.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate verified-positive localization benchmark "
            + "using full-image and oracle-ROI Grounding DINO."
        )
    )

    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=DEFAULT_BENCHMARK_FILE,
        help="Verified-positive benchmark CSV.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Grounding DINO detection threshold " + f"(default: {DEFAULT_THRESHOLD})."
        ),
    )

    parser.add_argument(
        "--roi-margin-ratio",
        type=float,
        default=DEFAULT_ROI_MARGIN_RATIO,
        help=(
            "Relative margin added around the annotated garment ROI "
            + f"(default: {DEFAULT_ROI_MARGIN_RATIO})."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional case limit for smoke testing.",
    )

    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip visualization and paired contact-sheet generation.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    if not 0.0 <= args.roi_margin_ratio <= 1.0:
        raise ValueError("--roi-margin-ratio must be between 0 and 1.")

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero.")

    return args


# ============================================================
# Device and model
# ============================================================


def get_device() -> torch.device:
    """根据参数选择当前实验的计算设备。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def synchronize_device(
    device: torch.device,
) -> None:
    """在 CUDA 设备上同步运算，避免异步执行低估耗时。

    Args:
        device: 当前计算设备，与输入张量和模型设备保持一致。
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_model(
    device: torch.device,
) -> tuple[Any, Any]:
    """加载当前实验指定的模型和权重。

    Args:
        device: 当前计算设备，与输入张量和模型设备保持一致。

    Returns:
        按顺序返回 processor, model 等结果。
    """
    print(f"Loading model: {MODEL_ID}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)

    model.to(device)
    model.eval()

    return processor, model


# ============================================================
# Benchmark loading
# ============================================================


def read_benchmark(
    benchmark_file: Path,
    limit: int | None,
) -> list[dict[str, str]]:
    """读取 基准。

    Args:
        benchmark_file: 对应文件的相对路径或当前解析后的路径。
        limit: limit。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。

    Raises:
        FileNotFoundError: 需要的文件不存在。
        ValueError: 输入或实验状态不符合检查条件。
    """
    if not benchmark_file.is_file():
        raise FileNotFoundError(f"Benchmark file not found: " + f"{benchmark_file}")

    with benchmark_file.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    required_fields = {
        "benchmark_id",
        "target",
        "image_name",
        "item_id",
        "category_name",
        "visibility",
        "verified_positive",
        "usable_for_evaluation",
    }

    if not rows:
        raise ValueError("Benchmark CSV is empty.")

    missing_fields = required_fields - set(rows[0].keys())

    if missing_fields:
        raise ValueError(
            "Benchmark CSV is missing fields: " + f"{sorted(missing_fields)}"
        )

    usable_rows = []

    seen_ids = set()

    for row in rows:
        benchmark_id = row["benchmark_id"].strip()

        if benchmark_id in seen_ids:
            raise ValueError("Duplicate benchmark_id: " + f"{benchmark_id}")

        seen_ids.add(benchmark_id)

        if row["verified_positive"].strip().lower() != "yes":
            continue

        if row["usable_for_evaluation"].strip().lower() != "yes":
            continue

        target = row["target"].strip().lower()

        if target not in TARGET_ORDER:
            raise ValueError(f"Unknown target: {target}")

        usable_rows.append(row)

    if limit is not None:
        usable_rows = usable_rows[:limit]

    return usable_rows


# ============================================================
# DeepFashion2 annotations
# ============================================================


def load_annotation(
    image_name: str,
) -> dict[str, Any]:
    """加载 标注。

    Args:
        image_name: 图像 name。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    annotation_path = ANNOTATION_DIR / f"{Path(image_name).stem}.json"

    if not annotation_path.is_file():
        raise FileNotFoundError("Annotation not found: " + f"{annotation_path}")

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(json_file)


def get_exact_garment(
    annotation: dict[str, Any],
    item_id: str,
) -> dict[str, Any]:
    """Retrieve the exact garment used by the benchmark.

    This is preferable to re-selecting the largest/preferred
    garment because the benchmark CSV already records the exact
    DeepFashion2 item_id.

    Args:
        annotation: 原始实例标注。
        item_id: item ID。

    Returns:
        结果字典，主要字段为 item_id, category_name, bbox。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    if item_id not in annotation:
        raise ValueError(f"item_id {item_id} was not found " + "in the annotation.")

    garment = annotation[item_id]

    if not isinstance(
        garment,
        dict,
    ):
        raise ValueError(f"Invalid garment record: {item_id}")

    bbox = garment.get("bounding_box")

    if bbox is None or len(bbox) != 4:
        raise ValueError(f"Invalid bounding box for {item_id}")

    category_name = garment.get(
        "category_name",
        "",
    )

    return {
        "item_id": item_id,
        "category_name": str(category_name),
        "bbox": [int(round(float(value))) for value in bbox],
    }


def clip_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
) -> list[int]:
    """把边界框限制在图像有效范围内。

    Args:
        bbox: xyxy 坐标的目标框。
        image_width: 图像 宽度。
        image_height: 图像 高度。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = bbox

    x_min = max(
        0,
        min(
            image_width - 1,
            x_min,
        ),
    )

    y_min = max(
        0,
        min(
            image_height - 1,
            y_min,
        ),
    )

    x_max = max(
        x_min,
        min(
            image_width - 1,
            x_max,
        ),
    )

    y_max = max(
        y_min,
        min(
            image_height - 1,
            y_max,
        ),
    )

    return [
        x_min,
        y_min,
        x_max,
        y_max,
    ]


def expand_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> list[int]:
    """按实验配置扩展目标框，保留局部识别需要的上下文。

    Args:
        bbox: xyxy 坐标的目标框。
        image_width: 图像 宽度。
        image_height: 图像 高度。
        margin_ratio: margin 比例。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = bbox

    width = max(
        1,
        x_max - x_min + 1,
    )

    height = max(
        1,
        y_max - y_min + 1,
    )

    margin_x = int(round(width * margin_ratio))

    margin_y = int(round(height * margin_ratio))

    expanded = [
        x_min - margin_x,
        y_min - margin_y,
        x_max + margin_x,
        y_max + margin_y,
    ]

    return clip_bbox(
        expanded,
        image_width,
        image_height,
    )


# ============================================================
# Grounding DINO inference
# ============================================================


def normalize_label(
    label: Any,
) -> str:
    """将预测或审核标签转换为本实验统一的标签表示。

    Args:
        label: 标签。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if isinstance(
        label,
        str,
    ):
        return label

    if torch.is_tensor(label):
        if label.numel() == 1:
            return str(label.item())

        return str(label.detach().cpu().tolist())

    return str(label)


def run_grounding_dino(
    image: Image.Image,
    prompt: str,
    processor: Any,
    model: Any,
    device: torch.device,
    threshold: float,
) -> tuple[
    list[dict[str, Any]],
    float,
]:
    """Run Grounding DINO on one image.

    Args:
        image: 本步骤处理的图像对象。
        prompt: 提示文本。
        processor: 与模型配套的输入处理器。
        model: 已构建的模型对象，由调用方负责选择权重。
        device: 当前计算设备，与输入张量和模型设备保持一致。
        threshold: 当前判定规则使用的阈值。

    Returns:
        按顺序返回 detections, elapsed_seconds 等结果。
    """
    synchronize_device(device)

    start_time = time.perf_counter()

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        target_sizes=[image.size[::-1]],
    )

    synchronize_device(device)

    elapsed_seconds = time.perf_counter() - start_time

    result = results[0]

    scores = result.get(
        "scores",
        [],
    )

    boxes = result.get(
        "boxes",
        [],
    )

    if "text_labels" in result:
        labels = result["text_labels"]
    elif "labels" in result:
        labels = result["labels"]
    else:
        labels = [prompt] * len(scores)

    detections = []

    for (
        score,
        label,
        box,
    ) in zip(
        scores,
        labels,
        boxes,
    ):
        if torch.is_tensor(score):
            score_value = float(score.detach().cpu().item())
        else:
            score_value = float(score)

        if torch.is_tensor(box):
            box_values = box.detach().cpu().tolist()
        else:
            box_values = list(box)

        detections.append(
            {
                "label": (normalize_label(label)),
                "score": (score_value),
                "box": [float(value) for value in box_values],
            }
        )

    detections.sort(
        key=lambda detection: (detection["score"]),
        reverse=True,
    )

    return (
        detections,
        elapsed_seconds,
    )


def map_roi_detections_to_full_image(
    detections: list[dict[str, Any]],
    roi_bbox: list[int],
) -> list[dict[str, Any]]:
    """map roi detections 转换 整图 图像。

    Args:
        detections: detections。
        roi_bbox: roi 边界框。

    Returns:
        返回 mapped，由函数体中同名变量的计算/收集过程得到。
    """
    x_offset = roi_bbox[0]
    y_offset = roi_bbox[1]

    mapped = []

    for detection in detections:
        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = detection["box"]

        mapped.append(
            {
                "label": (detection["label"]),
                "score": (detection["score"]),
                "box": [
                    x_min + x_offset,
                    y_min + y_offset,
                    x_max + x_offset,
                    y_max + y_offset,
                ],
            }
        )

    return mapped


# ============================================================
# Detection statistics
# ============================================================


def detection_box_area_ratio(
    detection: dict[str, Any],
    image_width: int,
    image_height: int,
) -> float:
    """detection 边界框 面积 比例。

    Args:
        detection: 记录字段，使用 box。
        image_width: 图像 宽度。
        image_height: 图像 高度。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = detection["box"]

    width = max(
        0.0,
        x_max - x_min,
    )

    height = max(
        0.0,
        y_max - y_min,
    )

    image_area = max(
        1.0,
        float(image_width * image_height),
    )

    return width * height / image_area


def summarize_single_condition(
    benchmark_row: dict[str, str],
    condition: str,
    detections: list[dict[str, Any]],
    inference_seconds: float,
    image_width: int,
    image_height: int,
    threshold: float,
    roi_margin_ratio: float,
    roi_bbox: list[int],
) -> dict[str, Any]:
    """分别汇总一个输入条件下的结果，避免不同条件混算。

    Args:
        benchmark_row: 记录字段，使用 benchmark_id, target, image_name, item_id, category_name,
        visibility。
        condition: condition。
        detections: detections。
        inference_seconds: inference seconds。
        image_width: 图像 宽度。
        image_height: 图像 高度。
        threshold: 当前判定规则使用的阈值。
        roi_margin_ratio: roi margin 比例。
        roi_bbox: roi 边界框。

    Returns:
        结果字典，主要字段为 benchmark_id, target, scale_group, condition, image_name, item_id,
        category_name, visibility, prompt, threshold。
    """
    detected = bool(detections)

    top_score = ""

    mean_score = ""

    top_box_area_ratio = ""

    if detected:
        top_score = detections[0]["score"]

        mean_score = sum(detection["score"] for detection in detections) / len(
            detections
        )

        top_box_area_ratio = detection_box_area_ratio(
            detections[0],
            image_width,
            image_height,
        )

    return {
        "benchmark_id": (benchmark_row["benchmark_id"]),
        "target": (benchmark_row["target"]),
        "scale_group": (SCALE_GROUPS[benchmark_row["target"]]),
        "condition": condition,
        "image_name": (benchmark_row["image_name"]),
        "item_id": (benchmark_row["item_id"]),
        "category_name": (benchmark_row["category_name"]),
        "visibility": (benchmark_row["visibility"]),
        "prompt": (benchmark_row["target"]),
        "threshold": (threshold),
        "roi_margin_ratio": (roi_margin_ratio),
        "roi_bbox": (json.dumps(roi_bbox)),
        "detected": ("yes" if detected else "no"),
        "num_detections": (len(detections)),
        "top_score": top_score,
        "mean_score": mean_score,
        "top_box_area_ratio": (top_box_area_ratio),
        "inference_seconds": (inference_seconds),
        "detections_json": (
            json.dumps(
                detections,
                ensure_ascii=False,
            )
        ),
    }


# ============================================================
# Visualization
# ============================================================


def draw_detection_boxes(
    image: Image.Image,
    detections: list[dict[str, Any]],
    title: str,
    roi_bbox: list[int] | None = None,
) -> Image.Image:
    """绘制候选框及其标签，保留原图坐标的对应关系。

    Args:
        image: 本步骤处理的图像对象。
        detections: detections。
        title: title。
        roi_bbox: roi 边界框。

    Returns:
        返回 visualization，由函数体中同名变量的计算/收集过程得到。
    """
    visualization = image.copy().convert("RGB")

    draw = ImageDraw.Draw(visualization)

    font = ImageFont.load_default()

    if roi_bbox is not None:
        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = roi_bbox

        draw.rectangle(
            [
                x_min,
                y_min,
                x_max,
                y_max,
            ],
            outline="yellow",
            width=4,
        )

        draw.text(
            (
                x_min + 3,
                max(
                    0,
                    y_min - 15,
                ),
            ),
            "garment ROI",
            fill="yellow",
            font=font,
        )

    for index, detection in enumerate(
        detections,
        start=1,
    ):
        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = detection["box"]

        draw.rectangle(
            [
                x_min,
                y_min,
                x_max,
                y_max,
            ],
            outline="red",
            width=4,
        )

        label = f"{index}: " + f"{detection['label']} " + f"{detection['score']:.3f}"

        text_y = max(
            0,
            int(y_min) - 15,
        )

        draw.text(
            (
                int(x_min) + 3,
                text_y,
            ),
            label,
            fill="red",
            font=font,
        )

    draw.rectangle(
        (
            0,
            0,
            min(
                visualization.width,
                500,
            ),
            20,
        ),
        fill="white",
    )

    draw.text(
        (5, 5),
        title,
        fill="black",
        font=font,
    )

    return visualization


def save_visualizations(
    benchmark_id: str,
    original_image: Image.Image,
    full_detections: list[dict[str, Any]],
    roi_detections: list[dict[str, Any]],
    roi_bbox: list[int],
    target: str,
) -> tuple[Path, Path]:
    """保存本次实验的预测可视化。

    Args:
        benchmark_id: 基准 ID。
        original_image: original 图像。
        full_detections: 整图 detections。
        roi_detections: roi detections。
        roi_bbox: roi 边界框。
        target: 目标。

    Returns:
        按顺序返回 full_path, roi_path 等结果。
    """
    FULL_IMAGE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ROI_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_visualization = draw_detection_boxes(
        original_image,
        full_detections,
        title=(f"{benchmark_id} | " + f"FULL | {target}"),
    )

    roi_visualization = draw_detection_boxes(
        original_image,
        roi_detections,
        title=(f"{benchmark_id} | " + f"ROI | {target}"),
        roi_bbox=roi_bbox,
    )

    full_path = FULL_IMAGE_OUTPUT_DIR / f"{benchmark_id}.jpg"

    roi_path = ROI_OUTPUT_DIR / f"{benchmark_id}.jpg"

    full_visualization.save(
        full_path,
        quality=92,
    )

    roi_visualization.save(
        roi_path,
        quality=92,
    )

    return (
        full_path,
        roi_path,
    )


# ============================================================
# Raw JSON
# ============================================================


def save_raw_case_result(
    benchmark_row: dict[str, str],
    roi_bbox: list[int],
    full_detections: list[dict[str, Any]],
    roi_detections: list[dict[str, Any]],
    threshold: float,
    roi_margin_ratio: float,
) -> Path:
    """保存单例原始预测，避免汇总后丢失错误线索。

    Args:
        benchmark_row: 记录字段，使用 benchmark_id, target, image_name, item_id。
        roi_bbox: roi 边界框。
        full_detections: 整图 detections。
        roi_detections: roi detections。
        threshold: 当前判定规则使用的阈值。
        roi_margin_ratio: roi margin 比例。

    Returns:
        返回 output_path，由函数体中同名变量的计算/收集过程得到。
    """
    RAW_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = RAW_OUTPUT_DIR / (f"{benchmark_row['benchmark_id']}" + ".json")

    payload = {
        "benchmark_id": (benchmark_row["benchmark_id"]),
        "target": (benchmark_row["target"]),
        "image_name": (benchmark_row["image_name"]),
        "item_id": (benchmark_row["item_id"]),
        "threshold": threshold,
        "roi_margin_ratio": (roi_margin_ratio),
        "oracle_garment_roi": (roi_bbox),
        "full_image_detections": (full_detections),
        "roi_conditioned_detections": (roi_detections),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            payload,
            json_file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# CSV writing
# ============================================================


def save_case_results(
    rows: list[dict[str, Any]],
) -> Path:
    """保存逐例结果，供总体指标回算与审核。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        返回 output_path，由函数体中同名变量的计算/收集过程得到。
    """
    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = REPORT_ROOT / "case_results.csv"

    fieldnames = [
        "benchmark_id",
        "target",
        "scale_group",
        "condition",
        "image_name",
        "item_id",
        "category_name",
        "visibility",
        "prompt",
        "threshold",
        "roi_margin_ratio",
        "roi_bbox",
        "detected",
        "num_detections",
        "top_score",
        "mean_score",
        "top_box_area_ratio",
        "inference_seconds",
        "detections_json",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    return output_path


# ============================================================
# Summary helpers
# ============================================================


def safe_mean(
    values: list[float],
) -> float:
    """在有效数值上计算均值，空组按本实验的缺失值约定处理。

    Args:
        values: 本步骤处理的数值或附加字段。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if not values:
        return float("nan")

    return statistics.mean(values)


def safe_median(
    values: list[float],
) -> float:
    """在有效数值上计算中位数，保留缺失值的语义。

    Args:
        values: 本步骤处理的数值或附加字段。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if not values:
        return float("nan")

    return statistics.median(values)


def float_or_none(
    value: Any,
) -> float | None:
    """尝试解析数值，无效或缺失输入返回 None。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if value in (
        "",
        None,
    ):
        return None

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def aggregate_group(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总同一分组中的结果，保留组内样本数。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        结果字典，主要字段为 n_cases, detected_count, detection_rate, mean_num_detections,
        mean_top_score_detected, median_top_score_detected,
        mean_top_box_area_ratio_detected, mean_inference_seconds。
    """
    n_cases = len(rows)

    detected_rows = [row for row in rows if row["detected"] == "yes"]

    detected_count = len(detected_rows)

    detection_rate = detected_count / n_cases if n_cases else 0.0

    top_scores = []

    box_ratios = []

    for row in detected_rows:
        top_score = float_or_none(row["top_score"])

        if top_score is not None:
            top_scores.append(top_score)

        box_ratio = float_or_none(row["top_box_area_ratio"])

        if box_ratio is not None:
            box_ratios.append(box_ratio)

    num_detections = [float(row["num_detections"]) for row in rows]

    inference_times = [float(row["inference_seconds"]) for row in rows]

    return {
        "n_cases": n_cases,
        "detected_count": (detected_count),
        "detection_rate": (detection_rate),
        "mean_num_detections": (safe_mean(num_detections)),
        "mean_top_score_detected": (safe_mean(top_scores)),
        "median_top_score_detected": (safe_median(top_scores)),
        "mean_top_box_area_ratio_detected": (safe_mean(box_ratios)),
        "mean_inference_seconds": (safe_mean(inference_times)),
    }


def format_number(
    value: Any,
    digits: int = 4,
) -> str:
    """将数值格式化为报告文本，显式处理缺失值。

    Args:
        value: 待解析或规范化的输入值。
        digits: digits。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if isinstance(
        value,
        float,
    ) and math.isnan(value):
        return ""

    if isinstance(
        value,
        float,
    ):
        return f"{value:.{digits}f}"

    return str(value)


# ============================================================
# Target summary
# ============================================================


def build_target_summary(
    case_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """构建 目标 汇总。

    Args:
        case_rows: 待处理的 case 记录。

    Returns:
        返回 summary_rows，由函数体中同名变量的计算/收集过程得到。
    """
    grouped = defaultdict(list)

    for row in case_rows:
        key = (
            row["target"],
            row["condition"],
        )

        grouped[key].append(row)

    summary_rows = []

    for target in TARGET_ORDER:
        for condition in (
            "full_image",
            "roi_conditioned",
        ):
            rows = grouped.get(
                (
                    target,
                    condition,
                ),
                [],
            )

            if not rows:
                continue

            metrics = aggregate_group(rows)

            summary_rows.append(
                {
                    "target": target,
                    "scale_group": (SCALE_GROUPS[target]),
                    "condition": (condition),
                    **metrics,
                }
            )

    return summary_rows


def save_target_summary(
    rows: list[dict[str, Any]],
) -> Path:
    """保存 目标 汇总。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        返回 output_path，由函数体中同名变量的计算/收集过程得到。
    """
    output_path = REPORT_ROOT / "target_summary.csv"

    fieldnames = [
        "target",
        "scale_group",
        "condition",
        "n_cases",
        "detected_count",
        "detection_rate",
        "mean_num_detections",
        "mean_top_score_detected",
        "median_top_score_detected",
        "mean_top_box_area_ratio_detected",
        "mean_inference_seconds",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({key: (format_number(row[key])) for key in fieldnames})

    return output_path


# ============================================================
# Scale-group summary
# ============================================================


def build_scale_group_summary(
    case_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """构建 scale group 汇总。

    Args:
        case_rows: 待处理的 case 记录。

    Returns:
        返回 summary_rows，由函数体中同名变量的计算/收集过程得到。
    """
    grouped = defaultdict(list)

    for row in case_rows:
        key = (
            row["scale_group"],
            row["condition"],
        )

        grouped[key].append(row)

    summary_rows = []

    for scale_group in (
        "larger_part",
        "small_object",
    ):
        for condition in (
            "full_image",
            "roi_conditioned",
        ):
            rows = grouped.get(
                (
                    scale_group,
                    condition,
                ),
                [],
            )

            if not rows:
                continue

            metrics = aggregate_group(rows)

            summary_rows.append(
                {
                    "scale_group": (scale_group),
                    "condition": (condition),
                    **metrics,
                }
            )

    return summary_rows


def save_scale_group_summary(
    rows: list[dict[str, Any]],
) -> Path:
    """保存 scale group 汇总。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        返回 output_path，由函数体中同名变量的计算/收集过程得到。
    """
    output_path = REPORT_ROOT / "scale_group_summary.csv"

    fieldnames = [
        "scale_group",
        "condition",
        "n_cases",
        "detected_count",
        "detection_rate",
        "mean_num_detections",
        "mean_top_score_detected",
        "median_top_score_detected",
        "mean_top_box_area_ratio_detected",
        "mean_inference_seconds",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({key: (format_number(row[key])) for key in fieldnames})

    return output_path


# ============================================================
# Manual paired audit
# ============================================================


def save_manual_audit(
    benchmark_rows: list[dict[str, str]],
    result_lookup: dict[
        tuple[str, str],
        dict[str, Any],
    ],
) -> Path:
    """保存 人工 检查。

    Args:
        benchmark_rows: 待处理的 benchmark 记录。
        result_lookup: 结果 lookup。

    Returns:
        返回 output_path，由函数体中同名变量的计算/收集过程得到。
    """
    output_path = REPORT_ROOT / "paired_manual_audit.csv"

    fieldnames = [
        "benchmark_id",
        "target",
        "scale_group",
        "image_name",
        "item_id",
        "category_name",
        "visibility",
        "full_detected",
        "full_top_score",
        "roi_detected",
        "roi_top_score",
        "full_localization_quality",
        "roi_localization_quality",
        "full_notes",
        "roi_notes",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for benchmark_row in benchmark_rows:
            benchmark_id = benchmark_row["benchmark_id"]

            full_result = result_lookup[
                (
                    benchmark_id,
                    "full_image",
                )
            ]

            roi_result = result_lookup[
                (
                    benchmark_id,
                    "roi_conditioned",
                )
            ]

            full_detected = full_result["detected"]

            roi_detected = roi_result["detected"]

            writer.writerow(
                {
                    "benchmark_id": (benchmark_id),
                    "target": (benchmark_row["target"]),
                    "scale_group": (SCALE_GROUPS[benchmark_row["target"]]),
                    "image_name": (benchmark_row["image_name"]),
                    "item_id": (benchmark_row["item_id"]),
                    "category_name": (benchmark_row["category_name"]),
                    "visibility": (benchmark_row["visibility"]),
                    "full_detected": (full_detected),
                    "full_top_score": (full_result["top_score"]),
                    "roi_detected": (roi_detected),
                    "roi_top_score": (roi_result["top_score"]),
                    "full_localization_quality": (
                        "missed" if full_detected == "no" else ""
                    ),
                    "roi_localization_quality": (
                        "missed" if roi_detected == "no" else ""
                    ),
                    "full_notes": "",
                    "roi_notes": "",
                }
            )

    return output_path


# ============================================================
# Paired contact sheets
# ============================================================


def resize_for_panel(
    image: Image.Image,
) -> Image.Image:
    """缩放 for panel。

    Args:
        image: 本步骤处理的图像对象。

    Returns:
        返回 resized，由函数体中同名变量的计算/收集过程得到。
    """
    resized = image.copy()

    max_width = PANEL_WIDTH - 20

    max_height = PANEL_HEIGHT - 20

    resized.thumbnail(
        (
            max_width,
            max_height,
        ),
        Image.Resampling.LANCZOS,
    )

    return resized


def build_paired_contact_sheets(
    visual_records: list[dict[str, Any]],
) -> list[Path]:
    """构建 paired 联系 sheets。

    Args:
        visual_records: 可视化 records。

    Returns:
        返回 output_paths，由函数体中同名变量的计算/收集过程得到。
    """
    PAIRED_CONTACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove only old paired sheet files
    # generated by this script.
    for old_path in PAIRED_CONTACT_DIR.glob("*_paired_*.jpg"):
        old_path.unlink()

    output_paths = []

    target_groups = defaultdict(list)

    for record in visual_records:
        target_groups[record["target"]].append(record)

    font = ImageFont.load_default()

    for target in TARGET_ORDER:
        records = target_groups.get(
            target,
            [],
        )

        if not records:
            continue

        for start_index in range(
            0,
            len(records),
            PAIRED_CASES_PER_PAGE,
        ):
            page_records = records[start_index : start_index + PAIRED_CASES_PER_PAGE]

            page_number = start_index // PAIRED_CASES_PER_PAGE + 1

            sheet_height = len(page_records) * PAIR_ROW_HEIGHT

            sheet = Image.new(
                "RGB",
                (
                    PAIRED_SHEET_WIDTH,
                    sheet_height,
                ),
                "white",
            )

            draw = ImageDraw.Draw(sheet)

            for row_index, record in enumerate(page_records):
                y_offset = row_index * PAIR_ROW_HEIGHT

                full_path = record["full_path"]

                roi_path = record["roi_path"]

                with Image.open(full_path) as image_file:
                    full_image = image_file.convert("RGB")

                with Image.open(roi_path) as image_file:
                    roi_image = image_file.convert("RGB")

                full_resized = resize_for_panel(full_image)

                roi_resized = resize_for_panel(roi_image)

                full_x = (PANEL_WIDTH - full_resized.width) // 2

                roi_x = PANEL_WIDTH + (PANEL_WIDTH - roi_resized.width) // 2

                image_y = (
                    y_offset + HEADER_HEIGHT + (PANEL_HEIGHT - full_resized.height) // 2
                )

                sheet.paste(
                    full_resized,
                    (
                        full_x,
                        image_y,
                    ),
                )

                roi_image_y = (
                    y_offset + HEADER_HEIGHT + (PANEL_HEIGHT - roi_resized.height) // 2
                )

                sheet.paste(
                    roi_resized,
                    (
                        roi_x,
                        roi_image_y,
                    ),
                )

                benchmark_id = record["benchmark_id"]

                target_name = record["target"]

                draw.text(
                    (
                        10,
                        y_offset + 10,
                    ),
                    (f"{benchmark_id} | " + f"{target_name} | FULL"),
                    fill="black",
                    font=font,
                )

                draw.text(
                    (
                        PANEL_WIDTH + 10,
                        y_offset + 10,
                    ),
                    (f"{benchmark_id} | " + f"{target_name} | ROI"),
                    fill="black",
                    font=font,
                )

                draw.line(
                    (
                        PANEL_WIDTH,
                        y_offset,
                        PANEL_WIDTH,
                        y_offset + PAIR_ROW_HEIGHT,
                    ),
                    fill="gray",
                    width=1,
                )

                draw.line(
                    (
                        0,
                        y_offset + PAIR_ROW_HEIGHT - 1,
                        PAIRED_SHEET_WIDTH,
                        y_offset + PAIR_ROW_HEIGHT - 1,
                    ),
                    fill="gray",
                    width=1,
                )

            output_path = PAIRED_CONTACT_DIR / (
                f"{target}_paired_" + f"{page_number:02d}.jpg"
            )

            sheet.save(
                output_path,
                quality=90,
            )

            output_paths.append(output_path)

    return output_paths


# ============================================================
# Markdown report
# ============================================================


def get_rate(
    summary_rows: list[dict[str, Any]],
    group_key: str,
    group_value: str,
    condition: str,
) -> float | None:
    """取得 rate。

    Args:
        summary_rows: 待处理的 summary 记录。
        group_key: group 标识。
        group_value: group value。
        condition: condition。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    for row in summary_rows:
        if row[group_key] == group_value and row["condition"] == condition:
            return float(row["detection_rate"])

    return None


def build_paired_outcomes(
    benchmark_rows: list[dict[str, str]],
    result_lookup: dict[
        tuple[str, str],
        dict[str, Any],
    ],
) -> dict[str, dict[str, int]]:
    """构建 paired outcomes。

    Args:
        benchmark_rows: 待处理的 benchmark 记录。
        result_lookup: 结果 lookup。

    Returns:
        返回 outcomes，由函数体中同名变量的计算/收集过程得到。
    """
    outcomes = {}

    for target in TARGET_ORDER:
        outcomes[target] = {
            "both_detected": 0,
            "full_only": 0,
            "roi_only": 0,
            "neither": 0,
        }

    for benchmark_row in benchmark_rows:
        benchmark_id = benchmark_row["benchmark_id"]

        target = benchmark_row["target"]

        full_detected = (
            result_lookup[
                (
                    benchmark_id,
                    "full_image",
                )
            ]["detected"]
            == "yes"
        )

        roi_detected = (
            result_lookup[
                (
                    benchmark_id,
                    "roi_conditioned",
                )
            ]["detected"]
            == "yes"
        )

        if full_detected and roi_detected:
            outcomes[target]["both_detected"] += 1

        elif full_detected and not roi_detected:
            outcomes[target]["full_only"] += 1

        elif not full_detected and roi_detected:
            outcomes[target]["roi_only"] += 1

        else:
            outcomes[target]["neither"] += 1

    return outcomes


def save_markdown_summary(
    benchmark_rows: list[dict[str, str]],
    target_summary: list[dict[str, Any]],
    scale_summary: list[dict[str, Any]],
    result_lookup: dict[
        tuple[str, str],
        dict[str, Any],
    ],
    threshold: float,
    roi_margin_ratio: float,
    manual_audit_path: Path,
) -> Path:
    """保存 markdown 汇总。

    Args:
        benchmark_rows: 待处理的 benchmark 记录。
        target_summary: 目标 汇总。
        scale_summary: scale 汇总。
        result_lookup: 结果 lookup。
        threshold: 当前判定规则使用的阈值。
        roi_margin_ratio: roi margin 比例。
        manual_audit_path: 对应文件的相对路径或当前解析后的路径。

    Returns:
        返回 output_path，由函数体中同名变量的计算/收集过程得到。
    """
    output_path = REPORT_ROOT / "evaluation_summary.md"

    paired_outcomes = build_paired_outcomes(
        benchmark_rows,
        result_lookup,
    )

    lines = []

    lines.append("# Verified-Positive Grounding Evaluation")

    lines.append("")
    lines.append("## Experimental Setup")

    lines.append("")
    lines.append(f"- Model: `{MODEL_ID}`")

    lines.append(f"- Detection threshold: {threshold:.2f}")

    lines.append("- Conditions: full image vs " + "annotation-assisted garment ROI")

    lines.append(f"- ROI margin ratio: " + f"{roi_margin_ratio:.3f}")

    lines.append(f"- Evaluated cases: " + f"{len(benchmark_rows)}")

    lines.append("- Fine-grained targets: " + "sleeve, collar, button, zipper")

    lines.append("- Larger-part group: " + "sleeve + collar")

    lines.append("- Small-object group: " + "button + zipper")

    lines.append("")
    lines.append(
        "> The annotation-assisted garment ROI is "
        + "an oracle diagnostic condition and does not "
        + "represent the final deployment pipeline."
    )

    lines.append("")
    lines.append("## Detection Rate by Target")

    lines.append("")
    lines.append("| Target | N | Full | ROI | Delta |")

    lines.append("|---|---:|---:|---:|---:|")

    for target in TARGET_ORDER:
        full_rate = get_rate(
            target_summary,
            "target",
            target,
            "full_image",
        )

        roi_rate = get_rate(
            target_summary,
            "target",
            target,
            "roi_conditioned",
        )

        n_cases = sum(row["target"] == target for row in benchmark_rows)

        if full_rate is None or roi_rate is None:
            continue

        delta = roi_rate - full_rate

        lines.append(
            "| "
            + f"{target} | "
            + f"{n_cases} | "
            + f"{full_rate:.1%} | "
            + f"{roi_rate:.1%} | "
            + f"{delta:+.1%} |"
        )

    lines.append("")
    lines.append("## Detection Rate by Scale Group")

    lines.append("")
    lines.append("| Group | Full | ROI | Delta |")

    lines.append("|---|---:|---:|---:|")

    for scale_group in (
        "larger_part",
        "small_object",
    ):
        full_rate = get_rate(
            scale_summary,
            "scale_group",
            scale_group,
            "full_image",
        )

        roi_rate = get_rate(
            scale_summary,
            "scale_group",
            scale_group,
            "roi_conditioned",
        )

        if full_rate is None or roi_rate is None:
            continue

        delta = roi_rate - full_rate

        lines.append(
            "| "
            + f"{scale_group} | "
            + f"{full_rate:.1%} | "
            + f"{roi_rate:.1%} | "
            + f"{delta:+.1%} |"
        )

    lines.append("")
    lines.append("## Paired Detection Outcomes")

    lines.append("")
    lines.append("| Target | Both | Full only | " + "ROI only | Neither |")

    lines.append("|---|---:|---:|---:|---:|")

    for target in TARGET_ORDER:
        values = paired_outcomes[target]

        lines.append(
            "| "
            + f"{target} | "
            + f"{values['both_detected']} | "
            + f"{values['full_only']} | "
            + f"{values['roi_only']} | "
            + f"{values['neither']} |"
        )

    lines.append("")
    lines.append("## Manual Localization Audit")

    lines.append("")
    lines.append(
        "Automatic detection alone does not establish "
        + "whether the predicted box correctly localizes "
        + "the fine-grained target."
    )

    lines.append("")
    lines.append("Complete the paired audit using:")

    lines.append("")
    lines.append(f"`{manual_audit_path}`")

    lines.append("")
    lines.append("For detections, fill:")

    lines.append("")
    lines.append("- `correct`: target localized appropriately")

    lines.append("- `coarse`: target covered, but box is too broad")

    lines.append("- `wrong`: detection exists but localization is wrong")

    lines.append("- `missed`: no detection " + "(already pre-filled automatically)")

    lines.append("")
    lines.append(
        "Final conclusions about small-object "
        + "localization should be based on this manual "
        + "quality audit, not detection rate alone."
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write("\n".join(lines))

    return output_path


# ============================================================
# Main evaluation
# ============================================================


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    args = parse_args()

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_rows = read_benchmark(
        benchmark_file=(args.benchmark_file),
        limit=args.limit,
    )

    print(f"Benchmark cases: " + f"{len(benchmark_rows)}")

    target_counts = defaultdict(int)

    for row in benchmark_rows:
        target_counts[row["target"]] += 1

    for target in TARGET_ORDER:
        if target in target_counts:
            print(f"  {target}: " + f"{target_counts[target]}")

    device = get_device()

    print(f"Using device: {device}")

    processor, model = load_model(device)

    case_rows = []

    result_lookup = {}

    visual_records = []

    for benchmark_row in tqdm(
        benchmark_rows,
        desc="Evaluating benchmark",
    ):
        benchmark_id = benchmark_row["benchmark_id"]

        target = benchmark_row["target"]

        image_name = benchmark_row["image_name"]

        item_id = benchmark_row["item_id"]

        image_path = IMAGE_DIR / image_name

        if not image_path.is_file():
            raise FileNotFoundError("Original image not found: " + f"{image_path}")

        with Image.open(image_path) as image_file:
            original_image = image_file.convert("RGB")

        image_width = original_image.width

        image_height = original_image.height

        annotation = load_annotation(image_name)

        garment = get_exact_garment(
            annotation,
            item_id,
        )

        garment_bbox = clip_bbox(
            garment["bbox"],
            image_width,
            image_height,
        )

        roi_bbox = expand_bbox(
            garment_bbox,
            image_width,
            image_height,
            args.roi_margin_ratio,
        )

        # ----------------------------------------------------
        # Condition 1: full image
        # ----------------------------------------------------

        (
            full_detections,
            full_seconds,
        ) = run_grounding_dino(
            image=original_image,
            prompt=target,
            processor=processor,
            model=model,
            device=device,
            threshold=args.threshold,
        )

        # ----------------------------------------------------
        # Condition 2: oracle garment ROI
        # ----------------------------------------------------

        (
            roi_x_min,
            roi_y_min,
            roi_x_max,
            roi_y_max,
        ) = roi_bbox

        roi_image = original_image.crop(
            (
                roi_x_min,
                roi_y_min,
                roi_x_max + 1,
                roi_y_max + 1,
            )
        )

        (
            roi_local_detections,
            roi_seconds,
        ) = run_grounding_dino(
            image=roi_image,
            prompt=target,
            processor=processor,
            model=model,
            device=device,
            threshold=args.threshold,
        )

        roi_detections = map_roi_detections_to_full_image(
            roi_local_detections,
            roi_bbox,
        )

        # ----------------------------------------------------
        # Store result rows
        # ----------------------------------------------------

        full_row = summarize_single_condition(
            benchmark_row=(benchmark_row),
            condition="full_image",
            detections=(full_detections),
            inference_seconds=(full_seconds),
            image_width=(image_width),
            image_height=(image_height),
            threshold=(args.threshold),
            roi_margin_ratio=(args.roi_margin_ratio),
            roi_bbox=(roi_bbox),
        )

        roi_row = summarize_single_condition(
            benchmark_row=(benchmark_row),
            condition=("roi_conditioned"),
            detections=(roi_detections),
            inference_seconds=(roi_seconds),
            image_width=(image_width),
            image_height=(image_height),
            threshold=(args.threshold),
            roi_margin_ratio=(args.roi_margin_ratio),
            roi_bbox=(roi_bbox),
        )

        case_rows.extend(
            [
                full_row,
                roi_row,
            ]
        )

        result_lookup[
            (
                benchmark_id,
                "full_image",
            )
        ] = full_row

        result_lookup[
            (
                benchmark_id,
                "roi_conditioned",
            )
        ] = roi_row

        save_raw_case_result(
            benchmark_row=(benchmark_row),
            roi_bbox=roi_bbox,
            full_detections=(full_detections),
            roi_detections=(roi_detections),
            threshold=(args.threshold),
            roi_margin_ratio=(args.roi_margin_ratio),
        )

        if not args.skip_visuals:
            (
                full_path,
                roi_path,
            ) = save_visualizations(
                benchmark_id=(benchmark_id),
                original_image=(original_image),
                full_detections=(full_detections),
                roi_detections=(roi_detections),
                roi_bbox=(roi_bbox),
                target=target,
            )

            visual_records.append(
                {
                    "benchmark_id": (benchmark_id),
                    "target": target,
                    "full_path": (full_path),
                    "roi_path": (roi_path),
                }
            )

    # ========================================================
    # Reports
    # ========================================================

    case_results_path = save_case_results(case_rows)

    target_summary = build_target_summary(case_rows)

    target_summary_path = save_target_summary(target_summary)

    scale_summary = build_scale_group_summary(case_rows)

    scale_summary_path = save_scale_group_summary(scale_summary)

    manual_audit_path = save_manual_audit(
        benchmark_rows=(benchmark_rows),
        result_lookup=(result_lookup),
    )

    paired_paths = []

    if not args.skip_visuals:
        paired_paths = build_paired_contact_sheets(visual_records)

    markdown_path = save_markdown_summary(
        benchmark_rows=(benchmark_rows),
        target_summary=(target_summary),
        scale_summary=(scale_summary),
        result_lookup=(result_lookup),
        threshold=(args.threshold),
        roi_margin_ratio=(args.roi_margin_ratio),
        manual_audit_path=(manual_audit_path),
    )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("Verified-positive evaluation completed.")

    print(f"Evaluated benchmark cases: " + f"{len(benchmark_rows)}")

    print(f"Inference runs: " + f"{len(case_rows)}")

    print()

    print("=== Detection Rate by Target ===")

    for target in TARGET_ORDER:
        full_rate = get_rate(
            target_summary,
            "target",
            target,
            "full_image",
        )

        roi_rate = get_rate(
            target_summary,
            "target",
            target,
            "roi_conditioned",
        )

        if full_rate is None or roi_rate is None:
            continue

        print(
            f"{target:8s}: "
            + f"full={full_rate:.1%}, "
            + f"roi={roi_rate:.1%}, "
            + f"delta="
            + f"{roi_rate - full_rate:+.1%}"
        )

    print()

    print("=== Detection Rate by Scale Group ===")

    for scale_group in (
        "larger_part",
        "small_object",
    ):
        full_rate = get_rate(
            scale_summary,
            "scale_group",
            scale_group,
            "full_image",
        )

        roi_rate = get_rate(
            scale_summary,
            "scale_group",
            scale_group,
            "roi_conditioned",
        )

        if full_rate is None or roi_rate is None:
            continue

        print(
            f"{scale_group:12s}: "
            + f"full={full_rate:.1%}, "
            + f"roi={roi_rate:.1%}, "
            + f"delta="
            + f"{roi_rate - full_rate:+.1%}"
        )

    print()

    print(f"Case results: " + f"{case_results_path}")

    print(f"Target summary: " + f"{target_summary_path}")

    print(f"Scale summary: " + f"{scale_summary_path}")

    print(f"Manual audit: " + f"{manual_audit_path}")

    print(f"Markdown summary: " + f"{markdown_path}")

    if paired_paths:
        print("Paired contact sheets: " + f"{len(paired_paths)}")

        print(f"Paired sheets directory: " + f"{PAIRED_CONTACT_DIR}")

    print()
    print("NEXT STEP:")

    print(
        "Review paired contact sheets and fill "
        + "full_localization_quality / "
        + "roi_localization_quality using:"
    )

    print("correct / coarse / wrong / missed")


if __name__ == "__main__":
    main()
