"""Evaluate the verified-positive benchmark with Grounding DINO.

Three controlled diagnostic conditions are evaluated:

1. full_image
   Run Grounding DINO on the original image.

2. roi_conditioned
   Run Grounding DINO on the DeepFashion2 garment bounding box.
   This is an oracle garment ROI diagnostic, not a deployment pipeline.

3. local_enlarged
   Run Grounding DINO on deterministic target-conditioned local windows
   inside the garment ROI, enlarge each crop before inference, map detections
   back to the original image, and merge duplicate boxes.

Important:
The local windows use only coarse spatial priors derived from the target name.
They do NOT use a fine-grained ground-truth bbox for sleeve/collar/button/zipper.
Therefore this remains a diagnostic experiment rather than a supervised
part-localization evaluation.

Automatic metrics:
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

must still be manually audited because the benchmark has target-presence
verification but no manually annotated fine-grained target bbox.
"""

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


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

DEFAULT_THRESHOLD = 0.30
DEFAULT_ROI_MARGIN_RATIO = 0.05
DEFAULT_LOCAL_ENLARGE_FACTOR = 2.0
DEFAULT_LOCAL_NMS_IOU = 0.50

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT.parent
    / "fashion_data"
    / "raw"
    / "train"
    / "train"
)

IMAGE_DIR = DATASET_ROOT / "image"
ANNOTATION_DIR = DATASET_ROOT / "annos"

DEFAULT_BENCHMARK_FILE = (
    PROJECT_ROOT
    / "reports"
    / "verified_positive_benchmark"
    / "verified_positive_benchmark.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "verified_positive_evaluation"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "verified_positive_evaluation"
)

FULL_IMAGE_OUTPUT_DIR = OUTPUT_ROOT / "full_image"
ROI_OUTPUT_DIR = OUTPUT_ROOT / "roi_conditioned"
LOCAL_OUTPUT_DIR = OUTPUT_ROOT / "local_enlarged"
RAW_OUTPUT_DIR = OUTPUT_ROOT / "raw_results"
TRIPLET_CONTACT_DIR = OUTPUT_ROOT / "triplet_contact_sheets"

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

CONDITIONS = [
    "full_image",
    "roi_conditioned",
    "local_enlarged",
]


# ============================================================
# Target-conditioned local crop rules
# ============================================================
#
# Each tuple is:
# (x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio)
#
# Ratios are relative to the expanded garment ROI.
#
# These windows are deliberately simple and deterministic.
# They are NOT fine-grained GT boxes.
#
# sleeve:
#   evaluate left and right upper-side garment regions separately.
#
# collar:
#   focus on the upper-center garment region.
#
# button:
#   focus on the central front torso strip.
#
# zipper:
#   focus on the central vertical garment strip.
# ============================================================

LOCAL_WINDOW_RULES = {
    "sleeve": [
        (0.00, 0.00, 0.48, 0.82),
        (0.52, 0.00, 1.00, 0.82),
    ],
    "collar": [
        (0.12, 0.00, 0.88, 0.48),
    ],
    "button": [
        (0.18, 0.08, 0.82, 0.92),
    ],
    "zipper": [
        (0.18, 0.04, 0.82, 0.96),
    ],
}


# ============================================================
# Contact-sheet configuration
# ============================================================

TRIPLET_CASES_PER_PAGE = 3

PANEL_WIDTH = 420
PANEL_HEIGHT = 320
HEADER_HEIGHT = 45
TRIPLET_ROW_HEIGHT = PANEL_HEIGHT + HEADER_HEIGHT
TRIPLET_SHEET_WIDTH = PANEL_WIDTH * 3


# ============================================================
# Arguments
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate verified-positive localization benchmark "
            "using full-image, garment-ROI, and local-enlarged "
            "Grounding DINO diagnostics."
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
            "Grounding DINO detection threshold "
            f"(default: {DEFAULT_THRESHOLD})."
        ),
    )

    parser.add_argument(
        "--roi-margin-ratio",
        type=float,
        default=DEFAULT_ROI_MARGIN_RATIO,
        help=(
            "Relative margin added around annotated garment ROI "
            f"(default: {DEFAULT_ROI_MARGIN_RATIO})."
        ),
    )

    parser.add_argument(
        "--local-enlarge-factor",
        type=float,
        default=DEFAULT_LOCAL_ENLARGE_FACTOR,
        help=(
            "Pixel enlargement factor applied to each local crop "
            f"before inference (default: {DEFAULT_LOCAL_ENLARGE_FACTOR})."
        ),
    )

    parser.add_argument(
        "--local-nms-iou",
        type=float,
        default=DEFAULT_LOCAL_NMS_IOU,
        help=(
            "IoU threshold used to merge duplicate detections from "
            f"multiple local windows (default: {DEFAULT_LOCAL_NMS_IOU})."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional global case limit.",
    )

    parser.add_argument(
        "--per-target-limit",
        type=int,
        default=None,
        help=(
            "Optional per-target case limit. "
            "Use --per-target-limit 1 for a four-target smoke test."
        ),
    )

    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip visualization and triplet contact-sheet generation.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    if not 0.0 <= args.roi_margin_ratio <= 1.0:
        raise ValueError(
            "--roi-margin-ratio must be between 0 and 1."
        )

    if args.local_enlarge_factor < 1.0:
        raise ValueError(
            "--local-enlarge-factor must be >= 1.0."
        )

    if not 0.0 <= args.local_nms_iou <= 1.0:
        raise ValueError(
            "--local-nms-iou must be between 0 and 1."
        )

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit must be greater than zero."
        )

    if (
        args.per_target_limit is not None
        and args.per_target_limit <= 0
    ):
        raise ValueError(
            "--per-target-limit must be greater than zero."
        )

    if (
        args.limit is not None
        and args.per_target_limit is not None
    ):
        raise ValueError(
            "Use either --limit or --per-target-limit, not both."
        )

    return args


# ============================================================
# Device and model
# ============================================================


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def synchronize_device(
    device: torch.device,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_model(
    device: torch.device,
) -> tuple[Any, Any]:
    print(f"Loading model: {MODEL_ID}")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(
            MODEL_ID
        )
    )

    model.to(device)
    model.eval()

    return processor, model


# ============================================================
# Benchmark loading
# ============================================================


def read_benchmark(
    benchmark_file: Path,
    limit: int | None,
    per_target_limit: int | None,
) -> list[dict[str, str]]:
    if not benchmark_file.is_file():
        raise FileNotFoundError(
            f"Benchmark file not found: {benchmark_file}"
        )

    with benchmark_file.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        rows = list(
            csv.DictReader(csv_file)
        )

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
        raise ValueError(
            "Benchmark CSV is empty."
        )

    missing_fields = (
        required_fields
        - set(rows[0].keys())
    )

    if missing_fields:
        raise ValueError(
            "Benchmark CSV is missing fields: "
            f"{sorted(missing_fields)}"
        )

    usable_rows = []
    seen_ids = set()

    for row in rows:
        benchmark_id = (
            row["benchmark_id"].strip()
        )

        if benchmark_id in seen_ids:
            raise ValueError(
                f"Duplicate benchmark_id: {benchmark_id}"
            )

        seen_ids.add(
            benchmark_id
        )

        if (
            row["verified_positive"]
            .strip()
            .lower()
            != "yes"
        ):
            continue

        if (
            row["usable_for_evaluation"]
            .strip()
            .lower()
            != "yes"
        ):
            continue

        target = (
            row["target"]
            .strip()
            .lower()
        )

        if target not in TARGET_ORDER:
            raise ValueError(
                f"Unknown target: {target}"
            )

        usable_rows.append(
            row
        )

    if per_target_limit is not None:
        grouped = defaultdict(list)

        for row in usable_rows:
            grouped[
                row["target"]
            ].append(
                row
            )

        selected = []

        for target in TARGET_ORDER:
            selected.extend(
                grouped[target][
                    :per_target_limit
                ]
            )

        usable_rows = selected

    if limit is not None:
        usable_rows = (
            usable_rows[:limit]
        )

    return usable_rows


# ============================================================
# DeepFashion2 annotations
# ============================================================


def load_annotation(
    image_name: str,
) -> dict[str, Any]:
    annotation_path = (
        ANNOTATION_DIR
        / f"{Path(image_name).stem}.json"
    )

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Annotation not found: {annotation_path}"
        )

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(
            json_file
        )


def get_exact_garment(
    annotation: dict[str, Any],
    item_id: str,
) -> dict[str, Any]:
    if item_id not in annotation:
        raise ValueError(
            f"item_id {item_id} was not found "
            "in the annotation."
        )

    garment = annotation[
        item_id
    ]

    if not isinstance(
        garment,
        dict,
    ):
        raise ValueError(
            f"Invalid garment record: {item_id}"
        )

    bbox = garment.get(
        "bounding_box"
    )

    if bbox is None or len(bbox) != 4:
        raise ValueError(
            f"Invalid bounding box for {item_id}"
        )

    category_name = garment.get(
        "category_name",
        "",
    )

    return {
        "item_id": item_id,
        "category_name": str(
            category_name
        ),
        "bbox": [
            int(
                round(
                    float(value)
                )
            )
            for value in bbox
        ],
    }


def clip_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
) -> list[int]:
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

    margin_x = int(
        round(
            width
            * margin_ratio
        )
    )

    margin_y = int(
        round(
            height
            * margin_ratio
        )
    )

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
# Local crop construction
# ============================================================


def relative_window_to_bbox(
    parent_bbox: list[int],
    relative_window: tuple[
        float,
        float,
        float,
        float,
    ],
    image_width: int,
    image_height: int,
) -> list[int]:
    (
        parent_x_min,
        parent_y_min,
        parent_x_max,
        parent_y_max,
    ) = parent_bbox

    parent_width = max(
        1,
        parent_x_max - parent_x_min + 1,
    )

    parent_height = max(
        1,
        parent_y_max - parent_y_min + 1,
    )

    (
        rel_x_min,
        rel_y_min,
        rel_x_max,
        rel_y_max,
    ) = relative_window

    x_min = (
        parent_x_min
        + int(
            round(
                rel_x_min
                * parent_width
            )
        )
    )

    y_min = (
        parent_y_min
        + int(
            round(
                rel_y_min
                * parent_height
            )
        )
    )

    x_max = (
        parent_x_min
        + int(
            round(
                rel_x_max
                * parent_width
            )
        )
        - 1
    )

    y_max = (
        parent_y_min
        + int(
            round(
                rel_y_max
                * parent_height
            )
        )
        - 1
    )

    return clip_bbox(
        [
            x_min,
            y_min,
            x_max,
            y_max,
        ],
        image_width,
        image_height,
    )


def build_local_windows(
    target: str,
    garment_roi: list[int],
    image_width: int,
    image_height: int,
) -> list[list[int]]:
    rules = LOCAL_WINDOW_RULES.get(
        target
    )

    if not rules:
        raise ValueError(
            f"No local-window rules for target: {target}"
        )

    return [
        relative_window_to_bbox(
            garment_roi,
            rule,
            image_width,
            image_height,
        )
        for rule in rules
    ]


# ============================================================
# Grounding DINO inference
# ============================================================


def normalize_label(
    label: Any,
) -> str:
    if isinstance(
        label,
        str,
    ):
        return label

    if torch.is_tensor(
        label
    ):
        if label.numel() == 1:
            return str(
                label.item()
            )

        return str(
            label.detach()
            .cpu()
            .tolist()
        )

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
    synchronize_device(
        device
    )

    start_time = (
        time.perf_counter()
    )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(
            **inputs
        )

    results = (
        processor
        .post_process_grounded_object_detection(
            outputs,
            inputs[
                "input_ids"
            ],
            threshold=threshold,
            target_sizes=[
                image.size[::-1]
            ],
        )
    )

    synchronize_device(
        device
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

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
        labels = result[
            "text_labels"
        ]
    elif "labels" in result:
        labels = result[
            "labels"
        ]
    else:
        labels = [
            prompt
        ] * len(scores)

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
        if torch.is_tensor(
            score
        ):
            score_value = float(
                score.detach()
                .cpu()
                .item()
            )
        else:
            score_value = float(
                score
            )

        if torch.is_tensor(
            box
        ):
            box_values = (
                box.detach()
                .cpu()
                .tolist()
            )
        else:
            box_values = list(
                box
            )

        detections.append(
            {
                "label": (
                    normalize_label(
                        label
                    )
                ),
                "score": score_value,
                "box": [
                    float(value)
                    for value
                    in box_values
                ],
            }
        )

    detections.sort(
        key=lambda detection: (
            detection["score"]
        ),
        reverse=True,
    )

    return (
        detections,
        elapsed_seconds,
    )


def map_crop_detections_to_full_image(
    detections: list[
        dict[str, Any]
    ],
    crop_bbox: list[int],
    scale_factor: float = 1.0,
) -> list[dict[str, Any]]:
    x_offset = crop_bbox[0]
    y_offset = crop_bbox[1]

    mapped = []

    for detection in detections:
        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = detection["box"]

        x_min /= scale_factor
        y_min /= scale_factor
        x_max /= scale_factor
        y_max /= scale_factor

        mapped.append(
            {
                "label": (
                    detection[
                        "label"
                    ]
                ),
                "score": (
                    detection[
                        "score"
                    ]
                ),
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
# NMS for multi-window local results
# ============================================================


def box_iou(
    box_a: list[float],
    box_b: list[float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(
        ax1,
        bx1,
    )

    inter_y1 = max(
        ay1,
        by1,
    )

    inter_x2 = min(
        ax2,
        bx2,
    )

    inter_y2 = min(
        ay2,
        by2,
    )

    inter_width = max(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_height = max(
        0.0,
        inter_y2 - inter_y1,
    )

    intersection = (
        inter_width
        * inter_height
    )

    area_a = (
        max(
            0.0,
            ax2 - ax1,
        )
        * max(
            0.0,
            ay2 - ay1,
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1,
        )
        * max(
            0.0,
            by2 - by1,
        )
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return (
        intersection
        / union
    )


def nms_detections(
    detections: list[
        dict[str, Any]
    ],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    ordered = sorted(
        detections,
        key=lambda detection: (
            detection["score"]
        ),
        reverse=True,
    )

    kept = []

    for detection in ordered:
        suppress = False

        for selected in kept:
            if (
                box_iou(
                    detection["box"],
                    selected["box"],
                )
                >= iou_threshold
            ):
                suppress = True
                break

        if not suppress:
            kept.append(
                detection
            )

    return kept


# ============================================================
# Local-enlarged inference
# ============================================================


def run_local_enlarged(
    original_image: Image.Image,
    target: str,
    garment_roi: list[int],
    processor: Any,
    model: Any,
    device: torch.device,
    threshold: float,
    enlarge_factor: float,
    nms_iou: float,
) -> tuple[
    list[dict[str, Any]],
    float,
    list[list[int]],
]:
    local_windows = build_local_windows(
        target=target,
        garment_roi=garment_roi,
        image_width=original_image.width,
        image_height=original_image.height,
    )

    all_mapped_detections = []
    total_seconds = 0.0

    for crop_bbox in local_windows:
        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = crop_bbox

        local_crop = (
            original_image.crop(
                (
                    x_min,
                    y_min,
                    x_max + 1,
                    y_max + 1,
                )
            )
        )

        if enlarge_factor > 1.0:
            enlarged_width = max(
                1,
                int(
                    round(
                        local_crop.width
                        * enlarge_factor
                    )
                ),
            )

            enlarged_height = max(
                1,
                int(
                    round(
                        local_crop.height
                        * enlarge_factor
                    )
                ),
            )

            local_input = (
                local_crop.resize(
                    (
                        enlarged_width,
                        enlarged_height,
                    ),
                    Image.Resampling.BICUBIC,
                )
            )
        else:
            local_input = local_crop

        (
            local_detections,
            local_seconds,
        ) = run_grounding_dino(
            image=local_input,
            prompt=target,
            processor=processor,
            model=model,
            device=device,
            threshold=threshold,
        )

        total_seconds += (
            local_seconds
        )

        mapped = (
            map_crop_detections_to_full_image(
                detections=local_detections,
                crop_bbox=crop_bbox,
                scale_factor=enlarge_factor,
            )
        )

        all_mapped_detections.extend(
            mapped
        )

    merged = nms_detections(
        detections=all_mapped_detections,
        iou_threshold=nms_iou,
    )

    return (
        merged,
        total_seconds,
        local_windows,
    )


# ============================================================
# Detection statistics
# ============================================================


def detection_box_area_ratio(
    detection: dict[str, Any],
    image_width: int,
    image_height: int,
) -> float:
    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = detection[
        "box"
    ]

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
        float(
            image_width
            * image_height
        ),
    )

    return (
        width
        * height
        / image_area
    )


def summarize_single_condition(
    benchmark_row: dict[str, str],
    condition: str,
    detections: list[
        dict[str, Any]
    ],
    inference_seconds: float,
    image_width: int,
    image_height: int,
    threshold: float,
    roi_margin_ratio: float,
    roi_bbox: list[int],
    local_windows: list[list[int]] | None,
    local_enlarge_factor: float,
) -> dict[str, Any]:
    detected = bool(
        detections
    )

    top_score = ""
    mean_score = ""
    top_box_area_ratio = ""

    if detected:
        top_score = (
            detections[0][
                "score"
            ]
        )

        mean_score = (
            sum(
                detection[
                    "score"
                ]
                for detection
                in detections
            )
            / len(detections)
        )

        top_box_area_ratio = (
            detection_box_area_ratio(
                detections[0],
                image_width,
                image_height,
            )
        )

    return {
        "benchmark_id": (
            benchmark_row[
                "benchmark_id"
            ]
        ),
        "target": (
            benchmark_row[
                "target"
            ]
        ),
        "scale_group": (
            SCALE_GROUPS[
                benchmark_row[
                    "target"
                ]
            ]
        ),
        "condition": condition,
        "image_name": (
            benchmark_row[
                "image_name"
            ]
        ),
        "item_id": (
            benchmark_row[
                "item_id"
            ]
        ),
        "category_name": (
            benchmark_row[
                "category_name"
            ]
        ),
        "visibility": (
            benchmark_row[
                "visibility"
            ]
        ),
        "prompt": (
            benchmark_row[
                "target"
            ]
        ),
        "threshold": threshold,
        "roi_margin_ratio": (
            roi_margin_ratio
        ),
        "roi_bbox": (
            json.dumps(
                roi_bbox
            )
        ),
        "local_windows": (
            json.dumps(
                local_windows or []
            )
        ),
        "local_enlarge_factor": (
            local_enlarge_factor
        ),
        "detected": (
            "yes"
            if detected
            else "no"
        ),
        "num_detections": (
            len(detections)
        ),
        "top_score": top_score,
        "mean_score": mean_score,
        "top_box_area_ratio": (
            top_box_area_ratio
        ),
        "inference_seconds": (
            inference_seconds
        ),
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
    detections: list[
        dict[str, Any]
    ],
    title: str,
    garment_roi: list[int] | None = None,
    local_windows: list[list[int]] | None = None,
) -> Image.Image:
    visualization = (
        image.copy()
        .convert("RGB")
    )

    draw = ImageDraw.Draw(
        visualization
    )

    font = (
        ImageFont.load_default()
    )

    if garment_roi is not None:
        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = garment_roi

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

    if local_windows:
        for index, window in enumerate(
            local_windows,
            start=1,
        ):
            (
                x_min,
                y_min,
                x_max,
                y_max,
            ) = window

            draw.rectangle(
                [
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                ],
                outline="cyan",
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
                f"local {index}",
                fill="cyan",
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
        ) = detection[
            "box"
        ]

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

        label = (
            f"{index}: "
            f"{detection['label']} "
            f"{detection['score']:.3f}"
        )

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
                560,
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
    full_detections: list[
        dict[str, Any]
    ],
    roi_detections: list[
        dict[str, Any]
    ],
    local_detections: list[
        dict[str, Any]
    ],
    roi_bbox: list[int],
    local_windows: list[list[int]],
    target: str,
) -> tuple[
    Path,
    Path,
    Path,
]:
    FULL_IMAGE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ROI_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOCAL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_visualization = (
        draw_detection_boxes(
            image=original_image,
            detections=full_detections,
            title=(
                f"{benchmark_id} | "
                f"FULL | {target}"
            ),
        )
    )

    roi_visualization = (
        draw_detection_boxes(
            image=original_image,
            detections=roi_detections,
            title=(
                f"{benchmark_id} | "
                f"ROI | {target}"
            ),
            garment_roi=roi_bbox,
        )
    )

    local_visualization = (
        draw_detection_boxes(
            image=original_image,
            detections=local_detections,
            title=(
                f"{benchmark_id} | "
                f"LOCAL x2 | {target}"
            ),
            garment_roi=roi_bbox,
            local_windows=local_windows,
        )
    )

    full_path = (
        FULL_IMAGE_OUTPUT_DIR
        / f"{benchmark_id}.jpg"
    )

    roi_path = (
        ROI_OUTPUT_DIR
        / f"{benchmark_id}.jpg"
    )

    local_path = (
        LOCAL_OUTPUT_DIR
        / f"{benchmark_id}.jpg"
    )

    full_visualization.save(
        full_path,
        quality=92,
    )

    roi_visualization.save(
        roi_path,
        quality=92,
    )

    local_visualization.save(
        local_path,
        quality=92,
    )

    return (
        full_path,
        roi_path,
        local_path,
    )


# ============================================================
# Raw JSON
# ============================================================


def save_raw_case_result(
    benchmark_row: dict[str, str],
    roi_bbox: list[int],
    local_windows: list[list[int]],
    full_detections: list[
        dict[str, Any]
    ],
    roi_detections: list[
        dict[str, Any]
    ],
    local_detections: list[
        dict[str, Any]
    ],
    threshold: float,
    roi_margin_ratio: float,
    local_enlarge_factor: float,
) -> Path:
    RAW_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RAW_OUTPUT_DIR
        / (
            f"{benchmark_row['benchmark_id']}"
            ".json"
        )
    )

    payload = {
        "benchmark_id": (
            benchmark_row[
                "benchmark_id"
            ]
        ),
        "target": (
            benchmark_row[
                "target"
            ]
        ),
        "image_name": (
            benchmark_row[
                "image_name"
            ]
        ),
        "item_id": (
            benchmark_row[
                "item_id"
            ]
        ),
        "threshold": threshold,
        "roi_margin_ratio": (
            roi_margin_ratio
        ),
        "local_enlarge_factor": (
            local_enlarge_factor
        ),
        "oracle_garment_roi": (
            roi_bbox
        ),
        "local_windows": (
            local_windows
        ),
        "full_image_detections": (
            full_detections
        ),
        "roi_conditioned_detections": (
            roi_detections
        ),
        "local_enlarged_detections": (
            local_detections
        ),
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
    rows: list[
        dict[str, Any]
    ],
) -> Path:
    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_ROOT
        / "case_results.csv"
    )

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
        "local_windows",
        "local_enlarge_factor",
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
        writer.writerows(
            rows
        )

    return output_path


# ============================================================
# Summary helpers
# ============================================================


def safe_mean(
    values: list[float],
) -> float:
    if not values:
        return float("nan")

    return statistics.mean(
        values
    )


def safe_median(
    values: list[float],
) -> float:
    if not values:
        return float("nan")

    return statistics.median(
        values
    )


def float_or_none(
    value: Any,
) -> float | None:
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
    rows: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    n_cases = len(rows)

    detected_rows = [
        row
        for row in rows
        if row[
            "detected"
        ] == "yes"
    ]

    detected_count = len(
        detected_rows
    )

    detection_rate = (
        detected_count
        / n_cases
        if n_cases
        else 0.0
    )

    top_scores = []
    box_ratios = []

    for row in detected_rows:
        top_score = (
            float_or_none(
                row[
                    "top_score"
                ]
            )
        )

        if top_score is not None:
            top_scores.append(
                top_score
            )

        box_ratio = (
            float_or_none(
                row[
                    "top_box_area_ratio"
                ]
            )
        )

        if box_ratio is not None:
            box_ratios.append(
                box_ratio
            )

    num_detections = [
        float(
            row[
                "num_detections"
            ]
        )
        for row in rows
    ]

    inference_times = [
        float(
            row[
                "inference_seconds"
            ]
        )
        for row in rows
    ]

    return {
        "n_cases": n_cases,
        "detected_count": (
            detected_count
        ),
        "detection_rate": (
            detection_rate
        ),
        "mean_num_detections": (
            safe_mean(
                num_detections
            )
        ),
        "mean_top_score_detected": (
            safe_mean(
                top_scores
            )
        ),
        "median_top_score_detected": (
            safe_median(
                top_scores
            )
        ),
        "mean_top_box_area_ratio_detected": (
            safe_mean(
                box_ratios
            )
        ),
        "mean_inference_seconds": (
            safe_mean(
                inference_times
            )
        ),
    }


def format_number(
    value: Any,
    digits: int = 4,
) -> str:
    if isinstance(
        value,
        float,
    ) and math.isnan(
        value
    ):
        return ""

    if isinstance(
        value,
        float,
    ):
        return f"{value:.{digits}f}"

    return str(value)


def build_summary(
    case_rows: list[
        dict[str, Any]
    ],
    group_field: str,
    group_order: list[str],
) -> list[dict[str, Any]]:
    grouped = defaultdict(
        list
    )

    for row in case_rows:
        key = (
            row[group_field],
            row["condition"],
        )

        grouped[
            key
        ].append(
            row
        )

    summary_rows = []

    for group_value in group_order:
        for condition in CONDITIONS:
            rows = grouped.get(
                (
                    group_value,
                    condition,
                ),
                [],
            )

            if not rows:
                continue

            metrics = (
                aggregate_group(
                    rows
                )
            )

            row = {
                group_field: (
                    group_value
                ),
                "condition": (
                    condition
                ),
                **metrics,
            }

            if group_field == "target":
                row[
                    "scale_group"
                ] = (
                    SCALE_GROUPS[
                        group_value
                    ]
                )

            summary_rows.append(
                row
            )

    return summary_rows


def save_summary_csv(
    rows: list[
        dict[str, Any]
    ],
    output_name: str,
    fieldnames: list[str],
) -> Path:
    output_path = (
        REPORT_ROOT
        / output_name
    )

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
            writer.writerow(
                {
                    key: (
                        format_number(
                            row[key]
                        )
                    )
                    for key
                    in fieldnames
                }
            )

    return output_path


# ============================================================
# Manual audit
# ============================================================


def save_manual_audit(
    benchmark_rows: list[
        dict[str, str]
    ],
    result_lookup: dict[
        tuple[str, str],
        dict[str, Any],
    ],
) -> Path:
    output_path = (
        REPORT_ROOT
        / "triplet_manual_audit.csv"
    )

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
        "local_detected",
        "local_top_score",
        "full_localization_quality",
        "roi_localization_quality",
        "local_localization_quality",
        "full_notes",
        "roi_notes",
        "local_notes",
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

        for benchmark_row in (
            benchmark_rows
        ):
            benchmark_id = (
                benchmark_row[
                    "benchmark_id"
                ]
            )

            condition_rows = {
                condition: (
                    result_lookup[
                        (
                            benchmark_id,
                            condition,
                        )
                    ]
                )
                for condition
                in CONDITIONS
            }

            full_result = (
                condition_rows[
                    "full_image"
                ]
            )

            roi_result = (
                condition_rows[
                    "roi_conditioned"
                ]
            )

            local_result = (
                condition_rows[
                    "local_enlarged"
                ]
            )

            writer.writerow(
                {
                    "benchmark_id": (
                        benchmark_id
                    ),
                    "target": (
                        benchmark_row[
                            "target"
                        ]
                    ),
                    "scale_group": (
                        SCALE_GROUPS[
                            benchmark_row[
                                "target"
                            ]
                        ]
                    ),
                    "image_name": (
                        benchmark_row[
                            "image_name"
                        ]
                    ),
                    "item_id": (
                        benchmark_row[
                            "item_id"
                        ]
                    ),
                    "category_name": (
                        benchmark_row[
                            "category_name"
                        ]
                    ),
                    "visibility": (
                        benchmark_row[
                            "visibility"
                        ]
                    ),
                    "full_detected": (
                        full_result[
                            "detected"
                        ]
                    ),
                    "full_top_score": (
                        full_result[
                            "top_score"
                        ]
                    ),
                    "roi_detected": (
                        roi_result[
                            "detected"
                        ]
                    ),
                    "roi_top_score": (
                        roi_result[
                            "top_score"
                        ]
                    ),
                    "local_detected": (
                        local_result[
                            "detected"
                        ]
                    ),
                    "local_top_score": (
                        local_result[
                            "top_score"
                        ]
                    ),
                    "full_localization_quality": (
                        "missed"
                        if (
                            full_result[
                                "detected"
                            ]
                            == "no"
                        )
                        else ""
                    ),
                    "roi_localization_quality": (
                        "missed"
                        if (
                            roi_result[
                                "detected"
                            ]
                            == "no"
                        )
                        else ""
                    ),
                    "local_localization_quality": (
                        "missed"
                        if (
                            local_result[
                                "detected"
                            ]
                            == "no"
                        )
                        else ""
                    ),
                    "full_notes": "",
                    "roi_notes": "",
                    "local_notes": "",
                }
            )

    return output_path


# ============================================================
# Triplet contact sheets
# ============================================================


def resize_for_panel(
    image: Image.Image,
) -> Image.Image:
    resized = image.copy()

    resized.thumbnail(
        (
            PANEL_WIDTH - 20,
            PANEL_HEIGHT - 20,
        ),
        Image.Resampling.LANCZOS,
    )

    return resized


def build_triplet_contact_sheets(
    visual_records: list[
        dict[str, Any]
    ],
) -> list[Path]:
    TRIPLET_CONTACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old_path in (
        TRIPLET_CONTACT_DIR.glob(
            "*_triplet_*.jpg"
        )
    ):
        old_path.unlink()

    output_paths = []
    target_groups = defaultdict(
        list
    )

    for record in visual_records:
        target_groups[
            record["target"]
        ].append(
            record
        )

    font = (
        ImageFont.load_default()
    )

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
            TRIPLET_CASES_PER_PAGE,
        ):
            page_records = (
                records[
                    start_index:
                    start_index
                    + TRIPLET_CASES_PER_PAGE
                ]
            )

            page_number = (
                start_index
                // TRIPLET_CASES_PER_PAGE
                + 1
            )

            sheet_height = (
                len(page_records)
                * TRIPLET_ROW_HEIGHT
            )

            sheet = Image.new(
                "RGB",
                (
                    TRIPLET_SHEET_WIDTH,
                    sheet_height,
                ),
                "white",
            )

            draw = (
                ImageDraw.Draw(
                    sheet
                )
            )

            for row_index, record in enumerate(
                page_records
            ):
                y_offset = (
                    row_index
                    * TRIPLET_ROW_HEIGHT
                )

                panel_specs = [
                    (
                        "FULL",
                        record[
                            "full_path"
                        ],
                    ),
                    (
                        "ROI",
                        record[
                            "roi_path"
                        ],
                    ),
                    (
                        "LOCAL",
                        record[
                            "local_path"
                        ],
                    ),
                ]

                for panel_index, (
                    panel_name,
                    image_path,
                ) in enumerate(
                    panel_specs
                ):
                    with Image.open(
                        image_path
                    ) as image_file:
                        panel_image = (
                            image_file
                            .convert("RGB")
                        )

                    resized = (
                        resize_for_panel(
                            panel_image
                        )
                    )

                    x_base = (
                        panel_index
                        * PANEL_WIDTH
                    )

                    image_x = (
                        x_base
                        + (
                            PANEL_WIDTH
                            - resized.width
                        )
                        // 2
                    )

                    image_y = (
                        y_offset
                        + HEADER_HEIGHT
                        + (
                            PANEL_HEIGHT
                            - resized.height
                        )
                        // 2
                    )

                    sheet.paste(
                        resized,
                        (
                            image_x,
                            image_y,
                        ),
                    )

                    draw.text(
                        (
                            x_base + 10,
                            y_offset + 10,
                        ),
                        (
                            f"{record['benchmark_id']} | "
                            f"{record['target']} | "
                            f"{panel_name}"
                        ),
                        fill="black",
                        font=font,
                    )

                draw.line(
                    (
                        PANEL_WIDTH,
                        y_offset,
                        PANEL_WIDTH,
                        y_offset
                        + TRIPLET_ROW_HEIGHT,
                    ),
                    fill="gray",
                    width=1,
                )

                draw.line(
                    (
                        PANEL_WIDTH * 2,
                        y_offset,
                        PANEL_WIDTH * 2,
                        y_offset
                        + TRIPLET_ROW_HEIGHT,
                    ),
                    fill="gray",
                    width=1,
                )

                draw.line(
                    (
                        0,
                        y_offset
                        + TRIPLET_ROW_HEIGHT
                        - 1,
                        TRIPLET_SHEET_WIDTH,
                        y_offset
                        + TRIPLET_ROW_HEIGHT
                        - 1,
                    ),
                    fill="gray",
                    width=1,
                )

            output_path = (
                TRIPLET_CONTACT_DIR
                / (
                    f"{target}_triplet_"
                    f"{page_number:02d}.jpg"
                )
            )

            sheet.save(
                output_path,
                quality=90,
            )

            output_paths.append(
                output_path
            )

    return output_paths


# ============================================================
# Markdown summary
# ============================================================


def get_rate(
    summary_rows: list[
        dict[str, Any]
    ],
    group_key: str,
    group_value: str,
    condition: str,
) -> float | None:
    for row in summary_rows:
        if (
            row[group_key]
            == group_value
            and row["condition"]
            == condition
        ):
            return float(
                row[
                    "detection_rate"
                ]
            )

    return None


def save_markdown_summary(
    benchmark_rows: list[
        dict[str, str]
    ],
    target_summary: list[
        dict[str, Any]
    ],
    scale_summary: list[
        dict[str, Any]
    ],
    threshold: float,
    roi_margin_ratio: float,
    local_enlarge_factor: float,
    manual_audit_path: Path,
) -> Path:
    output_path = (
        REPORT_ROOT
        / "evaluation_summary.md"
    )

    lines = []

    lines.append(
        "# Verified-Positive Grounding Evaluation"
    )

    lines.append("")
    lines.append(
        "## Experimental Setup"
    )

    lines.append("")
    lines.append(
        f"- Model: `{MODEL_ID}`"
    )

    lines.append(
        f"- Detection threshold: {threshold:.2f}"
    )

    lines.append(
        "- Conditions: full image, oracle garment ROI, "
        "target-conditioned local enlarged crop"
    )

    lines.append(
        f"- Garment ROI margin ratio: "
        f"{roi_margin_ratio:.3f}"
    )

    lines.append(
        f"- Local crop enlargement factor: "
        f"{local_enlarge_factor:.2f}x"
    )

    lines.append(
        f"- Evaluated cases: "
        f"{len(benchmark_rows)}"
    )

    lines.append(
        "- Larger-part group: sleeve + collar"
    )

    lines.append(
        "- Small-object group: button + zipper"
    )

    lines.append("")
    lines.append(
        "> The garment ROI uses DeepFashion2 annotation "
        "and is an oracle diagnostic condition."
    )

    lines.append("")
    lines.append(
        "> The local enlarged condition uses only fixed "
        "target-conditioned spatial priors inside the garment ROI. "
        "It does not use a fine-grained part ground-truth bbox."
    )

    lines.append("")
    lines.append(
        "## Detection Rate by Target"
    )

    lines.append("")
    lines.append(
        "| Target | N | Full | ROI | Local | "
        "ROI-Full | Local-Full |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )

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

        local_rate = get_rate(
            target_summary,
            "target",
            target,
            "local_enlarged",
        )

        if (
            full_rate is None
            or roi_rate is None
            or local_rate is None
        ):
            continue

        n_cases = sum(
            row["target"] == target
            for row in benchmark_rows
        )

        lines.append(
            "| "
            f"{target} | "
            f"{n_cases} | "
            f"{full_rate:.1%} | "
            f"{roi_rate:.1%} | "
            f"{local_rate:.1%} | "
            f"{roi_rate - full_rate:+.1%} | "
            f"{local_rate - full_rate:+.1%} |"
        )

    lines.append("")
    lines.append(
        "## Detection Rate by Scale Group"
    )

    lines.append("")
    lines.append(
        "| Group | Full | ROI | Local | "
        "ROI-Full | Local-Full |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )

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

        local_rate = get_rate(
            scale_summary,
            "scale_group",
            scale_group,
            "local_enlarged",
        )

        if (
            full_rate is None
            or roi_rate is None
            or local_rate is None
        ):
            continue

        lines.append(
            "| "
            f"{scale_group} | "
            f"{full_rate:.1%} | "
            f"{roi_rate:.1%} | "
            f"{local_rate:.1%} | "
            f"{roi_rate - full_rate:+.1%} | "
            f"{local_rate - full_rate:+.1%} |"
        )

    lines.append("")
    lines.append(
        "## Manual Localization Audit"
    )

    lines.append("")
    lines.append(
        "Detection rate is not localization accuracy."
    )

    lines.append("")
    lines.append(
        "Review the triplet contact sheets and fill:"
    )

    lines.append("")
    lines.append(
        f"`{manual_audit_path}`"
    )

    lines.append("")
    lines.append(
        "Allowed quality labels:"
    )

    lines.append("")
    lines.append(
        "- `correct`: target localized appropriately"
    )

    lines.append(
        "- `coarse`: target covered, but box is too broad"
    )

    lines.append(
        "- `wrong`: detection exists but localization is wrong"
    )

    lines.append(
        "- `missed`: no detection"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            "\n".join(lines)
        )

    return output_path


# ============================================================
# Main evaluation
# ============================================================


def main() -> None:
    args = parse_args()

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_rows = (
        read_benchmark(
            benchmark_file=(
                args.benchmark_file
            ),
            limit=args.limit,
            per_target_limit=(
                args.per_target_limit
            ),
        )
    )

    print(
        f"Benchmark cases: "
        f"{len(benchmark_rows)}"
    )

    target_counts = (
        defaultdict(int)
    )

    for row in benchmark_rows:
        target_counts[
            row["target"]
        ] += 1

    for target in TARGET_ORDER:
        if target in target_counts:
            print(
                f"  {target}: "
                f"{target_counts[target]}"
            )

    device = get_device()

    print(
        f"Using device: {device}"
    )

    processor, model = (
        load_model(
            device
        )
    )

    case_rows = []
    result_lookup = {}
    visual_records = []

    model_forward_calls = 0

    for benchmark_row in tqdm(
        benchmark_rows,
        desc="Evaluating benchmark",
    ):
        benchmark_id = (
            benchmark_row[
                "benchmark_id"
            ]
        )

        target = (
            benchmark_row[
                "target"
            ]
        )

        image_name = (
            benchmark_row[
                "image_name"
            ]
        )

        item_id = (
            benchmark_row[
                "item_id"
            ]
        )

        image_path = (
            IMAGE_DIR
            / image_name
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Original image not found: {image_path}"
            )

        with Image.open(
            image_path
        ) as image_file:
            original_image = (
                image_file
                .convert("RGB")
            )

        image_width = (
            original_image.width
        )

        image_height = (
            original_image.height
        )

        annotation = (
            load_annotation(
                image_name
            )
        )

        garment = (
            get_exact_garment(
                annotation,
                item_id,
            )
        )

        garment_bbox = (
            clip_bbox(
                garment["bbox"],
                image_width,
                image_height,
            )
        )

        roi_bbox = (
            expand_bbox(
                garment_bbox,
                image_width,
                image_height,
                args.roi_margin_ratio,
            )
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

        model_forward_calls += 1

        # ----------------------------------------------------
        # Condition 2: garment ROI
        # ----------------------------------------------------

        (
            roi_x_min,
            roi_y_min,
            roi_x_max,
            roi_y_max,
        ) = roi_bbox

        roi_image = (
            original_image.crop(
                (
                    roi_x_min,
                    roi_y_min,
                    roi_x_max + 1,
                    roi_y_max + 1,
                )
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

        model_forward_calls += 1

        roi_detections = (
            map_crop_detections_to_full_image(
                detections=(
                    roi_local_detections
                ),
                crop_bbox=roi_bbox,
                scale_factor=1.0,
            )
        )

        # ----------------------------------------------------
        # Condition 3: target-conditioned local enlarged crop
        # ----------------------------------------------------

        (
            local_detections,
            local_seconds,
            local_windows,
        ) = run_local_enlarged(
            original_image=original_image,
            target=target,
            garment_roi=roi_bbox,
            processor=processor,
            model=model,
            device=device,
            threshold=args.threshold,
            enlarge_factor=(
                args.local_enlarge_factor
            ),
            nms_iou=(
                args.local_nms_iou
            ),
        )

        model_forward_calls += len(
            local_windows
        )

        # ----------------------------------------------------
        # Store result rows
        # ----------------------------------------------------

        full_row = (
            summarize_single_condition(
                benchmark_row=benchmark_row,
                condition="full_image",
                detections=full_detections,
                inference_seconds=(
                    full_seconds
                ),
                image_width=image_width,
                image_height=image_height,
                threshold=args.threshold,
                roi_margin_ratio=(
                    args.roi_margin_ratio
                ),
                roi_bbox=roi_bbox,
                local_windows=None,
                local_enlarge_factor=(
                    args.local_enlarge_factor
                ),
            )
        )

        roi_row = (
            summarize_single_condition(
                benchmark_row=benchmark_row,
                condition=(
                    "roi_conditioned"
                ),
                detections=roi_detections,
                inference_seconds=(
                    roi_seconds
                ),
                image_width=image_width,
                image_height=image_height,
                threshold=args.threshold,
                roi_margin_ratio=(
                    args.roi_margin_ratio
                ),
                roi_bbox=roi_bbox,
                local_windows=None,
                local_enlarge_factor=(
                    args.local_enlarge_factor
                ),
            )
        )

        local_row = (
            summarize_single_condition(
                benchmark_row=benchmark_row,
                condition=(
                    "local_enlarged"
                ),
                detections=(
                    local_detections
                ),
                inference_seconds=(
                    local_seconds
                ),
                image_width=image_width,
                image_height=image_height,
                threshold=args.threshold,
                roi_margin_ratio=(
                    args.roi_margin_ratio
                ),
                roi_bbox=roi_bbox,
                local_windows=(
                    local_windows
                ),
                local_enlarge_factor=(
                    args.local_enlarge_factor
                ),
            )
        )

        case_rows.extend(
            [
                full_row,
                roi_row,
                local_row,
            ]
        )

        for condition_row in (
            full_row,
            roi_row,
            local_row,
        ):
            result_lookup[
                (
                    benchmark_id,
                    condition_row[
                        "condition"
                    ],
                )
            ] = condition_row

        save_raw_case_result(
            benchmark_row=(
                benchmark_row
            ),
            roi_bbox=roi_bbox,
            local_windows=(
                local_windows
            ),
            full_detections=(
                full_detections
            ),
            roi_detections=(
                roi_detections
            ),
            local_detections=(
                local_detections
            ),
            threshold=args.threshold,
            roi_margin_ratio=(
                args.roi_margin_ratio
            ),
            local_enlarge_factor=(
                args.local_enlarge_factor
            ),
        )

        if not args.skip_visuals:
            (
                full_path,
                roi_path,
                local_path,
            ) = save_visualizations(
                benchmark_id=(
                    benchmark_id
                ),
                original_image=(
                    original_image
                ),
                full_detections=(
                    full_detections
                ),
                roi_detections=(
                    roi_detections
                ),
                local_detections=(
                    local_detections
                ),
                roi_bbox=roi_bbox,
                local_windows=(
                    local_windows
                ),
                target=target,
            )

            visual_records.append(
                {
                    "benchmark_id": (
                        benchmark_id
                    ),
                    "target": target,
                    "full_path": (
                        full_path
                    ),
                    "roi_path": (
                        roi_path
                    ),
                    "local_path": (
                        local_path
                    ),
                }
            )

    # ========================================================
    # Reports
    # ========================================================

    case_results_path = (
        save_case_results(
            case_rows
        )
    )

    target_summary = (
        build_summary(
            case_rows=case_rows,
            group_field="target",
            group_order=TARGET_ORDER,
        )
    )

    target_summary_path = (
        save_summary_csv(
            rows=target_summary,
            output_name=(
                "target_summary.csv"
            ),
            fieldnames=[
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
            ],
        )
    )

    scale_summary = (
        build_summary(
            case_rows=case_rows,
            group_field=(
                "scale_group"
            ),
            group_order=[
                "larger_part",
                "small_object",
            ],
        )
    )

    scale_summary_path = (
        save_summary_csv(
            rows=scale_summary,
            output_name=(
                "scale_group_summary.csv"
            ),
            fieldnames=[
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
            ],
        )
    )

    manual_audit_path = (
        save_manual_audit(
            benchmark_rows=(
                benchmark_rows
            ),
            result_lookup=(
                result_lookup
            ),
        )
    )

    triplet_paths = []

    if not args.skip_visuals:
        triplet_paths = (
            build_triplet_contact_sheets(
                visual_records
            )
        )

    markdown_path = (
        save_markdown_summary(
            benchmark_rows=(
                benchmark_rows
            ),
            target_summary=(
                target_summary
            ),
            scale_summary=(
                scale_summary
            ),
            threshold=args.threshold,
            roi_margin_ratio=(
                args.roi_margin_ratio
            ),
            local_enlarge_factor=(
                args.local_enlarge_factor
            ),
            manual_audit_path=(
                manual_audit_path
            ),
        )
    )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print(
        "Verified-positive evaluation completed."
    )

    print(
        f"Evaluated benchmark cases: "
        f"{len(benchmark_rows)}"
    )

    print(
        f"Condition result rows: "
        f"{len(case_rows)}"
    )

    print(
        f"Model forward calls: "
        f"{model_forward_calls}"
    )

    print()

    print(
        "=== Detection Rate by Target ==="
    )

    for target in TARGET_ORDER:
        rates = {
            condition: get_rate(
                target_summary,
                "target",
                target,
                condition,
            )
            for condition
            in CONDITIONS
        }

        if any(
            value is None
            for value
            in rates.values()
        ):
            continue

        print(
            f"{target:8s}: "
            f"full={rates['full_image']:.1%}, "
            f"roi={rates['roi_conditioned']:.1%}, "
            f"local={rates['local_enlarged']:.1%}, "
            f"local-full="
            f"{rates['local_enlarged'] - rates['full_image']:+.1%}"
        )

    print()

    print(
        "=== Detection Rate by Scale Group ==="
    )

    for scale_group in (
        "larger_part",
        "small_object",
    ):
        rates = {
            condition: get_rate(
                scale_summary,
                "scale_group",
                scale_group,
                condition,
            )
            for condition
            in CONDITIONS
        }

        if any(
            value is None
            for value
            in rates.values()
        ):
            continue

        print(
            f"{scale_group:12s}: "
            f"full={rates['full_image']:.1%}, "
            f"roi={rates['roi_conditioned']:.1%}, "
            f"local={rates['local_enlarged']:.1%}, "
            f"local-full="
            f"{rates['local_enlarged'] - rates['full_image']:+.1%}"
        )

    print()

    print(
        f"Case results: "
        f"{case_results_path}"
    )

    print(
        f"Target summary: "
        f"{target_summary_path}"
    )

    print(
        f"Scale summary: "
        f"{scale_summary_path}"
    )

    print(
        f"Manual audit: "
        f"{manual_audit_path}"
    )

    print(
        f"Markdown summary: "
        f"{markdown_path}"
    )

    if triplet_paths:
        print(
            "Triplet contact sheets: "
            f"{len(triplet_paths)}"
        )

        print(
            f"Triplet directory: "
            f"{TRIPLET_CONTACT_DIR}"
        )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Review FULL / ROI / LOCAL triplet sheets and fill:"
    )

    print(
        "correct / coarse / wrong / missed"
    )


if __name__ == "__main__":
    main()
