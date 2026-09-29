"""Top-k candidate-ranking diagnostic for core_part_detector_v1.

Purpose
-------
Check whether correct collar / cuff / hem boxes already exist among the
detector's candidates but are buried below many higher-scoring false positives.

For each image and each class independently, retain only:
    Top-1, Top-3, Top-5
predictions ranked by model confidence.

This deliberately does NOT apply an external score threshold. The torchvision
Faster R-CNN internal score threshold is forced to 0.0 so low-confidence
candidates remain available for ranking analysis.

Expected checkpoint
-------------------
models/core_part_detector_v1/best_loss_model.pt

Inputs
------
reports/prd_region_coverage/core_part_annotation_labels.csv
reports/prd_region_coverage/core_part_detector_split_v1.csv

Outputs
-------
reports/prd_region_coverage/core_part_detector_topk_diagnostic/
    checkpoint_info.csv
    metrics_by_split_topk.csv
    per_gt_topk_diagnostic.csv
    gt_rank_summary.csv
    experiment_notes.md

outputs/core_part_detector_topk_diagnostic/
    train/
        top1/
        top3/
        top5/
    val/
        top1/
        top3/
        top5/

Each top-k folder contains per-image visualizations and a contact sheet.

Visualization
-------------
green = ground-truth box
red   = retained top-k prediction
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "prd_region_coverage"

LABELS_FILE = REPORT_ROOT / "core_part_annotation_labels.csv"
SPLIT_FILE = REPORT_ROOT / "core_part_detector_split_v1.csv"

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "core_part_detector_v1"
    / "best_loss_model.pt"
)

REPORT_DIR = (
    REPORT_ROOT
    / "core_part_detector_topk_diagnostic"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "core_part_detector_topk_diagnostic"
)

CLASS_TO_ID = {
    "collar": 1,
    "cuff": 2,
    "hem": 3,
}
ID_TO_CLASS = {
    value: key
    for key, value in CLASS_TO_ID.items()
}
NUM_CLASSES = 1 + len(CLASS_TO_ID)

TOP_K_VALUES = [1, 3, 5]
IOU_THRESHOLD = 0.50

# Keep very-low-confidence detections so ranking can be inspected.
INTERNAL_SCORE_THRESHOLD = 0.0
DETECTIONS_PER_IMAGE = 300


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

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
    if not rows:
        raise ValueError(
            f"No rows to write: {path}"
        )

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


def resolve_project_path(
    raw_path: str,
) -> Path:
    return PROJECT_ROOT / Path(
        raw_path.replace("\\", "/")
    )


def sha256_file(path: Path) -> str:
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
    data = json.loads(raw)

    if not isinstance(data, list) or not data:
        raise ValueError(
            "Labeled row must contain at least one bbox."
        )

    return [
        [float(value) for value in box]
        for box in data
    ]


def build_samples(
    split_name: str,
) -> list[dict[str, Any]]:
    labels = {
        row["annotation_id"]: row
        for row in read_csv(LABELS_FILE)
    }

    split_rows = [
        row
        for row in read_csv(SPLIT_FILE)
        if row["split"] == split_name
    ]

    raw_items = []

    for split_row in split_rows:
        annotation_id = split_row["annotation_id"]

        row = labels.get(annotation_id)
        if row is None:
            raise KeyError(
                f"Missing annotation row: {annotation_id}"
            )

        if row["label_status"] != "labeled":
            raise ValueError(
                "Split contains non-labeled row: "
                f"{annotation_id}"
            )

        image_path = resolve_project_path(
            row["annotation_image_path"]
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        boxes = parse_boxes(
            row["boxes_json"]
        )

        region = row["region"]
        if region not in CLASS_TO_ID:
            raise ValueError(
                f"Unknown region: {region}"
            )

        class_id = CLASS_TO_ID[region]

        raw_items.append(
            {
                "annotation_id": annotation_id,
                "image_name": row["image_name"],
                "image_path": image_path,
                "image_hash": sha256_file(
                    image_path
                ),
                "boxes": boxes,
                "labels": (
                    [class_id] * len(boxes)
                ),
            }
        )

    # Merge rows only when the underlying image bytes are identical.
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

        grouped[key]["annotation_ids"].append(
            item["annotation_id"]
        )
        grouped[key]["boxes"].extend(
            item["boxes"]
        )
        grouped[key]["labels"].extend(
            item["labels"]
        )

    samples = list(
        grouped.values()
    )

    print(
        f"{split_name}: "
        f"{len(split_rows)} annotation rows -> "
        f"{len(samples)} unique image samples"
    )

    return samples


class DiagnosticDataset(Dataset):
    """Dataset without augmentation for deterministic diagnosis."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
    ) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        sample = self.samples[index]

        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        boxes = torch.tensor(
            sample["boxes"],
            dtype=torch.float32,
        )

        labels = torch.tensor(
            sample["labels"],
            dtype=torch.int64,
        )

        image_tensor = (
            tvf.pil_to_tensor(image)
            .float()
            / 255.0
        )

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image_tensor, target


def build_model() -> torch.nn.Module:
    """Build architecture only; trained weights are loaded from checkpoint."""
    try:
        model = (
            fasterrcnn_mobilenet_v3_large_fpn(
                weights=None,
                weights_backbone=None,
                min_size=640,
                max_size=960,
            )
        )
    except TypeError:
        model = (
            fasterrcnn_mobilenet_v3_large_fpn(
                pretrained=False,
                pretrained_backbone=False,
                min_size=640,
                max_size=960,
            )
        )

    in_features = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            NUM_CLASSES,
        )
    )

    # Preserve low-confidence detections for ranking analysis.
    model.roi_heads.score_thresh = (
        INTERNAL_SCORE_THRESHOLD
    )
    model.roi_heads.detections_per_img = (
        DETECTIONS_PER_IMAGE
    )

    return model


def load_checkpoint(
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {MODEL_FILE}"
        )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint


def safe_class_name(
    class_id: int,
) -> str:
    return ID_TO_CLASS.get(
        int(class_id),
        f"class_{class_id}",
    )


def predict_dataset(
    model: torch.nn.Module,
    dataset: DiagnosticDataset,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    predictions = []

    model.eval()

    with torch.no_grad():
        for index in range(
            len(dataset)
        ):
            tensor, _ = dataset[index]

            prediction = model(
                [tensor.to(device)]
            )[0]

            predictions.append(
                {
                    "boxes": (
                        prediction["boxes"]
                        .detach()
                        .cpu()
                    ),
                    "labels": (
                        prediction["labels"]
                        .detach()
                        .cpu()
                    ),
                    "scores": (
                        prediction["scores"]
                        .detach()
                        .cpu()
                    ),
                }
            )

    return predictions


def select_topk_per_class(
    prediction: dict[str, torch.Tensor],
    top_k: int,
) -> dict[str, torch.Tensor]:
    """Keep top-k scoring detections independently for each target class."""
    selected_indices = []

    for class_id in ID_TO_CLASS:
        class_indices = torch.nonzero(
            prediction["labels"] == class_id,
            as_tuple=False,
        ).flatten()

        if len(class_indices) == 0:
            continue

        class_scores = (
            prediction["scores"][
                class_indices
            ]
        )

        order = torch.argsort(
            class_scores,
            descending=True,
        )

        keep_count = min(
            top_k,
            len(order),
        )

        selected_indices.extend(
            class_indices[
                order[:keep_count]
            ].tolist()
        )

    if not selected_indices:
        empty_boxes = (
            prediction["boxes"][:0]
        )
        empty_labels = (
            prediction["labels"][:0]
        )
        empty_scores = (
            prediction["scores"][:0]
        )

        return {
            "boxes": empty_boxes,
            "labels": empty_labels,
            "scores": empty_scores,
        }

    # Sort the final retained set globally by score for clean visualization.
    selected = torch.tensor(
        selected_indices,
        dtype=torch.long,
    )

    selected_scores = (
        prediction["scores"][
            selected
        ]
    )

    global_order = torch.argsort(
        selected_scores,
        descending=True,
    )

    selected = selected[
        global_order
    ]

    return {
        "boxes": (
            prediction["boxes"][
                selected
            ]
        ),
        "labels": (
            prediction["labels"][
                selected
            ]
        ),
        "scores": (
            prediction["scores"][
                selected
            ]
        ),
    }


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

    for gt_index in range(
        ious.shape[0]
    ):
        for pred_index in range(
            ious.shape[1]
        ):
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

        if (
            gt_index in used_gt
            or pred_index in used_pred
        ):
            continue

        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matched_ious.append(
            iou_value
        )

    tp = len(used_gt)
    fp = (
        len(pred_boxes)
        - len(used_pred)
    )
    fn = (
        len(gt_boxes)
        - len(used_gt)
    )

    return (
        tp,
        fp,
        fn,
        matched_ious,
    )


def first_good_candidate_rank(
    gt_box: torch.Tensor,
    gt_label: int,
    prediction: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Inspect same-class candidates ranked by confidence.

    Returns:
    - first_good_rank: first score-rank whose IoU >= 0.5
    - first_good_score / IoU
    - best_iou among all same-class candidates
    - score-rank of that best-IoU candidate
    """
    class_mask = (
        prediction["labels"]
        == gt_label
    )

    class_boxes = (
        prediction["boxes"][
            class_mask
        ]
    )
    class_scores = (
        prediction["scores"][
            class_mask
        ]
    )

    result = {
        "same_class_candidate_count": (
            len(class_boxes)
        ),
        "first_good_rank": "",
        "first_good_score": "",
        "first_good_iou": "",
        "best_iou": 0.0,
        "best_iou_score": "",
        "best_iou_score_rank": "",
    }

    if len(class_boxes) == 0:
        return result

    score_order = torch.argsort(
        class_scores,
        descending=True,
    )

    ranked_boxes = (
        class_boxes[
            score_order
        ]
    )
    ranked_scores = (
        class_scores[
            score_order
        ]
    )

    ranked_ious = box_iou(
        gt_box.unsqueeze(0),
        ranked_boxes,
    )[0]

    best_iou_index = int(
        torch.argmax(
            ranked_ious
        ).item()
    )

    result["best_iou"] = float(
        ranked_ious[
            best_iou_index
        ].item()
    )
    result["best_iou_score"] = float(
        ranked_scores[
            best_iou_index
        ].item()
    )
    result["best_iou_score_rank"] = (
        best_iou_index + 1
    )

    good_indices = torch.nonzero(
        ranked_ious
        >= IOU_THRESHOLD,
        as_tuple=False,
    ).flatten()

    if len(good_indices) > 0:
        first_index = int(
            good_indices[0].item()
        )

        result["first_good_rank"] = (
            first_index + 1
        )
        result["first_good_score"] = float(
            ranked_scores[
                first_index
            ].item()
        )
        result["first_good_iou"] = float(
            ranked_ious[
                first_index
            ].item()
        )

    return result


def evaluate_split_topk(
    split_name: str,
    dataset: DiagnosticDataset,
    raw_predictions: list[
        dict[str, torch.Tensor]
    ],
    top_k: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
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
    total_retained_predictions = 0

    per_gt_rows = []

    for index, sample in enumerate(
        dataset.samples
    ):
        _, target = dataset[index]

        raw_prediction = (
            raw_predictions[index]
        )

        prediction = (
            select_topk_per_class(
                raw_prediction,
                top_k,
            )
        )

        pred_boxes = (
            prediction["boxes"]
        )
        pred_labels = (
            prediction["labels"]
        )
        pred_scores = (
            prediction["scores"]
        )

        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        total_images += 1
        total_retained_predictions += (
            len(pred_boxes)
        )

        image_all_gt_matched = True

        for class_id in ID_TO_CLASS:
            class_gt = gt_boxes[
                gt_labels == class_id
            ]
            class_pred = pred_boxes[
                pred_labels == class_id
            ]
            class_scores = pred_scores[
                pred_labels == class_id
            ]

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

            stats = class_stats[
                class_id
            ]

            stats["tp"] += tp
            stats["fp"] += fp
            stats["fn"] += fn
            stats[
                "matched_ious"
            ].extend(
                matched_ious
            )

            if (
                len(class_gt) > 0
                and fn > 0
            ):
                image_all_gt_matched = (
                    False
                )

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
            gt_label_int = int(
                gt_label.item()
            )

            rank_info = (
                first_good_candidate_rank(
                    gt_box=gt_box,
                    gt_label=gt_label_int,
                    prediction=raw_prediction,
                )
            )

            first_good_rank = (
                rank_info[
                    "first_good_rank"
                ]
            )

            if first_good_rank == "":
                topk_contains_good_box = (
                    "no"
                )
            else:
                topk_contains_good_box = (
                    "yes"
                    if int(
                        first_good_rank
                    )
                    <= top_k
                    else "no"
                )

            per_gt_rows.append(
                {
                    "split": split_name,
                    "top_k_per_class": top_k,
                    "image_name": (
                        sample[
                            "image_name"
                        ]
                    ),
                    "annotation_ids": ";".join(
                        sample[
                            "annotation_ids"
                        ]
                    ),
                    "gt_index": gt_index,
                    "gt_class": (
                        safe_class_name(
                            gt_label_int
                        )
                    ),
                    "gt_box_json": (
                        json.dumps(
                            gt_box.tolist()
                        )
                    ),
                    "same_class_candidate_count": (
                        rank_info[
                            "same_class_candidate_count"
                        ]
                    ),
                    "first_good_rank": (
                        rank_info[
                            "first_good_rank"
                        ]
                    ),
                    "first_good_score": (
                        rank_info[
                            "first_good_score"
                        ]
                    ),
                    "first_good_iou": (
                        rank_info[
                            "first_good_iou"
                        ]
                    ),
                    "best_iou": (
                        rank_info[
                            "best_iou"
                        ]
                    ),
                    "best_iou_score": (
                        rank_info[
                            "best_iou_score"
                        ]
                    ),
                    "best_iou_score_rank": (
                        rank_info[
                            "best_iou_score_rank"
                        ]
                    ),
                    "topk_contains_good_box": (
                        topk_contains_good_box
                    ),
                }
            )

    metric_rows = []
    recalls = []

    for (
        class_id,
        class_name,
    ) in ID_TO_CLASS.items():
        stats = class_stats[
            class_id
        ]

        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        precision = (
            tp / max(
                1,
                tp + fp,
            )
        )

        recall = (
            tp / max(
                1,
                tp + fn,
            )
        )

        if (
            precision + recall
            > 0
        ):
            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )
        else:
            f1 = 0.0

        if stats[
            "matched_ious"
        ]:
            mean_iou = (
                sum(
                    stats[
                        "matched_ious"
                    ]
                )
                / len(
                    stats[
                        "matched_ious"
                    ]
                )
            )
        else:
            mean_iou = 0.0

        recalls.append(
            recall
        )

        metric_rows.append(
            {
                "split": split_name,
                "top_k_per_class": top_k,
                "class": class_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_matched_iou": (
                    mean_iou
                ),
                "macro_recall": "",
                "image_success_rate": "",
                "avg_retained_predictions_per_image": "",
            }
        )

    macro_recall = (
        sum(recalls)
        / len(recalls)
    )

    image_success_rate = (
        successful_images
        / max(
            1,
            total_images,
        )
    )

    avg_retained = (
        total_retained_predictions
        / max(
            1,
            total_images,
        )
    )

    metric_rows.append(
        {
            "split": split_name,
            "top_k_per_class": top_k,
            "class": "OVERALL",
            "tp": sum(
                class_stats[
                    class_id
                ]["tp"]
                for class_id
                in ID_TO_CLASS
            ),
            "fp": sum(
                class_stats[
                    class_id
                ]["fp"]
                for class_id
                in ID_TO_CLASS
            ),
            "fn": sum(
                class_stats[
                    class_id
                ]["fn"]
                for class_id
                in ID_TO_CLASS
            ),
            "precision": "",
            "recall": "",
            "f1": "",
            "mean_matched_iou": "",
            "macro_recall": (
                macro_recall
            ),
            "image_success_rate": (
                image_success_rate
            ),
            "avg_retained_predictions_per_image": (
                avg_retained
            ),
        }
    )

    return (
        metric_rows,
        per_gt_rows,
    )


def build_gt_rank_summary(
    per_gt_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize where the first IoU>=0.5 same-class candidate ranks."""
    # Use one copy per GT only; top-k rows repeat the same raw rank information.
    unique = {}

    for row in per_gt_rows:
        key = (
            row["split"],
            row["image_name"],
            row["gt_index"],
            row["gt_class"],
        )

        if key not in unique:
            unique[key] = row

    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in unique.values():
        grouped[
            (
                row["split"],
                row["gt_class"],
            )
        ].append(
            row
        )

    summary_rows = []

    split_names = [
        "train",
        "val",
    ]

    class_names = [
        "collar",
        "cuff",
        "hem",
    ]

    for split_name in split_names:
        overall_rows = []

        for class_name in class_names:
            rows = grouped.get(
                (
                    split_name,
                    class_name,
                ),
                [],
            )

            overall_rows.extend(
                rows
            )

            summary_rows.append(
                summarize_rank_group(
                    split_name,
                    class_name,
                    rows,
                )
            )

        summary_rows.append(
            summarize_rank_group(
                split_name,
                "OVERALL",
                overall_rows,
            )
        )

    return summary_rows


def summarize_rank_group(
    split_name: str,
    class_name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gt_count = len(rows)

    ranks = []

    for row in rows:
        rank = row[
            "first_good_rank"
        ]

        if rank != "":
            ranks.append(
                int(rank)
            )

    if ranks:
        median_rank = float(
            statistics.median(
                ranks
            )
        )
        mean_rank = (
            sum(ranks)
            / len(ranks)
        )
    else:
        median_rank = ""
        mean_rank = ""

    result = {
        "split": split_name,
        "class": class_name,
        "gt_count": gt_count,
        "gt_with_any_iou_ge_0_5_candidate": (
            len(ranks)
        ),
        "candidate_oracle_recall": (
            len(ranks)
            / max(
                1,
                gt_count,
            )
        ),
        "recall_if_top1": (
            sum(
                rank <= 1
                for rank in ranks
            )
            / max(
                1,
                gt_count,
            )
        ),
        "recall_if_top3": (
            sum(
                rank <= 3
                for rank in ranks
            )
            / max(
                1,
                gt_count,
            )
        ),
        "recall_if_top5": (
            sum(
                rank <= 5
                for rank in ranks
            )
            / max(
                1,
                gt_count,
            )
        ),
        "mean_first_good_rank": (
            mean_rank
        ),
        "median_first_good_rank": (
            median_rank
        ),
    }

    return result


def fit_tile(
    image: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
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

    x = (
        size[0]
        - preview.width
    ) // 2

    y = (
        size[1]
        - preview.height
    ) // 2

    tile.paste(
        preview,
        (x, y),
    )

    return tile


def make_contact_sheet(
    paths: list[Path],
    output_path: Path,
) -> None:
    if not paths:
        return

    tiles = []

    for path in paths:
        with Image.open(
            path
        ) as source:
            tiles.append(
                fit_tile(
                    source.convert(
                        "RGB"
                    ),
                    (520, 560),
                )
            )

    columns = 2
    rows = math.ceil(
        len(tiles)
        / columns
    )

    sheet = Image.new(
        "RGB",
        (
            520 * columns,
            560 * rows,
        ),
        "white",
    )

    for index, tile in enumerate(
        tiles
    ):
        x = (
            index
            % columns
        ) * 520

        y = (
            index
            // columns
        ) * 560

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


def draw_topk_visuals(
    split_name: str,
    dataset: DiagnosticDataset,
    raw_predictions: list[
        dict[str, torch.Tensor]
    ],
    top_k: int,
) -> None:
    topk_dir = (
        OUTPUT_DIR
        / split_name
        / f"top{top_k}"
    )

    per_image_dir = (
        topk_dir
        / "per_image"
    )

    per_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    visual_paths = []

    for index, sample in enumerate(
        dataset.samples
    ):
        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        _, target = dataset[index]

        prediction = (
            select_topk_per_class(
                raw_predictions[index],
                top_k,
            )
        )

        canvas = image.copy()
        draw = ImageDraw.Draw(
            canvas
        )

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
                (
                    "GT "
                    f"{safe_class_name(label)}"
                ),
                fill="green",
            )

        # Retained top-k predictions: red.
        for (
            box,
            label,
            score,
        ) in zip(
            prediction[
                "boxes"
            ].tolist(),
            prediction[
                "labels"
            ].tolist(),
            prediction[
                "scores"
            ].tolist(),
        ):
            draw.rectangle(
                box,
                outline="red",
                width=3,
            )

            draw.text(
                (
                    box[0] + 2,
                    max(
                        0,
                        box[1] - 12,
                    ),
                ),
                (
                    f"P "
                    f"{safe_class_name(label)} "
                    f"{score:.3f}"
                ),
                fill="red",
            )

        save_path = (
            per_image_dir
            / (
                f"{index:02d}_"
                f"{sample['image_name']}"
            )
        )

        canvas.save(
            save_path,
            quality=92,
        )

        visual_paths.append(
            save_path
        )

    make_contact_sheet(
        visual_paths,
        topk_dir
        / "contact_sheet.jpg",
    )


def write_notes(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> None:
    notes = f"""# Core part detector top-k ranking diagnostic

## Purpose

Test whether correct collar / cuff / hem boxes are already present among the
model's candidates but are buried below higher-scoring false positives.

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
- top-k values per class: {TOP_K_VALUES}
- IoU threshold: {IOU_THRESHOLD}
- no external score threshold is applied

## Interpretation

- High Top-1 / Top-3 recall:
  candidate ranking is usable and the previous low-threshold FP explosion can
  likely be controlled with ranking / NMS / priors.
- Large gain from Top-1 to Top-5:
  correct boxes exist, but their confidence ranking is weak.
- Low Top-5 recall despite high candidate-oracle recall:
  correct boxes exist but are buried deeper than rank 5.
- Low candidate-oracle recall:
  the detector is not generating a sufficiently accurate same-class box, so
  ranking alone cannot solve the problem.
"""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        REPORT_DIR
        / "experiment_notes.md"
    ).write_text(
        notes,
        encoding="utf-8",
    )


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    train_samples = (
        build_samples("train")
    )
    val_samples = (
        build_samples("val")
    )

    train_dataset = (
        DiagnosticDataset(
            train_samples
        )
    )
    val_dataset = (
        DiagnosticDataset(
            val_samples
        )
    )

    model = build_model().to(
        device
    )

    checkpoint = (
        load_checkpoint(
            model,
            device,
        )
    )

    model.eval()

    print(
        "Loaded checkpoint: "
        f"epoch={checkpoint.get('epoch')}, "
        f"train_loss={checkpoint.get('train_loss')}, "
        f"macro_recall={checkpoint.get('macro_recall')}, "
        f"image_success={checkpoint.get('image_success_rate')}"
    )

    checkpoint_rows = [
        {
            "checkpoint_file": (
                MODEL_FILE.name
            ),
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
        REPORT_DIR
        / "checkpoint_info.csv",
        checkpoint_rows,
    )

    print(
        "\nRunning model once on TRAIN..."
    )
    train_predictions = (
        predict_dataset(
            model,
            train_dataset,
            device,
        )
    )

    print(
        "Running model once on VAL..."
    )
    val_predictions = (
        predict_dataset(
            model,
            val_dataset,
            device,
        )
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
        for top_k in TOP_K_VALUES:
            print(
                f"Evaluating "
                f"{split_name.upper()} "
                f"with Top-{top_k} "
                f"per class..."
            )

            (
                metric_rows,
                per_gt_rows,
            ) = evaluate_split_topk(
                split_name=split_name,
                dataset=dataset,
                raw_predictions=(
                    predictions
                ),
                top_k=top_k,
            )

            all_metric_rows.extend(
                metric_rows
            )
            all_per_gt_rows.extend(
                per_gt_rows
            )

            draw_topk_visuals(
                split_name=split_name,
                dataset=dataset,
                raw_predictions=(
                    predictions
                ),
                top_k=top_k,
            )

    rank_summary_rows = (
        build_gt_rank_summary(
            all_per_gt_rows
        )
    )

    write_csv(
        REPORT_DIR
        / "metrics_by_split_topk.csv",
        all_metric_rows,
    )

    write_csv(
        REPORT_DIR
        / "per_gt_topk_diagnostic.csv",
        all_per_gt_rows,
    )

    write_csv(
        REPORT_DIR
        / "gt_rank_summary.csv",
        rank_summary_rows,
    )

    write_notes(
        checkpoint,
        device,
    )

    print(
        "\nFinished."
    )
    print(
        "Metrics: "
        f"{REPORT_DIR / 'metrics_by_split_topk.csv'}"
    )
    print(
        "Per-GT ranking: "
        f"{REPORT_DIR / 'per_gt_topk_diagnostic.csv'}"
    )
    print(
        "Rank summary: "
        f"{REPORT_DIR / 'gt_rank_summary.csv'}"
    )
    print(
        "Visuals: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
