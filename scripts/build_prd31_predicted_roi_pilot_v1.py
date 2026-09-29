"""
PRD 3.1 end-to-end integration pilot v1:
3.1.1 v2-balanced predicted instances -> predicted ROI manifest for 3.1.3.

This script DOES NOT use GT masks/crops as attribute input.
GT from the frozen validation manifest is used only for diagnostic matching.

Default pilot:
- fixed validation manifest: configs/prd_8class_val_v1.csv
- v2-balanced checkpoint
- 20 validation images
- score threshold = 0.40
- mask threshold = 0.50
- deterministic category-aware image selection

Outputs
-------
configs/prd_3_1_predicted_roi_pilot20_v1.csv

reports/prd_integration/predicted_roi_pilot20_v1/
├── pilot_image_selection.csv
├── predicted_instances.csv
├── gt_match_diagnostic.csv
└── summary.txt

outputs/prd_integration/predicted_roi_pilot20_v1/
├── <image_stem>/
│   ├── pred_XX_<category>_crop.jpg
│   ├── pred_XX_<category>_mask.png
│   └── pred_XX_<category>_masked.jpg
└── contact_sheet.jpg

The output manifest contains the fields expected by the existing 3.1.3
attribute scripts:
    source_image
    garment_id
    garment_category
    crop_path
    mask_path

Run
---
cd /workspace/fashion_multimodal_analysis

python scripts/build_prd31_predicted_roi_pilot_v1.py \
  --device cuda \
  --num-images 20
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VAL_CSV = (
    PROJECT_ROOT
    / "configs"
    / "prd_8class_val_v1.csv"
)

DEFAULT_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "prd_instance_segmentation"
    / "maskrcnn_8class_baseline_v2"
    / "checkpoint_last.pth"
)

MANIFEST_OUT = (
    PROJECT_ROOT
    / "configs"
    / "prd_3_1_predicted_roi_pilot20_v1.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_integration"
    / "predicted_roi_pilot20_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_integration"
    / "predicted_roi_pilot20_v1"
)

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

ID_TO_CLASS = {
    v: k
    for k, v in CLASS_TO_ID.items()
}

NUM_CLASSES = 9

PRD_CLASSES = list(
    CLASS_TO_ID.keys()
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT),
    )

    p.add_argument(
        "--val-csv",
        default=str(VAL_CSV),
    )

    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
    )

    p.add_argument(
        "--num-images",
        type=int,
        default=20,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260922,
    )

    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.40,
    )

    p.add_argument(
        "--mask-threshold",
        type=float,
        default=0.50,
    )

    p.add_argument(
        "--match-bbox-iou",
        type=float,
        default=0.50,
    )

    p.add_argument(
        "--max-predictions-per-image",
        type=int,
        default=10,
    )

    return p.parse_args()


def resolve_path(raw: str | Path) -> Path:
    p = Path(raw).expanduser()

    if p.is_absolute():
        return p.resolve()

    return (
        PROJECT_ROOT / p
    ).resolve()


def project_relative(
    path: Path,
) -> str:
    try:
        return str(
            path.resolve().relative_to(
                PROJECT_ROOT.resolve()
            )
        ).replace(
            "\\",
            "/",
        )
    except Exception:
        return str(
            path.resolve()
        ).replace(
            "\\",
            "/",
        )


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    if fields is None:
        fields = list(
            rows[0].keys()
        )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(
            rows
        )


def choose_device(
    name: str,
) -> torch.device:
    if name == "cpu":
        return torch.device(
            "cpu"
        )

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def build_model(
    min_size: int,
    max_size: int,
):
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    box_in = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            box_in,
            NUM_CLASSES,
        )
    )

    mask_in = (
        model.roi_heads
        .mask_predictor
        .conv5_mask
        .in_channels
    )

    model.roi_heads.mask_predictor = (
        MaskRCNNPredictor(
            mask_in,
            256,
            NUM_CLASSES,
        )
    )

    model.transform.min_size = (
        min_size,
    )

    model.transform.max_size = (
        max_size
    )

    return model


def group_val_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = defaultdict(
        list
    )

    for row in rows:
        key = (
            row[
                "source_dataset"
            ].strip(),
            row[
                "source_image"
            ].strip(),
        )

        grouped[
            key
        ].append(
            row
        )

    samples = []

    for (
        source_dataset,
        source_image,
    ), instances in grouped.items():

        gt_classes = sorted(
            {
                x[
                    "garment_category"
                ]
                .strip()
                .lower()
                for x in instances
            }
        )

        samples.append(
            {
                "source_dataset": source_dataset,
                "source_image": source_image,
                "instances": instances,
                "gt_classes": gt_classes,
            }
        )

    samples.sort(
        key=lambda x: (
            x[
                "source_dataset"
            ],
            x[
                "source_image"
            ],
        )
    )

    return samples


def category_aware_select(
    samples: list[dict[str, Any]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    if n >= len(
        samples
    ):
        return samples

    rng = random.Random(
        seed
    )

    remaining = list(
        samples
    )

    rng.shuffle(
        remaining
    )

    selected = []
    covered = set()

    # First cover all 8 PRD categories where possible.
    while (
        remaining
        and len(
            covered
        )
        < len(
            PRD_CLASSES
        )
        and len(
            selected
        )
        < n
    ):
        best_idx = None
        best_gain = -1

        for idx, sample in enumerate(
            remaining
        ):
            gain = len(
                set(
                    sample[
                        "gt_classes"
                    ]
                )
                - covered
            )

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if (
            best_idx is None
            or best_gain <= 0
        ):
            break

        sample = remaining.pop(
            best_idx
        )

        selected.append(
            sample
        )

        covered.update(
            sample[
                "gt_classes"
            ]
        )

    # Fill remaining slots deterministically.
    while (
        remaining
        and len(
            selected
        )
        < n
    ):
        selected.append(
            remaining.pop(
                0
            )
        )

    selected.sort(
        key=lambda x: (
            x[
                "source_dataset"
            ],
            x[
                "source_image"
            ],
        )
    )

    return selected


def box_iou(
    a,
    b,
) -> float:
    ax1, ay1, ax2, ay2 = map(
        float,
        a,
    )

    bx1, by1, bx2, by2 = map(
        float,
        b,
    )

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )

    inter = (
        max(
            0.0,
            ix2 - ix1,
        )
        * max(
            0.0,
            iy2 - iy1,
        )
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
        - inter
    )

    return (
        inter / union
        if union > 0
        else 0.0
    )


def best_gt_match(
    pred_box,
    pred_class: str,
    gt_rows: list[dict[str, str]],
) -> tuple[
    int | None,
    float,
    str,
]:
    best_idx = None
    best_iou = 0.0
    best_class = ""

    for idx, row in enumerate(
        gt_rows
    ):
        gt_box = [
            float(
                row[
                    "bbox_x1"
                ]
            ),
            float(
                row[
                    "bbox_y1"
                ]
            ),
            float(
                row[
                    "bbox_x2"
                ]
            ),
            float(
                row[
                    "bbox_y2"
                ]
            ),
        ]

        iou = box_iou(
            pred_box,
            gt_box,
        )

        if iou > best_iou:
            best_iou = iou
            best_idx = idx
            best_class = (
                row[
                    "garment_category"
                ]
                .strip()
                .lower()
            )

    return (
        best_idx,
        best_iou,
        best_class,
    )


def save_predicted_instance(
    image: Image.Image,
    full_mask: np.ndarray,
    bbox,
    category: str,
    score: float,
    image_dir: Path,
    pred_idx: int,
) -> tuple[
    Path,
    Path,
    Path,
    tuple[int, int, int, int],
]:
    width, height = image.size

    x1, y1, x2, y2 = [
        int(
            round(
                float(
                    x
                )
            )
        )
        for x in bbox
    ]

    x1 = max(
        0,
        min(
            width - 1,
            x1,
        ),
    )

    y1 = max(
        0,
        min(
            height - 1,
            y1,
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            width,
            x2,
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            height,
            y2,
        ),
    )

    crop = image.crop(
        (
            x1,
            y1,
            x2,
            y2,
        )
    )

    mask_crop_arr = full_mask[
        y1:y2,
        x1:x2,
    ].astype(
        np.uint8
    )

    if not mask_crop_arr.any():
        # Conservative fallback to bbox foreground so downstream scripts do not crash.
        mask_crop_arr = np.ones(
            (
                y2 - y1,
                x2 - x1,
            ),
            dtype=np.uint8,
        )

    mask_crop = Image.fromarray(
        mask_crop_arr
        * 255,
        mode="L",
    )

    arr = np.asarray(
        crop
    ).copy()

    mask_bool = (
        mask_crop_arr > 0
    )

    masked_arr = np.full_like(
        arr,
        255,
    )

    masked_arr[
        mask_bool
    ] = arr[
        mask_bool
    ]

    masked = Image.fromarray(
        masked_arr
    )

    stem = (
        f"pred_{pred_idx:02d}_"
        f"{category}"
    )

    crop_path = (
        image_dir
        / f"{stem}_crop.jpg"
    )

    mask_path = (
        image_dir
        / f"{stem}_mask.png"
    )

    masked_path = (
        image_dir
        / f"{stem}_masked.jpg"
    )

    crop.save(
        crop_path,
        quality=95,
    )

    mask_crop.save(
        mask_path,
    )

    masked.save(
        masked_path,
        quality=95,
    )

    return (
        crop_path,
        mask_path,
        masked_path,
        (
            x1,
            y1,
            x2,
            y2,
        ),
    )


def make_contact_sheet(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not rows:
        return

    rows = rows[
        : min(
            len(
                rows
            ),
            40,
        )
    ]

    cols = 4
    tile_w = 280
    tile_h = 330

    nrows = math.ceil(
        len(
            rows
        )
        / cols
    )

    sheet = Image.new(
        "RGB",
        (
            cols * tile_w,
            nrows * tile_h,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        sheet
    )

    font = ImageFont.load_default()

    for idx, row in enumerate(
        rows
    ):
        r = idx // cols
        c = idx % cols

        x0 = c * tile_w
        y0 = r * tile_h

        preview = Image.open(
            resolve_path(
                row[
                    "masked_preview_path"
                ]
            )
        ).convert(
            "RGB"
        )

        preview.thumbnail(
            (
                250,
                245,
            ),
            Image.Resampling.LANCZOS,
        )

        px = (
            x0
            + (
                tile_w
                - preview.width
            )
            // 2
        )

        py = (
            y0
            + 5
        )

        sheet.paste(
            preview,
            (
                px,
                py,
            ),
        )

        text_y = (
            y0
            + 255
        )

        for text in [
            (
                f"{row['garment_category']} "
                f"{float(row['prediction_score']):.2f}"
            ),
            (
                f"GT={row['best_gt_class']} "
                f"IoU={float(row['best_gt_bbox_iou']):.2f}"
            ),
            Path(
                row[
                    "source_image"
                ]
            ).name,
        ]:
            draw.text(
                (
                    x0 + 7,
                    text_y,
                ),
                text,
                fill="black",
                font=font,
            )

            text_y += 18

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        quality=92,
    )


def main() -> None:
    args = parse_args()

    device = choose_device(
        args.device
    )

    checkpoint = resolve_path(
        args.checkpoint
    )

    val_csv = resolve_path(
        args.val_csv
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            checkpoint
        )

    if not val_csv.is_file():
        raise FileNotFoundError(
            val_csv
        )

    val_rows = read_csv(
        val_csv
    )

    samples = group_val_rows(
        val_rows
    )

    selected = category_aware_select(
        samples,
        min(
            args.num_images,
            len(
                samples
            ),
        ),
        args.seed,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_rows = []

    for idx, sample in enumerate(
        selected,
        start=1,
    ):
        selection_rows.append(
            {
                "selection_index": idx,
                "source_dataset": sample[
                    "source_dataset"
                ],
                "source_image": sample[
                    "source_image"
                ],
                "gt_instance_count": len(
                    sample[
                        "instances"
                    ]
                ),
                "gt_classes": "|".join(
                    sample[
                        "gt_classes"
                    ]
                ),
            }
        )

    write_csv(
        REPORT_DIR
        / "pilot_image_selection.csv",
        selection_rows,
    )

    cp = torch.load(
        checkpoint,
        map_location="cpu",
    )

    saved_args = cp.get(
        "args",
        {},
    )

    min_size = int(
        saved_args.get(
            "min_size",
            640,
        )
    )

    max_size = int(
        saved_args.get(
            "max_size",
            1024,
        )
    )

    model = build_model(
        min_size,
        max_size,
    )

    model.load_state_dict(
        cp[
            "model_state_dict"
        ]
    )

    model.to(
        device
    ).eval()

    epoch = int(
        cp.get(
            "epoch",
            -1,
        )
    )

    print(
        f"device={device}"
    )

    print(
        f"checkpoint_epoch={epoch}"
    )

    print(
        f"pilot_images={len(selected)}"
    )

    # Warm-up.
    first_path = resolve_path(
        selected[
            0
        ][
            "source_image"
        ]
    )

    first_tensor = F.to_tensor(
        Image.open(
            first_path
        ).convert(
            "RGB"
        )
    ).to(
        device
    )

    with torch.inference_mode():
        for _ in range(
            3
        ):
            _ = model(
                [
                    first_tensor
                ]
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

    manifest_rows = []
    diagnostic_rows = []
    predicted_counts = Counter()
    source_counts = Counter()

    with torch.inference_mode():
        for image_idx, sample in enumerate(
            selected,
            start=1,
        ):
            source_image = sample[
                "source_image"
            ]

            source_dataset = sample[
                "source_dataset"
            ]

            image_path = resolve_path(
                source_image
            )

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

            tensor = F.to_tensor(
                image
            ).to(
                device
            )

            out = model(
                [
                    tensor
                ]
            )[0]

            scores = (
                out[
                    "scores"
                ]
                .detach()
                .cpu()
                .numpy()
            )

            keep = np.where(
                scores
                >= args.score_threshold
            )[0]

            keep = keep[
                : args.max_predictions_per_image
            ]

            boxes = (
                out[
                    "boxes"
                ]
                .detach()
                .cpu()
                .numpy()
            )

            labels = (
                out[
                    "labels"
                ]
                .detach()
                .cpu()
                .numpy()
            )

            masks = (
                out[
                    "masks"
                ]
                .detach()
                .cpu()
                .numpy()
                [:, 0]
            )

            image_dir = (
                OUTPUT_DIR
                / Path(
                    source_image
                ).stem
            )

            image_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for local_idx, pred_idx in enumerate(
                keep,
                start=1,
            ):
                label_id = int(
                    labels[
                        pred_idx
                    ]
                )

                category = ID_TO_CLASS.get(
                    label_id,
                    f"class_{label_id}",
                )

                score = float(
                    scores[
                        pred_idx
                    ]
                )

                full_mask = (
                    masks[
                        pred_idx
                    ]
                    >= args.mask_threshold
                )

                bbox = boxes[
                    pred_idx
                ]

                (
                    crop_path,
                    mask_path,
                    masked_path,
                    clipped_bbox,
                ) = save_predicted_instance(
                    image,
                    full_mask,
                    bbox,
                    category,
                    score,
                    image_dir,
                    local_idx,
                )

                (
                    gt_idx,
                    best_iou,
                    best_gt_class,
                ) = best_gt_match(
                    clipped_bbox,
                    category,
                    sample[
                        "instances"
                    ],
                )

                gt_row = (
                    sample[
                        "instances"
                    ][
                        gt_idx
                    ]
                    if gt_idx is not None
                    else {}
                )

                garment_id = (
                    f"pred_{image_idx:02d}_"
                    f"{local_idx:02d}_"
                    f"{category}"
                )

                manifest_row = {
                    "source_dataset": source_dataset,
                    "source_image": project_relative(
                        image_path
                    ),
                    "garment_id": garment_id,
                    "garment_category": category,
                    "fine_category": "",
                    "crop_path": project_relative(
                        crop_path
                    ),
                    "mask_path": project_relative(
                        mask_path
                    ),
                    "masked_preview_path": project_relative(
                        masked_path
                    ),
                    "bbox_x1": clipped_bbox[
                        0
                    ],
                    "bbox_y1": clipped_bbox[
                        1
                    ],
                    "bbox_x2": clipped_bbox[
                        2
                    ],
                    "bbox_y2": clipped_bbox[
                        3
                    ],
                    "prediction_score": f"{score:.6f}",
                    "prediction_model": (
                        "maskrcnn_8class_baseline_v2_balanced"
                    ),
                    "score_threshold": args.score_threshold,
                    "mask_threshold": args.mask_threshold,
                    "best_gt_class": best_gt_class,
                    "best_gt_bbox_iou": f"{best_iou:.6f}",
                    "best_gt_garment_id": gt_row.get(
                        "garment_id",
                        "",
                    ),
                    "gt_match_bbox50": int(
                        best_iou
                        >= args.match_bbox_iou
                    ),
                    "gt_match_class_correct": int(
                        bool(
                            best_gt_class
                        )
                        and (
                            category
                            == best_gt_class
                        )
                    ),
                }

                manifest_rows.append(
                    manifest_row
                )

                diagnostic_rows.append(
                    {
                        "source_dataset": source_dataset,
                        "source_image": project_relative(
                            image_path
                        ),
                        "predicted_garment_id": garment_id,
                        "predicted_category": category,
                        "prediction_score": f"{score:.6f}",
                        "best_gt_class": best_gt_class,
                        "best_gt_garment_id": gt_row.get(
                            "garment_id",
                            "",
                        ),
                        "best_gt_bbox_iou": f"{best_iou:.6f}",
                        "bbox50_match": int(
                            best_iou
                            >= args.match_bbox_iou
                        ),
                        "class_correct_to_best_gt": int(
                            bool(
                                best_gt_class
                            )
                            and category
                            == best_gt_class
                        ),
                    }
                )

                predicted_counts[
                    category
                ] += 1

                source_counts[
                    source_dataset
                ] += 1

            print(
                f"[{image_idx:02d}/{len(selected):02d}] "
                f"{Path(source_image).name}: "
                f"{len(keep)} predictions"
            )

    manifest_fields = [
        "source_dataset",
        "source_image",
        "garment_id",
        "garment_category",
        "fine_category",
        "crop_path",
        "mask_path",
        "masked_preview_path",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "prediction_score",
        "prediction_model",
        "score_threshold",
        "mask_threshold",
        "best_gt_class",
        "best_gt_bbox_iou",
        "best_gt_garment_id",
        "gt_match_bbox50",
        "gt_match_class_correct",
    ]

    write_csv(
        MANIFEST_OUT,
        manifest_rows,
        manifest_fields,
    )

    write_csv(
        REPORT_DIR
        / "predicted_instances.csv",
        manifest_rows,
        manifest_fields,
    )

    write_csv(
        REPORT_DIR
        / "gt_match_diagnostic.csv",
        diagnostic_rows,
    )

    make_contact_sheet(
        manifest_rows,
        OUTPUT_DIR
        / "contact_sheet.jpg",
    )

    bbox50_count = sum(
        int(
            row[
                "gt_match_bbox50"
            ]
        )
        for row in manifest_rows
    )

    bbox50_class_count = sum(
        int(
            row[
                "gt_match_bbox50"
            ]
        )
        * int(
            row[
                "gt_match_class_correct"
            ]
        )
        for row in manifest_rows
    )

    summary = [
        "PRD 3.1 Predicted ROI Integration Pilot v1",
        "==========================================",
        "",
        f"checkpoint={project_relative(checkpoint)}",
        f"checkpoint_epoch={epoch}",
        f"device={device}",
        f"validation_manifest={project_relative(val_csv)}",
        f"selected_images={len(selected)}",
        f"predicted_instances={len(manifest_rows)}",
        f"score_threshold={args.score_threshold}",
        f"mask_threshold={args.mask_threshold}",
        "",
        "Predicted category counts",
        "-------------------------",
    ]

    for cls in PRD_CLASSES:
        summary.append(
            f"{cls}={predicted_counts.get(cls, 0)}"
        )

    summary += [
        "",
        "Source counts",
        "-------------",
    ]

    for source, count in sorted(
        source_counts.items()
    ):
        summary.append(
            f"{source}={count}"
        )

    summary += [
        "",
        "Diagnostic matching (GT used only for analysis)",
        "-----------------------------------------------",
        f"predictions_with_bbox50_match={bbox50_count}",
        f"predictions_with_bbox50_and_class_match={bbox50_class_count}",
        "",
        "Important",
        "---------",
        "- crop_path and mask_path are generated from v2-balanced predictions.",
        "- GT crops/masks are NOT used as attribute input.",
        "- GT annotations are retained only as diagnostic metadata for later comparison.",
        "- This manifest can now be passed to existing 3.1.3 scripts that accept --manifest.",
        "- shoe/bag/accessory may be unsupported by some style-specific attribute scripts; those scripts should skip or mark not_applicable rather than invent labels.",
    ]

    (
        REPORT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(
            summary
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=== PREDICTED ROI PILOT FINISHED ==="
    )

    print(
        "Manifest:",
        MANIFEST_OUT.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Summary :",
        (
            REPORT_DIR
            / "summary.txt"
        ).relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Contact :",
        (
            OUTPUT_DIR
            / "contact_sheet.jpg"
        ).relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()
