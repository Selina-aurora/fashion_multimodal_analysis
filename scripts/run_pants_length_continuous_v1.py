"""PRD 3.1.3 continuous pants-length feature baseline v1.

Goal
----
Extract a continuous pants_length_ratio in [0, 1] instead of directly predicting
hard labels such as shorts / cropped / trousers.

This follows the mentor's suggestion:
    image information -> continuous style dimension -> thresholds later

Method
------
CLIP compares each pants instance with ordered semantic anchors:

0.10 very short shorts
0.28 mid-thigh shorts
0.45 knee-length bottoms
0.62 below-knee / capri
0.78 cropped ankle-length pants
0.92 ankle-length trousers
1.00 full-length trousers

The final scalar is the expected anchor value:
    pants_length_ratio = sum(prob_i * anchor_i)

DeepFashion2 provides only coarse weak supervision here:
    shorts    -> shorter group
    trousers  -> longer group

So the diagnostic question is NOT "is the exact ratio ground truth?"
It is:
    does trousers tend to receive a larger continuous ratio than shorts?

Example
-------
python scripts/run_pants_length_continuous_v1.py \
    --manifest configs/garment_instances_gt_pilot_100.csv

Outputs
-------
reports/prd_attribute_extraction/pants_continuous_v1/
    pants_continuous_predictions.csv
    pants_group_summary.csv
    pants_order_diagnostics.txt
    run_info.txt

outputs/prd_attribute_extraction/pants_continuous_v1/
    per_instance/
    contact_sheet_sorted_by_ratio.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "openai/clip-vit-base-patch32"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "pants_continuous_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_attribute_extraction"
    / "pants_continuous_v1"
)

PANTS_CATEGORIES = {
    "pants",
    "trousers",
    "jeans",
    "shorts",
}

PANTS_ANCHORS: Dict[str, tuple[float, str]] = {
    "very_short": (
        0.10,
        "very short fashion shorts ending high on the upper thigh",
    ),
    "mid_thigh": (
        0.28,
        "fashion shorts ending around the middle of the thigh",
    ),
    "knee_length": (
        0.45,
        "knee-length shorts or bottoms ending around the knee",
    ),
    "capri": (
        0.62,
        "below-knee capri pants ending around the upper calf",
    ),
    "cropped": (
        0.78,
        "cropped pants ending above the ankle",
    ),
    "ankle_length": (
        0.92,
        "ankle-length trousers reaching the ankle",
    ),
    "full_length": (
        1.00,
        "full-length trousers extending to the bottom of the leg",
    ),
}


def resolve_path(raw: str) -> Path:
    p = Path(raw).expanduser()

    if p.is_absolute() and p.exists():
        return p.resolve()

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = (PROJECT_ROOT / p).resolve()
    if project_candidate.exists():
        return project_candidate

    raise FileNotFoundError(
        "Path not found.\n"
        f"typed path       : {raw}\n"
        f"cwd candidate    : {cwd_candidate}\n"
        f"project candidate: {project_candidate}"
    )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {
        "source_image",
        "garment_id",
        "garment_category",
        "crop_path",
        "mask_path",
    }

    if not rows:
        raise ValueError("Manifest has no rows.")

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Manifest missing required columns: {sorted(missing)}"
        )

    return rows


def load_masked_garment(row: dict[str, str]) -> Image.Image:
    crop_path = resolve_path(row["crop_path"])
    crop = Image.open(crop_path).convert("RGB")

    mask_raw = row.get("mask_path", "").strip()

    if not mask_raw:
        return crop

    mask_path = resolve_path(mask_raw)
    mask = Image.open(mask_path).convert("L")

    if mask.size != crop.size:
        mask = mask.resize(
            crop.size,
            Image.Resampling.NEAREST,
        )

    white = Image.new(
        "RGB",
        crop.size,
        "white",
    )

    return Image.composite(
        crop,
        white,
        mask,
    )


def deepfashion2_gt_group(
    garment_id: str,
    row: dict[str, str],
) -> str:
    """Weak group label for diagnostic only."""
    fine_category = (
        row.get("fine_category", "")
        .strip()
        .lower()
    )

    gid = garment_id.lower()

    text = f"{fine_category} {gid}"

    if "shorts" in text:
        return "shorts"

    if "trousers" in text:
        return "trousers"

    return "unknown"


def infer_continuous_ratio(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> dict:
    labels: List[str] = list(
        PANTS_ANCHORS.keys()
    )

    anchors = torch.tensor(
        [
            PANTS_ANCHORS[label][0]
            for label in labels
        ],
        dtype=torch.float32,
    )

    prompts = [
        PANTS_ANCHORS[label][1]
        for label in labels
    ]

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    probs = (
        outputs.logits_per_image[0]
        .softmax(dim=0)
        .cpu()
    )

    ratio = float(
        torch.sum(
            probs * anchors
        ).item()
    )

    order = torch.argsort(
        probs,
        descending=True,
    )

    top3 = []

    for idx in order[:3].tolist():
        idx = int(idx)
        label = labels[idx]

        top3.append(
            {
                "anchor_label": label,
                "anchor_value": (
                    PANTS_ANCHORS[
                        label
                    ][0]
                ),
                "probability": float(
                    probs[idx].item()
                ),
            }
        )

    entropy = float(
        -torch.sum(
            probs
            * torch.log(
                probs.clamp_min(
                    1e-12
                )
            )
        ).item()
    )

    normalized_entropy = (
        entropy
        / math.log(
            len(labels)
        )
    )

    return {
        "pants_length_ratio": ratio,
        "top_anchor": (
            top3[0][
                "anchor_label"
            ]
        ),
        "top_anchor_probability": (
            top3[0][
                "probability"
            ]
        ),
        "normalized_entropy": (
            normalized_entropy
        ),
        "top3": top3,
    }


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        return float("nan")

    values = sorted(values)
    n = len(values)
    mid = n // 2

    if n % 2:
        return values[mid]

    return (
        values[mid - 1]
        + values[mid]
    ) / 2.0


def pairwise_order_accuracy(
    shorter_values: list[float],
    longer_values: list[float],
) -> tuple[int, int, float]:
    total = 0
    correct = 0

    for short_value in shorter_values:
        for long_value in longer_values:
            total += 1

            if long_value > short_value:
                correct += 1

    return (
        correct,
        total,
        (
            correct / total
            if total
            else float("nan")
        ),
    )


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def save_visual(
    image: Image.Image,
    record: dict,
    save_path: Path,
) -> None:
    panel_h = 170

    canvas = Image.new(
        "RGB",
        (
            image.width,
            image.height
            + panel_h,
        ),
        "white",
    )

    canvas.paste(
        image,
        (0, panel_h),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    font = (
        ImageFont.load_default()
    )

    lines = [
        (
            f"{record['garment_id']} | "
            f"GT={record['gt_group']}"
        ),
        (
            "pants_length_ratio: "
            f"{record['pants_length_ratio']:.3f}"
        ),
        (
            "top_anchor: "
            f"{record['top_anchor']}"
        ),
        (
            "top_anchor_p: "
            f"{record['top_anchor_probability']:.3f}"
        ),
        (
            "uncertainty: "
            f"{record['normalized_entropy']:.3f}"
        ),
    ]

    y = 10

    for line in lines:
        draw.text(
            (10, y),
            line,
            fill="black",
            font=font,
        )
        y += 29

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        save_path,
        quality=92,
    )


def make_contact_sheet(
    visual_records: list[
        tuple[float, Path]
    ],
    output_path: Path,
) -> None:
    if not visual_records:
        return

    visual_records = sorted(
        visual_records,
        key=lambda item: item[0],
    )

    tiles = []

    for _, path in visual_records:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail(
                (390, 500)
            )

            tile = Image.new(
                "RGB",
                (400, 510),
                "white",
            )

            x = (
                400 - img.width
            ) // 2

            y = (
                510 - img.height
            ) // 2

            tile.paste(
                img,
                (x, y),
            )

            tiles.append(tile)

    cols = 4

    rows = (
        len(tiles)
        + cols - 1
    ) // cols

    sheet = Image.new(
        "RGB",
        (
            cols * 400,
            rows * 510,
        ),
        "white",
    )

    for idx, tile in enumerate(
        tiles
    ):
        x = (
            idx % cols
        ) * 400

        y = (
            idx // cols
        ) * 510

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


def main() -> None:
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    torch.set_num_threads(
        max(
            1,
            args.threads,
        )
    )

    manifest_path = (
        resolve_path(
            args.manifest
        )
    )

    all_rows = (
        read_manifest(
            manifest_path
        )
    )

    rows = [
        row
        for row in all_rows
        if (
            row.get(
                "garment_category",
                "",
            )
            .strip()
            .lower()
            in PANTS_CATEGORIES
        )
    ]

    if not rows:
        raise ValueError(
            "No pants instances found in manifest."
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_instance_dir = (
        OUTPUT_DIR
        / "per_instance"
    )

    per_instance_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cpu"
    )

    print(
        f"manifest      : {args.manifest}"
    )
    print(
        f"all instances : {len(all_rows)}"
    )
    print(
        f"pants cases   : {len(rows)}"
    )
    print(
        f"model         : {MODEL_NAME}"
    )
    print(
        "feature       : pants_length_ratio [0,1]"
    )
    print(
        "loading CLIP..."
    )

    processor = (
        CLIPProcessor
        .from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        CLIPModel
        .from_pretrained(
            MODEL_NAME
        )
        .to(device)
    )

    model.eval()

    prediction_rows = []

    grouped: dict[
        str,
        list[float],
    ] = defaultdict(list)

    visual_records = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        garment_id = (
            row[
                "garment_id"
            ]
            .strip()
        )

        gt_group = (
            deepfashion2_gt_group(
                garment_id,
                row,
            )
        )

        image = (
            load_masked_garment(
                row
            )
        )

        result = (
            infer_continuous_ratio(
                image=image,
                model=model,
                processor=processor,
                device=device,
            )
        )

        ratio = result[
            "pants_length_ratio"
        ]

        if gt_group != "unknown":
            grouped[
                gt_group
            ].append(
                ratio
            )

        record = {
            "source_image": (
                row[
                    "source_image"
                ]
            ),
            "garment_id": (
                garment_id
            ),
            "garment_category": (
                row[
                    "garment_category"
                ]
            ),
            "gt_group": (
                gt_group
            ),
            "pants_length_ratio": (
                ratio
            ),
            "top_anchor": (
                result[
                    "top_anchor"
                ]
            ),
            "top_anchor_probability": (
                result[
                    "top_anchor_probability"
                ]
            ),
            "normalized_entropy": (
                result[
                    "normalized_entropy"
                ]
            ),
            "top3_json": (
                json.dumps(
                    result["top3"],
                    ensure_ascii=False,
                )
            ),
        }

        prediction_rows.append(
            record
        )

        print(
            f"[{index:02d}/{len(rows):02d}] "
            f"{garment_id:<30} "
            f"GT={gt_group:<9} "
            f"ratio={ratio:.3f}"
        )

        visual_path = (
            per_instance_dir
            / (
                f"{index:02d}_"
                f"{garment_id}.jpg"
            )
        )

        save_visual(
            image,
            record,
            visual_path,
        )

        visual_records.append(
            (
                ratio,
                visual_path,
            )
        )

    write_csv(
        REPORT_DIR
        / "pants_continuous_predictions.csv",
        prediction_rows,
        [
            "source_image",
            "garment_id",
            "garment_category",
            "gt_group",
            "pants_length_ratio",
            "top_anchor",
            "top_anchor_probability",
            "normalized_entropy",
            "top3_json",
        ],
    )

    summary_rows = []

    for group_name in [
        "shorts",
        "trousers",
    ]:
        values = grouped.get(
            group_name,
            [],
        )

        if not values:
            continue

        summary_rows.append(
            {
                "gt_group": (
                    group_name
                ),
                "n": len(values),
                "mean_ratio": (
                    mean(values)
                ),
                "median_ratio": (
                    median(values)
                ),
                "min_ratio": (
                    min(values)
                ),
                "max_ratio": (
                    max(values)
                ),
            }
        )

    write_csv(
        REPORT_DIR
        / "pants_group_summary.csv",
        summary_rows,
        [
            "gt_group",
            "n",
            "mean_ratio",
            "median_ratio",
            "min_ratio",
            "max_ratio",
        ],
    )

    shorts = grouped.get(
        "shorts",
        [],
    )

    trousers = grouped.get(
        "trousers",
        [],
    )

    correct, total, accuracy = (
        pairwise_order_accuracy(
            shorts,
            trousers,
        )
    )

    diagnostics = (
        "PRD 3.1.3 pants continuous feature v1\n"
        "======================================\n"
        f"model={MODEL_NAME}\n"
        f"pants_instances={len(rows)}\n"
        f"shorts_n={len(shorts)}\n"
        f"trousers_n={len(trousers)}\n"
        "\n"
        "Group means\n"
        "-----------\n"
        f"shorts={mean(shorts):.4f}\n"
        f"trousers={mean(trousers):.4f}\n"
        "\n"
        "Ordinal diagnostic\n"
        "------------------\n"
        f"shorts < trousers: "
        f"{correct}/{total} "
        + (
            f"pairwise-order accuracy={accuracy:.1%}\n"
            if total
            else "pairwise-order accuracy=N/A\n"
        )
        + "\n"
        "Interpretation\n"
        "--------------\n"
        "- The scalar is a continuous semantic feature, not exact physical GT.\n"
        "- DeepFashion2 only gives shorts/trousers here, so this validates order only.\n"
        "- Do not choose downstream thresholds from this same diagnostic set.\n"
    )

    (
        REPORT_DIR
        / "pants_order_diagnostics.txt"
    ).write_text(
        diagnostics,
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 continuous pants feature baseline v1\n"
        f"model={MODEL_NAME}\n"
        f"manifest={args.manifest}\n"
        f"pants_instances={len(rows)}\n"
        "output_feature=pants_length_ratio_[0,1]\n"
        "aggregation=expected_value_over_ordered_CLIP_semantic_anchors\n"
        "gt_source=DeepFashion2_shorts_vs_trousers_for_ordinal_diagnostic_only\n"
        "hard_thresholds=not_set\n"
        "material_and_craftsmanship=excluded\n"
    )

    (
        REPORT_DIR
        / "run_info.txt"
    ).write_text(
        run_info,
        encoding="utf-8",
    )

    make_contact_sheet(
        visual_records,
        OUTPUT_DIR
        / "contact_sheet_sorted_by_ratio.jpg",
    )

    print(
        "\n=== FINISHED ==="
    )
    print(
        "PREDICTIONS : "
        "reports/prd_attribute_extraction/pants_continuous_v1/"
        "pants_continuous_predictions.csv"
    )
    print(
        "SUMMARY     : "
        "reports/prd_attribute_extraction/pants_continuous_v1/"
        "pants_group_summary.csv"
    )
    print(
        "DIAGNOSTIC  : "
        "reports/prd_attribute_extraction/pants_continuous_v1/"
        "pants_order_diagnostics.txt"
    )
    print(
        "CONTACT     : "
        "outputs/prd_attribute_extraction/pants_continuous_v1/"
        "contact_sheet_sorted_by_ratio.jpg"
    )
    print(
        "================"
    )


if __name__ == "__main__":
    main()
