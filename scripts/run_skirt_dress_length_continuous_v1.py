"""PRD 3.1.3 skirt/dress length continuous semantic baseline v1.

Purpose
-------
Add another continuous style dimension for skirts and dresses:
    lower_garment_length_score in [0,1]

Unlike pants, DeepFashion2 does not provide direct mini/midi/maxi ground truth
for these items in this pilot. Therefore this is a semantic continuous feature
with qualitative validation only.

Ordered semantic anchors
------------------------
0.10 very short / mini
0.30 above-knee
0.50 knee-length
0.70 midi / mid-calf
0.88 ankle-length
1.00 maxi / floor-length

The score is the expected anchor value under CLIP candidate probabilities.

Important
---------
- No hard mini/midi/maxi labels are produced.
- No business thresholds are set.
- No direct GT is available; inspect the sorted contact sheet.
- Material/fabric and craftsmanship remain excluded.

Example
-------
python scripts/run_skirt_dress_length_continuous_v1.py \
    --manifest configs/garment_instances_gt_pilot_100.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "openai/clip-vit-base-patch32"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "skirt_dress_length_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_attribute_extraction"
    / "skirt_dress_length_v1"
)

TARGET_CATEGORIES = {
    "skirt",
    "dress",
}

LENGTH_ANCHORS: Dict[str, tuple[float, str]] = {
    "very_short": (
        0.10,
        "a very short mini skirt or mini dress ending high on the thigh",
    ),
    "above_knee": (
        0.30,
        "a short skirt or dress ending clearly above the knee",
    ),
    "knee_length": (
        0.50,
        "a skirt or dress ending around the knee",
    ),
    "midi": (
        0.70,
        "a midi skirt or dress ending around the middle of the calf",
    ),
    "ankle_length": (
        0.88,
        "a long skirt or dress reaching near the ankle",
    ),
    "maxi": (
        1.00,
        "a maxi skirt or floor-length dress extending to the bottom of the legs",
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


def infer_length_score(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> dict:
    labels: List[str] = list(
        LENGTH_ANCHORS.keys()
    )

    values = torch.tensor(
        [
            LENGTH_ANCHORS[label][0]
            for label in labels
        ],
        dtype=torch.float32,
    )

    prompts = [
        LENGTH_ANCHORS[label][1]
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

    score = float(
        torch.sum(
            probs * values
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
                "anchor_value": LENGTH_ANCHORS[label][0],
                "probability": float(
                    probs[idx].item()
                ),
            }
        )

    entropy = float(
        -torch.sum(
            probs
            * torch.log(
                probs.clamp_min(1e-12)
            )
        ).item()
    )

    normalized_entropy = (
        entropy
        / math.log(len(labels))
    )

    return {
        "length_score": score,
        "top_anchor": top3[0]["anchor_label"],
        "top_anchor_probability": top3[0]["probability"],
        "normalized_entropy": normalized_entropy,
        "top3": top3,
    }


def save_visual(
    image: Image.Image,
    record: dict,
    save_path: Path,
) -> None:
    panel_h = 175

    canvas = Image.new(
        "RGB",
        (
            image.width,
            image.height + panel_h,
        ),
        "white",
    )

    canvas.paste(
        image,
        (0, panel_h),
    )

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    lines = [
        (
            f"{record['garment_id']} | "
            f"{record['garment_category']}"
        ),
        (
            "length_score: "
            f"{record['lower_garment_length_score']:.3f}"
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
        y += 30

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        save_path,
        quality=92,
    )


def make_contact_sheet(
    records: list[tuple[float, Path]],
    output_path: Path,
) -> None:
    if not records:
        return

    records = sorted(
        records,
        key=lambda item: item[0],
    )

    tiles = []

    for _, path in records:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((390, 510))

            tile = Image.new(
                "RGB",
                (400, 520),
                "white",
            )

            x = (400 - img.width) // 2
            y = (520 - img.height) // 2

            tile.paste(img, (x, y))
            tiles.append(tile)

    cols = 4
    rows = (
        len(tiles) + cols - 1
    ) // cols

    sheet = Image.new(
        "RGB",
        (
            cols * 400,
            rows * 520,
        ),
        "white",
    )

    for idx, tile in enumerate(tiles):
        x = (idx % cols) * 400
        y = (idx // cols) * 520
        sheet.paste(tile, (x, y))

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        quality=92,
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


def main() -> None:
    parser = argparse.ArgumentParser()

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
        max(1, args.threads)
    )

    manifest_path = resolve_path(
        args.manifest
    )

    all_rows = read_manifest(
        manifest_path
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
            in TARGET_CATEGORIES
        )
    ]

    if not rows:
        raise ValueError(
            "No skirt/dress instances found."
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

    device = torch.device("cpu")

    print(f"manifest       : {args.manifest}")
    print(f"all instances  : {len(all_rows)}")
    print(f"skirt+dress    : {len(rows)}")
    print(f"model          : {MODEL_NAME}")
    print(
        "feature        : lower_garment_length_score [0,1]"
    )
    print("loading CLIP...")

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = (
        CLIPModel
        .from_pretrained(
            MODEL_NAME
        )
        .to(device)
    )

    model.eval()

    output_rows = []
    visuals = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        image = load_masked_garment(row)

        result = infer_length_score(
            image,
            model,
            processor,
            device,
        )

        record = {
            "source_image": row["source_image"],
            "garment_id": row["garment_id"],
            "garment_category": (
                row[
                    "garment_category"
                ]
                .strip()
                .lower()
            ),
            "lower_garment_length_score": (
                result["length_score"]
            ),
            "top_anchor": (
                result["top_anchor"]
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
            "top3_json": json.dumps(
                result["top3"],
                ensure_ascii=False,
            ),
        }

        output_rows.append(record)

        print(
            f"[{index:02d}/{len(rows):02d}] "
            f"{record['garment_id']:<30} "
            f"{record['garment_category']:<6} "
            f"score={record['lower_garment_length_score']:.3f} "
            f"anchor={record['top_anchor']}"
        )

        visual_path = (
            per_instance_dir
            / f"{index:02d}_{record['garment_id']}.jpg"
        )

        save_visual(
            image,
            record,
            visual_path,
        )

        visuals.append(
            (
                record[
                    "lower_garment_length_score"
                ],
                visual_path,
            )
        )

    write_csv(
        REPORT_DIR
        / "skirt_dress_length_predictions.csv",
        output_rows,
        [
            "source_image",
            "garment_id",
            "garment_category",
            "lower_garment_length_score",
            "top_anchor",
            "top_anchor_probability",
            "normalized_entropy",
            "top3_json",
        ],
    )

    scores = np.asarray(
        [
            row[
                "lower_garment_length_score"
            ]
            for row in output_rows
        ],
        dtype=np.float64,
    )

    summary = (
        "PRD 3.1.3 skirt/dress continuous length semantic v1\n"
        "===================================================\n"
        f"model={MODEL_NAME}\n"
        f"instances={len(output_rows)}\n"
        "hard_classification=false\n"
        "gt_available=false\n"
        "\n"
        "lower_garment_length_score\n"
        "--------------------------\n"
        f"mean={scores.mean():.4f}\n"
        f"median={np.median(scores):.4f}\n"
        f"min={scores.min():.4f}\n"
        f"max={scores.max():.4f}\n"
        "\n"
        "Interpretation\n"
        "--------------\n"
        "- The score is an ordered semantic continuous descriptor, not physical length ground truth.\n"
        "- DeepFashion2 provides no mini/midi/maxi GT for this pilot.\n"
        "- Validate qualitatively using the contact sheet sorted from short to long.\n"
        "- Do not define business thresholds from this pilot alone.\n"
        "- Material/fabric and craftsmanship remain excluded.\n"
    )

    (
        REPORT_DIR
        / "skirt_dress_length_summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 skirt/dress continuous length semantic v1\n"
        f"model={MODEL_NAME}\n"
        f"manifest={args.manifest}\n"
        f"instances={len(output_rows)}\n"
        "garment_categories=skirt,dress\n"
        "output_feature=lower_garment_length_score_[0,1]\n"
        "aggregation=expected_value_over_ordered_CLIP_semantic_anchors\n"
        "hard_classification=false\n"
        "gt_available=false\n"
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
        visuals,
        OUTPUT_DIR
        / "contact_sheet_sorted_by_length_score.jpg",
    )

    print("\n=== FINISHED ===")
    print(
        "PREDICTIONS : "
        "reports/prd_attribute_extraction/skirt_dress_length_v1/"
        "skirt_dress_length_predictions.csv"
    )
    print(
        "SUMMARY     : "
        "reports/prd_attribute_extraction/skirt_dress_length_v1/"
        "skirt_dress_length_summary.txt"
    )
    print(
        "CONTACT     : "
        "outputs/prd_attribute_extraction/skirt_dress_length_v1/"
        "contact_sheet_sorted_by_length_score.jpg"
    )
    print("================")


if __name__ == "__main__":
    main()
