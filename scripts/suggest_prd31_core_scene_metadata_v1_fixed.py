"""
Generate CLIP-assisted coverage metadata suggestions for PRD 3.1.1 Core.

IMPORTANT
---------
This is a REVIEW ASSISTANT, not ground truth.

It proposes:
- scene_type
- occlusion_level
- visibility_level

for the 437 currently eligible Core candidates, then marks low-confidence
cases for human review. It NEVER overwrites the human-final metadata fields.

Why this step
-------------
The final Core must preserve:
- exact 50/class balance;
- proportional size coverage;
- broad scene coverage.

Manual labelling of all candidates is expensive. These suggestions are used
only to prioritize and accelerate human coverage review before freeze.

Input
-----
benchmark/prd_3_1_v1/freeze_v1/coverage_review_v1/
    core_coverage_review_v1.csv

Outputs
-------
benchmark/prd_3_1_v1/freeze_v1/coverage_review_v1/scene_suggestions_v1/
├── core_coverage_review_with_suggestions_v1.csv
├── coverage_suggestion_summary_v1.csv
├── low_confidence_review_v1.csv
├── run_info.txt
└── low_confidence_contact_sheets/
    ├── top_01.jpg
    ├── ...
    └── accessory_XX.jpg

Run
---
cd /workspace/fashion_multimodal_analysis

python scripts/suggest_prd31_core_scene_metadata_v1.py \
  --device cuda

Notes
-----
- Uses openai/clip-vit-base-patch32.
- Confidence is RELATIVE CLIP softmax over the candidate prompts and is not a
  calibrated probability.
- Human review is still required before final freeze.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCH_ROOT = PROJECT_ROOT / "benchmark" / "prd_3_1_v1"
COVERAGE_ROOT = (
    BENCH_ROOT
    / "freeze_v1"
    / "coverage_review_v1"
)

INPUT_CSV = COVERAGE_ROOT / "core_coverage_review_v1.csv"

OUT_DIR = COVERAGE_ROOT / "scene_suggestions_v1"
LOW_SHEET_DIR = OUT_DIR / "low_confidence_contact_sheets"

CLASSES = [
    "top",
    "pants",
    "skirt",
    "outerwear",
    "dress",
    "shoe",
    "bag",
    "accessory",
]

SCENE_PROMPTS = {
    "worn_person": [
        "a fashion photo of a person clearly wearing the target clothing item",
        "a person wearing the fashion item in a normal outfit photo",
    ],
    "product_display": [
        "a clothing or fashion product displayed without a person wearing it",
        "a standalone fashion product display or catalog product photo",
    ],
    "partial_view": [
        "a close-up cropped fashion photo showing only part of a person or garment",
        "a partial view of the fashion item cropped by the image",
    ],
    "complex_scene": [
        "a complex fashion scene with clutter, multiple people, runway, crowd, or difficult background",
        "a difficult fashion scene with strong background clutter or multiple subjects",
    ],
}

OCCLUSION_PROMPTS = {
    "none": [
        "the target fashion item is clearly visible and not occluded",
        "the target garment is unobstructed and easy to see",
    ],
    "partial": [
        "the target fashion item is partly occluded by the body, another garment, hair, bag, or object",
        "part of the target garment is hidden by another object or body part",
    ],
    "heavy": [
        "the target fashion item is heavily occluded and much of it is hidden",
        "most of the target garment is obstructed or difficult to see",
    ],
}

VISIBILITY_PROMPTS = {
    "full": [
        "the complete target fashion item is visible in the image",
        "the whole target garment or accessory can be seen",
    ],
    "partial": [
        "only part of the target fashion item is visible or it is cropped by the image",
        "the target garment or accessory is only partially visible",
    ],
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    p.add_argument(
        "--scene-confidence-min",
        type=float,
        default=0.40,
    )
    p.add_argument(
        "--scene-margin-min",
        type=float,
        default=0.08,
    )
    p.add_argument(
        "--attribute-confidence-min",
        type=float,
        default=0.45,
    )
    p.add_argument(
        "--attribute-margin-min",
        type=float,
        default=0.08,
    )

    return p.parse_args()


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

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
        w.writerows(rows)


def resolve_project(raw: str) -> Path:
    p = Path(str(raw).strip())
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def pooled_output(output):
    """
    Compatibility helper for different transformers CLIP return objects.
    Avoids passing return_dict directly to CLIPTextTransformer /
    CLIPVisionTransformer, which caused compatibility issues in this server.
    """
    if hasattr(output, "pooler_output"):
        pooled = output.pooler_output
        if pooled is not None:
            return pooled

    if isinstance(output, (tuple, list)):
        if len(output) > 1 and output[1] is not None:
            return output[1]
        first = output[0]
        if first.ndim == 3:
            return first[:, 0, :]
        return first

    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]

    raise RuntimeError(
        f"Cannot obtain pooled output from {type(output).__name__}"
    )


@torch.inference_mode()
def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: list[str],
    device: torch.device,
):
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    output = model.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    pooled = pooled_output(output)
    features = model.text_projection(pooled)
    features = F.normalize(features, dim=-1)
    return features


@torch.inference_mode()
def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: list[Image.Image],
    device: torch.device,
):
    inputs = processor(
        images=images,
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].to(device)

    output = model.vision_model(
        pixel_values=pixel_values,
    )

    pooled = pooled_output(output)
    features = model.visual_projection(pooled)
    features = F.normalize(features, dim=-1)
    return features


def build_class_text_features(
    model,
    processor,
    prompt_map,
    device,
):
    labels = list(prompt_map.keys())
    label_features = []

    for label in labels:
        texts = prompt_map[label]
        feats = encode_texts(
            model,
            processor,
            texts,
            device,
        )
        proto = F.normalize(
            feats.mean(dim=0, keepdim=True),
            dim=-1,
        )
        label_features.append(proto)

    matrix = torch.cat(label_features, dim=0)
    return labels, matrix


@torch.inference_mode()
def probs_from_features(
    image_features,
    text_features,
    model,
):
    # Keep all similarity computation inside inference_mode.
    # Some PyTorch versions reject using inference tensors in operations
    # that autograd would otherwise track.
    scale = model.logit_scale.exp().clamp(max=100.0)
    logits = scale * (image_features @ text_features.T)
    return logits.softmax(dim=-1)


def top_prediction(
    probs: np.ndarray,
    labels: list[str],
):
    order = np.argsort(probs)[::-1]
    top1 = int(order[0])
    top2 = int(order[1]) if len(order) > 1 else top1

    return {
        "label": labels[top1],
        "confidence": float(probs[top1]),
        "second_label": labels[top2],
        "second_confidence": float(probs[top2]),
        "margin": float(probs[top1] - probs[top2]),
    }


def gt_crop_from_mask(
    source_path: Path,
    mask_path: Path,
    padding_ratio: float = 0.18,
):
    image = Image.open(source_path).convert("RGB")
    mask = np.asarray(
        Image.open(mask_path).convert("L")
    ) > 0

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return image.copy()

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    px = max(8, int(bw * padding_ratio))
    py = max(8, int(bh * padding_ratio))

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(image.width, x2 + px)
    y2 = min(image.height, y2 + py)

    return image.crop((x1, y1, x2, y2))


def make_low_conf_sheets(
    rows,
    out_dir: Path,
):
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for cls in CLASSES:
        rr = [
            r
            for r in rows
            if r["garment_category"] == cls
            and r["coverage_suggestion_review_flag"] == "REVIEW"
        ]

        if not rr:
            continue

        page_size = 8

        for start in range(0, len(rr), page_size):
            page_rows = rr[start:start + page_size]
            page_no = start // page_size + 1

            cols = 2
            rows_n = 4
            tile_w = 760
            tile_h = 330

            canvas = Image.new(
                "RGB",
                (cols * tile_w, rows_n * tile_h),
                "white",
            )

            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()

            for i, row in enumerate(page_rows):
                r = i // cols
                c = i % cols

                x0 = c * tile_w
                y0 = r * tile_h

                source_path = resolve_project(row["source_image"])
                mask_path = resolve_project(row["gt_mask_path"])

                try:
                    full = Image.open(source_path).convert("RGB")
                    crop = gt_crop_from_mask(
                        source_path,
                        mask_path,
                    )
                except Exception:
                    continue

                full.thumbnail((330, 220))
                crop.thumbnail((330, 220))

                canvas.paste(
                    full,
                    (
                        x0 + 10 + (330 - full.width)//2,
                        y0 + 60 + (220 - full.height)//2,
                    ),
                )
                canvas.paste(
                    crop,
                    (
                        x0 + 390 + (330 - crop.width)//2,
                        y0 + 60 + (220 - crop.height)//2,
                    ),
                )

                title = (
                    f"{row['review_sample_id']} | "
                    f"{row['garment_category']} | "
                    f"{row['fine_or_source_category']}"
                )

                meta1 = (
                    f"scene={row['scene_type_suggested']} "
                    f"p={float(row['scene_confidence']):.2f} "
                    f"m={float(row['scene_margin']):.2f}"
                )

                meta2 = (
                    f"occ={row['occlusion_level_suggested']} "
                    f"p={float(row['occlusion_confidence']):.2f} | "
                    f"vis={row['visibility_level_suggested']} "
                    f"p={float(row['visibility_confidence']):.2f}"
                )

                draw.text(
                    (x0 + 8, y0 + 8),
                    title,
                    fill="black",
                    font=font,
                )
                draw.text(
                    (x0 + 8, y0 + 25),
                    meta1,
                    fill="black",
                    font=font,
                )
                draw.text(
                    (x0 + 8, y0 + 42),
                    meta2,
                    fill="black",
                    font=font,
                )

                draw.rectangle(
                    (
                        x0 + 2,
                        y0 + 2,
                        x0 + tile_w - 3,
                        y0 + tile_h - 3,
                    ),
                    outline="gray",
                    width=1,
                )

            out_path = (
                out_dir
                / f"{cls}_{page_no:02d}.jpg"
            )

            canvas.save(
                out_path,
                quality=92,
            )


def main():
    args = parse_args()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(args.device)

    rows = read_csv(INPUT_CSV)

    print("Loading CLIP:", args.model)
    print("device:", device)

    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model)
    model.eval()
    model.to(device)

    scene_labels, scene_text = build_class_text_features(
        model,
        processor,
        SCENE_PROMPTS,
        device,
    )

    occ_labels, occ_text = build_class_text_features(
        model,
        processor,
        OCCLUSION_PROMPTS,
        device,
    )

    vis_labels, vis_text = build_class_text_features(
        model,
        processor,
        VISIBILITY_PROMPTS,
        device,
    )

    results = []

    for start in range(
        0,
        len(rows),
        args.batch_size,
    ):
        batch = rows[
            start:
            start
            + args.batch_size
        ]

        full_images = []
        crop_images = []

        for row in batch:
            source_path = resolve_project(
                row["source_image"]
            )
            mask_path = resolve_project(
                row["gt_mask_path"]
            )

            full_images.append(
                Image.open(
                    source_path
                ).convert("RGB")
            )

            crop_images.append(
                gt_crop_from_mask(
                    source_path,
                    mask_path,
                )
            )

        full_feat = encode_images(
            model,
            processor,
            full_images,
            device,
        )

        crop_feat = encode_images(
            model,
            processor,
            crop_images,
            device,
        )

        scene_probs = probs_from_features(
            full_feat,
            scene_text,
            model,
        ).detach().cpu().numpy()

        occ_probs = probs_from_features(
            crop_feat,
            occ_text,
            model,
        ).detach().cpu().numpy()

        vis_probs = probs_from_features(
            crop_feat,
            vis_text,
            model,
        ).detach().cpu().numpy()

        for i, row in enumerate(batch):
            scene = top_prediction(
                scene_probs[i],
                scene_labels,
            )
            occ = top_prediction(
                occ_probs[i],
                occ_labels,
            )
            vis = top_prediction(
                vis_probs[i],
                vis_labels,
            )

            reasons = []

            if (
                scene["confidence"]
                < args.scene_confidence_min
                or scene["margin"]
                < args.scene_margin_min
            ):
                reasons.append("LOW_SCENE_CONFIDENCE")

            if (
                occ["confidence"]
                < args.attribute_confidence_min
                or occ["margin"]
                < args.attribute_margin_min
            ):
                reasons.append("LOW_OCCLUSION_CONFIDENCE")

            if (
                vis["confidence"]
                < args.attribute_confidence_min
                or vis["margin"]
                < args.attribute_margin_min
            ):
                reasons.append("LOW_VISIBILITY_CONFIDENCE")

            out = dict(row)

            out.update({
                "scene_type_suggested": scene["label"],
                "scene_confidence": f"{scene['confidence']:.6f}",
                "scene_second_label": scene["second_label"],
                "scene_margin": f"{scene['margin']:.6f}",

                "occlusion_level_suggested": occ["label"],
                "occlusion_confidence": f"{occ['confidence']:.6f}",
                "occlusion_second_label": occ["second_label"],
                "occlusion_margin": f"{occ['margin']:.6f}",

                "visibility_level_suggested": vis["label"],
                "visibility_confidence": f"{vis['confidence']:.6f}",
                "visibility_second_label": vis["second_label"],
                "visibility_margin": f"{vis['margin']:.6f}",

                "coverage_suggestion_review_flag": (
                    "REVIEW"
                    if reasons
                    else "HIGH_CONF_SUGGESTION"
                ),
                "coverage_suggestion_review_reasons": "|".join(reasons),

                # Human-final fields remain explicitly blank/unreviewed.
                "scene_type": "",
                "occlusion_level": "",
                "visibility_level": "",
                "coverage_review_status": "unreviewed",
            })

            results.append(out)

        print(
            f"processed {min(start + len(batch), len(rows))}/{len(rows)}"
        )

    out_csv = (
        OUT_DIR
        / "core_coverage_review_with_suggestions_v1.csv"
    )

    write_csv(
        out_csv,
        results,
    )

    low = [
        r
        for r in results
        if r[
            "coverage_suggestion_review_flag"
        ]
        == "REVIEW"
    ]

    write_csv(
        OUT_DIR
        / "low_confidence_review_v1.csv",
        low,
    )

    # Summary by class and proposed labels.
    summary_rows = []

    for cls in CLASSES:
        rr = [
            r
            for r in results
            if r[
                "garment_category"
            ]
            == cls
        ]

        scene_counts = Counter(
            r[
                "scene_type_suggested"
            ]
            for r in rr
        )
        occ_counts = Counter(
            r[
                "occlusion_level_suggested"
            ]
            for r in rr
        )
        vis_counts = Counter(
            r[
                "visibility_level_suggested"
            ]
            for r in rr
        )

        summary_rows.append({
            "garment_category": cls,
            "n": len(rr),
            "high_conf_suggestions": sum(
                r[
                    "coverage_suggestion_review_flag"
                ]
                == "HIGH_CONF_SUGGESTION"
                for r in rr
            ),
            "needs_review": sum(
                r[
                    "coverage_suggestion_review_flag"
                ]
                == "REVIEW"
                for r in rr
            ),
            "scene_distribution_suggested": "; ".join(
                f"{k}={v}"
                for k, v in sorted(
                    scene_counts.items()
                )
            ),
            "occlusion_distribution_suggested": "; ".join(
                f"{k}={v}"
                for k, v in sorted(
                    occ_counts.items()
                )
            ),
            "visibility_distribution_suggested": "; ".join(
                f"{k}={v}"
                for k, v in sorted(
                    vis_counts.items()
                )
            ),
        })

    write_csv(
        OUT_DIR
        / "coverage_suggestion_summary_v1.csv",
        summary_rows,
    )

    make_low_conf_sheets(
        results,
        LOW_SHEET_DIR,
    )

    run_info = f"""PRD 3.1.1 Core Coverage Metadata Suggestions v1
================================================

model={args.model}
device={device}
cases={len(results)}
low_confidence_cases={len(low)}

IMPORTANT
---------
- Suggestions are CLIP-assisted review metadata only.
- Confidence is relative softmax over the prompt candidates, not a calibrated probability.
- Human-final scene_type / occlusion_level / visibility_level fields remain blank.
- Do not freeze Core directly from these suggestions.
- Review low-confidence cases first, then spot-check high-confidence suggestions.
- Final 400-case membership must be chosen without looking at candidate-model performance.

Thresholds
----------
scene_confidence_min={args.scene_confidence_min}
scene_margin_min={args.scene_margin_min}
attribute_confidence_min={args.attribute_confidence_min}
attribute_margin_min={args.attribute_margin_min}
"""

    (
        OUT_DIR
        / "run_info.txt"
    ).write_text(
        run_info,
        encoding="utf-8",
    )

    print()
    print("FINISHED")

    for p in [
        OUT_DIR
        / "run_info.txt",
        OUT_DIR
        / "coverage_suggestion_summary_v1.csv",
        OUT_DIR
        / "low_confidence_review_v1.csv",
        out_csv,
    ]:
        print(
            p.relative_to(
                PROJECT_ROOT
            )
        )

    print(
        "low-confidence contact sheets:",
        LOW_SHEET_DIR.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()
