"""PRD 3.1.3 pattern recognition v3: four-way presence gate + subtype classifier.

Motivation
----------
Pattern v2 showed that Stage 2 subtype classification was strong once a true
patterned garment was correctly routed, while Stage 1 pattern-presence gating
was the main bottleneck.

V3 therefore changes ONLY Stage 1.

Stage 1 semantic buckets
------------------------
1. plain_non_pattern
   Plain / solid garment with no meaningful visible surface pattern.

2. texture_structure_non_pattern
   Non-pattern visual effects such as:
   denim wash, fading, distressing, ripped areas, seams, pleats, folds,
   wrinkles, color blocking, panels, zippers, buttons, studs, lace trim,
   small isolated logo, shadows, hair/occlusion.

3. local_graphic_pattern
   Prominent local graphic / text / logo / cartoon / illustration.

4. repeated_surface_pattern
   Repeated surface pattern such as:
   stripes, checks/plaid, flowers, polka dots, camouflage, animal print,
   geometric or abstract repeated motif.

Binary mapping
--------------
plain_non_pattern
texture_structure_non_pattern
    -> non_pattern -> final_pattern = solid

local_graphic_pattern
repeated_surface_pattern
    -> visible_pattern -> Stage 2 subtype classifier

Stage 2
-------
Unchanged from v2:
striped, checked_plaid, floral, graphic_logo, polka_dot, animal_print,
camouflage, geometric_abstract, other_pattern

Validation policy
-----------------
Do NOT use the v1 focused 42 cases or the v2 holdout 40 cases as the official
v3 validation set. Pass both with --exclude-audit. V3 will draw a fresh
holdout from the remaining instances.

Example
-------
python scripts/run_pattern_hierarchical_v3.py \
    --manifest configs/garment_instances_gt_pilot_100.csv \
    --exclude-audit reports/prd_attribute_extraction/pattern_v1/pattern_manual_audit_focus_v1_final.csv \
    --exclude-audit reports/prd_attribute_extraction/pattern_v2_hierarchical/pattern_v2_holdout_audit_prefilled.csv

Outputs
-------
reports/prd_attribute_extraction/pattern_v3_hierarchical/
    pattern_v3_predictions.csv
    pattern_v3_summary.txt
    pattern_v3_holdout_audit_template.csv
    run_info.txt

outputs/prd_attribute_extraction/pattern_v3_hierarchical/
    per_instance/
    contact_sheet_stage1_ambiguous.jpg
    contact_sheet_stage1_texture_structure.jpg
    contact_sheet_stage2_ambiguous.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
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
    / "pattern_v3_hierarchical"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_attribute_extraction"
    / "pattern_v3_hierarchical"
)


STAGE1_PROMPTS: Dict[str, List[str]] = {
    "plain_non_pattern": [
        "a plain solid-color garment with no visible decorative surface pattern",
        "a simple garment whose main fabric surface is visually plain",
        "a mostly plain garment with no repeated motif and no prominent graphic print",
    ],
    "texture_structure_non_pattern": [
        "a garment with denim wash, fading, distressing or ripped areas but no printed pattern",
        "a garment with seams, stitching, folds, wrinkles or pleats but no printed surface pattern",
        "a garment with color blocking or panel construction but no repeated decorative print",
        "a garment with buttons, zippers, studs or small embellishments that are not a repeated print",
        "a garment with lace trim or structural mesh detail rather than a printed pattern",
        "a mostly plain garment with a tiny isolated logo or small chest mark",
        "a garment where shadows, hair or occlusion create visual texture but the fabric itself is plain",
    ],
    "local_graphic_pattern": [
        "a garment with a prominent large graphic print",
        "a garment with large visible text or lettering",
        "a garment with a prominent logo, cartoon, illustration, face or printed picture",
        "a garment whose main visual motif is a large local graphic rather than a repeated pattern",
    ],
    "repeated_surface_pattern": [
        "a garment with a clear repeated decorative surface pattern",
        "a garment with repeated stripes, checks, plaid, flowers or polka dots",
        "a garment with repeated camouflage, animal print, geometric or abstract motifs",
        "a garment where the pattern repeats across a meaningful area of the fabric",
    ],
}


STAGE2_PROMPTS: Dict[str, List[str]] = {
    "striped": [
        "a garment with clear repeated horizontal stripes",
        "a garment with clear repeated vertical stripes",
        "a garment with repeated diagonal stripe bands",
    ],
    "checked_plaid": [
        "a garment with a clear checked or plaid pattern",
        "a garment with tartan, gingham, square grid or criss-cross check pattern",
    ],
    "floral": [
        "a garment with flower motifs or floral print",
        "a garment with leaves, botanical motifs, vines or floral ornamental pattern",
    ],
    "graphic_logo": [
        "a garment with a prominent large graphic print",
        "a garment with large visible text, letters, logo, cartoon, illustration or printed picture",
        "a garment dominated by a large graphic motif rather than a tiny isolated logo",
    ],
    "polka_dot": [
        "a garment with repeated polka dots",
        "a garment with a repeated round dot pattern",
    ],
    "animal_print": [
        "a garment with leopard print texture",
        "a garment with zebra, tiger, snake or other repeated animal-skin print",
        "an animal-print texture, not a single picture of an animal",
    ],
    "camouflage": [
        "a garment with military camouflage pattern",
        "a garment with repeated camo blotches in a camouflage print",
    ],
    "geometric_abstract": [
        "a garment with repeated geometric shapes",
        "a garment with an abstract repeated decorative print",
    ],
    "other_pattern": [
        "a garment with a visible decorative pattern that does not fit stripes, checks, floral, graphic, dots, animal print, camouflage or geometric print",
    ],
}


NON_PATTERN_BUCKETS = {
    "plain_non_pattern",
    "texture_structure_non_pattern",
}

VISIBLE_PATTERN_BUCKETS = {
    "local_graphic_pattern",
    "repeated_surface_pattern",
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
        f"Path not found: {raw}\n"
        f"cwd candidate: {cwd_candidate}\n"
        f"project candidate: {project_candidate}"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_manifest(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)

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
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    return rows


def load_masked_garment(row: dict[str, str]) -> Image.Image:
    crop = Image.open(resolve_path(row["crop_path"])).convert("RGB")
    mask_raw = row.get("mask_path", "").strip()

    if not mask_raw:
        return crop

    mask = Image.open(resolve_path(mask_raw)).convert("L")

    if mask.size != crop.size:
        mask = mask.resize(crop.size, Image.Resampling.NEAREST)

    white = Image.new("RGB", crop.size, "white")
    return Image.composite(crop, white, mask)


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor for normalization, got {type(x).__name__}"
        )

    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def encode_texts(
    prompts: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    text_inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
    )

    text_inputs = {
        key: value.to(device)
        for key, value in text_inputs.items()
    }

    with torch.inference_mode():
        text_outputs = model.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask"),
            )
        pooled = text_outputs[1]
        text_features = model.text_projection(pooled)

    return l2_normalize(text_features)


def encode_image(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    image_inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = image_inputs["pixel_values"].to(device)

    with torch.inference_mode():
        vision_outputs = model.vision_model(
            pixel_values=pixel_values,
            )
        pooled = vision_outputs[1]
        image_features = model.visual_projection(pooled)

    return l2_normalize(image_features)


def build_class_prototypes(
    prompt_map: Dict[str, List[str]],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    labels = list(prompt_map.keys())
    prototypes = []

    for label in labels:
        features = encode_texts(
            prompt_map[label],
            model,
            processor,
            device,
        )

        prototype = features.mean(dim=0, keepdim=True)
        prototype = l2_normalize(prototype)
        prototypes.append(prototype)

    return labels, torch.cat(prototypes, dim=0)


def classify_with_prototypes(
    image_feature: torch.Tensor,
    labels: list[str],
    prototypes: torch.Tensor,
    model: CLIPModel,
) -> dict:
    scale = model.logit_scale.exp().detach()
    logits = scale * image_feature @ prototypes.T
    probs = logits[0].softmax(dim=0).cpu()

    order = torch.argsort(probs, descending=True)

    top3 = []

    for idx in order[: min(3, len(labels))].tolist():
        idx = int(idx)
        top3.append(
            {
                "label": labels[idx],
                "score": float(probs[idx].item()),
            }
        )

    top1 = top3[0]
    top2 = top3[1] if len(top3) >= 2 else {"label": "", "score": 0.0}

    entropy = float(
        -torch.sum(
            probs * torch.log(probs.clamp_min(1e-12))
        ).item()
    )

    normalized_entropy = (
        entropy / math.log(len(labels))
        if len(labels) > 1
        else 0.0
    )

    return {
        "top1_label": top1["label"],
        "top1_score": top1["score"],
        "top2_label": top2["label"],
        "top2_score": top2["score"],
        "margin": top1["score"] - top2["score"],
        "normalized_entropy": normalized_entropy,
        "top3": top3,
    }


def binary_route(stage1_label: str) -> str:
    if stage1_label in NON_PATTERN_BUCKETS:
        return "non_pattern"

    if stage1_label in VISIBLE_PATTERN_BUCKETS:
        return "visible_pattern"

    raise ValueError(
        f"Unknown Stage-1 label: {stage1_label}"
    )


def save_visual(
    image: Image.Image,
    record: dict,
    save_path: Path,
) -> None:
    panel_h = 300

    canvas = Image.new(
        "RGB",
        (image.width, image.height + panel_h),
        "white",
    )
    canvas.paste(image, (0, panel_h))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    lines = [
        f"{record['garment_id']} | {record['garment_category']}",
        (
            f"bucket: {record['stage1_bucket']} "
            f"({record['stage1_top1_score']:.3f})"
        ),
        (
            f"bucket2: {record['stage1_top2']} "
            f"({record['stage1_top2_score']:.3f})"
        ),
        (
            f"bucket margin/H: {record['stage1_margin']:.3f} / "
            f"{record['stage1_entropy']:.3f}"
        ),
        f"binary route: {record['pattern_presence']}",
        f"final: {record['final_pattern']}",
        (
            f"subtype margin/H: {record['stage2_margin']:.3f} / "
            f"{record['stage2_entropy']:.3f}"
        ),
    ]

    y = 10
    for line in lines:
        draw.text((10, y), line, fill="black", font=font)
        y += 36

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, quality=92)


def make_contact_sheet(
    records: list[tuple[float, Path]],
    output_path: Path,
    descending: bool,
    limit: int = 40,
) -> None:
    if not records:
        return

    records = sorted(
        records,
        key=lambda item: item[0],
        reverse=descending,
    )[:limit]

    tiles = []

    for _, path in records:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((390, 560))

            tile = Image.new("RGB", (400, 570), "white")
            x = (400 - img.width) // 2
            y = (570 - img.height) // 2
            tile.paste(img, (x, y))
            tiles.append(tile)

    cols = 4
    rows = math.ceil(len(tiles) / cols)

    sheet = Image.new(
        "RGB",
        (cols * 400, rows * 570),
        "white",
    )

    for idx, tile in enumerate(tiles):
        x = (idx % cols) * 400
        y = (idx // cols) * 570
        sheet.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def excluded_keys_from_audits(
    paths: list[Path],
) -> set[tuple[str, str]]:
    keys = set()

    for path in paths:
        rows = read_csv(path)

        for row in rows:
            source_image = row.get("source_image", "").strip()
            garment_id = row.get("garment_id", "").strip()

            if source_image and garment_id:
                keys.add(
                    (
                        source_image,
                        garment_id,
                    )
                )

    return keys


def write_holdout_audit(
    rows: list[dict],
    excluded_keys: set[tuple[str, str]],
    seed: int,
    n: int,
) -> int:
    candidates = [
        row
        for row in rows
        if (
            row["source_image"],
            row["garment_id"],
        )
        not in excluded_keys
    ]

    rng = random.Random(seed)
    rng.shuffle(candidates)

    selected = candidates[
        : min(n, len(candidates))
    ]

    fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "final_pattern",
        "stage1_bucket",
        "pattern_presence",
        "stage1_top1_score",
        "stage1_margin",
        "stage1_entropy",
        "stage2_top1",
        "stage2_top1_score",
        "stage2_margin",
        "stage2_entropy",
        "manual_pattern",
        "manual_correct",
        "error_type",
        "notes",
    ]

    path = (
        REPORT_DIR
        / "pattern_v3_holdout_audit_template.csv"
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in selected:
            writer.writerow(
                {
                    **{
                        key: row.get(key, "")
                        for key in fields
                    },
                    "manual_pattern": "",
                    "manual_correct": "",
                    "error_type": "",
                    "notes": "",
                }
            )

    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--exclude-audit",
        action="append",
        default=[],
        help=(
            "Audit CSV to exclude from the fresh v3 holdout. "
            "Repeat this option for multiple prior audit sets."
        ),
    )

    parser.add_argument(
        "--holdout-size",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260917,
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

    exclude_paths = [
        resolve_path(raw)
        for raw in args.exclude_audit
    ]

    rows = read_manifest(
        manifest_path
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

    print(
        f"manifest       : {args.manifest}"
    )
    print(
        f"instances      : {len(rows)}"
    )
    print(
        f"model          : {MODEL_NAME}"
    )
    print(
        "stage1         : 4 semantic buckets -> binary pattern presence"
    )
    print(
        "stage2         : unchanged pattern subtype classifier"
    )
    print(
        f"exclude audits : {len(exclude_paths)}"
    )
    print(
        "loading CLIP..."
    )

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

    print(
        "building stage-1 prototypes..."
    )

    stage1_labels, stage1_proto = (
        build_class_prototypes(
            STAGE1_PROMPTS,
            model,
            processor,
            device,
        )
    )

    print(
        "building stage-2 prototypes..."
    )

    stage2_labels, stage2_proto = (
        build_class_prototypes(
            STAGE2_PROMPTS,
            model,
            processor,
            device,
        )
    )

    output_rows = []

    bucket_counts = Counter()
    route_counts = Counter()
    final_counts = Counter()

    stage1_ambiguous_visuals = []
    texture_structure_visuals = []
    stage2_ambiguous_visuals = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        image = load_masked_garment(
            row
        )

        image_feature = encode_image(
            image,
            model,
            processor,
            device,
        )

        stage1 = classify_with_prototypes(
            image_feature,
            stage1_labels,
            stage1_proto,
            model,
        )

        bucket = stage1[
            "top1_label"
        ]

        presence = binary_route(
            bucket
        )

        bucket_counts[
            bucket
        ] += 1

        route_counts[
            presence
        ] += 1

        if presence == "visible_pattern":
            stage2 = classify_with_prototypes(
                image_feature,
                stage2_labels,
                stage2_proto,
                model,
            )

            final_pattern = stage2[
                "top1_label"
            ]

        else:
            stage2 = {
                "top1_label": "",
                "top1_score": 0.0,
                "top2_label": "",
                "top2_score": 0.0,
                "margin": 0.0,
                "normalized_entropy": 0.0,
                "top3": [],
            }

            final_pattern = "solid"

        final_counts[
            final_pattern
        ] += 1

        stage1_quality = (
            "ambiguous"
            if stage1["margin"] < 0.10
            else "usable"
        )

        if presence == "visible_pattern":
            stage2_quality = (
                "ambiguous"
                if stage2["margin"] < 0.10
                else "usable"
            )
        else:
            stage2_quality = (
                "not_applicable"
            )

        record = {
            "source_image": (
                row["source_image"]
            ),
            "garment_id": (
                row["garment_id"]
            ),
            "garment_category": (
                row[
                    "garment_category"
                ]
                .strip()
                .lower()
            ),
            "stage1_bucket": (
                bucket
            ),
            "pattern_presence": (
                presence
            ),
            "stage1_top1_score": (
                stage1[
                    "top1_score"
                ]
            ),
            "stage1_top2": (
                stage1[
                    "top2_label"
                ]
            ),
            "stage1_top2_score": (
                stage1[
                    "top2_score"
                ]
            ),
            "stage1_margin": (
                stage1[
                    "margin"
                ]
            ),
            "stage1_entropy": (
                stage1[
                    "normalized_entropy"
                ]
            ),
            "stage1_quality": (
                stage1_quality
            ),
            "stage2_top1": (
                stage2[
                    "top1_label"
                ]
            ),
            "stage2_top1_score": (
                stage2[
                    "top1_score"
                ]
            ),
            "stage2_top2": (
                stage2[
                    "top2_label"
                ]
            ),
            "stage2_top2_score": (
                stage2[
                    "top2_score"
                ]
            ),
            "stage2_margin": (
                stage2[
                    "margin"
                ]
            ),
            "stage2_entropy": (
                stage2[
                    "normalized_entropy"
                ]
            ),
            "stage2_quality": (
                stage2_quality
            ),
            "final_pattern": (
                final_pattern
            ),
            "stage1_top3_json": (
                json.dumps(
                    stage1[
                        "top3"
                    ],
                    ensure_ascii=False,
                )
            ),
            "stage2_top3_json": (
                json.dumps(
                    stage2[
                        "top3"
                    ],
                    ensure_ascii=False,
                )
            ),
        }

        output_rows.append(
            record
        )

        print(
            f"[{index:03d}/{len(rows):03d}] "
            f"{record['garment_id']:<30} "
            f"bucket={bucket:<30} "
            f"route={presence:<15} "
            f"final={final_pattern:<18} "
            f"M1={record['stage1_margin']:.3f} "
            f"M2={record['stage2_margin']:.3f}"
        )

        visual_path = (
            per_instance_dir
            / (
                f"{index:03d}_"
                f"{record['garment_id']}.jpg"
            )
        )

        save_visual(
            image,
            record,
            visual_path,
        )

        stage1_ambiguous_visuals.append(
            (
                record[
                    "stage1_margin"
                ],
                visual_path,
            )
        )

        if bucket == "texture_structure_non_pattern":
            texture_structure_visuals.append(
                (
                    record[
                        "stage1_top1_score"
                    ],
                    visual_path,
                )
            )

        if presence == "visible_pattern":
            stage2_ambiguous_visuals.append(
                (
                    record[
                        "stage2_margin"
                    ],
                    visual_path,
                )
            )

    fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "stage1_bucket",
        "pattern_presence",
        "stage1_top1_score",
        "stage1_top2",
        "stage1_top2_score",
        "stage1_margin",
        "stage1_entropy",
        "stage1_quality",
        "stage2_top1",
        "stage2_top1_score",
        "stage2_top2",
        "stage2_top2_score",
        "stage2_margin",
        "stage2_entropy",
        "stage2_quality",
        "final_pattern",
        "stage1_top3_json",
        "stage2_top3_json",
    ]

    pred_path = (
        REPORT_DIR
        / "pattern_v3_predictions.csv"
    )

    with pred_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    excluded_keys = (
        excluded_keys_from_audits(
            exclude_paths
        )
    )

    holdout_n = write_holdout_audit(
        output_rows,
        excluded_keys,
        seed=args.seed,
        n=args.holdout_size,
    )

    make_contact_sheet(
        stage1_ambiguous_visuals,
        OUTPUT_DIR
        / "contact_sheet_stage1_ambiguous.jpg",
        descending=False,
        limit=40,
    )

    make_contact_sheet(
        texture_structure_visuals,
        OUTPUT_DIR
        / "contact_sheet_stage1_texture_structure.jpg",
        descending=True,
        limit=40,
    )

    make_contact_sheet(
        stage2_ambiguous_visuals,
        OUTPUT_DIR
        / "contact_sheet_stage2_ambiguous.jpg",
        descending=False,
        limit=40,
    )

    stage1_margins = np.asarray(
        [
            row[
                "stage1_margin"
            ]
            for row in output_rows
        ],
        dtype=np.float64,
    )

    visible_rows = [
        row
        for row in output_rows
        if row[
            "pattern_presence"
        ]
        == "visible_pattern"
    ]

    stage2_margins = np.asarray(
        [
            row[
                "stage2_margin"
            ]
            for row in visible_rows
        ],
        dtype=np.float64,
    )

    summary_lines = [
        "PRD 3.1.3 garment pattern hierarchical baseline v3",
        "=================================================",
        f"model={MODEL_NAME}",
        f"instances={len(output_rows)}",
        "stage1=4_semantic_buckets_then_binary_mapping",
        "stage2=unchanged_from_v2",
        "gt_available=false_until_new_holdout_manual_audit",
        "",
        "Stage-1 bucket distribution",
        "---------------------------",
    ]

    for label in STAGE1_PROMPTS:
        summary_lines.append(
            f"{label}={bucket_counts[label]}"
        )

    summary_lines += [
        "",
        "Binary pattern-presence distribution",
        "------------------------------------",
        f"non_pattern={route_counts['non_pattern']}",
        f"visible_pattern={route_counts['visible_pattern']}",
        "",
        "Final distribution",
        "------------------",
    ]

    for label in [
        "solid",
        *STAGE2_PROMPTS.keys(),
    ]:
        summary_lines.append(
            f"{label}={final_counts[label]}"
        )

    summary_lines += [
        "",
        "Uncertainty",
        "-----------",
        (
            "stage1_margin: "
            f"mean={stage1_margins.mean():.4f}, "
            f"median={np.median(stage1_margins):.4f}, "
            f"min={stage1_margins.min():.4f}, "
            f"max={stage1_margins.max():.4f}"
        ),
    ]

    if len(stage2_margins):
        summary_lines.append(
            "stage2_margin_visible_pattern_only: "
            f"mean={stage2_margins.mean():.4f}, "
            f"median={np.median(stage2_margins):.4f}, "
            f"min={stage2_margins.min():.4f}, "
            f"max={stage2_margins.max():.4f}"
        )

    summary_lines += [
        "",
        "Validation policy",
        "-----------------",
        f"prior_audit_files_excluded={len(exclude_paths)}",
        f"excluded_unique_instances={len(excluded_keys)}",
        f"new_holdout_audit_rows={holdout_n}",
        "- v1 focused audit and v2 holdout must not be reused as official v3 validation.",
        "- manually label the fresh v3 holdout before reporting v3 accuracy.",
        "",
        "Interpretation",
        "--------------",
        "- V3 modifies Stage 1 only.",
        "- plain and texture/structure effects are both mapped to non_pattern.",
        "- prominent local graphics and repeated surface patterns are mapped to visible_pattern.",
        "- Stage 2 taxonomy remains unchanged because v2 subtype performance was already strong after correct routing.",
        "- material/fabric and craftsmanship remain excluded as target attributes.",
    ]

    (
        REPORT_DIR
        / "pattern_v3_summary.txt"
    ).write_text(
        "\n".join(
            summary_lines
        )
        + "\n",
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 garment pattern hierarchical baseline v3\n"
        f"model={MODEL_NAME}\n"
        f"manifest={args.manifest}\n"
        f"instances={len(output_rows)}\n"
        "stage1=plain_non_pattern,texture_structure_non_pattern,local_graphic_pattern,repeated_surface_pattern\n"
        "binary_mapping=first_two_to_non_pattern,last_two_to_visible_pattern\n"
        "stage2=unchanged_v2_subtype_classifier\n"
        f"exclude_audit_count={len(exclude_paths)}\n"
        f"excluded_unique_instances={len(excluded_keys)}\n"
        f"holdout_size={holdout_n}\n"
        f"seed={args.seed}\n"
        "transformers_compatibility=explicit_text_and_vision_tower_projection\n"
        "gt_available=false_until_new_holdout_manual_audit\n"
        "material_and_craftsmanship=excluded_as_targets\n"
    )

    (
        REPORT_DIR
        / "run_info.txt"
    ).write_text(
        run_info,
        encoding="utf-8",
    )

    print(
        "\n=== FINISHED ==="
    )

    print(
        "PREDICTIONS : "
        "reports/prd_attribute_extraction/"
        "pattern_v3_hierarchical/"
        "pattern_v3_predictions.csv"
    )

    print(
        "SUMMARY     : "
        "reports/prd_attribute_extraction/"
        "pattern_v3_hierarchical/"
        "pattern_v3_summary.txt"
    )

    print(
        "HOLDOUT     : "
        "reports/prd_attribute_extraction/"
        "pattern_v3_hierarchical/"
        "pattern_v3_holdout_audit_template.csv"
    )

    print(
        "STAGE1 AMB  : "
        "outputs/prd_attribute_extraction/"
        "pattern_v3_hierarchical/"
        "contact_sheet_stage1_ambiguous.jpg"
    )

    print(
        "TEXTURE     : "
        "outputs/prd_attribute_extraction/"
        "pattern_v3_hierarchical/"
        "contact_sheet_stage1_texture_structure.jpg"
    )

    print(
        "STAGE2 AMB  : "
        "outputs/prd_attribute_extraction/"
        "pattern_v3_hierarchical/"
        "contact_sheet_stage2_ambiguous.jpg"
    )

    print(
        "================"
    )


if __name__ == "__main__":
    main()
