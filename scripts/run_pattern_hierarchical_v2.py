"""PRD 3.1.3 pattern recognition v2: hierarchical CLIP baseline.

Motivation
----------
Pattern v1 directly classified every garment into one of 10 pattern classes.
Focused audit showed an important failure mode: non-pattern visual effects
(denim wash, distress, folds, pleats, seams, color blocks, tiny logos) could be
mistaken for stripes / graphics / dots.

Pattern v2 therefore separates the problem into two stages:

Stage 1: pattern presence
    non_pattern
    visible_pattern

Stage 2: pattern subtype, only when Stage 1 selects visible_pattern
    striped
    checked_plaid
    floral
    graphic_logo
    polka_dot
    animal_print
    camouflage
    geometric_abstract
    other_pattern

Notes
-----
- "solid" is represented by Stage-1 non_pattern.
- denim wash / distress / folds / pleats / seams / color-block construction
  are explicitly described as non-pattern visual effects.
- a tiny isolated logo does NOT automatically make the garment graphic_logo;
  a prominent graphic/text/logo is required.
- CLIP softmax scores are candidate-relative, not calibrated probabilities.
- no production threshold is claimed.
- use a NEW manual audit set for v2 evaluation; do not report the v1 focused
  42-case set as unbiased v2 performance.

Example
-------
python scripts/run_pattern_hierarchical_v2.py \
    --manifest configs/garment_instances_gt_pilot_100.csv

Optional, if the v1 focused-audit CSV is placed in the project:
python scripts/run_pattern_hierarchical_v2.py \
    --manifest configs/garment_instances_gt_pilot_100.csv \
    --exclude-audit reports/prd_attribute_extraction/pattern_v1/pattern_manual_audit_focus_v1_final.csv

Outputs
-------
reports/prd_attribute_extraction/pattern_v2_hierarchical/
    pattern_v2_predictions.csv
    pattern_v2_summary.txt
    pattern_v2_holdout_audit_template.csv
    run_info.txt

outputs/prd_attribute_extraction/pattern_v2_hierarchical/
    per_instance/
    contact_sheet_stage1_ambiguous.jpg
    contact_sheet_pattern_subtype_ambiguous.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

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
    / "pattern_v2_hierarchical"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_attribute_extraction"
    / "pattern_v2_hierarchical"
)

# Multiple prompts per class are intentionally used so one wording does not
# dominate the result.
STAGE1_PROMPTS: Dict[str, List[str]] = {
    "non_pattern": [
        "a plain garment with no repeated or prominent printed pattern",
        "a solid-color garment; folds, seams, pleats and shadows are not patterns",
        "a denim garment where wash, fading, ripped areas and distressing are texture or craftsmanship, not a printed pattern",
        "a color-block or panelled garment without a repeated printed motif",
        "a mostly plain garment with only a tiny isolated logo or small chest mark",
    ],
    "visible_pattern": [
        "a garment with a clear visible repeated or decorative surface pattern",
        "a garment with obvious stripes, checks, flowers, dots, camouflage, animal print or geometric print",
        "a garment with a prominent large graphic, text, logo, cartoon or illustration",
        "a garment whose visible motif covers a meaningful area of the garment",
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
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def build_class_prototypes(
    prompt_map: Dict[str, List[str]],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    labels = list(prompt_map.keys())
    prototypes = []

    with torch.inference_mode():
        for label in labels:
            prompts = prompt_map[label]

            text_inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
            )

            text_inputs = {
                key: value.to(device)
                for key, value in text_inputs.items()
            }

            text_features = model.get_text_features(**text_inputs)
            text_features = l2_normalize(text_features)

            prototype = text_features.mean(dim=0, keepdim=True)
            prototype = l2_normalize(prototype)
            prototypes.append(prototype)

    return labels, torch.cat(prototypes, dim=0)


def image_embedding(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    image_inputs = processor(
        images=image,
        return_tensors="pt",
    )

    image_inputs = {
        key: value.to(device)
        for key, value in image_inputs.items()
    }

    with torch.inference_mode():
        image_features = model.get_image_features(**image_inputs)

    return l2_normalize(image_features)


def classify_with_prototypes(
    image_feature: torch.Tensor,
    labels: list[str],
    prototypes: torch.Tensor,
    model: CLIPModel,
) -> dict:
    # Use CLIP's learned logit scale, but scores remain candidate-relative.
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


def save_visual(
    image: Image.Image,
    record: dict,
    save_path: Path,
) -> None:
    panel_h = 260
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
            "gate: "
            f"{record['stage1_top1']} "
            f"({record['stage1_top1_score']:.3f})"
        ),
        (
            "gate2: "
            f"{record['stage1_top2']} "
            f"({record['stage1_top2_score']:.3f})"
        ),
        (
            "gate margin/H: "
            f"{record['stage1_margin']:.3f} / "
            f"{record['stage1_entropy']:.3f}"
        ),
        (
            "final: "
            f"{record['final_pattern']}"
        ),
        (
            "subtype margin/H: "
            f"{record['stage2_margin']:.3f} / "
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
            img.thumbnail((390, 540))

            tile = Image.new("RGB", (400, 550), "white")
            x = (400 - img.width) // 2
            y = (550 - img.height) // 2
            tile.paste(img, (x, y))
            tiles.append(tile)

    cols = 4
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new(
        "RGB",
        (cols * 400, rows * 550),
        "white",
    )

    for idx, tile in enumerate(tiles):
        x = (idx % cols) * 400
        y = (idx // cols) * 550
        sheet.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def excluded_keys_from_audit(path: Path | None) -> set[tuple[str, str]]:
    if path is None:
        return set()

    rows = read_csv(path)

    return {
        (
            row.get("source_image", "").strip(),
            row.get("garment_id", "").strip(),
        )
        for row in rows
        if row.get("source_image", "").strip()
        and row.get("garment_id", "").strip()
    }


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
    selected = candidates[: min(n, len(candidates))]

    fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "final_pattern",
        "stage1_top1",
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

    path = REPORT_DIR / "pattern_v2_holdout_audit_template.csv"

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in selected:
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fields},
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
        default=None,
        help=(
            "Optional v1 tuning/focused-audit CSV. "
            "These garments are excluded from the new v2 holdout audit."
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
        default=20260916,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))

    manifest_path = resolve_path(args.manifest)

    exclude_path = (
        resolve_path(args.exclude_audit)
        if args.exclude_audit
        else None
    )

    rows = read_manifest(manifest_path)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per_instance_dir = OUTPUT_DIR / "per_instance"
    per_instance_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    print(f"manifest       : {args.manifest}")
    print(f"instances      : {len(rows)}")
    print(f"model          : {MODEL_NAME}")
    print("pipeline       : pattern presence -> pattern subtype")
    print("loading CLIP...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print("building stage-1 prototypes...")
    stage1_labels, stage1_proto = build_class_prototypes(
        STAGE1_PROMPTS,
        model,
        processor,
        device,
    )

    print("building stage-2 prototypes...")
    stage2_labels, stage2_proto = build_class_prototypes(
        STAGE2_PROMPTS,
        model,
        processor,
        device,
    )

    output_rows = []
    stage1_ambiguous_visuals = []
    stage2_ambiguous_visuals = []

    final_counts = Counter()
    gate_counts = Counter()

    for index, row in enumerate(rows, start=1):
        image = load_masked_garment(row)

        image_feature = image_embedding(
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

        gate_counts[stage1["top1_label"]] += 1

        if stage1["top1_label"] == "visible_pattern":
            stage2 = classify_with_prototypes(
                image_feature,
                stage2_labels,
                stage2_proto,
                model,
            )

            final_pattern = stage2["top1_label"]
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

        final_counts[final_pattern] += 1

        # Quality flag only; it is not a production threshold.
        if stage1["margin"] < 0.10:
            gate_quality = "ambiguous"
        else:
            gate_quality = "usable"

        if (
            stage1["top1_label"] == "visible_pattern"
            and stage2["margin"] < 0.10
        ):
            subtype_quality = "ambiguous"
        elif stage1["top1_label"] == "visible_pattern":
            subtype_quality = "usable"
        else:
            subtype_quality = "not_applicable"

        record = {
            "source_image": row["source_image"],
            "garment_id": row["garment_id"],
            "garment_category": row["garment_category"].strip().lower(),
            "stage1_top1": stage1["top1_label"],
            "stage1_top1_score": stage1["top1_score"],
            "stage1_top2": stage1["top2_label"],
            "stage1_top2_score": stage1["top2_score"],
            "stage1_margin": stage1["margin"],
            "stage1_entropy": stage1["normalized_entropy"],
            "stage1_quality": gate_quality,
            "stage2_top1": stage2["top1_label"],
            "stage2_top1_score": stage2["top1_score"],
            "stage2_top2": stage2["top2_label"],
            "stage2_top2_score": stage2["top2_score"],
            "stage2_margin": stage2["margin"],
            "stage2_entropy": stage2["normalized_entropy"],
            "stage2_quality": subtype_quality,
            "final_pattern": final_pattern,
            "stage1_top3_json": json.dumps(
                stage1["top3"],
                ensure_ascii=False,
            ),
            "stage2_top3_json": json.dumps(
                stage2["top3"],
                ensure_ascii=False,
            ),
        }

        output_rows.append(record)

        print(
            f"[{index:03d}/{len(rows):03d}] "
            f"{record['garment_id']:<30} "
            f"gate={record['stage1_top1']:<15} "
            f"final={record['final_pattern']:<18} "
            f"gM={record['stage1_margin']:.3f} "
            f"sM={record['stage2_margin']:.3f}"
        )

        visual_path = (
            per_instance_dir
            / f"{index:03d}_{record['garment_id']}.jpg"
        )

        save_visual(
            image,
            record,
            visual_path,
        )

        stage1_ambiguous_visuals.append(
            (
                record["stage1_margin"],
                visual_path,
            )
        )

        if record["stage1_top1"] == "visible_pattern":
            stage2_ambiguous_visuals.append(
                (
                    record["stage2_margin"],
                    visual_path,
                )
            )

    fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "stage1_top1",
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

    pred_path = REPORT_DIR / "pattern_v2_predictions.csv"

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
        writer.writerows(output_rows)

    excluded_keys = excluded_keys_from_audit(exclude_path)

    holdout_n = write_holdout_audit(
        output_rows,
        excluded_keys,
        seed=args.seed,
        n=args.holdout_size,
    )

    make_contact_sheet(
        stage1_ambiguous_visuals,
        OUTPUT_DIR / "contact_sheet_stage1_ambiguous.jpg",
        descending=False,
        limit=40,
    )

    make_contact_sheet(
        stage2_ambiguous_visuals,
        OUTPUT_DIR / "contact_sheet_pattern_subtype_ambiguous.jpg",
        descending=False,
        limit=40,
    )

    stage1_margins = np.asarray(
        [row["stage1_margin"] for row in output_rows],
        dtype=np.float64,
    )

    visible_rows = [
        row
        for row in output_rows
        if row["stage1_top1"] == "visible_pattern"
    ]

    stage2_margins = np.asarray(
        [row["stage2_margin"] for row in visible_rows],
        dtype=np.float64,
    )

    summary_lines = [
        "PRD 3.1.3 garment pattern hierarchical baseline v2",
        "=================================================",
        f"model={MODEL_NAME}",
        f"instances={len(output_rows)}",
        "pipeline=pattern_presence_then_pattern_subtype",
        "gt_available=false",
        "",
        "Stage-1 distribution",
        "--------------------",
    ]

    for label in STAGE1_PROMPTS:
        summary_lines.append(
            f"{label}={gate_counts[label]}"
        )

    summary_lines += [
        "",
        "Final distribution",
        "------------------",
    ]

    for label in ["solid", *STAGE2_PROMPTS.keys()]:
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
        f"v1_tuning_audit_excluded_from_holdout={bool(excluded_keys)}",
        f"excluded_instances={len(excluded_keys)}",
        f"new_holdout_audit_rows={holdout_n}",
        "- Do not claim v2 improvement from the same 42 focused v1 cases used to design the hierarchy.",
        "- Manually label the new holdout audit before reporting v2 accuracy.",
        "",
        "Interpretation",
        "--------------",
        "- Stage 1 separates true visible patterns from non-pattern texture/craftsmanship/structure effects.",
        "- Stage 2 predicts subtype only for garments routed as visible_pattern.",
        "- Scores are candidate-relative CLIP similarities, not calibrated probabilities.",
        "- material/fabric and craftsmanship remain excluded as target attributes.",
    ]

    (REPORT_DIR / "pattern_v2_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 garment pattern hierarchical baseline v2\n"
        f"model={MODEL_NAME}\n"
        f"manifest={args.manifest}\n"
        f"instances={len(output_rows)}\n"
        "stage1=non_pattern_vs_visible_pattern\n"
        "stage2=striped,checked_plaid,floral,graphic_logo,polka_dot,animal_print,camouflage,geometric_abstract,other_pattern\n"
        f"exclude_audit={args.exclude_audit or ''}\n"
        f"holdout_size={holdout_n}\n"
        f"seed={args.seed}\n"
        "gt_available=false_until_manual_holdout_audit\n"
        "material_and_craftsmanship=excluded_as_targets\n"
    )

    (REPORT_DIR / "run_info.txt").write_text(
        run_info,
        encoding="utf-8",
    )

    print("\n=== FINISHED ===")
    print(
        "PREDICTIONS : "
        "reports/prd_attribute_extraction/pattern_v2_hierarchical/"
        "pattern_v2_predictions.csv"
    )
    print(
        "SUMMARY     : "
        "reports/prd_attribute_extraction/pattern_v2_hierarchical/"
        "pattern_v2_summary.txt"
    )
    print(
        "HOLDOUT     : "
        "reports/prd_attribute_extraction/pattern_v2_hierarchical/"
        "pattern_v2_holdout_audit_template.csv"
    )
    print(
        "STAGE1 IMG  : "
        "outputs/prd_attribute_extraction/pattern_v2_hierarchical/"
        "contact_sheet_stage1_ambiguous.jpg"
    )
    print(
        "STAGE2 IMG  : "
        "outputs/prd_attribute_extraction/pattern_v2_hierarchical/"
        "contact_sheet_pattern_subtype_ambiguous.jpg"
    )
    print("================")


if __name__ == "__main__":
    main()
