"""PRD 3.1.3 per-garment CLIP attribute baseline.

This version fixes the full-outfit color problem by treating EACH garment
instance as a separate record.

Input manifest columns
----------------------
source_image,garment_id,garment_category,crop_path,mask_path

One row = one garment instance.

Recommended upstream source:
    3.1.1 garment instance segmentation / garment crop.

Current attributes
------------------
- primary_color
- secondary_colors
- color_mode
- pattern
- sleeve_length
- neckline

Excluded for current stage
--------------------------
- material / fabric composition
- manufacturing process / craftsmanship

Example
-------
python scripts/run_attribute_baseline_clip_instances.py \
    --manifest configs/garment_instances.csv

Outputs
-------
reports/prd_attribute_extraction/per_garment/
    garment_attribute_predictions.csv
    garment_attribute_predictions.json
    manual_audit_template.csv
    run_info.txt

outputs/prd_attribute_extraction/per_garment/
    per_instance/
    contact_sheet.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
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
    / "per_garment"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_attribute_extraction"
    / "per_garment"
)

BASIC_COLORS = {
    "black": "a fashion garment whose visible main color is black",
    "white": "a fashion garment whose visible main color is white",
    "gray": "a fashion garment whose visible main color is gray",
    "red": "a fashion garment whose visible main color is red",
    "orange": "a fashion garment whose visible main color is orange",
    "yellow": "a fashion garment whose visible main color is yellow",
    "green": "a fashion garment whose visible main color is green",
    "blue": "a fashion garment whose visible main color is blue",
    "purple": "a fashion garment whose visible main color is purple",
    "pink": "a fashion garment whose visible main color is pink",
    "brown": "a fashion garment whose visible main color is brown",
    "beige": "a fashion garment whose visible main color is beige",
}

COLOR_MODE_PROMPTS = {
    "single_color": (
        "a garment with one clearly dominant visible color"
    ),
    "multicolor": (
        "a garment with two or more visually prominent colors"
    ),
}

PATTERN_PROMPTS = {
    "solid": "a fashion garment with a plain solid appearance",
    "striped": "a fashion garment with stripes",
    "checked": "a fashion garment with a checkered plaid pattern",
    "floral": "a fashion garment with a floral flower pattern",
    "graphic": "a fashion garment with printed graphics letters or logos",
    "polka_dot": "a fashion garment with polka dots",
    "animal": "a fashion garment with an animal print pattern",
    "camouflage": "a fashion garment with camouflage pattern",
    "geometric": "a fashion garment with geometric shapes pattern",
    "abstract": "a fashion garment with an abstract pattern",
    "other": "a fashion garment with another visible decorative pattern",
    "unknown": "a fashion garment whose pattern cannot be determined",
}

SLEEVE_PROMPTS = {
    "sleeveless": "a sleeveless fashion garment with no sleeves",
    "cap": "a fashion garment with very short cap sleeves",
    "short": "a fashion garment with short sleeves above the elbow",
    "elbow": "a fashion garment with sleeves ending around the elbow",
    "three_quarter": "a fashion garment with three quarter sleeves below the elbow",
    "long": "a fashion garment with long sleeves reaching the wrist",
    "unknown": "a fashion garment whose sleeve length cannot be determined",
}

NECKLINE_PROMPTS = {
    "crew": "a fashion garment with a crew neck neckline",
    "v_neck": "a fashion garment with a V neck neckline",
    "round": "a fashion garment with a round scoop neckline",
    "square": "a fashion garment with a square neckline",
    "boat": "a fashion garment with a boat neckline",
    "halter": "a fashion garment with a halter neckline",
    "off_shoulder": "an off shoulder fashion garment",
    "one_shoulder": "a one shoulder fashion garment",
    "turtleneck": "a fashion garment with a turtleneck",
    "mock_neck": "a fashion garment with a mock neck",
    "polo": "a fashion garment with a polo collar",
    "shirt_collar": "a fashion garment with a shirt collar",
    "hooded": "a fashion garment with a hood",
    "other": "a fashion garment with another visible neckline type",
    "unknown": "a fashion garment whose neckline cannot be determined",
}

NO_UPPER_BODY_ATTRS = {
    "pants",
    "trousers",
    "jeans",
    "skirt",
    "shoes",
    "shoe",
    "bag",
    "accessory",
    "accessories",
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
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
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
            f"Manifest missing columns: {sorted(missing)}"
        )

    return rows


def apply_optional_mask(
    crop: Image.Image,
    mask_path_raw: str,
) -> Image.Image:
    if not mask_path_raw.strip():
        return crop

    mask_path = resolve_path(
        mask_path_raw
    )

    mask = Image.open(
        mask_path
    ).convert("L")

    if mask.size != crop.size:
        mask = mask.resize(
            crop.size
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


def clip_rank(
    image: Image.Image,
    label_prompts: Dict[str, str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> list[dict]:
    labels: List[str] = list(
        label_prompts.keys()
    )

    prompts: List[str] = [
        label_prompts[label]
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
    )

    order = torch.argsort(
        probs,
        descending=True,
    )

    ranked = []

    for idx in order.tolist():
        idx = int(idx)

        ranked.append(
            {
                "label": labels[idx],
                "score": float(
                    probs[idx].item()
                ),
            }
        )

    return ranked


def infer_colors(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> dict:
    mode_ranked = clip_rank(
        image,
        COLOR_MODE_PROMPTS,
        model,
        processor,
        device,
    )

    color_ranked = clip_rank(
        image,
        BASIC_COLORS,
        model,
        processor,
        device,
    )

    color_mode = (
        mode_ranked[0]["label"]
    )

    primary = color_ranked[0]

    secondary = []

    if color_mode == "multicolor":
        primary_score = max(
            primary["score"],
            1e-8,
        )

        for item in color_ranked[1:3]:
            # Baseline heuristic only.
            # This is intentionally conservative and should be manually audited.
            if (
                item["score"] >= 0.10
                and item["score"]
                >= primary_score * 0.55
            ):
                secondary.append(
                    item["label"]
                )

    return {
        "primary_color": primary,
        "secondary_colors": secondary,
        "color_mode": mode_ranked[0],
        "color_top3": color_ranked[:3],
        "color_mode_top2": mode_ranked[:2],
        "score_note": (
            "CLIP relative candidate-set scores; not calibrated probabilities"
        ),
    }


def infer_single_attribute(
    image: Image.Image,
    prompts: Dict[str, str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> dict:
    ranked = clip_rank(
        image,
        prompts,
        model,
        processor,
        device,
    )

    return {
        "label": ranked[0]["label"],
        "score": ranked[0]["score"],
        "top3": ranked[:3],
        "score_note": (
            "CLIP relative candidate-set scores; not calibrated probabilities"
        ),
    }


def save_visual(
    image: Image.Image,
    record: dict,
    save_path: Path,
) -> None:
    panel_h = 205

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

    draw = ImageDraw.Draw(
        canvas
    )
    font = ImageFont.load_default()

    draw.text(
        (12, 10),
        (
            f"{record['garment_id']} | "
            f"{record['garment_category']}"
        ),
        fill="black",
        font=font,
    )

    attrs = record[
        "attributes"
    ]

    primary = attrs[
        "primary_color"
    ]["label"]

    secondary = attrs[
        "secondary_colors"
    ]

    draw.text(
        (12, 38),
        f"primary_color: {primary}",
        fill="black",
        font=font,
    )

    draw.text(
        (12, 64),
        (
            "secondary_colors: "
            + (
                ", ".join(secondary)
                if secondary
                else "[]"
            )
        ),
        fill="black",
        font=font,
    )

    draw.text(
        (12, 90),
        (
            "pattern: "
            f"{attrs['pattern']['label']}"
        ),
        fill="black",
        font=font,
    )

    draw.text(
        (12, 116),
        (
            "sleeve_length: "
            f"{attrs['sleeve_length']['label']}"
        ),
        fill="black",
        font=font,
    )

    draw.text(
        (12, 142),
        (
            "neckline: "
            f"{attrs['neckline']['label']}"
        ),
        fill="black",
        font=font,
    )

    draw.text(
        (12, 168),
        (
            "color_mode: "
            f"{attrs['color_mode']['label']}"
        ),
        fill="black",
        font=font,
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        save_path,
        quality=92,
    )


def make_contact_sheet(
    paths: list[Path],
    output_path: Path,
) -> None:
    if not paths:
        return

    tiles = []

    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((420, 540))

            tile = Image.new(
                "RGB",
                (430, 550),
                "white",
            )

            x = (
                430 - img.width
            ) // 2
            y = (
                550 - img.height
            ) // 2

            tile.paste(
                img,
                (x, y),
            )

            tiles.append(tile)

    cols = 3
    rows = (
        len(tiles) + cols - 1
    ) // cols

    sheet = Image.new(
        "RGB",
        (
            430 * cols,
            550 * rows,
        ),
        "white",
    )

    for idx, tile in enumerate(
        tiles
    ):
        x = (
            idx % cols
        ) * 430
        y = (
            idx // cols
        ) * 550

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

    device = torch.device(
        "cpu"
    )

    print(
        f"manifest     : {args.manifest}"
    )
    print(
        f"instances    : {len(rows)}"
    )
    print(
        f"model        : {MODEL_NAME}"
    )
    print(
        "loading CLIP..."
    )

    processor = (
        CLIPProcessor.from_pretrained(
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

    records = []
    csv_rows = []
    visual_paths = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        garment_id = (
            row["garment_id"].strip()
            or f"garment_{index:02d}"
        )

        category = (
            row["garment_category"]
            .strip()
            .lower()
            or "unknown"
        )

        crop_path = resolve_path(
            row["crop_path"]
        )

        print(
            f"[{index:02d}/{len(rows):02d}] "
            f"{garment_id} | "
            f"{category} | "
            f"{crop_path.name}"
        )

        crop = Image.open(
            crop_path
        ).convert("RGB")

        crop = apply_optional_mask(
            crop,
            row["mask_path"],
        )

        color_result = infer_colors(
            crop,
            model,
            processor,
            device,
        )

        pattern_result = (
            infer_single_attribute(
                crop,
                PATTERN_PROMPTS,
                model,
                processor,
                device,
            )
        )

        if category in NO_UPPER_BODY_ATTRS:
            sleeve_result = {
                "label": "not_applicable",
                "score": 1.0,
                "top3": [],
                "score_note": (
                    "forced by garment category"
                ),
            }

            neckline_result = {
                "label": "not_applicable",
                "score": 1.0,
                "top3": [],
                "score_note": (
                    "forced by garment category"
                ),
            }

        else:
            sleeve_result = (
                infer_single_attribute(
                    crop,
                    SLEEVE_PROMPTS,
                    model,
                    processor,
                    device,
                )
            )

            neckline_result = (
                infer_single_attribute(
                    crop,
                    NECKLINE_PROMPTS,
                    model,
                    processor,
                    device,
                )
            )

        record = {
            "source_image": (
                row["source_image"]
            ),
            "garment_id": garment_id,
            "garment_category": category,
            "region_source": (
                "mask"
                if row["mask_path"].strip()
                else "crop"
            ),
            "crop_path": (
                row["crop_path"]
            ),
            "mask_path": (
                row["mask_path"]
            ),
            "model": MODEL_NAME,
            "attributes": {
                "primary_color": (
                    color_result[
                        "primary_color"
                    ]
                ),
                "secondary_colors": (
                    color_result[
                        "secondary_colors"
                    ]
                ),
                "color_mode": (
                    color_result[
                        "color_mode"
                    ]
                ),
                "color_top3": (
                    color_result[
                        "color_top3"
                    ]
                ),
                "pattern": pattern_result,
                "sleeve_length": (
                    sleeve_result
                ),
                "neckline": (
                    neckline_result
                ),
            },
        }

        records.append(
            record
        )

        csv_rows.append(
            {
                "source_image": row[
                    "source_image"
                ],
                "garment_id": garment_id,
                "garment_category": category,
                "crop_path": row[
                    "crop_path"
                ],
                "mask_path": row[
                    "mask_path"
                ],
                "primary_color": (
                    record[
                        "attributes"
                    ][
                        "primary_color"
                    ][
                        "label"
                    ]
                ),
                "secondary_colors": ";".join(
                    record[
                        "attributes"
                    ][
                        "secondary_colors"
                    ]
                ),
                "color_mode": (
                    record[
                        "attributes"
                    ][
                        "color_mode"
                    ][
                        "label"
                    ]
                ),
                "pattern": (
                    pattern_result[
                        "label"
                    ]
                ),
                "sleeve_length": (
                    sleeve_result[
                        "label"
                    ]
                ),
                "neckline": (
                    neckline_result[
                        "label"
                    ]
                ),
            }
        )

        visual_path = (
            per_instance_dir
            / f"{index:02d}_{garment_id}.jpg"
        )

        save_visual(
            crop,
            record,
            visual_path,
        )

        visual_paths.append(
            visual_path
        )

    json_path = (
        REPORT_DIR
        / "garment_attribute_predictions.json"
    )

    json_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "3.1.3-v2-per-garment"
                ),
                "model": MODEL_NAME,
                "scope": (
                    "per garment instance; "
                    "material and craftsmanship excluded"
                ),
                "garments": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = (
        REPORT_DIR
        / "garment_attribute_predictions.csv"
    )

    csv_fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "crop_path",
        "mask_path",
        "primary_color",
        "secondary_colors",
        "color_mode",
        "pattern",
        "sleeve_length",
        "neckline",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )
        writer.writeheader()
        writer.writerows(
            csv_rows
        )

    audit_path = (
        REPORT_DIR
        / "manual_audit_template.csv"
    )

    audit_fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "primary_color_pred",
        "primary_color_gt",
        "primary_color_correct",
        "secondary_colors_pred",
        "secondary_colors_gt",
        "secondary_colors_correct",
        "pattern_pred",
        "pattern_gt",
        "pattern_correct",
        "sleeve_length_pred",
        "sleeve_length_gt",
        "sleeve_length_correct",
        "neckline_pred",
        "neckline_gt",
        "neckline_correct",
        "overall_note",
    ]

    audit_rows = []

    for row in csv_rows:
        audit_rows.append(
            {
                "source_image": (
                    row[
                        "source_image"
                    ]
                ),
                "garment_id": (
                    row[
                        "garment_id"
                    ]
                ),
                "garment_category": (
                    row[
                        "garment_category"
                    ]
                ),
                "primary_color_pred": (
                    row[
                        "primary_color"
                    ]
                ),
                "primary_color_gt": "",
                "primary_color_correct": "",
                "secondary_colors_pred": (
                    row[
                        "secondary_colors"
                    ]
                ),
                "secondary_colors_gt": "",
                "secondary_colors_correct": "",
                "pattern_pred": (
                    row[
                        "pattern"
                    ]
                ),
                "pattern_gt": "",
                "pattern_correct": "",
                "sleeve_length_pred": (
                    row[
                        "sleeve_length"
                    ]
                ),
                "sleeve_length_gt": "",
                "sleeve_length_correct": "",
                "neckline_pred": (
                    row[
                        "neckline"
                    ]
                ),
                "neckline_gt": "",
                "neckline_correct": "",
                "overall_note": "",
            }
        )

    with audit_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=audit_fields,
        )
        writer.writeheader()
        writer.writerows(
            audit_rows
        )

    make_contact_sheet(
        visual_paths,
        OUTPUT_DIR
        / "contact_sheet.jpg",
    )

    run_info = (
        "PRD 3.1.3 per-garment CLIP baseline\n"
        f"model={MODEL_NAME}\n"
        f"instances={len(rows)}\n"
        "unit_of_analysis=one garment instance per row\n"
        "attributes=primary_color,secondary_colors,color_mode,pattern,sleeve_length,neckline\n"
        "material_and_craftsmanship=excluded\n"
        "scores=relative CLIP candidate-set scores, not calibrated probabilities\n"
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
        "JSON    : "
        "reports/prd_attribute_extraction/per_garment/"
        "garment_attribute_predictions.json"
    )
    print(
        "CSV     : "
        "reports/prd_attribute_extraction/per_garment/"
        "garment_attribute_predictions.csv"
    )
    print(
        "AUDIT   : "
        "reports/prd_attribute_extraction/per_garment/"
        "manual_audit_template.csv"
    )
    print(
        "CONTACT : "
        "outputs/prd_attribute_extraction/per_garment/"
        "contact_sheet.jpg"
    )
    print(
        "================"
    )


if __name__ == "__main__":
    main()
