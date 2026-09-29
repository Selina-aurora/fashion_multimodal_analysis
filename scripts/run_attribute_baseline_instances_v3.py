"""PRD 3.1.3 per-garment baseline v3.

Key decisions
-------------
1. One garment instance = one attribute record.
2. primary_color returns to CLIP zero-shot classification.
   Reason: the masked-pixel quantization pilot was too sensitive to
   illumination, shadows, skin/occlusion and annotation-mask contamination.
3. secondary_colors/color_mode are removed from the current v1 baseline.
   The original multi-color outfit problem is handled by per-garment records:
       top.color != pants.color
4. pattern and sleeve_length use the garment instance crop/mask.
5. neckline is evaluated twice:
       - whole garment
       - top-center local proxy crop
   The local result is used as the current final neckline prediction.
   This is only a proxy for the future 3.1.2 collar ROI integration.

Example
-------
python scripts/run_attribute_baseline_instances_v3.py \
    --manifest configs/garment_instances_gt_pilot.csv

Outputs
-------
reports/prd_attribute_extraction/per_garment_v3/
    garment_attribute_predictions_v3.json
    garment_attribute_predictions_v3.csv
    manual_audit_template_v3.csv
    run_info.txt

outputs/prd_attribute_extraction/per_garment_v3/
    per_instance/
    neckline_local/
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
    / "per_garment_v3"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_attribute_extraction"
    / "per_garment_v3"
)

COLOR_PROMPTS = {
    "black": "a fashion garment whose dominant color is black",
    "white": "a fashion garment whose dominant color is white",
    "gray": "a fashion garment whose dominant color is gray",
    "red": "a fashion garment whose dominant color is red",
    "orange": "a fashion garment whose dominant color is orange",
    "yellow": "a fashion garment whose dominant color is yellow",
    "green": "a fashion garment whose dominant color is green",
    "blue": "a fashion garment whose dominant color is blue",
    "purple": "a fashion garment whose dominant color is purple",
    "pink": "a fashion garment whose dominant color is pink",
    "brown": "a fashion garment whose dominant color is brown",
    "beige": "a fashion garment whose dominant color is beige",
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
    "crew": "a garment with a crew neck collar opening",
    "v_neck": "a garment with a V-shaped neckline",
    "round": "a garment with a wide round scoop neckline",
    "square": "a garment with a square neckline",
    "boat": "a garment with a wide boat neckline",
    "halter": "a garment with a halter neckline",
    "off_shoulder": "a garment with an off shoulder neckline",
    "one_shoulder": "a garment with a one shoulder neckline",
    "turtleneck": "a garment with a high folded turtleneck collar",
    "mock_neck": "a garment with a short high mock neck collar",
    "polo": "a garment with a polo collar and neckline",
    "shirt_collar": "a garment with a folded shirt collar",
    "hooded": "a garment with a hood around the neckline",
    "other": "a garment with another neckline type",
    "unknown": "a garment whose neckline cannot be determined",
}

NO_UPPER_BODY_ATTRS = {
    "pants",
    "trousers",
    "jeans",
    "shorts",
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


def load_crop_and_mask(
    row: dict[str, str],
) -> tuple[Image.Image, Image.Image | None]:
    crop_path = resolve_path(row["crop_path"])
    crop = Image.open(crop_path).convert("RGB")

    mask_raw = row.get("mask_path", "").strip()
    if not mask_raw:
        return crop, None

    mask_path = resolve_path(mask_raw)
    mask = Image.open(mask_path).convert("L")

    if mask.size != crop.size:
        mask = mask.resize(
            crop.size,
            Image.Resampling.NEAREST,
        )

    return crop, mask


def apply_mask(
    crop: Image.Image,
    mask: Image.Image | None,
) -> Image.Image:
    if mask is None:
        return crop

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


def make_neckline_local_proxy(
    crop: Image.Image,
    mask: Image.Image | None,
) -> tuple[Image.Image, Image.Image | None]:
    """Top-center local crop used only as a proxy for a future 3.1.2 collar ROI."""
    width, height = crop.size

    x1 = int(round(width * 0.10))
    x2 = int(round(width * 0.90))
    y1 = 0
    y2 = max(
        1,
        int(round(height * 0.45)),
    )

    local_crop = crop.crop(
        (x1, y1, x2, y2)
    )

    local_mask = None

    if mask is not None:
        local_mask = mask.crop(
            (x1, y1, x2, y2)
        )

    return local_crop, local_mask


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


def infer_attribute(
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
            "relative CLIP candidate-set score; "
            "not calibrated probability"
        ),
    }


def save_instance_visual(
    image: Image.Image,
    record: dict,
    save_path: Path,
) -> None:
    panel_h = 225

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

    attrs = record["attributes"]

    lines = [
        (
            f"{record['garment_id']} | "
            f"{record['garment_category']}"
        ),
        (
            "primary_color: "
            f"{attrs['primary_color']['label']} "
            f"score={attrs['primary_color']['score']:.3f}"
        ),
        (
            "pattern: "
            f"{attrs['pattern']['label']}"
        ),
        (
            "sleeve_length: "
            f"{attrs['sleeve_length']['label']}"
        ),
        (
            "neckline_final: "
            f"{attrs['neckline']['label']}"
        ),
        (
            "neckline_whole: "
            f"{attrs['neckline_whole']['label']}"
        ),
    ]

    y = 10

    for line in lines:
        draw.text(
            (12, y),
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
    paths: list[Path],
    output_path: Path,
) -> None:
    if not paths:
        return

    tiles = []

    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((420, 570))

            tile = Image.new(
                "RGB",
                (430, 580),
                "white",
            )

            x = (430 - img.width) // 2
            y = (580 - img.height) // 2

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
            580 * rows,
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
        ) * 580
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

    local_dir = (
        OUTPUT_DIR
        / "neckline_local"
    )

    per_instance_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_dir.mkdir(
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
        "primary color: per-garment CLIP"
    )
    print(
        "neckline     : top-center local proxy + whole-garment comparison"
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

        print(
            f"[{index:02d}/{len(rows):02d}] "
            f"{garment_id} | {category}"
        )

        crop, mask = load_crop_and_mask(
            row
        )

        garment_image = apply_mask(
            crop,
            mask,
        )

        color_result = infer_attribute(
            garment_image,
            COLOR_PROMPTS,
            model,
            processor,
            device,
        )

        pattern_result = infer_attribute(
            garment_image,
            PATTERN_PROMPTS,
            model,
            processor,
            device,
        )

        if category in NO_UPPER_BODY_ATTRS:
            sleeve_result = {
                "label": "not_applicable",
                "score": 1.0,
                "top3": [],
                "score_note": "forced by garment category",
            }

            neckline_whole_result = {
                "label": "not_applicable",
                "score": 1.0,
                "top3": [],
                "score_note": "forced by garment category",
            }

            neckline_local_result = dict(
                neckline_whole_result
            )

            neckline_final = dict(
                neckline_whole_result
            )

            local_preview = None

        else:
            sleeve_result = infer_attribute(
                garment_image,
                SLEEVE_PROMPTS,
                model,
                processor,
                device,
            )

            neckline_whole_result = infer_attribute(
                garment_image,
                NECKLINE_PROMPTS,
                model,
                processor,
                device,
            )

            local_crop, local_mask = (
                make_neckline_local_proxy(
                    crop,
                    mask,
                )
            )

            local_preview = apply_mask(
                local_crop,
                local_mask,
            )

            neckline_local_result = infer_attribute(
                local_preview,
                NECKLINE_PROMPTS,
                model,
                processor,
                device,
            )

            # Current final = local proxy result.
            # This will later be replaced by the actual 3.1.2 collar ROI result.
            neckline_final = dict(
                neckline_local_result
            )

        record = {
            "source_image": row["source_image"],
            "garment_id": garment_id,
            "garment_category": category,
            "region_source": (
                "mask"
                if row.get("mask_path", "").strip()
                else "crop"
            ),
            "crop_path": row["crop_path"],
            "mask_path": row.get(
                "mask_path",
                "",
            ),
            "model": MODEL_NAME,
            "attributes": {
                "primary_color": color_result,
                "pattern": pattern_result,
                "sleeve_length": sleeve_result,
                "neckline": neckline_final,
                "neckline_whole": (
                    neckline_whole_result
                ),
                "neckline_local_proxy": (
                    neckline_local_result
                ),
            },
        }

        records.append(
            record
        )

        csv_rows.append(
            {
                "source_image": (
                    row["source_image"]
                ),
                "garment_id": garment_id,
                "garment_category": category,
                "primary_color": (
                    color_result["label"]
                ),
                "primary_color_score": (
                    color_result["score"]
                ),
                "pattern": (
                    pattern_result["label"]
                ),
                "sleeve_length": (
                    sleeve_result["label"]
                ),
                "neckline_final": (
                    neckline_final["label"]
                ),
                "neckline_whole": (
                    neckline_whole_result["label"]
                ),
                "neckline_local_proxy": (
                    neckline_local_result["label"]
                ),
            }
        )

        if local_preview is not None:
            local_preview.save(
                local_dir
                / f"{index:02d}_{garment_id}_neckline.jpg",
                quality=95,
            )

        visual_path = (
            per_instance_dir
            / f"{index:02d}_{garment_id}.jpg"
        )

        save_instance_visual(
            garment_image,
            record,
            visual_path,
        )

        visual_paths.append(
            visual_path
        )

    json_path = (
        REPORT_DIR
        / "garment_attribute_predictions_v3.json"
    )

    json_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "3.1.3-v3-per-garment"
                ),
                "model": MODEL_NAME,
                "scope": (
                    "one garment instance per record; "
                    "dominant color only; "
                    "material and craftsmanship excluded"
                ),
                "neckline_note": (
                    "final neckline currently uses a top-center local proxy crop; "
                    "replace with actual 3.1.2 collar ROI in integration stage"
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
        / "garment_attribute_predictions_v3.csv"
    )

    csv_fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "primary_color",
        "primary_color_score",
        "pattern",
        "sleeve_length",
        "neckline_final",
        "neckline_whole",
        "neckline_local_proxy",
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
        / "manual_audit_template_v3.csv"
    )

    audit_fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "primary_color_pred",
        "primary_color_gt",
        "primary_color_correct",
        "pattern_pred",
        "pattern_gt",
        "pattern_correct",
        "sleeve_length_pred",
        "sleeve_length_gt",
        "sleeve_length_correct",
        "neckline_final_pred",
        "neckline_gt",
        "neckline_correct",
        "neckline_whole_pred",
        "neckline_local_proxy_pred",
        "overall_note",
    ]

    audit_rows = []

    for row in csv_rows:
        audit_rows.append(
            {
                "source_image": (
                    row["source_image"]
                ),
                "garment_id": (
                    row["garment_id"]
                ),
                "garment_category": (
                    row["garment_category"]
                ),
                "primary_color_pred": (
                    row["primary_color"]
                ),
                "primary_color_gt": "",
                "primary_color_correct": "",
                "pattern_pred": (
                    row["pattern"]
                ),
                "pattern_gt": "",
                "pattern_correct": "",
                "sleeve_length_pred": (
                    row["sleeve_length"]
                ),
                "sleeve_length_gt": "",
                "sleeve_length_correct": "",
                "neckline_final_pred": (
                    row["neckline_final"]
                ),
                "neckline_gt": "",
                "neckline_correct": "",
                "neckline_whole_pred": (
                    row["neckline_whole"]
                ),
                "neckline_local_proxy_pred": (
                    row["neckline_local_proxy"]
                ),
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
        "PRD 3.1.3 per-garment baseline v3\n"
        f"model={MODEL_NAME}\n"
        f"instances={len(rows)}\n"
        "unit_of_analysis=one garment instance per row\n"
        "primary_color=CLIP dominant-color classification\n"
        "secondary_colors=not included in current v1 scope\n"
        "pattern=CLIP on garment instance\n"
        "sleeve_length=CLIP on garment instance\n"
        "neckline=CLIP on top-center local proxy crop\n"
        "neckline_whole=retained for diagnostic comparison\n"
        "material_and_craftsmanship=excluded\n"
    )

    (
        REPORT_DIR
        / "run_info.txt"
    ).write_text(
        run_info,
        encoding="utf-8",
    )

    print("\n=== FINISHED ===")
    print(
        "JSON    : "
        "reports/prd_attribute_extraction/per_garment_v3/"
        "garment_attribute_predictions_v3.json"
    )
    print(
        "CSV     : "
        "reports/prd_attribute_extraction/per_garment_v3/"
        "garment_attribute_predictions_v3.csv"
    )
    print(
        "AUDIT   : "
        "reports/prd_attribute_extraction/per_garment_v3/"
        "manual_audit_template_v3.csv"
    )
    print(
        "LOCAL   : "
        "outputs/prd_attribute_extraction/per_garment_v3/"
        "neckline_local/"
    )
    print(
        "CONTACT : "
        "outputs/prd_attribute_extraction/per_garment_v3/"
        "contact_sheet.jpg"
    )
    print("================")


if __name__ == "__main__":
    main()
