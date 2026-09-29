"""PRD 3.1.3 CLIP batch pilot.

Runs a deterministic small-batch pilot for:
- color
- pattern
- sleeve_length
- neckline

Model:
    openai/clip-vit-base-patch32

Path policy:
- no machine-specific hard-coded paths
- project root derived from this script
- image directory passed as a relative path
- outputs saved under project-relative reports/ and outputs/

Example:
    python scripts/run_attribute_baseline_clip_batch.py \
        --image-dir ../fashion_data/raw/train/train/image \
        --sample-size 20 \
        --seed 20260915

Outputs:
    reports/prd_attribute_extraction/batch_20/
        predictions.csv
        manual_audit_template.csv
        run_info.txt

    outputs/prd_attribute_extraction/batch_20/
        per_image/
        contact_sheet.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "openai/clip-vit-base-patch32"

LABEL_PROMPTS: Dict[str, Dict[str, str]] = {
    "color": {
        "black": "a fashion garment whose main color is black",
        "white": "a fashion garment whose main color is white",
        "gray": "a fashion garment whose main color is gray",
        "red": "a fashion garment whose main color is red",
        "orange": "a fashion garment whose main color is orange",
        "yellow": "a fashion garment whose main color is yellow",
        "green": "a fashion garment whose main color is green",
        "blue": "a fashion garment whose main color is blue",
        "purple": "a fashion garment whose main color is purple",
        "pink": "a fashion garment whose main color is pink",
        "brown": "a fashion garment whose main color is brown",
        "beige": "a fashion garment whose main color is beige",
        "multicolor": "a fashion garment with multiple dominant colors",
        "unknown": "a fashion garment whose color cannot be determined",
    },
    "pattern": {
        "solid": "a fashion garment with a plain solid pattern",
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
    },
    "sleeve_length": {
        "sleeveless": "a sleeveless fashion garment with no sleeves",
        "cap": "a fashion garment with very short cap sleeves",
        "short": "a fashion garment with short sleeves above the elbow",
        "elbow": "a fashion garment with sleeves ending around the elbow",
        "three_quarter": "a fashion garment with three quarter sleeves below the elbow",
        "long": "a fashion garment with long sleeves reaching the wrist",
        "unknown": "a fashion garment whose sleeve length cannot be determined",
        "not_applicable": "a garment for which sleeve length is not applicable",
    },
    "neckline": {
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
        "not_applicable": "a garment for which neckline is not applicable",
    },
}


def resolve_dir(raw: str) -> Path:
    p = Path(raw).expanduser()

    if p.is_absolute() and p.is_dir():
        return p.resolve()

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.is_dir():
        return cwd_candidate

    project_candidate = (PROJECT_ROOT / p).resolve()
    if project_candidate.is_dir():
        return project_candidate

    raise FileNotFoundError(
        "Image directory not found.\n"
        f"typed path       : {raw}\n"
        f"cwd candidate    : {cwd_candidate}\n"
        f"project candidate: {project_candidate}"
    )


def classify_attribute(
    image: Image.Image,
    attribute: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> dict:
    label_to_prompt = LABEL_PROMPTS[attribute]
    labels: List[str] = list(label_to_prompt.keys())
    prompts: List[str] = [label_to_prompt[label] for label in labels]

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    probs = outputs.logits_per_image[0].softmax(dim=0)
    order = torch.argsort(probs, descending=True)

    top3 = []
    for idx in order[:3].tolist():
        idx = int(idx)
        top3.append(
            {
                "label": labels[idx],
                "score": float(probs[idx].item()),
            }
        )

    best_idx = int(order[0].item())

    return {
        "label": labels[best_idx],
        "score": float(probs[best_idx].item()),
        "top3": top3,
    }


def save_visual(
    image: Image.Image,
    predictions: dict,
    save_path: Path,
) -> None:
    panel_height = 170

    canvas = Image.new(
        "RGB",
        (image.width, image.height + panel_height),
        "white",
    )
    canvas.paste(image, (0, panel_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text(
        (12, 10),
        "PRD 3.1.3 CLIP batch pilot",
        fill="black",
        font=font,
    )

    y = 38

    for attr in [
        "color",
        "pattern",
        "sleeve_length",
        "neckline",
    ]:
        item = predictions[attr]

        draw.text(
            (12, y),
            f"{attr}: {item['label']}  score={item['score']:.3f}",
            fill="black",
            font=font,
        )

        y += 28

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        save_path,
        quality=92,
    )


def fit_tile(
    image: Image.Image,
    size: tuple[int, int] = (420, 540),
) -> Image.Image:
    tile = Image.new(
        "RGB",
        size,
        "white",
    )

    preview = image.copy()
    preview.thumbnail(
        (size[0] - 10, size[1] - 10)
    )

    x = (size[0] - preview.width) // 2
    y = (size[1] - preview.height) // 2

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
        with Image.open(path) as img:
            tiles.append(
                fit_tile(img.convert("RGB"))
            )

    columns = 4
    tile_w = 420
    tile_h = 540

    rows = (
        len(tiles) + columns - 1
    ) // columns

    sheet = Image.new(
        "RGB",
        (
            tile_w * columns,
            tile_h * rows,
        ),
        "white",
    )

    for idx, tile in enumerate(tiles):
        x = (idx % columns) * tile_w
        y = (idx // columns) * tile_h
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
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-dir",
        required=True,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260915,
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

    image_dir = resolve_dir(
        args.image_dir
    )

    image_paths = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }
        ]
    )

    if not image_paths:
        raise RuntimeError(
            f"No images found in: {image_dir}"
        )

    sample_size = min(
        args.sample_size,
        len(image_paths),
    )

    rng = random.Random(
        args.seed
    )

    selected = rng.sample(
        image_paths,
        sample_size,
    )

    batch_name = (
        f"batch_{sample_size}"
    )

    report_dir = (
        PROJECT_ROOT
        / "reports"
        / "prd_attribute_extraction"
        / batch_name
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "prd_attribute_extraction"
        / batch_name
    )

    per_image_dir = (
        output_dir
        / "per_image"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"image dir   : {args.image_dir}"
    )
    print(
        f"images found: {len(image_paths)}"
    )
    print(
        f"sample size : {sample_size}"
    )
    print(
        f"seed        : {args.seed}"
    )
    print(
        f"model       : {MODEL_NAME}"
    )
    print(
        "loading CLIP..."
    )

    device = torch.device(
        "cpu"
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

    rows = []
    visual_paths = []

    for index, image_path in enumerate(
        selected,
        start=1,
    ):
        print(
            f"[{index:02d}/{sample_size:02d}] "
            f"{image_path.name}"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        predictions = {}

        for attr in [
            "color",
            "pattern",
            "sleeve_length",
            "neckline",
        ]:
            predictions[attr] = (
                classify_attribute(
                    image=image,
                    attribute=attr,
                    model=model,
                    processor=processor,
                    device=device,
                )
            )

        row = {
            "case_id": index,
            "image_name": image_path.name,
        }

        for attr in [
            "color",
            "pattern",
            "sleeve_length",
            "neckline",
        ]:
            item = predictions[attr]

            row[
                f"{attr}_pred"
            ] = item["label"]

            row[
                f"{attr}_score"
            ] = item["score"]

            row[
                f"{attr}_top3_json"
            ] = json.dumps(
                item["top3"],
                ensure_ascii=False,
            )

        rows.append(row)

        visual_path = (
            per_image_dir
            / f"{index:02d}_{image_path.stem}_attributes.jpg"
        )

        save_visual(
            image=image,
            predictions=predictions,
            save_path=visual_path,
        )

        visual_paths.append(
            visual_path
        )

    prediction_fields = [
        "case_id",
        "image_name",
        "color_pred",
        "color_score",
        "color_top3_json",
        "pattern_pred",
        "pattern_score",
        "pattern_top3_json",
        "sleeve_length_pred",
        "sleeve_length_score",
        "sleeve_length_top3_json",
        "neckline_pred",
        "neckline_score",
        "neckline_top3_json",
    ]

    predictions_path = (
        report_dir
        / "predictions.csv"
    )

    write_csv(
        predictions_path,
        rows,
        prediction_fields,
    )

    audit_rows = []

    for row in rows:
        audit_rows.append(
            {
                "case_id": row["case_id"],
                "image_name": row["image_name"],
                "color_pred": row["color_pred"],
                "color_gt": "",
                "color_correct": "",
                "pattern_pred": row["pattern_pred"],
                "pattern_gt": "",
                "pattern_correct": "",
                "sleeve_length_pred": row["sleeve_length_pred"],
                "sleeve_length_gt": "",
                "sleeve_length_correct": "",
                "neckline_pred": row["neckline_pred"],
                "neckline_gt": "",
                "neckline_correct": "",
                "overall_note": "",
            }
        )

    audit_fields = [
        "case_id",
        "image_name",
        "color_pred",
        "color_gt",
        "color_correct",
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

    audit_path = (
        report_dir
        / "manual_audit_template.csv"
    )

    write_csv(
        audit_path,
        audit_rows,
        audit_fields,
    )

    make_contact_sheet(
        visual_paths,
        output_dir
        / "contact_sheet.jpg",
    )

    run_info = (
        "PRD 3.1.3 CLIP batch pilot\n"
        f"model={MODEL_NAME}\n"
        f"image_dir={args.image_dir}\n"
        f"sample_size={sample_size}\n"
        f"seed={args.seed}\n"
        "attributes=color,pattern,sleeve_length,neckline\n"
        "material_and_craftsmanship=excluded\n"
        "scores=relative candidate-set softmax, not calibrated probabilities\n"
    )

    (
        report_dir
        / "run_info.txt"
    ).write_text(
        run_info,
        encoding="utf-8",
    )

    print(
        "\n=== FINISHED ==="
    )
    print(
        f"PREDICTIONS : "
        f"reports/prd_attribute_extraction/"
        f"{batch_name}/predictions.csv"
    )
    print(
        f"AUDIT       : "
        f"reports/prd_attribute_extraction/"
        f"{batch_name}/manual_audit_template.csv"
    )
    print(
        f"CONTACT     : "
        f"outputs/prd_attribute_extraction/"
        f"{batch_name}/contact_sheet.jpg"
    )
    print(
        "================"
    )


if __name__ == "__main__":
    main()
