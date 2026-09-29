"""PRD 3.1.3 local CPU baseline using CLIP.

This replaces the tiny generative VLM for the first structured baseline because
SmolVLM-256M can see the image but does not reliably obey strict JSON output.

Attributes:
- color
- pattern
- sleeve_length
- neckline

Excluded:
- material/fabric composition
- manufacturing process/craftsmanship
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "openai/clip-vit-base-patch32"
REPORT_DIR = PROJECT_ROOT / "reports" / "prd_attribute_extraction"
VIS_DIR = PROJECT_ROOT / "outputs" / "prd_attribute_extraction"

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


def resolve_image_path(raw: str) -> Path:
    p = Path(raw).expanduser()

    if p.is_absolute() and p.is_file():
        return p.resolve()

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate

    project_candidate = (PROJECT_ROOT / p).resolve()
    if project_candidate.is_file():
        return project_candidate

    raise FileNotFoundError(
        "Image not found.\n"
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

    top_index = int(order[0].item())

    top3 = []
    for idx in order[:3].tolist():
        idx = int(idx)
        top3.append({
            "label": labels[idx],
            "score": float(probs[idx].item()),
        })

    return {
        "label": labels[top_index],
        "confidence": float(probs[top_index].item()),
        "top3": top3,
        "confidence_note": "relative softmax score over candidate labels; not calibrated",
    }


def make_visual(image: Image.Image, result: dict, save_path: Path) -> None:
    panel_height = 165
    canvas = Image.new("RGB", (image.width, image.height + panel_height), "white")
    canvas.paste(image, (0, panel_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((12, 10), "PRD 3.1.3 CLIP CPU baseline", fill="black", font=font)

    y = 36
    for attr in ["color", "pattern", "sleeve_length", "neckline"]:
        item = result["predictions"][attr]
        draw.text(
            (12, y),
            f"{attr}: {item['label']}  score={item['confidence']:.3f}",
            fill="black",
            font=font,
        )
        y += 27

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--region",
        default="whole_garment",
        choices=["whole_garment", "roi", "local_region"],
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")

    image_path = resolve_image_path(args.image)
    image = Image.open(image_path).convert("RGB")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    json_path = REPORT_DIR / f"{stem}_attributes_clip.json"
    input_copy_path = VIS_DIR / f"{stem}_input.jpg"
    visual_path = VIS_DIR / f"{stem}_attributes_clip.jpg"

    print(f"model       : {MODEL_NAME}")
    print("device      : cpu")
    print(f"input image : {args.image}")
    print(f"cpu threads : {args.threads}")
    print("loading processor...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    print("loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    predictions = {}

    for attr in ["color", "pattern", "sleeve_length", "neckline"]:
        print(f"classifying {attr}...")
        predictions[attr] = classify_attribute(
            image=image,
            attribute=attr,
            model=model,
            processor=processor,
            device=device,
        )

    final = {
        "image": args.image,
        "region": args.region,
        "model": MODEL_NAME,
        "scope": "visual attributes only; material and craftsmanship excluded",
        "predictions": predictions,
    }

    json_path.write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    shutil.copy2(image_path, input_copy_path)
    make_visual(image, final, visual_path)

    print("\nRESULT:")
    print(json.dumps(final, ensure_ascii=False, indent=2))

    print("\n=== SAVED FILES ===")
    print(f"JSON       : reports/prd_attribute_extraction/{json_path.name}")
    print(f"INPUT COPY : outputs/prd_attribute_extraction/{input_copy_path.name}")
    print(f"VISUAL     : outputs/prd_attribute_extraction/{visual_path.name}")
    print("===================")


if __name__ == "__main__":
    main()
