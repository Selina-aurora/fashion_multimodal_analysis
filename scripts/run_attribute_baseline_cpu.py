"""PRD 3.1.3 local CPU attribute baseline.

Lightweight model:
    HuggingFaceTB/SmolVLM-256M-Instruct

Purpose:
    Run a small local sanity-check baseline for visually observable fashion
    attributes without loading a multi-billion-parameter VLM.

Attributes:
    - color
    - pattern
    - sleeve_length
    - neckline

Intentionally excluded:
    - material / fabric composition
    - manufacturing process / craftsmanship

Path policy:
    - no hard-coded machine-specific paths
    - project root is derived from this script's location
    - input image can be passed with a relative path
    - outputs are saved relative to PROJECT_ROOT

Example:
    python scripts/run_attribute_baseline_cpu.py \
        --image ../fashion_data/raw/train/train/image/000001.jpg

Outputs:
    reports/prd_attribute_extraction/<stem>_attributes.json
    outputs/prd_attribute_extraction/<stem>_input.jpg
    outputs/prd_attribute_extraction/<stem>_attributes.jpg
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForVision2Seq, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

REPORT_DIR = PROJECT_ROOT / "reports" / "prd_attribute_extraction"
VIS_DIR = PROJECT_ROOT / "outputs" / "prd_attribute_extraction"

ALLOWED = {
    "color": [
        "black",
        "white",
        "gray",
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "pink",
        "brown",
        "beige",
        "multicolor",
        "unknown",
    ],
    "pattern": [
        "solid",
        "striped",
        "checked",
        "floral",
        "graphic",
        "polka_dot",
        "animal",
        "camouflage",
        "geometric",
        "abstract",
        "other",
        "unknown",
    ],
    "sleeve_length": [
        "sleeveless",
        "cap",
        "short",
        "elbow",
        "three_quarter",
        "long",
        "unknown",
        "not_applicable",
    ],
    "neckline": [
        "crew",
        "v_neck",
        "round",
        "square",
        "boat",
        "halter",
        "off_shoulder",
        "one_shoulder",
        "turtleneck",
        "mock_neck",
        "polo",
        "shirt_collar",
        "hooded",
        "other",
        "unknown",
        "not_applicable",
    ],
}


def resolve_image_path(raw: str) -> Path:
    """Resolve user input without hard-coding a machine-specific absolute path."""
    p = Path(raw).expanduser()

    if p.is_absolute() and p.is_file():
        return p.resolve()

    # Normal case: user runs from project root and supplies ../fashion_data/...
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate

    # Fallback: interpret relative to project root.
    project_candidate = (PROJECT_ROOT / p).resolve()
    if project_candidate.is_file():
        return project_candidate

    raise FileNotFoundError(
        "Image not found.\n"
        f"typed path       : {raw}\n"
        f"cwd candidate    : {cwd_candidate}\n"
        f"project candidate: {project_candidate}"
    )


def build_prompt() -> str:
    return f"""
You are a visual attribute extractor for e-commerce fashion images.

Inspect ONLY attributes that are directly visible in the image.

Extract exactly these four attributes:
1. color
2. pattern
3. sleeve_length
4. neckline

Do NOT infer:
- fabric or material composition
- manufacturing process or craftsmanship

Rules:
- Return one JSON object only.
- Use only the allowed labels below.
- If visual evidence is insufficient, return "unknown".
- If the attribute does not apply, return "not_applicable".
- confidence must be a number from 0.0 to 1.0.
- evidence must be a very short visual reason.

Allowed labels:
color = {ALLOWED["color"]}
pattern = {ALLOWED["pattern"]}
sleeve_length = {ALLOWED["sleeve_length"]}
neckline = {ALLOWED["neckline"]}

Required JSON:
{{
  "predictions": {{
    "color": {{
      "label": "...",
      "confidence": 0.0,
      "evidence": "..."
    }},
    "pattern": {{
      "label": "...",
      "confidence": 0.0,
      "evidence": "..."
    }},
    "sleeve_length": {{
      "label": "...",
      "confidence": 0.0,
      "evidence": "..."
    }},
    "neckline": {{
      "label": "...",
      "confidence": 0.0,
      "evidence": "..."
    }}
  }}
}}
""".strip()


def extract_json(text: str) -> dict:
    """Extract the outermost JSON object from model output."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "The model response did not contain a JSON object.\n"
            f"Raw response:\n{text}"
        )

    return json.loads(text[start : end + 1])


def validate_result(result: dict) -> dict:
    """Force the output back into our fixed schema."""
    raw_predictions = result.get("predictions", {})
    cleaned = {}

    for attr, allowed_values in ALLOWED.items():
        item = raw_predictions.get(attr, {})

        label = str(item.get("label", "unknown")).strip()
        if label not in allowed_values:
            label = "unknown"

        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))
        evidence = str(item.get("evidence", "")).strip()

        cleaned[attr] = {
            "label": label,
            "confidence": confidence,
            "evidence": evidence,
        }

    return {"predictions": cleaned}


def make_visual(
    image: Image.Image,
    result: dict,
    save_path: Path,
) -> None:
    """Save original image plus a simple text summary panel."""
    panel_height = 150

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
        "PRD 3.1.3 local CPU baseline",
        fill="black",
        font=font,
    )

    y = 34

    for attr in [
        "color",
        "pattern",
        "sleeve_length",
        "neckline",
    ]:
        item = result["predictions"][attr]

        line = (
            f"{attr}: {item['label']} "
            f"(conf={item['confidence']:.2f})"
        )

        draw.text(
            (12, y),
            line,
            fill="black",
            font=font,
        )

        y += 24

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        save_path,
        quality=92,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Input image path. Relative paths are recommended.",
    )

    parser.add_argument(
        "--region",
        default="whole_garment",
        choices=[
            "whole_garment",
            "roi",
            "local_region",
        ],
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU inference threads.",
    )

    args = parser.parse_args()

    torch.set_num_threads(
        max(1, args.threads)
    )

    image_path = resolve_image_path(
        args.image
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = image_path.stem

    json_path = (
        REPORT_DIR
        / f"{stem}_attributes.json"
    )

    input_copy_path = (
        VIS_DIR
        / f"{stem}_input.jpg"
    )

    visual_path = (
        VIS_DIR
        / f"{stem}_attributes.jpg"
    )

    # Keep this local sanity-check deliberately on CPU.
    device = torch.device("cpu")

    print(
        f"project root : {PROJECT_ROOT}"
    )
    print(
        f"device       : {device}"
    )
    print(
        f"model        : {args.model}"
    )
    print(
        f"input image  : {image_path}"
    )
    print(
        f"cpu threads  : {args.threads}"
    )
    print(
        "loading processor..."
    )

    processor = AutoProcessor.from_pretrained(
        args.model
    )

    print(
        "loading lightweight model..."
    )

    # float32 is the safest CPU dtype across different Windows CPUs.
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        _attn_implementation="eager",
    ).to(device)

    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": build_prompt(),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    print(
        "running inference..."
    )

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    input_length = (
        inputs["input_ids"].shape[1]
    )

    new_tokens = generated_ids[
        :,
        input_length:,
    ]

    generated_text = (
        processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    )

    print(
        "\nraw model response:"
    )
    print(
        generated_text
    )

    parsed = extract_json(
        generated_text
    )

    result = validate_result(
        parsed
    )

    final = {
        "image": str(
            image_path.relative_to(
                PROJECT_ROOT.parent
            )
            if image_path.is_relative_to(
                PROJECT_ROOT.parent
            )
            else image_path.name
        ),
        "region": args.region,
        "model": args.model,
        **result,
    }

    output_text = json.dumps(
        final,
        ensure_ascii=False,
        indent=2,
    )

    json_path.write_text(
        output_text,
        encoding="utf-8",
    )

    shutil.copy2(
        image_path,
        input_copy_path,
    )

    make_visual(
        image=image,
        result=result,
        save_path=visual_path,
    )

    print(
        "\nvalidated result:"
    )
    print(
        output_text
    )

    print(
        "\n=== SAVED FILES ==="
    )
    print(
        "JSON       : "
        f"reports/prd_attribute_extraction/"
        f"{json_path.name}"
    )
    print(
        "INPUT COPY : "
        f"outputs/prd_attribute_extraction/"
        f"{input_copy_path.name}"
    )
    print(
        "VISUAL     : "
        f"outputs/prd_attribute_extraction/"
        f"{visual_path.name}"
    )
    print(
        "==================="
    )


if __name__ == "__main__":
    main()
