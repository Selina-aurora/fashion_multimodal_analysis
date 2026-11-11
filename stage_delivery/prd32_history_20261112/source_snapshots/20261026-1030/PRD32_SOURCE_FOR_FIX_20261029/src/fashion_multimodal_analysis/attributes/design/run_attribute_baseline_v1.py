"""3.1.3 可视属性提取：关注颜色、图案和服装设计/几何属性；一致性不等于人工真值准确率。

PRD 3.1.3 fine-grained attribute extraction baseline v1.

Baseline scope
--------------
First-stage visually observable attributes only:
- color
- pattern
- sleeve_length
- neckline

Intentionally excluded for now:
- material / fabric composition
- manufacturing process / craftsmanship

Model
-----
Qwen2.5-VL-3B-Instruct via Hugging Face Transformers.

Usage
-----
python scripts/attributes/design/run_attribute_baseline_v1.py --image path/to/image.jpg

Optional ROI/crop input:
python scripts/attributes/design/run_attribute_baseline_v1.py --image path/to/crop.jpg
--region local_region

Outputs
-------
Prints JSON and optionally saves it with --output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

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


def build_prompt() -> str:
    """将当前标签与任务约束组织为模型文本提示。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return f"""
You are a visual attribute extractor for e-commerce fashion images.

Task:
Inspect ONLY directly visible visual evidence in the image.
Return exactly one JSON object and nothing else.

Current baseline attributes:
1. color
2. pattern
3. sleeve_length
4. neckline

IMPORTANT:
- Do NOT infer fabric/material composition.
- Do NOT infer manufacturing process/craftsmanship.
- If an attribute is not clearly visible, use "unknown".
- If the attribute does not apply to the garment, use "not_applicable".
- confidence must be a number from 0.0 to 1.0.
- evidence must be short and based only on visible cues.
- Use ONLY the allowed labels below.

Allowed labels:
color = {ALLOWED["color"]}
pattern = {ALLOWED["pattern"]}
sleeve_length = {ALLOWED["sleeve_length"]}
neckline = {ALLOWED["neckline"]}

Required JSON schema:
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
    """从模型响应中解析结构化 JSON 结果。

    Args:
        text: 文本。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
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
        raise ValueError(f"Model did not return a JSON object:\n{text}")

    return json.loads(text[start : end + 1])


def validate_result(result: dict) -> dict:
    """校验模型结果的字段和值是否符合本实验约定。

    Args:
        result: 结果。

    Returns:
        结果字典，主要字段为 predictions。
    """
    predictions = result.get("predictions", {})
    cleaned = {}

    for attr, allowed_values in ALLOWED.items():
        item = predictions.get(attr, {})

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


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--region",
        default="whole_garment",
        choices=["whole_garment", "roi", "local_region"],
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    image = Image.open(image_path).convert("RGB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"device: {device}")
    print(f"model: {args.model}")
    print(f"image: {image_path}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(args.model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt()},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_text = processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    result = validate_result(extract_json(generated_text))

    final = {
        "image": str(image_path),
        "region": args.region,
        **result,
    }

    output_text = json.dumps(
        final,
        ensure_ascii=False,
        indent=2,
    )

    print("\n" + output_text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            output_text,
            encoding="utf-8",
        )
        print(f"\nsaved: {output_path}")


if __name__ == "__main__":
    main()
