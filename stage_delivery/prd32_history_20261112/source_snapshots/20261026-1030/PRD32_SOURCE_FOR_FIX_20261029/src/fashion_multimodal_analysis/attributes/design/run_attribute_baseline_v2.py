"""3.1.3 可视属性提取：关注颜色、图案和服装设计/几何属性；一致性不等于人工真值准确率。

PRD 3.1.3 attribute extraction baseline v2.

Fixes compared with v1:
1. Output folders are anchored to the project root.
2. JSON is always saved.
3. The input image is copied into the output folder.
4. An annotated visualization image is always generated.
5. Absolute output paths are printed at the end.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
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
    """resolve 图像 路径。

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        当前工作副本可使用的路径。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
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
        + f"typed path: {raw}\n"
        + f"cwd candidate: {cwd_candidate}\n"
        + f"project candidate: {project_candidate}"
    )


def build_prompt() -> str:
    """将当前标签与任务约束组织为模型文本提示。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return f"""
You are a visual attribute extractor for e-commerce fashion images.

Inspect ONLY directly visible visual evidence.
Return exactly one JSON object and nothing else.

Extract:
1. color
2. pattern
3. sleeve_length
4. neckline

IMPORTANT:
- Do NOT infer fabric or material composition.
- Do NOT infer manufacturing process or craftsmanship.
- If unclear, use "unknown".
- If not applicable, use "not_applicable".
- confidence must be between 0.0 and 1.0.
- evidence must be short and based only on visible cues.
- Use ONLY the allowed labels.

Allowed labels:
color = {ALLOWED["color"]}
pattern = {ALLOWED["pattern"]}
sleeve_length = {ALLOWED["sleeve_length"]}
neckline = {ALLOWED["neckline"]}

Required JSON:
{{
  "predictions": {{
    "color": {{"label": "...", "confidence": 0.0, "evidence": "..."}},
    "pattern": {{"label": "...", "confidence": 0.0, "evidence": "..."}},
    "sleeve_length": {{"label": "...", "confidence": 0.0, "evidence": "..."}},
    "neckline": {{"label": "...", "confidence": 0.0, "evidence": "..."}}
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
        raise ValueError(f"Model did not return JSON:\n{text}")

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


def make_visual(image: Image.Image, result: dict, save_path: Path) -> None:
    """生成 可视化。

    Args:
        image: 本步骤处理的图像对象。
        result: 记录字段，使用 predictions。
        save_path: 对应文件的相对路径或当前解析后的路径。
    """
    panel_h = 150
    canvas = Image.new("RGB", (image.width, image.height + panel_h), "white")
    canvas.paste(image, (0, panel_h))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((12, 10), "PRD 3.1.3 attribute baseline v1", fill="black", font=font)

    y = 34
    for attr in ["color", "pattern", "sleeve_length", "neckline"]:
        item = result["predictions"][attr]
        line = f"{attr}: {item['label']}  conf={item['confidence']:.2f}"
        draw.text((12, y), line, fill="black", font=font)
        y += 24

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, quality=92)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
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

    image_path = resolve_image_path(args.image)
    image = Image.open(image_path).convert("RGB")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    if args.output:
        raw_output = Path(args.output).expanduser()
        if raw_output.is_absolute():
            json_path = raw_output
        else:
            json_path = (PROJECT_ROOT / raw_output).resolve()
    else:
        json_path = REPORT_DIR / f"{stem}_attributes.json"

    json_path.parent.mkdir(parents=True, exist_ok=True)

    input_copy_path = VIS_DIR / f"{stem}_input.jpg"
    visual_path = VIS_DIR / f"{stem}_attributes.jpg"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"project root : {PROJECT_ROOT}")
    print(f"device       : {device}")
    print(f"model        : {args.model}")
    print(f"input image  : {image_path}")

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

    output_text = json.dumps(final, ensure_ascii=False, indent=2)

    json_path.write_text(output_text, encoding="utf-8")
    shutil.copy2(image_path, input_copy_path)
    make_visual(image=image, result=result, save_path=visual_path)

    print("\n" + output_text)
    print("\n=== SAVED FILES ===")
    print(f"JSON        : {json_path.resolve()}")
    print(f"INPUT COPY  : {input_copy_path.resolve()}")
    print(f"VISUAL      : {visual_path.resolve()}")
    print("===================")


if __name__ == "__main__":
    main()
