"""可视化与审核：图片用于发现共性错误，人工判断需要回填到审核记录。"""

from __future__ import annotations

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    import json
    import os

    from PIL import Image, ImageDraw, ImageFont

    # ==========================
    # Configuration
    # ==========================

    IMAGE_PATH = "data/test/test.jpg"

    RESULT_PATH = (
        "outputs/grounding_dino_test_result/grounding_dino_result_left_sleeve.json"
    )

    OUTPUT_PATH = "outputs/grounding_dino_test_visualization_result/left_sleeve.jpg"

    # ==========================
    # Load image
    # ==========================

    image = Image.open(IMAGE_PATH).convert("RGB")

    draw = ImageDraw.Draw(image)

    # ==========================
    # Load detection result
    # ==========================

    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    # ==========================
    # Draw boxes
    # ==========================

    for item in results:

        label = item["label"]
        score = item["score"]
        box = item["box"]

        x1, y1, x2, y2 = box

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

        text = f"{label}: {score:.2f}"

        draw.text((x1, max(0, y1 - 20)), text)

    # ==========================
    # Save
    # ==========================

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    image.save(OUTPUT_PATH)

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
