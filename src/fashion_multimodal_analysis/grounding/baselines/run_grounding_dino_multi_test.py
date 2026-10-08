"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。"""

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

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    # ==========================
    # Configuration
    # ==========================

    MODEL_ID = "IDEA-Research/grounding-dino-tiny"

    IMAGE_DIR = "data/tests"

    TEXT_PROMPT = "sleeve cuff"

    OUTPUT_DIR = os.path.join("outputs", f"multi_test_{TEXT_PROMPT.replace(' ', '_')}")

    # ==========================
    # Device
    # ==========================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    # ==========================
    # Load Model
    # ==========================

    print("Loading Grounding DINO...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)

    model.to(device)
    model.eval()

    # ==========================
    # Prepare Output
    # ==========================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    # ==========================
    # Batch Inference
    # ==========================

    image_files = [
        f
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print("Number of images:", len(image_files))

    for image_name in image_files:

        print("\nProcessing:", image_name)

        image_path = os.path.join(IMAGE_DIR, image_name)

        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():

            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=0.3, target_sizes=[image.size[::-1]]
        )

        result = results[0]

        detections = []

        for score, label, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):

            detections.append(
                {
                    "label": label,
                    "score": float(score),
                    "box": [float(x) for x in box.tolist()],
                }
            )

        all_results[image_name] = detections

    # ==========================
    # Save
    # ==========================

    output_file = os.path.join(
        OUTPUT_DIR, f"{TEXT_PROMPT.replace(' ', '_')}_results.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print("\nFinished!")
    print("Saved:", output_file)


if __name__ == "__main__":
    main()
