import json
import os

import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)


# ==========================
# Configuration
# ==========================

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

IMAGE_DIR = "data/tests"

PROMPTS = [
    "shirt area",
    "shirt sleeve",
    "shirt collar area",
    "small button on clothing",
    "cuff at the end of a shirt sleeve",
    "zipper on a jacket or shirt",
]

BASE_OUTPUT_DIR = "outputs/prompt_optimization"

THRESHOLD = 0.3


# ==========================
# Device
# ==========================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# ==========================
# Load Model
# ==========================

print("Loading Grounding DINO...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_ID
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ==========================
# Load Image Files
# ==========================

image_files = [
    file_name
    for file_name in os.listdir(IMAGE_DIR)
    if file_name.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

image_files.sort()

print("Number of images:", len(image_files))


# ==========================
# Prompt Optimization Testing
# ==========================

for text_prompt in PROMPTS:

    print("\n==========================")
    print("Testing prompt:", text_prompt)
    print("==========================")

    prompt_folder_name = text_prompt.replace(
        " ",
        "_"
    )

    output_dir = os.path.join(
        BASE_OUTPUT_DIR,
        prompt_folder_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    all_results = {}

    for image_name in image_files:

        print("Processing:", image_name)

        image_path = os.path.join(
            IMAGE_DIR,
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=THRESHOLD,
            target_sizes=[image.size[::-1]]
        )

        result = results[0]

        detections = []

        for score, label, box in zip(
            result["scores"],
            result["labels"],
            result["boxes"]
        ):
            detections.append(
                {
                    "label": label,
                    "score": float(score),
                    "box": [
                        float(x)
                        for x in box.tolist()
                    ]
                }
            )

        all_results[image_name] = detections

    output_file = os.path.join(
        output_dir,
        f"{prompt_folder_name}_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            all_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Saved to:", output_file)


print("\nAll prompt optimization experiments finished.")