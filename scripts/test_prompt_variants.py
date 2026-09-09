from pathlib import Path
import json
import os

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


# ==========================
# Configuration
# ==========================

MODEL_ID = "IDEA-Research/grounding-dino-tiny"


IMAGE_DIR = Path(
    "data/tests"
)


OUTPUT_DIR = Path(
    "outputs/prompt_optimization"
)


CATEGORIES = {

    "sleeve": [
        "sleeve",
        "shirt sleeve",
        "long sleeve part of a shirt",
        "upper arm clothing sleeve",
    ],

    "collar": [
        "collar",
        "shirt collar area",
        "collar part of clothing",
        "upper clothing collar",
    ],

    "button": [
        "button",
        "small button on clothing",
        "shirt button detail",
        "clothing fastener button",
    ],

    "zipper": [
        "zipper",
        "zipper on a jacket or shirt",
        "clothing zipper detail",
        "front zipper closure",
    ],
}


# ==========================
# Device
# ==========================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using device:",
    device
)


# ==========================
# Load Model
# ==========================

print(
    "Loading Grounding DINO..."
)


processor = AutoProcessor.from_pretrained(
    MODEL_ID
)


model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_ID
)


model.to(device)

model.eval()



# ==========================
# Inference Function
# ==========================

def detect(
    image,
    prompt
):

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    )


    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }


    with torch.no_grad():

        outputs = model(
            **inputs
        )


    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        target_sizes=[
            image.size[::-1]
        ],
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


    return detections



# ==========================
# Run Experiment
# ==========================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


images = list(
    IMAGE_DIR.glob(
        "*.jpg"
    )
)


all_results = {}



for category, prompts in CATEGORIES.items():

    print(
        "\nCategory:",
        category
    )


    category_results = {}


    for prompt in prompts:

        print(
            "Testing:",
            prompt
        )


        prompt_result = {}


        for image_path in images:

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )


            detections = detect(
                image,
                prompt
            )


            prompt_result[
                image_path.name
            ] = detections


        category_results[
            prompt
        ] = prompt_result



    all_results[
        category
    ] = category_results



output_file = (
    OUTPUT_DIR /
    "prompt_variant_results.json"
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        indent=4,
        ensure_ascii=False
    )


print(
    "\nSaved:",
    output_file
)