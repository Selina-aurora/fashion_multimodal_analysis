import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import json
import os


# ==========================
# Configuration
# ==========================

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

IMAGE_PATH = "data/test/test.jpg"   
TEXT_PROMPT = "sleeve"

OUTPUT_PATH = "outputs/grounding_dino_test_result/sleeve_threshold_0.4.json"


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


# ==========================
# Load Image
# ==========================

image = Image.open(IMAGE_PATH).convert("RGB")

print("Image size:", image.size)


# ==========================
# Prepare Input
# ==========================

inputs = processor(
    images=image,
    text=TEXT_PROMPT,
    return_tensors="pt"
)


inputs = {
    k: v.to(device)
    for k, v in inputs.items()
}


# ==========================
# Inference
# ==========================

with torch.no_grad():

    outputs = model(**inputs)


# ==========================
# Post Process
# ==========================

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs["input_ids"],
    threshold=0.4,
    target_sizes=[image.size[::-1]]
)

result = results[0]

# ==========================
# Save Results
# ==========================

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


os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        detections,
        f,
        indent=4,
        ensure_ascii=False
    )


print("\nDetection Results:")

for d in detections:
    print(
        d["label"],
        d["score"],
        d["box"]
    )


print("\nSaved to:", OUTPUT_PATH)