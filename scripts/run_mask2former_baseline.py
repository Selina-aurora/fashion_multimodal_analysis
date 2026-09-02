from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor
from transformers import Mask2FormerForUniversalSegmentation


model_name = "facebook/mask2former-swin-tiny-coco-instance"

project_root = Path(__file__).resolve().parents[1]

image_path = (
    project_root.parent
    / "fashion_data"
    / "raw"
    / "train"
    / "train"
    / "image"
    / "000001.jpg"
)

output_dir = project_root / "outputs" / "mask2former_baseline"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "000001_prediction.jpg"


image = Image.open(image_path).convert("RGB")

print(f"image path: {image_path}")
print(f"image size: {image.size}")
print(f"image mode: {image.mode}")


print("loading image processor...")

processor = AutoImageProcessor.from_pretrained(
    model_name,
    use_fast=False,
)

print("loading mask2former model...")

model = Mask2FormerForUniversalSegmentation.from_pretrained(
    model_name,
)

model.eval()

print("model loaded successfully")


inputs = processor(
    images=image,
    return_tensors="pt",
)

print(f"pixel values shape: {inputs['pixel_values'].shape}")


with torch.no_grad():
    outputs = model(**inputs)

print("inference completed")

print(
    f"class logits shape: "
    f"{outputs.class_queries_logits.shape}"
)

print(
    f"mask logits shape: "
    f"{outputs.masks_queries_logits.shape}"
)


result = processor.post_process_instance_segmentation(
    outputs,
    target_sizes=[
        (
            image.height,
            image.width,
        )
    ],
)[0]


print(
    f"number of detected instances: "
    f"{len(result['segments_info'])}"
)


for segment in result["segments_info"]:
    label_id = segment["label_id"]
    label_name = model.config.id2label[label_id]
    score = segment["score"]

    print(
        f"id={segment['id']}, "
        f"label={label_name}, "
        f"score={score:.4f}"
    )


segmentation = result["segmentation"].cpu().numpy()

visualization = np.array(image).copy()


for segment in result["segments_info"]:
    segment_id = segment["id"]
    label_id = segment["label_id"]
    score = segment["score"]

    label_name = model.config.id2label[label_id]

    mask = segmentation == segment_id

    visualization[mask] = (
        visualization[mask] * 0.5
        + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)

    y_indices, x_indices = np.where(mask)

    if len(x_indices) == 0 or len(y_indices) == 0:
        continue

    x_min = int(x_indices.min())
    y_min = int(y_indices.min())
    x_max = int(x_indices.max())
    y_max = int(y_indices.max())

    cv2.rectangle(
        visualization,
        (x_min, y_min),
        (x_max, y_max),
        (255, 0, 0),
        2,
    )

    label_text = f"{label_name}: {score:.2f}"

    cv2.putText(
        visualization,
        label_text,
        (
            x_min,
            max(y_min - 10, 20),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )


visualization_bgr = cv2.cvtColor(
    visualization,
    cv2.COLOR_RGB2BGR,
)

success = cv2.imwrite(
    str(output_path),
    visualization_bgr,
)

if not success:
    raise RuntimeError(
        f"failed to save visualization: {output_path}"
    )

print(f"prediction saved to: {output_path}")