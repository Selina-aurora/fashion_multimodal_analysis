import json
from pathlib import Path

import cv2
import numpy as np


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

annotation_path = (
    project_root.parent
    / "fashion_data"
    / "raw"
    / "train"
    / "train"
    / "annos"
    / "000001.json"
)

output_dir = project_root / "outputs" / "annotation_visualization"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "000001_annotation.jpg"


image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"image not found: {image_path}")

with annotation_path.open("r", encoding="utf-8") as file:
    annotation = json.load(file)


for item_name, item_data in annotation.items():
    if not item_name.startswith("item"):
        continue

    category_name = item_data["category_name"]
    bounding_box = item_data["bounding_box"]
    segmentation = item_data["segmentation"]

    x_min, y_min, x_max, y_max = bounding_box

    cv2.rectangle(
        image,
        (x_min, y_min),
        (x_max, y_max),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        image,
        category_name,
        (x_min, max(y_min - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    for polygon in segmentation:
        points = np.array(polygon, dtype=np.int32).reshape(-1, 2)

        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=(255, 0, 0),
            thickness=2,
        )


cv2.imwrite(str(output_path), image)

print(f"visualization saved to: {output_path}")