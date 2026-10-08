"""可视化与审核：图片用于发现共性错误，人工判断需要回填到审核记录。"""

from __future__ import annotations

from pathlib import Path

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    import json
    from pathlib import Path

    import cv2
    import numpy as np

    project_root = get_project_root()

    image_path = data_root() / "raw" / "train" / "train" / "image" / "000001.jpg"

    annotation_path = data_root() / "raw" / "train" / "train" / "annos" / "000001.json"

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


if __name__ == "__main__":
    main()
