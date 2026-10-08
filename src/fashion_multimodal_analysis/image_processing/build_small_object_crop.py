"""项目公共接口：参数约定与返回结构供其他模块复用。"""

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
    """解析运行参数并执行本模块的实验入口。"""
    import csv
    from pathlib import Path

    from PIL import Image

    root = get_project_root()

    error = root / "reports/prd_3_1_v1/core_regression_v3_b1_dataexp_v1/error_cases.csv"
    val = root / "configs/prd_8class_val_v1.csv"

    out = root / "reports/prd_3_1_v1/small_object_analysis"
    out.mkdir(parents=True, exist_ok=True)

    targets = {"prd8_0175": "shoe", "prd8_0192": "accessory", "prd8_0181": "bag"}

    bbox = {}

    with open(val, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            bbox[r["record_id"]] = r

    with open(error, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):

            rid = r["record_id"]

            if rid not in targets:
                continue

            b = bbox[rid]

            img = root / r["source_image"]

            im = Image.open(img).convert("RGB")

            w, h = im.size

            x1 = float(b["bbox_x1"])
            y1 = float(b["bbox_y1"])
            x2 = float(b["bbox_x2"])
            y2 = float(b["bbox_y2"])

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            bw = (x2 - x1) * 3
            bh = (y2 - y1) * 3

            box = (
                max(0, int(cx - bw / 2)),
                max(0, int(cy - bh / 2)),
                min(w, int(cx + bw / 2)),
                min(h, int(cy + bh / 2)),
            )

            crop = im.crop(box)

            save = out / f"{rid}_{targets[rid]}.jpg"

            crop.save(save)

            print("saved", save)

    print("done")


if __name__ == "__main__":
    main()
