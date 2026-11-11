"""3.1.1 实例分割：数据与模型版本分别记录，指标按固定匹配和计时口径解释。"""

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
    import sys
    from pathlib import Path

    import torch
    import torchvision.transforms as T
    from PIL import Image

    from fashion_multimodal_analysis.segmentation.evaluation.eval_prd_8class_maskrcnn_v3_b1_dataexp import (
        build_model,
    )

    # 加入 scripts 路径

    device = "cuda"

    # =========================
    # 1. 加载训练好的8类Mask R-CNN
    # =========================

    model = build_model(640, 1024)

    checkpoint = torch.load(
        "outputs/prd_instance_segmentation/maskrcnn_8class_v3_b1_dataexp/checkpoint_last.pth",
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print("model loaded")

    # =========================
    # 2. 测试图片
    # =========================

    imgs = [
        "reports/prd_3_1_v1/small_object_analysis/resize640/prd8_0175_shoe.jpg",
        "reports/prd_3_1_v1/small_object_analysis/resize640/prd8_0181_bag.jpg",
        "reports/prd_3_1_v1/small_object_analysis/resize640/prd8_0192_accessory.jpg",
    ]

    transform = T.ToTensor()

    # =========================
    # 3. inference
    # =========================

    for img_path in imgs:

        img = Image.open(img_path).convert("RGB")

        x = transform(img).to(device)

        with torch.no_grad():
            pred = model([x])[0]

        print("\n====================")
        print("IMAGE:", img_path)

        if len(pred["scores"]) == 0:
            print("no prediction")
            continue

        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()

        for s, label_id in zip(scores[:5], labels[:5]):
            print("class:", int(label_id), "score:", round(float(s), 4))


if __name__ == "__main__":
    main()
