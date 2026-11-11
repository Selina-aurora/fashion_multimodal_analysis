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
    from pathlib import Path

    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    ROOT = get_project_root()

    IMG_DIR = ROOT / "reports/prd_3_1_v1/small_object_analysis"

    CKPT = (
        ROOT
        / "outputs/prd_instance_segmentation/maskrcnn_8class_v3_b1_dataexp/checkpoint_last.pth"
    )

    device = "cuda"

    classes = [
        "background",
        "top",
        "pants",
        "skirt",
        "outerwear",
        "dress",
        "shoe",
        "bag",
        "accessory",
    ]

    model = maskrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, len(classes)
    )

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])

    model.to(device)
    model.eval()

    transform = T.Compose([T.ToTensor()])

    for img_path in IMG_DIR.glob("*.jpg"):

        img = Image.open(img_path).convert("RGB")

        x = transform(img).to(device)

        with torch.no_grad():
            pred = model([x])[0]

        print("\nIMAGE:", img_path.name)

        if len(pred["scores"]) == 0:
            print("no prediction")
            continue

        for i, s in enumerate(pred["scores"][:5]):

            score = float(s)

            label = int(pred["labels"][i])

            print("class:", classes[label], "score:", round(score, 3))


if __name__ == "__main__":
    main()
