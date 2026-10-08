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

    # 你的模型文件
    CKPT = "outputs/prd_instance_segmentation/maskrcnn_8class_v3_b1_dataexp/checkpoint_last.pth"

    # crop目录
    CROP_DIR = Path("reports/prd_3_1_v1/small_object_analysis")

    device = "cuda"

    # 这里直接复用torchvision maskrcnn
    model = torch.hub.load("pytorch/vision", "maskrcnn_resnet50_fpn", weights=None)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    transform = T.Compose([T.ToTensor()])

    for img_path in CROP_DIR.glob("*.jpg"):

        img = Image.open(img_path).convert("RGB")

        # 核心：resize
        img = img.resize((640, 640))

        x = transform(img).to(device)

        with torch.no_grad():
            pred = model([x])[0]

        print("\nIMAGE:", img_path.name)

        if len(pred["scores"]) == 0:
            print("no prediction")
            continue

        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()

        for s, label_id in zip(scores[:5], labels[:5]):
            print("label:", int(label_id), "score:", float(s))


if __name__ == "__main__":
    main()
