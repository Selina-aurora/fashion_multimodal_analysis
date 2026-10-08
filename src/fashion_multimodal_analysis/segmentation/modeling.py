"""3.1.1 实例分割：数据与模型版本分别记录，指标按固定匹配和计时口径解释。

Standard PRD8 Mask R-CNN factories. Anchor ablations keep their own builders.
"""

from __future__ import annotations

from typing import Any

from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from fashion_multimodal_analysis.common.schema import NUM_CLASSES


def build_training_model(
    min_size: int,
    max_size: int,
) -> Any:
    """构造训练模型，将预训练分类/掩码头替换为八类任务头。

    Args:
        min_size: 图像预处理的最短边目标尺寸。
        max_size: 图像预处理的最长边限制。

    Returns:
        Mask R-CNN ResNet50-FPN；保留 COCO 预训练的特征提取与提案参数，
        分类头和掩码头替换为 NUM_CLASSES（8 个前景类别 + 背景）的新头。
        调用方负责设备迁移、train() 和优化器配置。首次初始化可能下载预训练权重。
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

    model = maskrcnn_resnet50_fpn(
        weights=weights,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # COCO 原分类数与 PRD8 不同：框分类/回归头和掩码头必须一起替换。
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES,
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        NUM_CLASSES,
    )

    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size

    return model


def build_evaluation_model(min_size: int, max_size: int) -> Any:
    """构造评估模型，权重由调用方从指定 checkpoint 加载。

    Args:
        min_size: 图像预处理的最短边目标尺寸。
        max_size: 图像预处理的最长边限制。

    Returns:
        未加载预训练权重的八类 Mask R-CNN。调用方必须加载匹配的 checkpoint，
        再设置设备和 eval()；本函数本身不执行推理，也不保证任意消融模型兼容。
    """
    # 评估权重全部来自 checkpoint，避免下载或混入另一套 backbone 初始化。
    m = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    inf = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(inf, NUM_CLASSES)
    infm = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(infm, 256, NUM_CLASSES)
    m.transform.min_size = (min_size,)
    m.transform.max_size = max_size
    return m
