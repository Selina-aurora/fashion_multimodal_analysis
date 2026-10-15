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
        返回 model，由函数体中同名变量的计算/收集过程得到。
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

    model = maskrcnn_resnet50_fpn(
        weights=weights,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

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
        返回 m，由函数体中同名变量的计算/收集过程得到。
    """
    m = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    inf = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(inf, NUM_CLASSES)
    infm = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(infm, 256, NUM_CLASSES)
    m.transform.min_size = (min_size,)
    m.transform.max_size = max_size
    return m
