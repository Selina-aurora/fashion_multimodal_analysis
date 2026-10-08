"""评估工具：逐例表是指标回算依据，计时只含预处理、前向和后处理。

Time preprocessing, model forward and CPU postprocessing, excluding disk/GT/report I/O.
"""

from __future__ import annotations

import time
from typing import Any


def postprocess_filtered(
    output: dict[str, Any], score_threshold: float, mask_threshold: float
) -> tuple[Any, ...]:
    """先在设备上筛选分数，再传回保留候选的预测。

    Args:
        output: torchvision 检测输出，含 boxes、labels、scores、masks。
        score_threshold: 有限置信度达到该值的候选才保留。
        mask_threshold: 在 CPU NumPy 上将概率掩码二值化的阈值。

    Returns:
        boxes (M,4)、labels (M,)、scores (M,)、masks (M,H,W)。
        四个数组使用同一布尔选择，顺序一致；masks 为 uint8。

    Notes:
        完整掩码传输是旧版后处理的开销。先筛选可减少 GPU→CPU 数据量，
        仍沿用原 NumPy 阈值操作，以保留边界值、数组 dtype 和预测结果。
    """
    import numpy as np
    import torch

    score_tensor = output["scores"].detach()
    keep = torch.isfinite(score_tensor) & (score_tensor >= score_threshold)
    boxes = output["boxes"].detach()[keep].cpu().numpy()
    labels = output["labels"].detach()[keep].cpu().numpy()
    scores = score_tensor[keep].cpu().numpy()
    # Keep the original NumPy mask threshold operation; transfer retained masks only.
    masks = (output["masks"].detach()[keep, 0].cpu().numpy() >= mask_threshold).astype(
        np.uint8
    )
    return boxes, labels, scores, masks


def predict_timed(
    raw_image: Any,
    model: Any,
    device: Any,
    score_threshold: float,
    mask_threshold: float,
) -> tuple[Any, ...]:
    """计时一次预处理、模型前向和预测后处理。

    Args:
        raw_image: 计时前已经读入的 PIL 原图。
        model: 已加载权重并处于 eval 模式的 Mask R-CNN。
        device: 模型和输入使用的计算设备。
        score_threshold: 候选保留阈值。
        mask_threshold: 概率掩码二值化阈值。

    Returns:
        框、类别 ID、置信度、整图二值掩码，以及本次推理的毫秒数。

    Notes:
        CUDA 运算异步执行，所以计时前后都同步。计时包含 RGB 转换、张量构造、
        模型前向及后处理；不包含图片磁盘读取、GT 解码、模型加载和报告写入。
        10 次预热和 5 遍计时由评估入口控制，本函数负责一次有效计时。
    """
    import torch
    from torchvision.transforms import functional as F

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    image = raw_image.convert("RGB")
    tensor = F.to_tensor(image).to(device)
    with torch.inference_mode():
        output = model([tensor])[0]
    result = postprocess_filtered(output, score_threshold, mask_threshold)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return (*result, elapsed_ms)
