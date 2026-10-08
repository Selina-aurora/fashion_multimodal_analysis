"""项目公共接口：参数约定与返回结构供其他模块复用。

Shared binary mask loading and crop-to-image reconstruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def open_binary_mask(path: Path) -> Image.Image:
    """open 二值 掩码。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    mask = Image.open(path).convert("L")
    arr = np.asarray(mask)
    arr = (arr > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def reconstruct_full_mask(
    mask_path: Path,
    image_size: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    """将裁剪掩码映射回整图，处理缩放及越界裁剪。

    Args:
        mask_path: 二值目标掩码的文件引用。
        image_size: 图像的宽、高，用于处理坐标和掩码尺寸。
        bbox: xyxy 坐标的目标框。

    Returns:
        返回 canvas，由函数体中同名变量的计算/收集过程得到。
    """
    image_w, image_h = image_size
    x1, y1, x2, y2 = bbox

    mask = open_binary_mask(mask_path)

    if mask.size == (image_w, image_h):
        return mask

    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)

    if mask.size != (bbox_w, bbox_h):
        mask = mask.resize(
            (bbox_w, bbox_h),
            resample=Image.Resampling.NEAREST,
        )

    canvas = Image.new("L", (image_w, image_h), 0)

    cx1 = max(0, min(image_w, x1))
    cy1 = max(0, min(image_h, y1))
    cx2 = max(0, min(image_w, x2))
    cy2 = max(0, min(image_h, y2))

    if cx2 <= cx1 or cy2 <= cy1:
        return canvas

    mx1 = cx1 - x1
    my1 = cy1 - y1
    mx2 = mx1 + (cx2 - cx1)
    my2 = my1 + (cy2 - cy1)

    canvas.paste(
        mask.crop((mx1, my1, mx2, my2)),
        (cx1, cy1),
    )

    return canvas
