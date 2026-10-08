"""二值掩码读取与裁剪掩码回填：统一使用 PIL 的 (宽, 高) 和整图像素坐标。

Shared binary mask loading and crop-to-image reconstruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def open_binary_mask(path: Path) -> Image.Image:
    """读取掩码，将所有非零灰度像素视为前景。

    Args:
        path: 掩码图片路径；先转为 L 灰度模式，再按像素值 > 0 二值化。

    Returns:
        PIL L 模式图片，背景为 0、前景为 255；不是 0 到 1 的模型概率图。
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
        mask_path: 整图掩码或目标框内裁剪掩码的图片路径。
        image_size: 原图 (width, height)，单位为像素，顺序与 PIL.Image.size 一致。
        bbox: 整图像素坐标 (x1, y1, x2, y2)，右下边界不包含在裁剪内。

    Returns:
        与原图同尺寸的 L 模式二值掩码。整图掩码直接返回；裁剪掩码先缩放到框尺寸，
        再按可见交集回填。框与原图无交集时返回全零画布。
    """
    image_w, image_h = image_size
    x1, y1, x2, y2 = bbox

    mask = open_binary_mask(mask_path)

    # 尺寸与原图一致时按整图掩码处理，不再套用 bbox 缩放。
    if mask.size == (image_w, image_h):
        return mask

    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)

    # 最近邻插值保持二值边界，不引入灰度插值产生的伪前景。
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

    # 从整图交集转换到裁剪掩码坐标，避免越界框截断后产生位置偏移。
    mx1 = cx1 - x1
    my1 = cy1 - y1
    mx2 = mx1 + (cx2 - cx1)
    my2 = my1 + (cy2 - cy1)

    canvas.paste(
        mask.crop((mx1, my1, mx2, my2)),
        (cx1, cy1),
    )

    return canvas
