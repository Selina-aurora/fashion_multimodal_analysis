"""数据处理：保存原图来源和实例标识，避免同一图片的不同实例跨训练与评估划分。

PRD8 instance manifests, image grouping and Mask R-CNN targets.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_project_path,
)
from fashion_multimodal_analysis.common.schema import CLASS_TO_ID
from fashion_multimodal_analysis.image_processing.masks import reconstruct_full_mask


def read_manifest(path: Path) -> list[dict[str, str]]:
    """读取实例清单，不在读取阶段改变样本划分。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest:\n  {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def group_manifest(
    rows: list[dict[str, str]],
) -> list[dict]:
    """将同一原图的全部实例合并为一个训练/评估样本。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        返回 samples，由函数体中同名变量的计算/收集过程得到。
    """
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        key = (
            str(row["source_dataset"]).strip(),
            str(row["source_image"]).strip(),
        )
        grouped[key].append(row)

    samples = []

    for (dataset, source_image), instances in grouped.items():
        samples.append(
            {
                "source_dataset": dataset,
                "source_image": source_image,
                "instances": instances,
            }
        )

    samples.sort(
        key=lambda x: (
            x["source_dataset"],
            x["source_image"],
        )
    )

    return samples


class PRD8MaskDataset(Dataset):
    """保存 PRD8MaskDataset 的数据/运行职责。

    Attributes:
        manifest_path: 对应文件的相对路径或当前解析后的路径。
        training: 是否使用训练时的数据增强。
        enable_hflip: 是否允许随机水平翻转，并同步翻转框和掩码。
        rows: 待处理的逐行记录。
        samples: samples。
    """

    def __init__(
        self,
        manifest_path: Path,
        training: bool,
        enable_hflip: bool,
    ) -> None:
        """保存当前对象需要的配置、数据引用和状态。

        Args:
            manifest_path: 对应文件的相对路径或当前解析后的路径。
            training: 是否使用训练时的数据增强。
            enable_hflip: 是否允许随机水平翻转，并同步翻转框和掩码。
        """
        self.manifest_path = manifest_path
        self.training = training
        self.enable_hflip = enable_hflip

        rows = read_manifest(manifest_path)
        self.rows = rows
        self.samples = group_manifest(rows)

    def __len__(self) -> int:
        """返回当前数据集或容器中的样本数量。

        Returns:
            本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        """取得指定样本，并生成本数据集约定的输入和目标。

        Args:
            index: 待访问样本的整数下标。

        Returns:
            按顺序返回 image_tensor, target, meta 等结果。

        Raises:
            FileNotFoundError: 需要的文件不存在。
            ValueError: 输入或实验状态不符合检查条件。
        """
        sample = self.samples[index]

        image_path = resolve_project_path(sample["source_image"])

        if not image_path.is_file():
            raise FileNotFoundError(f"Source image not found:\n  {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_w, image_h = image.size

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for row in sample["instances"]:
            category = str(row["garment_category"]).strip().lower()

            if category not in CLASS_TO_ID:
                raise ValueError(f"Unknown PRD class: {category!r}")

            x1 = int(float(row["bbox_x1"]))
            y1 = int(float(row["bbox_y1"]))
            x2 = int(float(row["bbox_x2"]))
            y2 = int(float(row["bbox_y2"]))

            x1 = max(0, min(image_w - 1, x1))
            y1 = max(0, min(image_h - 1, y1))
            x2 = max(x1 + 1, min(image_w, x2))
            y2 = max(y1 + 1, min(image_h, y2))

            mask_path = resolve_project_path(row["mask_path"])

            if not mask_path.is_file():
                raise FileNotFoundError(f"Mask not found:\n  {mask_path}")

            full_mask = reconstruct_full_mask(
                mask_path=mask_path,
                image_size=(image_w, image_h),
                bbox=(x1, y1, x2, y2),
            )

            mask_arr = (np.asarray(full_mask) > 0).astype(np.uint8)

            if mask_arr.sum() == 0:
                raise ValueError(
                    "Empty reconstructed mask:\n"
                    + f"  image={image_path}\n"
                    + f"  mask={mask_path}"
                )

            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(CLASS_TO_ID[category])
            masks.append(mask_arr)
            areas.append(float(mask_arr.sum()))
            iscrowd.append(0)

        image_tensor = F.to_tensor(image)

        boxes_t = torch.as_tensor(
            boxes,
            dtype=torch.float32,
        )
        labels_t = torch.as_tensor(
            labels,
            dtype=torch.int64,
        )
        masks_t = torch.as_tensor(
            np.stack(masks, axis=0),
            dtype=torch.uint8,
        )
        area_t = torch.as_tensor(
            areas,
            dtype=torch.float32,
        )
        iscrowd_t = torch.as_tensor(
            iscrowd,
            dtype=torch.int64,
        )

        if self.training and self.enable_hflip and random.random() < 0.5:
            image_tensor = torch.flip(
                image_tensor,
                dims=[2],
            )

            masks_t = torch.flip(
                masks_t,
                dims=[2],
            )

            old_x1 = boxes_t[:, 0].clone()
            old_x2 = boxes_t[:, 2].clone()

            boxes_t[:, 0] = image_w - old_x2
            boxes_t[:, 2] = image_w - old_x1

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor(
                [index],
                dtype=torch.int64,
            ),
            "area": area_t,
            "iscrowd": iscrowd_t,
        }

        meta = {
            "source_dataset": sample["source_dataset"],
            "source_image": sample["source_image"],
            "instance_count": len(boxes),
            "classes": sorted(
                {
                    str(r["garment_category"]).strip().lower()
                    for r in sample["instances"]
                }
            ),
        }

        return image_tensor, target, meta


def collate_fn(batch: Any) -> tuple[Any, ...]:
    """collate fn。

    Args:
        batch: 一批图像与实例目标；每张图的实例数量可以不同。

    Returns:
        按顺序返回 images, targets, metas 等结果。
    """
    images = []
    targets = []
    metas = []

    for image, target, meta in batch:
        images.append(image)
        targets.append(target)
        metas.append(meta)

    return images, targets, metas
