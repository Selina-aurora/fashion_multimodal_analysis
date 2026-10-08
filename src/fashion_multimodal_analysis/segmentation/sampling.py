"""3.1.1 实例分割：数据与模型版本分别记录，指标按固定匹配和计时口径解释。

Image-group sampler weights; pure Python and independent of model loading.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Iterator

from fashion_multimodal_analysis.common.schema import CLASS_TO_ID

if TYPE_CHECKING:
    from fashion_multimodal_analysis.datasets.prd8 import PRD8MaskDataset


def instance_class_counts(
    dataset: PRD8MaskDataset,
) -> Counter:
    """按实例统计八类训练分布。

    Args:
        dataset: 已读取训练清单的数据集；rows 中每行对应一个实例。

    Returns:
        类别名称到实例数量的 Counter。此计数用于权重计算，不等于图片数量。
    """
    return Counter(str(r["garment_category"]).strip().lower() for r in dataset.rows)


def build_sampler_weights(
    dataset: PRD8MaskDataset,
    alpha: float,
) -> tuple[list[float], list[dict]]:
    """给每张训练原图分配一个类别均衡权重。

    Args:
        dataset: 按原图分组的数据集，同一图片中的全部实例一起训练。
        alpha: 权重指数。0 不做均衡；0.5 对实例数量比取平方根。

    Returns:
        逐图权重列表和审计行。类别权重为 (最大类别实例数 / 本类实例数) ** alpha；
        多类别图片取其中最大权重，避免相乘后过度放大某张图片的采样概率。

    Raises:
        ValueError: alpha 为负、训练清单为空，或八类中有类别没有实例。

    Notes:
        此处计算权重；是否有放回采样由训练入口决定。V8 使用无放回全量遍历，
        不调用本函数来代替全量训练，不能把两者的 epoch 含义混在一起。
    """
    if alpha < 0:
        raise ValueError("--sampler-alpha must be >= 0")

    counts = instance_class_counts(dataset)

    if not counts:
        raise ValueError("Training manifest has no class rows.")

    max_count = max(counts.get(cls, 0) for cls in CLASS_TO_ID)

    class_weight = {}

    for cls in CLASS_TO_ID:
        count = counts.get(cls, 0)

        if count <= 0:
            raise ValueError(f"Training class missing: {cls}")

        class_weight[cls] = (max_count / count) ** alpha

    image_weights = []
    report_rows = []

    for index, sample in enumerate(dataset.samples):
        classes = sorted(
            {str(r["garment_category"]).strip().lower() for r in sample["instances"]}
        )

        # Use max weight so an image containing any rare class gets the
        # appropriate boost without multiplying weights for multi-instance imgs.
        weight = max(class_weight[c] for c in classes)

        image_weights.append(float(weight))

        report_rows.append(
            {
                "dataset_index": index,
                "source_dataset": sample["source_dataset"],
                "source_image": sample["source_image"],
                "instance_count": len(sample["instances"]),
                "classes": ";".join(classes),
                "sampler_weight": f"{weight:.6f}",
            }
        )

    return image_weights, report_rows
