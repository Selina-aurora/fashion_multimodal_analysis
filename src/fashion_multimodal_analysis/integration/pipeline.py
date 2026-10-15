"""3.1 接口联调：区分预测 ROI 传递和使用 GT 的受控诊断，保留对象 ID 与坐标对应关系。

GT-free product pipeline. Matching and human truth belong to evaluation only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from fashion_multimodal_analysis.evaluation.protocol import original_box, valid_box
from fashion_multimodal_analysis.integration.quality_guards import overlap_review_flags


@dataclass
class GarmentROI:
    """保存单个预测服饰的ID、类别、置信度、原图xyxy框和整图mask。"""

    instance_id: str
    category: str
    score: float
    bbox: list[float]
    mask: Any
    provenance: str = "segmentation_prediction"


def crop_bounds(box: Any, size: Any) -> Any:
    """裁剪 bounds。

    Args:
        box: xyxy 坐标的候选框。
        size: size。

    Returns:
        返回 bounds，由函数体中同名变量的计算/收集过程得到。

    Raises:
        ValueError: ROI is outside the source image
    """
    if not valid_box(box):
        raise ValueError("Invalid ROI bounding box")
    w, h = size
    bounds = [
        max(0, int(math.floor(box[0]))),
        max(0, int(math.floor(box[1]))),
        min(w, int(math.ceil(box[2]))),
        min(h, int(math.ceil(box[3]))),
    ]
    if not valid_box(bounds):
        raise ValueError("ROI is outside the source image")
    return bounds


def roi_images(image: Any, roi: Any) -> tuple[Any, ...]:
    """Return raw crop, binary crop mask and origin, all in source-image scale.

    Args:
        image: 本步骤处理的图像对象。
        roi: roi。

    Returns:
        按顺序返回 bounds 等结果。

    Raises:
        ValueError: Garment mask must have full original-image dimensions
    """
    import numpy as np
    from PIL import Image

    bounds = crop_bounds(roi.bbox, image.size)
    arr = np.asarray(roi.mask) > 0
    if arr.shape != (image.height, image.width):
        raise ValueError("Garment mask must have full original-image dimensions")
    x1, y1, x2, y2 = bounds
    return (
        image.crop(bounds),
        Image.fromarray(arr[y1:y2, x1:x2].astype("uint8") * 255),
        bounds,
    )


class FashionPipeline:
    """按服饰实例串联分割、区域定位及属性后端，隔离并记录单步失败。

    Attributes:
        backend: backend。
    """

    def __init__(self, backend: Any) -> None:
        """保存当前对象需要的配置、数据引用和状态。

        Args:
            backend: backend。
        """
        self.backend = backend

    def run(self, image: Any, queries: list[str], attributes: Any) -> Any:
        """Process every predicted garment. No GT boxes, labels or matching enter here.

        Args:
            image: 本步骤处理的图像对象。
            queries: 查询。
            attributes: 属性。

        Returns:
            结果字典包含 status, instances, timings, failures。
        """
        queries = list(dict.fromkeys(queries))
        attributes = list(dict.fromkeys(attributes))
        if "neckline" in attributes and "collar" not in queries:
            queries.append("collar")
        result = {"status": "completed", "instances": [], "timings": [], "failures": []}
        try:
            rois, elapsed = self.backend.segment(image)
            result["timings"].append(
                {
                    "module": "segmentation",
                    "ms": elapsed,
                    "unit": "image",
                    "instance_id": "",
                }
            )
        except Exception as exc:
            result.update(
                status="segmentation_failed", error=f"{type(exc).__name__}: {exc}"
            )
            return result
        downstream = self.run_downstream(image, rois, queries, attributes)
        result["instances"] = downstream["instances"]
        result["timings"].extend(downstream["timings"])
        result["failures"].extend(downstream["failures"])
        return result

    def run_downstream(
        self, image: Any, rois: Any, queries: list[str], attributes: Any
    ) -> Any:
        """执行 downstream。

        Args:
            image: 本步骤处理的图像对象。
            rois: rois。
            queries: 查询。
            attributes: 属性。

        Returns:
            结果字典包含 status, instances, timings, failures。
        """
        queries = list(dict.fromkeys(queries))
        attributes = list(dict.fromkeys(attributes))
        if "neckline" in attributes and "collar" not in queries:
            queries.append("collar")
        result = {"status": "completed", "instances": [], "timings": [], "failures": []}
        rois = list(rois)
        # 标记重叠供复核；叠穿常有框重叠，不能自动删除或合并。
        review_flags = overlap_review_flags(rois)
        for roi in rois:
            item = {
                "roi": roi,
                "regions": {},
                "attributes": {},
                "review_flags": review_flags[roi.instance_id],
            }
            for region in queries:
                try:
                    prediction, elapsed = self.backend.ground(image, roi, region)
                    item["regions"][region] = prediction
                    result["timings"].append(
                        {
                            "module": "grounding",
                            "ms": elapsed,
                            "unit": "query_instance",
                            "instance_id": roi.instance_id,
                            "target_region": region,
                        }
                    )
                except Exception as exc:
                    item["regions"][region] = {
                        "status": "grounding_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    result["failures"].append(
                        {
                            "instance_id": roi.instance_id,
                            "stage": "grounding",
                            "target": region,
                        }
                    )
            try:
                predictions, elapsed = self.backend.attributes(
                    image, roi, item["regions"], attributes, controlled=False
                )
                item["attributes"] = predictions
                sleeve = predictions.get("sleeve_length", {})
                if (
                    sleeve.get("status") == "predicted"
                    and sleeve.get("label") == "sleeveless"
                ):
                    # 无袖本身也是模型预测，不能据此生成“无袖口”的真值。
                    # 将 cuff missed 与真实漏检区分，交给人工确认适用性。
                    item["review_flags"].append(
                        {
                            "reason": "cuff_applicability_requires_review",
                            "predicted_sleeve_length": "sleeveless",
                            "cuff_status": item["regions"]
                            .get("cuff", {})
                            .get("status"),
                            "automatic_action": "none",
                            "note": "Do not count cuff missed as a confirmed miss before sleeve applicability review.",
                        }
                    )
                result["timings"].append(
                    {
                        "module": "attributes",
                        "ms": elapsed,
                        "unit": "instance",
                        "instance_id": roi.instance_id,
                    }
                )
            except Exception as exc:
                item["attributes"] = {
                    name: {
                        "status": "attribute_failed",
                        "label": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    for name in attributes
                }
                result["failures"].append(
                    {
                        "instance_id": roi.instance_id,
                        "stage": "attributes",
                        "target": "all",
                    }
                )
            # 默认不启用候选规则；只有显式配置开启时改变交付状态。
            # 属性已经计算完毕，再检查跨模块矛盾，防止伪造 GT 或覆盖类别。
            if getattr(self.backend, "config", {}).get(
                "semantic_consistency_review", False
            ):
                from fashion_multimodal_analysis.integration.semantic_consistency import (
                    apply_consistency_review,
                )

                item["review_flags"].extend(
                    apply_consistency_review(item["regions"], item["attributes"])
                )
            result["instances"].append(item)
        return result
