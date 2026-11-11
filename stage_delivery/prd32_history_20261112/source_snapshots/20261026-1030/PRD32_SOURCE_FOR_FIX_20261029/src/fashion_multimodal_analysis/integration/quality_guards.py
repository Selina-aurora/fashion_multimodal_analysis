"""可解释的几何检查；只筛查模型输出，不读取人工标签或替换类别。

面积阈值是开发阶段的可调规则，不代表经过校准的语义正确率。
被拦截的框保留诊断记录，不能计为部位不存在或正确检出。
"""
from __future__ import annotations

import numpy as np

LOCAL_REGIONS = frozenset({"collar", "cuff", "hem", "pocket", "shoulder", "waist"})


def box_area(box):
    """xyxy 框面积；调用方应先检查坐标合法性。"""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def proposal_area_ratio(box, crop_size):
    """候选框占服装裁剪的比例，用于识别整件衣服级的局部框。"""
    width, height = crop_size
    return box_area(box) / max(1, width * height)


def constrain_local_mask(mask, garment_mask, box):
    """把局部 SAM 输出限制在候选框及预测服装内，阻止向整件衣服扩张。

    这是局部定位保护，不是 GT 修正；错误候选框仍可能产生错误结果。
    pattern/decoration 的整体区域由调用方决定是否使用此限制。
    """
    raw = np.asarray(mask, dtype=bool)
    garment = np.asarray(garment_mask) > 0
    if raw.shape != garment.shape or raw.ndim != 2:
        raise ValueError("SAM mask shape differs from garment crop")
    window = np.zeros(raw.shape, dtype=bool)
    x1, y1, x2, y2 = box
    window[y1:y2, x1:x2] = True
    return raw & garment & window


def overlap_review_flags(rois, min_overlap=0.85):
    """按小框覆盖率标记可能重复实例，保留分层衣物及原预测。

    框重叠不能证明同一物体，故不自动删除或合并实例。
    同类裤腿被拆分且互不重叠时，此检查不能识别，需要人工归属审核。
    """
    flags = {roi.instance_id: [] for roi in rois}
    for index, first in enumerate(rois):
        for second in rois[index + 1:]:
            a, b = first.bbox, second.bbox
            intersection = box_area([max(a[0], b[0]), max(a[1], b[1]),
                                     min(a[2], b[2]), min(a[3], b[3])])
            ratio = intersection / max(1e-9, min(box_area(a), box_area(b)))
            if ratio < min_overlap:
                continue
            for current, other in ((first, second), (second, first)):
                flags[current.instance_id].append({
                    "reason": "overlapping_instances_require_review",
                    "other_instance_id": other.instance_id,
                    "other_category": other.category,
                    "smaller_box_overlap": float(ratio),
                    "automatic_action": "none",
                })
    return flags
