"""校验上下文范围与分歧处理；不测试模型识别准确率。"""

import sys
import types
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
from PIL import Image

from fashion_multimodal_analysis.integration.attribute_context import (
    neckline_context_bounds,
    paired_neckline_result,
)


class AttributeContextTests(unittest.TestCase):
    """领口上下文几何和候选配对的CPU测试。"""

    def test_context_contains_collar_and_stays_in_garment(self) -> None:
        """扩展范围应包含领口且被服饰框限制。"""
        result = neckline_context_bounds(
            [30, 20, 50, 40], [10, 10, 100, 110], (120, 120)
        )
        self.assertEqual(result, [20, 10, 60, 50])

    def test_tiny_box_retains_context(self) -> None:
        """极小领口框仍应保留最低宽高的上下文。"""
        result = neckline_context_bounds([49, 29, 51, 31], [0, 0, 100, 200], (100, 200))
        self.assertGreaterEqual(result[2] - result[0], 25)
        self.assertGreaterEqual(result[3] - result[1], 30)

    def test_image_edge_clipped(self) -> None:
        """越界服饰框的上下文应裁到原图范围。"""
        result = neckline_context_bounds([0, 0, 10, 10], [-3, -3, 20, 20], (20, 20))
        self.assertEqual(result[:2], [0, 0])
        self.assertLessEqual(result[2], 20)
        self.assertLessEqual(result[3], 20)

    def test_disjoint_is_error(self) -> None:
        """领口与服饰不相交时应拒绝构造上下文。"""
        with self.assertRaises(ValueError):
            neckline_context_bounds([50, 50, 60, 60], [0, 0, 10, 10], (100, 100))

    def test_nonfinite_is_error(self) -> None:
        """非有限坐标应触发明确异常。"""
        with self.assertRaises(ValueError):
            neckline_context_bounds(
                [0, 0, float("nan"), 10], [0, 0, 100, 100], (100, 100)
            )

    def test_disagreement_is_review_without_final_label(self) -> None:
        """视图分歧应保留候选并清空正式标签。"""
        a = {"top1_label": "crew", "top1_score": 0.8}
        b = {"top1_label": "v_neck", "top1_score": 0.9}
        r = paired_neckline_result(a, b, [1, 1, 2, 2], [0, 0, 3, 3])
        self.assertEqual(r["status"], "requires_review")
        self.assertEqual(r["label"], "")
        self.assertIsNone(r["confidence"])
        self.assertEqual(r["legacy_candidate"]["label"], "crew")

    def test_agreement_is_still_pending_review(self) -> None:
        """视图一致也不能自动认定为人工审核通过。"""
        a = {"top1_label": "crew", "top1_score": 0.8}
        r = paired_neckline_result(a, a, [1, 1, 2, 2], [0, 0, 3, 3])
        self.assertEqual(r["status"], "predicted")
        self.assertEqual(r["semantic_review_status"], "pending")


class BackendContextTests(unittest.TestCase):
    """用替身后端验证上下文开关与受控轨道隔离。"""

    def infer(
        self, enabled: bool, region_status: str = "detected", controlled: bool = False
    ) -> dict[str, Any]:
        """以可控假特征运行领型分支并返回属性记录，不加载真实模型。"""
        semantic = types.ModuleType(
            "fashion_multimodal_analysis.integration.region_semantics"
        )
        semantic.REGION_PROMPTS = {}
        semantic.phrase_matches = lambda *args: True
        semantic.supports_region = lambda *args: True
        with patch.dict(sys.modules, {semantic.__name__: semantic}):
            from fashion_multimodal_analysis.integration.model_backends import (
                ModelBackends,
            )
        from fashion_multimodal_analysis.integration.pipeline import GarmentROI

        backend = ModelBackends.__new__(ModelBackends)
        backend.config = {"neckline_context_pair": enabled}
        backend.measure = lambda action: (action(), 0)
        backend.clip_model = backend.clip_processor = backend.device = None
        backend.design_proto = {"neckline": ([], [])}
        labels = iter(["crew", "v_neck"])
        backend.pattern = SimpleNamespace(
            encode_image=lambda *args: None,
            classify_with_prototypes=lambda *args: {
                "top1_label": next(labels),
                "top1_score": 0.8,
            },
        )
        backend.design = SimpleNamespace(neckline_crop=lambda crop, mask: crop)
        image = Image.new("RGB", (100, 100), "red")
        roi = GarmentROI("a", "top", 0.9, [0, 0, 100, 100], np.ones((100, 100)))
        collar = {
            "status": region_status,
            "bbox": [30, 10, 60, 30],
            "mask": np.ones((100, 100)),
        }
        return backend.attributes(
            image, roi, {"collar": collar}, ["neckline"], controlled
        )[0]["neckline"]

    def test_backend_conflicting_views_need_review(self) -> None:
        """后端应将原裁剪与上下文分歧传为待复核。"""
        r = self.infer(True)
        self.assertEqual(r["status"], "requires_review")
        self.assertEqual(r["context_candidate"]["label"], "v_neck")

    def test_disabled_retains_original_prediction(self) -> None:
        """开关关闭时应保持原裁剪预测。"""
        r = self.infer(False)
        self.assertEqual(r["status"], "predicted")
        self.assertEqual(r["label"], "crew")
        self.assertNotIn("context_candidate", r)

    def test_missing_collar_does_not_fabricate_context(self) -> None:
        """缺少领口定位时不能伪造上下文候选。"""
        r = self.infer(True, "missed")
        self.assertEqual(r["status"], "localization_missing")
        self.assertNotIn("context_candidate", r)

    def test_controlled_path_keeps_existing_gt_proxy(self) -> None:
        """受控GT轨道不受预测领口上下文开关影响。"""
        r = self.infer(True, controlled=True)
        self.assertEqual(
            r["roi_provenance"], "controlled_gt_garment_roi_with_fixed_neckline_proxy"
        )
        self.assertNotIn("context_candidate", r)


if __name__ == "__main__":
    unittest.main()
