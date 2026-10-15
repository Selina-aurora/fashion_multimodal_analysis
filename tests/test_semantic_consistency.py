"""验证冲突复核保留原预测，不将规则输出当作人工真值。"""

import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from fashion_multimodal_analysis.integration.semantic_consistency import (
    apply_consistency_review,
    pattern_phrase_label,
)


class ConsistencyTests(unittest.TestCase):
    """验证跨模块冲突的状态传递与候选保留。"""

    def test_same_cuff_and_hem_keep_geometry_but_require_review(self) -> None:
        """重合袖口与下摆应保留几何并改为待复核。"""
        regions = {
            k: {"status": "detected", "bbox": [0, 0, 10, 10], "mask_path": "x.png"}
            for k in ("cuff", "hem")
        }
        flags = apply_consistency_review(regions, {})
        self.assertEqual(len(flags), 1)
        self.assertEqual(regions["cuff"]["status"], "requires_review")
        self.assertEqual(regions["hem"]["bbox"], [0, 0, 10, 10])
        self.assertEqual(regions["hem"]["model_status_before_review"], "detected")

    def test_separate_regions_not_changed(self) -> None:
        """分离的局部区域不应被冲突规则修改。"""
        regions = {
            "cuff": {"status": "detected", "bbox": [0, 0, 2, 2]},
            "hem": {"status": "detected", "bbox": [5, 5, 9, 9]},
        }
        self.assertEqual(apply_consistency_review(regions, {}), [])

    def test_generic_pattern_phrase_does_not_claim_floral(self) -> None:
        """泛称图案不得自动解释为印花类别。"""
        self.assertIsNone(pattern_phrase_label("printed pattern"))
        self.assertIsNone(pattern_phrase_label("floral striped pattern"))

    def test_pattern_conflict_preserves_both_candidates(self) -> None:
        """图案分歧应保留两方候选而不强行裁决。"""
        regions = {
            "pattern": {
                "status": "detected",
                "model_phrase": "floral pattern",
                "bbox": [0, 0, 10, 10],
            }
        }
        attrs = {
            "pattern": {"status": "predicted", "label": "solid", "confidence": 0.8}
        }
        apply_consistency_review(regions, attrs)
        self.assertEqual(attrs["pattern"]["label"], "")
        self.assertEqual(
            attrs["pattern"]["model_candidate_before_review"]["label"], "solid"
        )
        self.assertEqual(regions["pattern"]["model_phrase"], "floral pattern")

    def test_agreeing_pattern_is_retained(self) -> None:
        """一致图案预测应保留原输出。"""
        regions = {
            "pattern": {"status": "detected", "model_phrase": "printed floral pattern"}
        }
        attrs = {"pattern": {"status": "predicted", "label": "floral"}}
        self.assertEqual(apply_consistency_review(regions, attrs), [])

    def test_other_pattern_is_not_forced(self) -> None:
        """其他图案标签不应被强制映射为具体类别。"""
        regions = {"pattern": {"status": "detected", "model_phrase": "floral pattern"}}
        attrs = {"pattern": {"status": "predicted", "label": "other_pattern"}}
        self.assertEqual(apply_consistency_review(regions, attrs), [])

    def test_conflicted_collar_propagates_to_neckline(self) -> None:
        """领口冲突应传递为领型属性待复核。"""
        regions = {
            k: {"status": "detected", "bbox": [0, 0, 10, 10]}
            for k in ("collar", "pocket")
        }
        attrs = {
            "neckline": {"status": "predicted", "label": "crew", "confidence": 0.8}
        }
        apply_consistency_review(regions, attrs)
        self.assertEqual(attrs["neckline"]["status"], "requires_review")
        self.assertEqual(
            attrs["neckline"]["model_candidate_before_review"]["label"], "crew"
        )

    def test_missed_is_not_absence_or_review(self) -> None:
        """漏检不能被转换成不存在或正确复核。"""
        r = {
            "cuff": {"status": "missed", "bbox": None},
            "hem": {"status": "detected", "bbox": [0, 0, 10, 10]},
        }
        self.assertEqual(apply_consistency_review(r, {}), [])
        self.assertEqual(r["cuff"]["status"], "missed")

    def test_pipeline_explicit_option_checks_conflict_without_class_override(
        self,
    ) -> None:
        """显式开关应启用冲突检查但不更改服饰类别。"""
        from fashion_multimodal_analysis.integration.pipeline import (
            FashionPipeline,
            GarmentROI,
        )

        roi = GarmentROI("a", "top", 0.8, [0, 0, 10, 10], np.ones((10, 10)))
        backend = SimpleNamespace(
            config={"semantic_consistency_review": True},
            segment=lambda image: ([roi], 0),
            ground=lambda *args: ({"status": "detected", "bbox": [0, 0, 10, 10]}, 0),
            attributes=lambda *args, **kwargs: ({}, 0),
        )
        r = FashionPipeline(backend).run(
            Image.new("RGB", (10, 10)), ["cuff", "hem"], []
        )["instances"][0]
        self.assertEqual(r["roi"].category, "top")
        self.assertEqual(r["regions"]["cuff"]["status"], "requires_review")
        self.assertEqual(
            r["review_flags"][0]["reason"], "different_regions_share_near_identical_box"
        )


if __name__ == "__main__":
    unittest.main()
