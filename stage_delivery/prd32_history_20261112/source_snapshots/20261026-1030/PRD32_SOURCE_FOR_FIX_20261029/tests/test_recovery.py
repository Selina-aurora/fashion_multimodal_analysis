"""CPU 检查：验证几何、数据身份和入口行为，不生成新的 GPU 精度或速度成绩。

Behavior checks for protocol repair, GT isolation, review gates and ROI propagation.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fashion_multimodal_analysis.analysis.audit_dataset_overlap import image_key
from fashion_multimodal_analysis.analysis.summarize_prd31_acceptance import (
    strict_grounding,
    summarize,
)
from fashion_multimodal_analysis.benchmarking.freeze_prd31_reviewed_tests import (
    freeze,
    validate_reviews,
    verify_frozen,
)
from fashion_multimodal_analysis.benchmarking.prepare_prd31_recovery import (
    ATTRIBUTES,
    REGIONS,
    applies,
    digest,
    labels_for,
)
from fashion_multimodal_analysis.common.io import read_csv, write_csv, write_json
from fashion_multimodal_analysis.common.schema import CLASS_TO_ID
from fashion_multimodal_analysis.evaluation.protocol import (
    confidence_first_match,
    detection_summary,
    metric_status,
    original_box,
    per_attribute_metrics,
)
from fashion_multimodal_analysis.integration.model_backends import ModelBackends
from fashion_multimodal_analysis.integration.pipeline import (
    FashionPipeline,
    GarmentROI,
    roi_images,
)
from fashion_multimodal_analysis.integration.run_prd31_acceptance import evaluate


class ProtocolTests(unittest.TestCase):
    """保存 ProtocolTests 的数据/运行职责。"""

    def test_confidence_precedes_better_iou(self) -> None:
        """验证匹配优先级按置信度排序，而不是先取 IoU 最大的预测。"""
        matches = confidence_first_match(
            [[0, 0, 10, 10]], [[0, 0, 6, 10], [0, 0, 9, 10]], scores=[0.95, 0.80]
        )
        self.assertEqual(matches, [(0, 0, 0.6)])

    def test_one_to_one_stable_ties_and_nonfinite_predictions(self) -> None:
        """验证一对一匹配、稳定同分顺序及非有限预测的处理。"""
        matches = confidence_first_match(
            [[0, 0, 10, 10], [20, 0, 30, 10]],
            [[0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 30, 10], [20, 0, 30, 10]],
            scores=[0.9, 0.9, float("nan"), 0.8],
        )
        self.assertEqual([(gi, pi) for gi, pi, _ in matches], [(0, 0), (1, 3)])
        self.assertEqual(
            confidence_first_match([None, [0, 0, 0, 1]], [[0, 0, 1, 1]], scores=[1]), []
        )
        with self.assertRaises(ValueError):
            confidence_first_match([], [])

    def test_wrong_class_counts_as_fp_and_fn(self) -> None:
        """验证类别错误同时计入误报 FP 与漏检 FN。"""
        rows = [
            {"localized_bbox50": True, "class_correct": False, "mask_iou": 1},
            {"localized_bbox50": True, "class_correct": True, "mask_iou": 0.9},
            {"localized_bbox50": False, "class_correct": False, "mask_iou": 0},
        ]
        m = detection_summary(rows, 4)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 3, 2))
        self.assertEqual(m["precision"], 0.25)
        self.assertEqual(m["mean_mask_iou_correct_class"], 0.9)

    def test_empty_or_missing_coverage_cannot_pass(self) -> None:
        """验证空结果和缺少覆盖的结果不能被验收为通过。"""
        m = detection_summary([], 0)
        self.assertIsNone(m["mean_mask_iou_correct_class"])
        self.assertEqual(metric_status(None, 0.85), "NOT_EVALUATED")
        self.assertEqual(
            metric_status(1, 0.85, coverage=False), "INSUFFICIENT_COVERAGE"
        )
        self.assertEqual(
            metric_status(1, 50, higher=False, evaluated=False), "NOT_EVALUATED"
        )

    def test_all_eligible_attribute_truth_including_segmentation_miss_remains(
        self,
    ) -> None:
        """验证分割漏检时，符合条件的属性 GT 仍留在分母。"""
        truth = [
            {
                "sample_id": f"s{i}",
                "attribute_name": "pattern",
                "garment_category": "top",
                "gt_label": "solid",
                "review_status": "reviewed",
                "applicability": "eligible",
                "ambiguous": "0",
            }
            for i in range(3)
        ]
        predictions = {
            ("s0", "pattern"): {
                "status": "predicted",
                "label": "solid",
                "garment_class_correct": True,
            },
            ("s1", "pattern"): {
                "status": "predicted",
                "label": "solid",
                "garment_class_correct": False,
            },
        }
        m, rows = per_attribute_metrics(truth, predictions, "B")
        self.assertEqual(m["n"], 3)
        self.assertEqual(m["micro_accuracy"], 1 / 3)
        self.assertEqual(rows[-1]["prediction_status"], "missing")

    def test_coarse_and_unreviewed_grounding_cannot_pass(self) -> None:
        """验证粗粒度或未审核局部标注不能直接通过验收。"""
        rows = [
            {
                "track": t,
                "query_id": f"q{i}",
                "sample_id": f"s{i}",
                "target_region": "collar",
                "prediction_status": "detected",
                "garment_class_correct": "True",
                "bbox_iou": ".9",
                "manual_grade": g,
                "reviewer": "human" if g else "",
                "review_status": "reviewed" if g else "pending",
            }
            for t in ("A", "B")
            for i, g in enumerate(("correct", "coarse", ""))
        ]
        m, detail = strict_grounding(rows, None)
        self.assertIsNone(m["A"]["strict_accuracy"])
        self.assertEqual(m["A"]["correct"], 1)
        self.assertEqual([r["strict_correct"] for r in detail[:3]], [1, 0, 0])
        reviewed = [
            {
                **r,
                "manual_grade": r["manual_grade"] or "wrong",
                "reviewer": "human",
                "review_status": "reviewed",
            }
            for r in rows
        ]
        m, _ = strict_grounding(rows, reviewed)
        self.assertEqual(m["A"]["strict_accuracy"], 1 / 3)
        bad = [{**r, "bbox_iou": "1"} for r in reviewed]
        with self.assertRaisesRegex(ValueError, "Immutable"):
            strict_grounding(rows, bad)


class DataRepairTests(unittest.TestCase):
    """保存 DataRepairTests 的数据/运行职责。"""

    def test_clean_manifest_excludes_all_core_and_validation_source_images(
        self,
    ) -> None:
        """验证清理后的训练清单排除全部 Core 和开发集原图。"""
        clean = read_csv(ROOT / "configs/prd_8class_train_v4_clean.csv")
        original = read_csv(ROOT / "configs/prd_8class_train_v3.csv")
        protected = {
            image_key(r["source_image"])
            for p in (
                "configs/prd_8class_val_v1.csv",
                "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv",
            )
            for r in read_csv(ROOT / p)
        }
        self.assertEqual(
            (
                len(original),
                len(clean),
                len({image_key(r["source_image"]) for r in clean}),
            ),
            (580, 547, 475),
        )
        self.assertFalse({image_key(r["source_image"]) for r in clean} & protected)
        self.assertEqual(set(r["garment_category"] for r in clean), set(CLASS_TO_ID))

    def test_legacy_reviews_keep_actual_accuracy_as_development(self) -> None:
        """验证旧审核的实际准确率保留为开发结果。"""
        m = json.loads(
            (
                ROOT / "benchmark/prd_3_1_v2/development/legacy_attribute_metrics.json"
            ).read_text()
        )
        self.assertEqual(
            (m["primary_color"]["correct"], m["primary_color"]["eligible_rows"]),
            (32, 37),
        )
        self.assertEqual(
            (m["pattern"]["correct"], m["pattern"]["eligible_rows"]), (34, 40)
        )
        self.assertTrue(
            all(
                v["split"] == "development" and v["formal_acceptance"] == "NOT_CLAIMED"
                for v in m.values()
            )
        )


class ReviewTests(unittest.TestCase):
    """保存 ReviewTests 的数据/运行职责。"""

    @classmethod
    def setUpClass(cls) -> None:
        """setUpClass。"""
        cls.ground = read_csv(ROOT / "benchmark/prd_3_1_v2/drafts/grounding_test.csv")
        cls.attrs = read_csv(ROOT / "benchmark/prd_3_1_v2/drafts/attribute_test.csv")

    def reviews(self) -> tuple[Any, ...]:
        """reviews。

        Returns:
            按顺序返回 ground, attrs 等结果。
        """
        ground = [
            {
                **r,
                "target_present": "yes",
                "review_status": "reviewed",
                "reviewer": "test_fixture",
                **{
                    "gt_region_bbox_" + k: r["gt_bbox_" + k]
                    for k in ("x1", "y1", "x2", "y2")
                },
            }
            for r in self.ground
        ]
        attrs = [
            (
                {
                    **r,
                    "gt_label": labels_for(r["attribute_name"], r["garment_category"])[
                        0
                    ],
                    "review_status": "reviewed",
                    "reviewer": "test_fixture",
                }
                if r["applicability"] == "eligible"
                else dict(r)
            )
            for r in self.attrs
        ]
        return ground, attrs

    def test_pending_reviews_are_rejected(self) -> None:
        """验证尚未完成的审核记录不能用于冻结测试。"""
        with self.assertRaisesRegex(ValueError, "pending"):
            validate_reviews(self.ground, self.attrs, self.ground, self.attrs)

    def test_candidate_identity_and_original_image_coordinates_are_locked(self) -> None:
        """验证候选身份和原图坐标受到冻结约束。"""
        ground, attrs = self.reviews()
        ground[0]["source_image"] = "replacement.jpg"
        with self.assertRaisesRegex(ValueError, "Immutable"):
            validate_reviews(self.ground, self.attrs, ground, attrs)
        ground, attrs = self.reviews()
        ground[0]["gt_region_bbox_x2"] = str(float(ground[0]["image_width"]) + 1)
        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            validate_reviews(self.ground, self.attrs, ground, attrs)

    def test_full_coverage_and_category_specific_labels_are_required(self) -> None:
        """验证冻结要求完整覆盖及适用于对应衣物类别的标签。"""
        ground, attrs = self.reviews()
        for r in ground:
            if r["target_region"] == "collar":
                r["target_present"] = "no"
        with self.assertRaisesRegex(ValueError, "Insufficient positive"):
            validate_reviews(self.ground, self.attrs, ground, attrs)
        ground, attrs = self.reviews()
        wrong = next(
            r
            for r in attrs
            if r["attribute_name"] == "silhouette_fit"
            and r["garment_category"] == "pants"
        )
        wrong["gt_label"] = "a_line"
        with self.assertRaisesRegex(ValueError, "taxonomy"):
            validate_reviews(self.ground, self.attrs, ground, attrs)

    def test_complete_reviews_select_fixed_five_per_region(self) -> None:
        """验证完整审核按每个区域固定选择五例。"""
        ground, attrs = self.reviews()
        selected, _, summary, _ = validate_reviews(
            self.ground, self.attrs, ground, attrs
        )
        self.assertEqual(len(selected), 40)
        self.assertEqual(
            Counter(r["target_region"] for r in selected), {r: 5 for r in REGIONS}
        )
        self.assertEqual(summary["eligible_attributes"], 160)


class PipelineTests(unittest.TestCase):
    """保存 PipelineTests 的数据/运行职责。"""

    def test_two_predictions_are_processed_without_gt_filtering(self) -> None:
        """验证两个预测实例均处理，不能先按 GT 筛选预测。"""
        image = Image.new("RGB", (20, 10), "blue")
        mask = np.ones((10, 20), dtype=np.uint8)
        rois = [
            GarmentROI("p0", "top", 0.9, [0, 0, 10, 10], mask),
            GarmentROI("p1", "bag", 0.8, [10, 0, 20, 10], mask),
        ]
        calls = []

        class Backend:
            """保存 Backend 的数据/运行职责。"""

            def segment(self, image: Any) -> tuple[Any, ...]:
                """segment。

                Args:
                    image: 本步骤处理的图像对象。

                Returns:
                    按顺序返回 rois 等结果。
                """
                return rois, 1

            def ground(self, image: Any, roi: Any, region: str) -> tuple[Any, ...]:
                """ground。

                Args:
                    image: 本步骤处理的图像对象。
                    roi: roi。
                    region: 区域。

                Returns:
                    本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
                """
                calls.append((roi.instance_id, region))
                return {"status": "missed", "bbox": None}, 2

            def attributes(
                self,
                image: Any,
                roi: Any,
                regions: Any,
                names: list[str],
                controlled: bool = False,
            ) -> tuple[Any, ...]:
                """属性。

                Args:
                    image: 本步骤处理的图像对象。
                    roi: roi。
                    regions: 区域。
                    names: names。
                    controlled: controlled。

                Returns:
                    本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
                """
                return {
                    name: {
                        "status": (
                            "localization_missing"
                            if name == "neckline"
                            else "predicted"
                        ),
                        "label": "",
                    }
                    for name in names
                }, 3

        out = FashionPipeline(Backend()).run(
            image, ["cuff"], ["primary_color", "neckline"]
        )
        self.assertEqual(len(out["instances"]), 2)
        self.assertEqual(
            set(calls),
            {("p0", "cuff"), ("p0", "collar"), ("p1", "cuff"), ("p1", "collar")},
        )
        self.assertEqual(
            out["instances"][0]["attributes"]["neckline"]["status"],
            "localization_missing",
        )

    def test_stage_failure_keeps_next_garment_and_failure_record(self) -> None:
        """验证某实例阶段失败后仍处理下一实例并保留错误。"""
        rois = [
            GarmentROI(
                str(i), "top", 0.9, [0, 0, 4, 4], np.ones((4, 4), dtype=np.uint8)
            )
            for i in range(2)
        ]

        class Backend:
            """保存 Backend 的数据/运行职责。"""

            def segment(self, image: Any) -> tuple[Any, ...]:
                """segment。

                Args:
                    image: 本步骤处理的图像对象。

                Returns:
                    按顺序返回 rois 等结果。
                """
                return rois, 1

            def ground(self, image: Any, roi: Any, region: str) -> None:
                """ground。

                Args:
                    image: 本步骤处理的图像对象。
                    roi: roi。
                    region: 区域。

                Raises:
                    RuntimeError: localizer error
                """
                raise RuntimeError("localizer error")

            def attributes(
                self,
                image: Any,
                roi: Any,
                regions: Any,
                names: list[str],
                controlled: bool = False,
            ) -> tuple[Any, ...]:
                """属性。

                Args:
                    image: 本步骤处理的图像对象。
                    roi: roi。
                    regions: 区域。
                    names: names。
                    controlled: controlled。

                Returns:
                    本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

                Raises:
                    RuntimeError: attribute error
                """
                if roi.instance_id == "0":
                    raise RuntimeError("attribute error")
                return {"pattern": {"status": "predicted", "label": "solid"}}, 1

        out = FashionPipeline(Backend()).run(
            Image.new("RGB", (4, 4)), ["collar"], ["pattern"]
        )
        self.assertEqual(len(out["instances"]), 2)
        self.assertEqual(len(out["failures"]), 3)
        self.assertEqual(
            out["instances"][0]["attributes"]["pattern"]["status"], "attribute_failed"
        )
        self.assertEqual(out["instances"][1]["attributes"]["pattern"]["label"], "solid")

    def test_roi_crop_and_coordinate_roundtrip(self) -> None:
        """验证 ROI 裁剪与原图坐标转换往返一致。"""
        image = Image.new("RGB", (20, 10), "blue")
        mask = np.zeros((10, 20), dtype=np.uint8)
        mask[2:7, 3:13] = 1
        crop, local, origin = roi_images(
            image, GarmentROI("p", "top", 0.9, [3, 2, 13, 7], mask)
        )
        self.assertEqual(crop.size, (10, 5))
        self.assertEqual(local.size, crop.size)
        self.assertEqual(original_box([1, 1, 4, 3], origin), [4, 3, 7, 5])
        with self.assertRaisesRegex(ValueError, "full original"):
            roi_images(
                image, GarmentROI("p", "top", 0.9, [3, 2, 13, 7], np.ones((5, 10)))
            )

    def test_neckline_uses_predicted_part_and_never_gt_proxy_in_track_b(self) -> None:
        """验证 Track B 领口使用预测局部区域，不能用 GT 代理。"""
        backend = object.__new__(ModelBackends)
        backend.measure = lambda action: (action(), 1)
        seen = []
        backend.pattern = SimpleNamespace(
            encode_image=lambda image, *args: seen.append(image.size) or "feature",
            classify_with_prototypes=lambda *args: {
                "top1_label": "crew",
                "top1_score": 0.9,
            },
        )
        backend.design = SimpleNamespace(
            neckline_crop=lambda *args: (_ for _ in ()).throw(
                AssertionError("GT proxy called in Track B")
            )
        )
        backend.clip_model = backend.clip_processor = backend.device = None
        backend.design_proto = {"neckline": ("labels", "prototypes")}
        image = Image.new("RGB", (20, 20), "red")
        mask = np.ones((20, 20), dtype=np.uint8)
        roi = GarmentROI("p", "top", 0.9, [2, 2, 18, 18], mask)
        result, _ = backend.attributes(image, roi, {}, ["neckline"], controlled=False)
        self.assertEqual(result["neckline"]["status"], "localization_missing")
        self.assertEqual(seen, [])
        result, _ = backend.attributes(
            image,
            roi,
            {"collar": {"status": "detected", "bbox": [5, 3, 11, 7], "mask": mask}},
            ["neckline"],
            controlled=False,
        )
        self.assertEqual(seen, [(6, 4)])
        self.assertEqual(
            result["neckline"]["roi_provenance"], "3_1_2_predicted_collar_roi"
        )


class WorkflowIntegrationTests(unittest.TestCase):
    """保存 WorkflowIntegrationTests 的数据/运行职责。"""

    def fixture(self, root: Path) -> None:
        """Synthetic images and fixture reviewers never become user benchmark truth.

        Args:
            root: 当前项目根目录。
        """
        core = []
        for category, category_id in CLASS_TO_ID.items():
            for i in range(50):
                sid = f"{category}_{i:02d}"
                source = f"fixture_images/{sid}.png"
                mask = f"fixture_masks/{sid}.png"
                (root / source).parent.mkdir(parents=True, exist_ok=True)
                (root / mask).parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (10, 10), (category_id * 20, 0, i)).save(root / source)
                arr = np.zeros((10, 10), dtype=np.uint8)
                arr[1:9, 1:9] = 255
                Image.fromarray(arr).save(root / mask)
                core.append(
                    {
                        "final_sample_id": sid,
                        "source_dataset": "synthetic_test_fixture",
                        "source_image": source,
                        "garment_id": sid,
                        "garment_category": category,
                        "image_width": 10,
                        "image_height": 10,
                        "gt_bbox_x1": 1,
                        "gt_bbox_y1": 1,
                        "gt_bbox_x2": 9,
                        "gt_bbox_y2": 9,
                        "gt_mask_path": mask,
                        "gt_overlay_path": source,
                    }
                )
        core_path = root / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv"
        write_csv(core_path, core)
        train = root / "configs/prd_8class_train_v4_clean.csv"
        val = root / "configs/prd_8class_val_v1.csv"
        write_csv(
            train, [{"source_image": "fixture_training.jpg", "garment_category": "top"}]
        )
        write_csv(val, [{"source_image": "fixture_val.jpg", "garment_category": "top"}])
        config = json.loads((ROOT / "configs/prd_3_1_recovery_v2.json").read_text())
        config["reference_segmentation_sha256"] = digest(core_path)
        write_json(root / "configs/prd_3_1_recovery_v2.json", config)
        from fashion_multimodal_analysis.benchmarking.prepare_prd31_recovery import (
            sample_case,
        )

        attr = []
        for row in core:
            if int(row["final_sample_id"].rsplit("_", 1)[1]) >= 5:
                continue
            case = sample_case(row)
            for name in ATTRIBUTES:
                eligible = applies(name, row["garment_category"])
                attr.append(
                    {
                        **case,
                        "attribute_name": name,
                        "applicability": "eligible" if eligible else "not_applicable",
                        "gt_label": "",
                        "ambiguous": "0",
                        "review_status": "pending" if eligible else "not_applicable",
                        "reviewer": "",
                        "annotation_note": "",
                        "exclude_reason": "",
                    }
                )
        ground = []
        for region in REGIONS:
            for i in range(5):
                row = core[i]
                ground.append(
                    {
                        **sample_case(row),
                        "query_id": f"{region}_{i:03d}",
                        "target_region": region,
                        "query_text": region,
                        "target_present": "",
                        "gt_region_bbox_x1": "",
                        "gt_region_bbox_y1": "",
                        "gt_region_bbox_x2": "",
                        "gt_region_bbox_y2": "",
                        "gt_region_mask_path": "",
                        "ambiguous": "0",
                        "review_status": "pending",
                        "reviewer": "",
                        "annotation_note": "",
                        "exclude_reason": "",
                    }
                )
        base = root / "benchmark/prd_3_1_v2"
        write_csv(base / "drafts/grounding_test.csv", ground)
        write_csv(base / "drafts/attribute_test.csv", attr)
        write_csv(
            base / "drafts/end_to_end_test.csv",
            [
                {
                    **sample_case(core[0]),
                    "required_3_1_2_queries": "[]",
                    "required_3_1_3_attributes": "[]",
                    "review_status": "pending",
                    "exclude_reason": "",
                }
            ],
        )
        write_csv(
            base / "review/grounding_reviewed.csv",
            [
                {
                    **r,
                    "target_present": "yes",
                    "review_status": "reviewed",
                    "reviewer": "synthetic_fixture",
                    "gt_region_bbox_x1": "2",
                    "gt_region_bbox_y1": "2",
                    "gt_region_bbox_x2": "5",
                    "gt_region_bbox_y2": "5",
                }
                for r in ground
            ],
        )
        write_csv(
            base / "review/attribute_reviewed.csv",
            [
                (
                    {
                        **r,
                        "gt_label": labels_for(
                            r["attribute_name"], r["garment_category"]
                        )[0],
                        "review_status": "reviewed",
                        "reviewer": "synthetic_fixture",
                    }
                    if r["applicability"] == "eligible"
                    else r
                )
                for r in attr
            ],
        )

    def test_freeze_pipeline_and_report_keep_misses_in_denominator(self) -> None:
        """验证冻结链路和结果汇总都保留漏检 GT 的分母。"""

        class Backend:
            """保存 Backend 的数据/运行职责。"""

            device = "cpu"

            def segment(self, image: Any) -> tuple[Any, ...]:
                """segment。

                Args:
                    image: 本步骤处理的图像对象。

                Returns:
                    本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
                """
                red, _, blue = image.getpixel((0, 0))
                category = next(c for c, n in CLASS_TO_ID.items() if n * 20 == red)
                if category == "top" and blue == 0:
                    return [], 1
                if category == "top" and blue == 1:
                    category = "bag"
                mask = np.zeros((10, 10), dtype=np.uint8)
                mask[1:9, 1:9] = 1
                return [GarmentROI("p", category, 0.9, [1, 1, 9, 9], mask)], 1

            def ground(self, image: Any, roi: Any, region: str) -> tuple[Any, ...]:
                """ground。

                Args:
                    image: 本步骤处理的图像对象。
                    roi: roi。
                    region: 区域。

                Returns:
                    本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
                """
                mask = np.zeros((10, 10), dtype=np.uint8)
                mask[2:5, 2:5] = 1
                return {
                    "status": "detected",
                    "bbox": [2, 2, 5, 5],
                    "mask": mask,
                    "model_phrase": region,
                    "score": 0.9,
                }, 2

            def attributes(
                self,
                image: Any,
                roi: Any,
                regions: Any,
                names: list[str],
                controlled: bool = False,
            ) -> tuple[Any, ...]:
                """属性。

                Args:
                    image: 本步骤处理的图像对象。
                    roi: roi。
                    regions: 区域。
                    names: names。
                    controlled: controlled。

                Returns:
                    本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
                """
                return {
                    name: (
                        {
                            "status": "predicted",
                            "label": labels_for(name, roi.category)[0],
                            "roi_provenance": (
                                "controlled_gt_roi"
                                if controlled
                                else "segmentation_prediction"
                            ),
                        }
                        if applies(name, roi.category)
                        else {"status": "not_applicable", "label": ""}
                    )
                    for name in names
                }, 3

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"FASHION_PROJECT_ROOT": str(root)}):
                self.fixture(root)
                meta = freeze(root)
                self.assertEqual(meta["review_summary"]["eligible_attributes"], 160)
                self.assertEqual(
                    verify_frozen(root)["status"], "FROZEN_REVIEWED_REGRESSION_TEST"
                )
                with self.assertRaises(FileExistsError):
                    freeze(root)
                output = root / "reports/fixture"
                with contextlib.redirect_stdout(io.StringIO()):
                    initial = evaluate(root, output, backend=Backend())
                self.assertEqual(initial["attributes"]["A"]["n"], 160)
                self.assertEqual(initial["attributes"]["B"]["n"], 160)
                self.assertEqual(
                    initial["attributes"]["B"]["micro_accuracy"], 148 / 160
                )
                self.assertEqual(initial["end_to_end"]["failure_stages"]["SEG_MISS"], 1)
                self.assertEqual(
                    initial["end_to_end"]["failure_stages"]["SEG_WRONG_CLASS"], 1
                )
                self.assertEqual(
                    initial["grounding"]["A"]["pending_semantic_reviews"], 40
                )
                self.assertEqual(initial["prd_module_status"], "NOT_EVALUATED")
                reviews = read_csv(output / "grounding_semantic_audit_template.csv")
                for row in reviews:
                    row.update(
                        manual_grade="correct",
                        reviewer="synthetic_fixture",
                        review_status="reviewed",
                    )
                # A human coarse judgment must fail even with a geometrically perfect box.
                row = next(
                    r
                    for r in reviews
                    if r["track"] == "B" and r["query_id"] == "collar_004"
                )
                row["manual_grade"] = "coarse"
                write_csv(output / "grounding_semantic_audit_reviewed.csv", reviews)
                result = summarize(root, output)
                self.assertEqual(result["grounding"]["A"]["strict_accuracy"], 1)
                self.assertEqual(
                    result["end_to_end"]["failure_stages"]["GROUNDING_WRONG_OR_COARSE"],
                    1,
                )
                self.assertTrue(
                    all(
                        g["status"] == "NOT_EVALUATED"
                        for g in result["gates"]
                        if "latency" in g["metric"]
                    )
                )
                self.assertEqual(result["prd_module_status"], "NOT_EVALUATED")
                # Frozen GT tampering is detected before metrics can be finalized.
                with (root / "benchmark/prd_3_1_v2/frozen/attribute_test.csv").open(
                    "a"
                ) as handle:
                    handle.write("tampered")
                with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                    summarize(root, output)


if __name__ == "__main__":
    unittest.main()
