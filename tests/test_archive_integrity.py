"""核对整理包的冻结清单、路径迁移和 V8 结果分母。"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashion_multimodal_analysis.common.paths import resolve_record_path
from fashion_multimodal_analysis.maintenance.check_assets import (
    EXPECTED,
    inspect_assets,
)


class ArchiveIntegrityTests(unittest.TestCase):
    """从原始结果及真实文件检验迁移后的可用性。"""

    def test_frozen_and_recovered_manifests_keep_their_original_hashes(self) -> None:
        """冻结 Core 和恢复的 V5 清单须保持训练回执对应的原始字节。"""
        for name in ("core400_manifest", "v5_training_manifest"):
            relative, expected = EXPECTED[name]
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected
            )

    def test_renamed_core_masks_resolve_without_rewriting_csv(self) -> None:
        """原 CSV 中的大写实例名通过别名找到实际小写文件。"""
        relative, _ = EXPECTED["core400_manifest"]
        with (ROOT / relative).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        count = 0
        for row in rows:
            for key, value in row.items():
                if "mask" in key and value.startswith("benchmark/"):
                    self.assertTrue(
                        resolve_record_path(
                            value, ROOT, ROOT.parent / "fashion_data"
                        ).is_file(),
                        value,
                    )
                    count += 1
        self.assertGreater(count, 0)

    def test_full_result_metrics_match_saved_per_gt_rows(self) -> None:
        """逐 GT 表须支持全量模型的召回和全部 GT 掩码通过率。"""
        base = ROOT / "reports/reruns/recovery_v2/v8_full_dataset"
        for directory in sorted(base.glob("eval_*")):
            metrics = json.loads((directory / "protocol_metrics.json").read_text())[
                "micro"
            ]
            with (directory / "per_gt_results.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            correct = [row for row in rows if row["class_correct"].lower() == "true"]
            passed = [row for row in correct if float(row["mask_iou"]) >= 0.85]
            self.assertEqual(len(rows), metrics["gt_count"])
            self.assertEqual(len(correct), metrics["tp"])
            self.assertEqual(len(passed), metrics["mask_iou_ge_0_85_count"])
            self.assertAlmostEqual(len(correct) / len(rows), metrics["recall"])
            self.assertAlmostEqual(
                len(passed) / len(rows), metrics["mask_iou_ge_0_85_rate_all_gt"]
            )
            self.assertAlmostEqual(
                sum(float(row["mask_iou"]) for row in correct) / len(correct),
                metrics["mean_mask_iou_correct_class"],
            )

    def test_final_completion_is_distinct_from_the_old_prepare_failure(self) -> None:
        """保留旧失败日志，但最终状态须按完整训练和工作流记录解释。"""
        base = ROOT / "reports/reruns/recovery_v2/v8_full_dataset"
        training = json.loads((base / "training_summary.json").read_text())
        preparation = json.loads((base / "data_preparation.json").read_text())
        workflow = json.loads((base / "workflow_status.json").read_text())
        self.assertEqual(training["status"], "COMPLETED")
        self.assertEqual(workflow["status"], "TRAINING_AND_REGRESSION_COMPLETED")
        self.assertEqual(workflow["formal_acceptance"], "NOT_CLAIMED")
        self.assertEqual(training["epochs"], 1)
        self.assertEqual(
            training["epoch_log"][0]["images"], preparation["training_images"]
        )
        self.assertEqual(
            sum(preparation["class_counts"].values()), preparation["training_instances"]
        )
        self.assertTrue((base / "failure.txt").is_file())

    def test_receipts_do_not_count_as_missing_model_bytes(self) -> None:
        """只读检查必须按文件存在状态判断权重，不能只看到训练回执就判齐备。"""
        report = inspect_assets(ROOT, ROOT.parent / "fashion_data")
        for asset in report["assets"]:
            if asset["name"].endswith("checkpoint"):
                expected_path = ROOT / asset["path"]
                if not expected_path.is_file():
                    self.assertEqual(asset["status"], "MISSING")


if __name__ == "__main__":
    unittest.main()
