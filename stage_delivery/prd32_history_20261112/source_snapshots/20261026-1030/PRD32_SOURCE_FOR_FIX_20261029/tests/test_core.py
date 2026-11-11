"""CPU 检查：验证几何、数据身份和入口行为，不生成新的 GPU 精度或速度成绩。

CPU regression checks for the shared logic and moved command runtime.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from PIL import Image

from fashion_multimodal_analysis.analysis.audit_dataset_overlap import audit, image_key
from fashion_multimodal_analysis.cli import registry, run_entry
from fashion_multimodal_analysis.common.paths import (
    data_root,
    project_root,
    resolve_path,
)
from fashion_multimodal_analysis.common.schema import CLASS_TO_ID
from fashion_multimodal_analysis.evaluation.metrics import box_iou, mask_iou
from fashion_multimodal_analysis.image_processing.masks import reconstruct_full_mask
from fashion_multimodal_analysis.maintenance.packaging.runtime import copy_runtime
from fashion_multimodal_analysis.segmentation.sampling import build_sampler_weights


class SemanticAst(ast.NodeTransformer):
    """Remove documentation and API annotations while preserving executed statements."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Normalize a function and all its nested definitions.

        Args:
            node: Parsed function whose executable statements are compared.

        Returns:
            The function with documentation and type hints removed.
        """
        self.generic_visit(node)
        node.returns = None
        node.type_comment = None
        if node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body.pop(0)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Normalize methods and remove the class documentation.

        Args:
            node: Parsed class definition.

        Returns:
            A class containing the original executable statements.
        """
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body.pop(0)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Remove an argument annotation.

        Args:
            node: Parsed function argument.

        Returns:
            The original argument name without its annotation.
        """
        node.annotation = None
        node.type_comment = None
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """Fold explicit adjacent literal additions for a stable comparison.

        Args:
            node: Parsed binary operation.

        Returns:
            The same operation, or its identical constant string value.
        """
        self.generic_visit(node)
        if isinstance(node.op, ast.Add):
            left = self.string_fragments(node.left)
            right = self.string_fragments(node.right)
            if left is not None and right is not None:
                return self.visit_JoinedStr(ast.JoinedStr(values=left + right))
        return node

    def string_fragments(self, node: ast.expr) -> list[ast.expr] | None:
        """Extract only literal or formatted string fragments for normalization.

        Args:
            node: A parsed string literal, formatted string, or other expression.

        Returns:
            String fragments, or None when the expression is not a string literal.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node]
        if isinstance(node, ast.JoinedStr):
            return node.values
        return None

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.expr:
        """Normalize equivalent implicit and explicit formatted string joins.

        Args:
            node: A parsed formatted string whose expression order is preserved.

        Returns:
            The combined string fragments with adjacent constant text merged.
        """
        self.generic_visit(node)
        values: list[ast.expr] = []
        for value in node.values:
            if (
                values
                and isinstance(values[-1], ast.Constant)
                and isinstance(value, ast.Constant)
                and isinstance(values[-1].value, str)
                and isinstance(value.value, str)
            ):
                values[-1] = ast.Constant(values[-1].value + value.value)
            else:
                values.append(value)
        if all(isinstance(value, ast.Constant) for value in values):
            return ast.Constant("".join(value.value for value in values))
        node.values = values
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Normalize the descriptive replacement for an ambiguous label name.

        Args:
            node: Parsed identifier.

        Returns:
            The identifier used consistently on both sides of the comparison.
        """
        if node.id == "l":
            node.id = "label_id"
        return node


class MetricsTests(unittest.TestCase):
    """保存 MetricsTests 的数据/运行职责。"""

    def test_box_geometry(self) -> None:
        """验证相交、相离和退化边界框的 IoU。"""
        self.assertEqual(box_iou([0, 0, 2, 2], [0, 0, 2, 2]), 1)
        self.assertEqual(box_iou([0, 0, 2, 2], [2, 0, 4, 2]), 0)
        self.assertAlmostEqual(box_iou([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7)
        self.assertEqual(box_iou([1, 1, 1, 1], [1, 1, 1, 1]), 0)

    def test_mask_geometry(self) -> None:
        """验证二值掩码交并比及两个空掩码的约定。"""
        a = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        b = np.array([[255, 255], [0, 0]], dtype=np.uint8)
        self.assertAlmostEqual(mask_iou(a, b), 1 / 3)
        self.assertEqual(mask_iou(a, a), 1)
        self.assertEqual(mask_iou(a * 0, a * 0), 0)

    def test_crop_mask_clipping_and_resize(self) -> None:
        """验证裁剪掩码的缩放、边界裁切和原图回填。"""
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "mask.png"
            Image.fromarray(np.array([[255, 0], [0, 255]], dtype=np.uint8)).save(p)
            resized = np.asarray(reconstruct_full_mask(p, (6, 6), (1, 1, 5, 5))) > 0
            expected = np.zeros((6, 6), dtype=bool)
            expected[1:3, 1:3] = True
            expected[3:5, 3:5] = True
            np.testing.assert_array_equal(resized, expected)
            clipped = np.asarray(reconstruct_full_mask(p, (4, 4), (-1, -1, 1, 1))) > 0
            self.assertEqual(clipped.sum(), 1)
            self.assertTrue(clipped[0, 0])

    def test_full_mask_ignores_crop_bbox(self) -> None:
        """验证全图掩码不被裁剪框再次错误变换。"""
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "mask.png"
            arr = np.zeros((4, 4), dtype=np.uint8)
            arr[0, 0] = 255
            Image.fromarray(arr).save(p)
            np.testing.assert_array_equal(
                np.asarray(reconstruct_full_mask(p, (4, 4), (2, 2, 3, 3))), arr
            )


class SamplerTests(unittest.TestCase):
    """保存 SamplerTests 的数据/运行职责。"""

    def dataset(self) -> Any:
        """数据集。

        Returns:
            本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
        """
        rows = [{"garment_category": c} for c in CLASS_TO_ID]
        rows.append({"garment_category": "top"})
        samples = [
            {"source_dataset": "test", "source_image": str(i), "instances": [row]}
            for i, row in enumerate(rows)
        ]
        return SimpleNamespace(rows=rows, samples=samples)

    def test_no_balancing_and_sqrt_balancing(self) -> None:
        """验证不均衡采样和平方根均衡权重的公式。"""
        dataset = self.dataset()
        equal, _ = build_sampler_weights(dataset, 0)
        self.assertEqual(equal, [1.0] * 9)
        damped, report = build_sampler_weights(dataset, 0.5)
        self.assertEqual(damped[0], 1)
        self.assertAlmostEqual(damped[1], math.sqrt(2))
        self.assertEqual(len(report), 9)

    def test_multiclass_image_uses_max_not_product(self) -> None:
        """验证多类同图使用最大类别权重，不能把权重相乘。"""
        dataset = self.dataset()
        dataset.samples = [
            {
                "source_dataset": "test",
                "source_image": "combined",
                "instances": dataset.rows[:3],
            }
        ]
        weights, _ = build_sampler_weights(dataset, 0.5)
        self.assertAlmostEqual(weights[0], math.sqrt(2))

    def test_invalid_training_distribution(self) -> None:
        """验证无效训练类别分布及时报错。"""
        with self.assertRaises(ValueError):
            build_sampler_weights(self.dataset(), -0.1)
        with self.assertRaises(ValueError):
            build_sampler_weights(SimpleNamespace(rows=[], samples=[]), 0.5)
        dataset = self.dataset()
        dataset.rows = dataset.rows[1:8]
        with self.assertRaisesRegex(ValueError, "class missing"):
            build_sampler_weights(dataset, 0.5)


class RuntimeTests(unittest.TestCase):
    """保存 RuntimeTests 的数据/运行职责。"""

    def test_paths_can_relocate(self) -> None:
        """验证项目和外部数据可迁移到另一工作目录。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkout"
            (root / "configs").mkdir(parents=True)
            external = Path(directory) / "dataset"
            with patch.dict(
                os.environ,
                {"FASHION_PROJECT_ROOT": str(root), "FASHION_DATA_ROOT": str(external)},
            ):
                self.assertEqual(project_root(), root)
                self.assertEqual(data_root(), external)
                self.assertEqual(resolve_path("configs/x.csv"), root / "configs/x.csv")
                self.assertEqual(
                    resolve_path("../fashion_data/raw/x.jpg"), external / "raw/x.jpg"
                )
                self.assertEqual(
                    resolve_path("..\\fashion_data\\raw\\x.jpg"), external / "raw/x.jpg"
                )
                self.assertEqual(
                    resolve_path(
                        "/workspace/fashion_multimodal_analysis/reports/x.csv"
                    ),
                    root / "reports/x.csv",
                )

    def test_launcher_restores_cwd_and_argv_after_failure(self) -> None:
        """验证入口失败后仍恢复调用者工作目录和参数。"""
        before_cwd, before_argv = Path.cwd(), sys.argv[:]

        def fail(*args: Any, **kwargs: Any) -> None:
            """fail。

            Raises:
                RuntimeError: expected failure
            """
            self.assertEqual(Path.cwd(), ROOT)
            raise RuntimeError("expected failure")

        with patch(
            "fashion_multimodal_analysis.cli.runpy.run_module", side_effect=fail
        ):
            with self.assertRaisesRegex(RuntimeError, "expected failure"):
                run_entry("audit_dataset_overlap", [])
        self.assertEqual(Path.cwd(), before_cwd)
        self.assertEqual(sys.argv, before_argv)

    def test_copied_gpu_runtime_has_functioning_categorized_entrypoint(self) -> None:
        """验证复制出的运行目录仍能使用分类入口。"""
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "relocated_project"
            copy_runtime(ROOT, destination)
            script = (
                destination
                / registry()["train_prd_8class_maskrcnn_v3_b1_dataexp"]["entrypoint"]
            )
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--sampler-alpha", result.stdout)


class EvidenceTests(unittest.TestCase):
    """保存 EvidenceTests 的数据/运行职责。"""

    def test_overlap_audit_preserves_the_known_issue(self) -> None:
        """验证重叠审计如实保留已知的 33 张 Core 重叠。"""
        result, overlaps = audit(ROOT)
        self.assertEqual(result["status"], "OVERLAP_FOUND")
        self.assertEqual(result["training"]["v3"]["core_overlap_images"], 33)
        self.assertEqual(len(overlaps), 33)
        self.assertEqual(result["training"]["v1"]["core_overlap_images"], 0)
        self.assertEqual(result["training"]["v2"]["core_overlap_images"], 0)
        self.assertEqual(result["validation_core_overlap_images"], 0)
        self.assertEqual(
            Counter(row["core_category"] for row in overlaps),
            {"shoe": 12, "bag": 9, "accessory": 12},
        )

    def test_overlap_keys_do_not_match_different_directories(self) -> None:
        """验证同名但不同数据目录的图片不会被误判为同图。"""
        self.assertEqual(
            image_key("/workspace/fashion_data/raw/a.jpg"),
            image_key("../fashion_data/raw/a.jpg"),
        )
        self.assertNotEqual(
            image_key("../fashion_data/raw/a.jpg"),
            image_key("../fashion_data/test/a.jpg"),
        )

    def test_extracted_algorithms_match_uploaded_ast_fingerprints(self) -> None:
        """验证上传算法指纹，并比较补注释后的执行结构。"""
        records = json.loads((ROOT / "docs/refactor_record.json").read_text())[
            "shared_extractions"
        ]
        checked = set()
        for record in records:
            identity = (
                record["shared_module"],
                record["shared_symbol"],
                record["symbol"],
            )
            if identity in checked:
                continue
            checked.add(identity)
            p = (
                ROOT
                / "src/fashion_multimodal_analysis"
                / (record["shared_module"].replace(".", "/") + ".py")
            )
            node = next(
                n
                for n in ast.parse(p.read_text()).body
                if getattr(n, "name", "") == record["shared_symbol"]
            )
            node.name = record["symbol"]
            reference_path = (
                ROOT
                / "archive/reference_algorithms"
                / (record["shared_module"].replace(".", "/") + ".py.txt")
            )
            reference = next(
                n
                for n in ast.parse(reference_path.read_text()).body
                if getattr(n, "name", "") == record["shared_symbol"]
            )
            reference.name = record["symbol"]
            # The uploaded AST fingerprint first authenticates the reference.
            digest = hashlib.sha256(ast.dump(reference).encode()).hexdigest()
            self.assertEqual(digest, record["original_ast_sha256"], identity)
            # Comments, docstrings and hints may change; executed logic must match.
            self.assertEqual(
                ast.dump(SemanticAst().visit(node)),
                ast.dump(SemanticAst().visit(reference)),
                identity,
            )

    def test_documented_runtime_preserves_uploaded_filtered_transfer_logic(
        self,
    ) -> None:
        """验证注释版 runtime 保留上传速度补丁的筛选与计时逻辑。"""
        reference_path = (
            ROOT
            / "archive/reference_algorithms/evaluation/runtime_fast_uploaded.py.txt"
        )
        original = reference_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "60021df022b16fbe42c88ed57635e180cfd397077803403cbb6513169daabaca",
        )
        references = {
            node.name: node
            for node in ast.parse(original).body
            if isinstance(node, ast.FunctionDef)
        }
        current = ast.parse(
            (ROOT / "src/fashion_multimodal_analysis/evaluation/runtime.py").read_text()
        )
        for node in current.body:
            if isinstance(node, ast.FunctionDef):
                self.assertIn(node.name, references)
                self.assertEqual(
                    ast.dump(SemanticAst().visit(node)),
                    ast.dump(SemanticAst().visit(references.pop(node.name))),
                    node.name,
                )
        self.assertFalse(references)


if __name__ == "__main__":
    unittest.main()
