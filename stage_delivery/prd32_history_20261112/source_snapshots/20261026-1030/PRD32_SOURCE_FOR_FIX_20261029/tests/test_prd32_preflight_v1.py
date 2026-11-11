"""Verify that PRD3.2 preparation rejects leakage and wrong image identities."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fashion_multimodal_analysis.benchmarking.prd32_preflight_v1 import (
    inspect_requests,
    request_errors,
)


class Prd32PreflightTests(unittest.TestCase):
    """Check boundaries using synthetic file bytes, never image-derived GT."""

    def request(self, path: Path) -> dict[str, Any]:
        """Create one valid synthetic public request for file-behavior tests."""
        return {
            "case_id": "fixture",
            "task": "vqa",
            "split": "development",
            "image_id": "fixture_image",
            "image_path": str(path),
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_size": [100, 100],
            "target_instance_id": "fixture_pred",
            "target_bbox_xyxy": [1, 2, 90, 90],
            "target_bbox_origin": "prd31_frozen_prediction",
            "query": "Describe visible evidence.",
            "output_format": "text",
            "attributes_input": None,
        }

    def test_reference_answers_cannot_enter_public_inputs(self) -> None:
        """Unknown answer fields fail instead of leaking into a model prompt."""
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "fixture.bin"
            p.write_bytes(b"fixture")
            request = self.request(p)
            request["reference_answer"] = "leaked answer"
            self.assertTrue(
                any("reference_fields" in e for e in request_errors(request))
            )

    def test_hash_mismatch_fails_asset_check(self) -> None:
        """A similarly named but changed image is not accepted as the same input."""
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "fixture.bin"
            p.write_bytes(b"before")
            request = self.request(p)
            p.write_bytes(b"after")
            self.assertTrue(inspect_requests([request], True)["errors"])

    def test_schema_only_does_not_claim_assets_or_inference(self) -> None:
        """A missing image can pass syntax only, not asset readiness."""
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "fixture.bin"
            p.write_bytes(b"fixture")
            request = self.request(p)
            p.unlink()
            report = inspect_requests([request], False)
            self.assertEqual(report["status"], "PRD32_REQUEST_SCHEMA_PASS")
            self.assertFalse(report["asset_checks_requested"])
            self.assertFalse(report["model_inference_performed"])

    def test_duplicate_requests_are_rejected(self) -> None:
        """Duplicated cases cannot silently inflate a benchmark count."""
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "fixture.bin"
            p.write_bytes(b"fixture")
            request = self.request(p)
            self.assertTrue(inspect_requests([request, request], False)["errors"])

    def test_box_outside_declared_image_is_rejected(self) -> None:
        """Source-image coordinates must not cross the declared image bounds."""
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "fixture.bin"
            p.write_bytes(b"fixture")
            request = self.request(p)
            request["target_bbox_xyxy"] = [1, 2, 101, 90]
            self.assertIn("invalid_target_box_or_source_size", request_errors(request))

    def test_sealed_path_is_rejected_before_file_access(self) -> None:
        """Protected path names fail validation without opening the path."""
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "fixture.bin"
            p.write_bytes(b"fixture")
            request = self.request(p)
            request["image_path"] = "/missing/holdout_SEALED/image.jpg"
            report = inspect_requests([request], True)
            self.assertEqual(report["image_assets"], [])
            self.assertIn("protected_path_not_allowed", report["errors"][0]["errors"])


if __name__ == "__main__":
    unittest.main()
