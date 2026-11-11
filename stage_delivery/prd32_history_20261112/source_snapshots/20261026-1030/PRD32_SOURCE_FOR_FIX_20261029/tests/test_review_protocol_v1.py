"""Check that draft review preparation cannot turn ambiguity into measured success."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fashion_multimodal_analysis.benchmarking.review_protocol_v1 import (
    main,
    new_record,
    prepare_json_review,
    prepare_legacy,
    priority_reason,
    summarize,
    validate_record,
)


class ReviewProtocolTests(unittest.TestCase):
    """Exercise semantic conflicts, legacy preservation and safe file behavior."""

    def reviewed(self) -> dict[str, str]:
        """Return a hand-authored test fixture, never an image assessment."""
        row = new_record("fixture_image", "fixture_instance", "cuff")
        row.update(
            target_status="correct",
            presence="present",
            visibility="visible",
            geometry_evaluable="no",
            prediction_status="detected",
            localization="reasonable",
            reference_origin="none",
            reviewer="fixture",
            reviewed_at="2026-10-26",
            evidence_note="Synthetic unit-test fixture",
            review_status="reviewed",
        )
        return row

    def test_pending_does_not_become_a_review(self) -> None:
        """Unfilled template rows cannot count as reviewed or accurate."""
        result = summarize([new_record("i", "p", "cuff")])
        self.assertEqual(result["reviewed_records"], 0)
        self.assertIsNone(result["formal_accuracy"])

    def test_occlusion_is_not_a_miss(self) -> None:
        """Hidden parts remain non-evaluable even when the model missed them."""
        row = self.reviewed()
        row.update(
            visibility="occluded", prediction_status="missed", localization="missed"
        )
        self.assertTrue(validate_record(row))
        row["localization"] = "not_evaluable"
        self.assertEqual(validate_record(row), [])

    def test_wrong_target_is_upstream(self) -> None:
        """A wrong selected garment cannot produce a local localization grade."""
        row = self.reviewed()
        row["target_status"] = "wrong"
        self.assertIn("wrong_target_cannot_score_localization", validate_record(row))

    def test_absence_is_scored_separately(self) -> None:
        """A detected mask for a truly absent region is a false positive."""
        row = self.reviewed()
        row.update(
            presence="absent",
            visibility="not_applicable",
            localization="false_positive",
        )
        self.assertEqual(validate_record(row), [])
        row["localization"] = "reasonable"
        self.assertTrue(validate_record(row))

    def test_model_adopted_reference_cannot_enter_geometry(self) -> None:
        """Accepting a candidate does not make it independent geometric GT."""
        row = self.reviewed()
        row.update(
            geometry_evaluable="yes",
            reference_origin="model_adopted",
            reference_geometry_path="fixture.json",
        )
        self.assertIn(
            "geometry_requires_independent_observable_reference", validate_record(row)
        )

    def test_empty_mask_on_negative_requires_separate_output_review(self) -> None:
        """A candidate box plus empty mask does not establish a true negative."""
        row = self.reviewed()
        row.update(
            presence="absent",
            visibility="not_applicable",
            prediction_status="empty_mask",
            localization="correct_rejection",
        )
        self.assertTrue(validate_record(row))
        row["localization"] = "not_evaluable"
        self.assertEqual(validate_record(row), [])

    def test_visible_positive_cannot_be_a_correct_rejection(self) -> None:
        """Negatives cannot hide observable positive failures in a summary."""
        row = self.reviewed()
        row["localization"] = "correct_rejection"
        self.assertIn(
            "negative_grade_requires_confirmed_negative", validate_record(row)
        )

    def test_validation_rejects_duplicate_completed_rows(self) -> None:
        """Duplicate reviews do not inflate reviewed-record counts."""
        result = summarize([self.reviewed(), self.reviewed()])
        self.assertEqual(result["reviewed_records"], 0)
        self.assertEqual(len(result["decision_conflicts"]), 2)

    def test_legacy_labels_are_preserved_and_new_decisions_pending(self) -> None:
        """Historical labels stay opaque until a new reviewer supplies decisions."""
        with tempfile.TemporaryDirectory() as name:
            source = Path(name) / "dev.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["image", "instance", "part", "grade"]
                )
                writer.writeheader()
                writer.writerow(
                    {"image": "i", "instance": "p", "part": "cuff", "grade": "基本合理"}
                )
            rows = prepare_legacy(source, "image", "instance", "part")
            self.assertEqual(
                json.loads(rows[0]["legacy_record_json"])["grade"], "基本合理"
            )
            self.assertEqual(rows[0]["localization"], "pending")
            self.assertEqual(rows[0]["review_status"], "pending")

    def test_duplicate_id_is_not_silently_deduplicated(self) -> None:
        """Overlapping identities need explicit part-side handling."""
        with tempfile.TemporaryDirectory() as name:
            source = Path(name) / "dev.csv"
            source.write_text("i,p,r\nx,y,cuff\nx,y,cuff\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_legacy(source, "i", "p", "r")

    def test_output_directory_is_not_overwritten(self) -> None:
        """An existing audit run is preserved."""
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(FileExistsError):
                main(["--output", name])

    def test_native_json_import_excludes_unselected_pending_records(self) -> None:
        """Only explicitly reviewed identities enter the new pending queue."""
        with tempfile.TemporaryDirectory() as name:
            source = Path(name) / "development.json"
            data = {
                "schema": "prd31_new20_visual_review_v1",
                "targets": {"x/y": {"verdict": "correct"}},
                "reviews": {"x/y/cuff": {"presence": "occluded", "verdict": "wrong"}},
                "records": [
                    {"image_id": "x", "instance_id": "y", "region": "cuff"},
                    {"image_id": "x", "instance_id": "z", "region": "cuff"},
                ],
            }
            source.write_text(json.dumps(data), encoding="utf-8")
            rows = prepare_json_review(source)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["review_status"], "pending")
            self.assertEqual(
                priority_reason(rows[0]), "p1_occlusion_and_localization_mixed"
            )

    def test_native_json_import_requires_exact_identity_link(self) -> None:
        """An unmatched historical review cannot become a new annotation row."""
        with tempfile.TemporaryDirectory() as name:
            source = Path(name) / "development.json"
            source.write_text(
                json.dumps(
                    {
                        "schema": "prd31_new20_visual_review_v1",
                        "targets": {},
                        "reviews": {"x/y/cuff": {}},
                        "records": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                prepare_json_review(source)

    def test_holdout_input_is_rejected_before_opening(self) -> None:
        """A sealed-looking path is rejected even when it does not exist."""
        with self.assertRaises(SystemExit):
            main(
                [
                    "--output",
                    "unused",
                    "--validate-csv",
                    "holdout_SEALED_DO_NOT_OPEN.csv",
                ]
            )


if __name__ == "__main__":
    unittest.main()
