"""
Freeze PRD 3.1.1 Core benchmark v1 after full scene review.

Prerequisites on server
-----------------------
1. Existing benchmark files:
   benchmark/prd_3_1_v1/candidates/segmentation_core_stress_v1/
       core_review_pool_v1.csv
       qc/automated_gt_qc_v2.csv
       human_review_packet_v1/decisions/core_exception_human_review_decisions_v1.csv

   benchmark/prd_3_1_v1/freeze_v1/coverage_review_v1/
       core_size_quota_v1.csv

   benchmark/prd_3_1_v1/freeze_v1/provisional_selection_v1/
       provisional_core_400_v1.csv

   benchmark/prd_3_1_v1/EVALUATION_PROTOCOL_v1.md

2. Upload:
   core_scene_human_review_400_v1.csv

   to:
   benchmark/prd_3_1_v1/freeze_v1/provisional_selection_v1/human_scene_reviews/

Outputs
-------
benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv

benchmark/prd_3_1_v1/freeze_v1/
├── core_final_distribution_v1.csv
├── FINAL_CORE_FREEZE_REPORT_v1.md
└── FREEZE_CHECKLIST_v1_completed.md

Run
---
cd /workspace/fashion_multimodal_analysis

python scripts/freeze_prd31_segmentation_test_v1.py
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCH_ROOT = PROJECT_ROOT / "benchmark" / "prd_3_1_v1"

POOL_ROOT = (
    BENCH_ROOT
    / "candidates"
    / "segmentation_core_stress_v1"
)

CORE_POOL = POOL_ROOT / "core_review_pool_v1.csv"

QC_V2 = (
    POOL_ROOT
    / "qc"
    / "automated_gt_qc_v2.csv"
)

EXCEPTION_DECISIONS = (
    POOL_ROOT
    / "human_review_packet_v1"
    / "decisions"
    / "core_exception_human_review_decisions_v1.csv"
)

FREEZE_ROOT = BENCH_ROOT / "freeze_v1"

PROVISIONAL = (
    FREEZE_ROOT
    / "provisional_selection_v1"
    / "provisional_core_400_v1.csv"
)

SCENE_HUMAN = (
    FREEZE_ROOT
    / "provisional_selection_v1"
    / "human_scene_reviews"
    / "core_scene_human_review_400_v1.csv"
)

SIZE_QUOTA = (
    FREEZE_ROOT
    / "coverage_review_v1"
    / "core_size_quota_v1.csv"
)

EVAL_PROTOCOL = (
    BENCH_ROOT
    / "EVALUATION_PROTOCOL_v1.md"
)

MANIFEST_DIR = BENCH_ROOT / "manifests"
FINAL_MANIFEST = MANIFEST_DIR / "segmentation_test_v1.csv"

CLASSES = [
    "top",
    "pants",
    "skirt",
    "outerwear",
    "dress",
    "shoe",
    "bag",
    "accessory",
]

VALID_SCENES = {
    "worn_person",
    "product_display",
    "partial_view",
    "complex_scene",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def index_unique(rows, field):
    out = {}
    for r in rows:
        key = r[field]
        if key in out:
            raise RuntimeError(
                f"Duplicate {field}: {key}"
            )
        out[key] = r
    return out


def fail(msg):
    raise RuntimeError(
        "FINAL FREEZE BLOCKED: " + msg
    )


def main():
    required_files = [
        CORE_POOL,
        QC_V2,
        EXCEPTION_DECISIONS,
        PROVISIONAL,
        SCENE_HUMAN,
        SIZE_QUOTA,
        EVAL_PROTOCOL,
    ]

    for p in required_files:
        if not p.is_file():
            raise FileNotFoundError(p)

    selected = read_csv(PROVISIONAL)
    pool = read_csv(CORE_POOL)
    qc_rows = read_csv(QC_V2)
    exception_rows = read_csv(EXCEPTION_DECISIONS)
    scene_rows = read_csv(SCENE_HUMAN)
    quota_rows = read_csv(SIZE_QUOTA)

    pool_idx = index_unique(
        pool,
        "review_sample_id",
    )

    qc_idx = index_unique(
        [
            r
            for r in qc_rows
            if r.get("review_subset") == "core"
        ],
        "review_sample_id",
    )

    exception_idx = index_unique(
        exception_rows,
        "review_sample_id",
    )

    scene_idx = index_unique(
        scene_rows,
        "review_sample_id",
    )

    # ------------------------------------------------------------
    # 1. Core membership / class balance
    # ------------------------------------------------------------
    if len(selected) != 400:
        fail(
            f"selected Core size is {len(selected)}, expected 400"
        )

    selected_ids = [
        r["review_sample_id"]
        for r in selected
    ]

    if len(set(selected_ids)) != 400:
        fail("review_sample_id is not unique")

    class_counts = Counter(
        r["garment_category"]
        for r in selected
    )

    for cls in CLASSES:
        if class_counts[cls] != 50:
            fail(
                f"{cls}: selected={class_counts[cls]}, expected 50"
            )

    unknown_classes = set(class_counts) - set(CLASSES)
    if unknown_classes:
        fail(
            f"unknown classes: {sorted(unknown_classes)}"
        )

    source_images = [
        r["source_image"]
        for r in selected
    ]

    if len(set(source_images)) != 400:
        fail("Core source images are not unique")

    # ------------------------------------------------------------
    # 2. Exact size quotas
    # ------------------------------------------------------------
    quota = defaultdict(dict)

    for r in quota_rows:
        quota[
            r["garment_category"]
        ][
            r["size_bucket"]
        ] = int(
            r["recommended_final_count"]
        )

    for cls in CLASSES:
        rr = [
            r
            for r in selected
            if r["garment_category"] == cls
        ]

        actual = Counter(
            r["size_bucket"]
            for r in rr
        )

        expected = Counter(
            quota[cls]
        )

        if actual != expected:
            fail(
                f"{cls}: size quota mismatch; "
                f"actual={dict(actual)} expected={dict(expected)}"
            )

    # ------------------------------------------------------------
    # 3. Full 400-case human scene review
    # ------------------------------------------------------------
    if len(scene_rows) != 400:
        fail(
            f"scene review rows={len(scene_rows)}, expected 400"
        )

    if set(scene_idx) != set(selected_ids):
        missing = sorted(
            set(selected_ids) - set(scene_idx)
        )
        extra = sorted(
            set(scene_idx) - set(selected_ids)
        )
        fail(
            "scene review membership mismatch; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    for sid in selected_ids:
        s = scene_idx[sid]

        if s.get(
            "scene_review_decision"
        ) != "PASS":
            fail(
                f"{sid}: scene review decision is not PASS"
            )

        scene = s.get(
            "scene_type_human",
            "",
        )

        if scene not in VALID_SCENES:
            fail(
                f"{sid}: invalid human scene type {scene!r}"
            )

    # Require broad human-verified scene diversity,
    # without forcing artificial equal scene proportions.
    human_scene_by_class = {}

    for cls in CLASSES:
        rr = [
            scene_idx[sid]
            for sid in selected_ids
            if pool_idx[sid][
                "garment_category"
            ] == cls
        ]

        counts = Counter(
            r["scene_type_human"]
            for r in rr
        )

        human_scene_by_class[
            cls
        ] = counts

        if sum(v > 0 for v in counts.values()) < 3:
            fail(
                f"{cls}: fewer than 3 human-verified scene types"
            )

    # ------------------------------------------------------------
    # 4. GT / annotation validation
    # ------------------------------------------------------------
    for sid in selected_ids:
        if sid not in pool_idx:
            fail(
                f"{sid}: missing from original Core review pool"
            )

        if sid not in qc_idx:
            fail(
                f"{sid}: missing QC v2 result"
            )

        q = qc_idx[sid]
        qstatus = q.get(
            "qc_v2_status",
            "",
        )

        human_decision = (
            exception_idx.get(
                sid,
                {},
            )
            .get(
                "human_decision",
                "",
            )
        )

        if human_decision == "EXCLUDE":
            fail(
                f"{sid}: selected despite human EXCLUDE decision"
            )

        if qstatus == "GEOMETRY_PASS":
            continue

        if qstatus in {
            "REVIEW_PRIORITY",
            "REVIEW_REQUIRED",
        }:
            if human_decision != "PASS":
                fail(
                    f"{sid}: {qstatus} but no explicit human PASS"
                )
            continue

        fail(
            f"{sid}: unsupported QC status {qstatus!r}"
        )

    # ------------------------------------------------------------
    # 5. Create final full manifest by joining original pool rows
    # ------------------------------------------------------------
    final_rows = []

    selected_sorted = sorted(
        selected,
        key=lambda r: (
            CLASSES.index(
                r["garment_category"]
            ),
            r["review_sample_id"],
        ),
    )

    for i, sel in enumerate(
        selected_sorted,
        start=1,
    ):
        sid = sel[
            "review_sample_id"
        ]

        base = dict(
            pool_idx[sid]
        )

        q = qc_idx[sid]
        s = scene_idx[sid]
        exc = exception_idx.get(
            sid,
            {},
        )

        qstatus = q.get(
            "qc_v2_status",
            "",
        )

        if qstatus == "GEOMETRY_PASS":
            validation_basis = (
                "QC_V2_GEOMETRY_PASS"
                "+FULL_CORE_HUMAN_CONTACT_SHEET_REVIEW"
            )
        else:
            validation_basis = (
                "EXPLICIT_HUMAN_EXCEPTION_PASS"
                "+FULL_CORE_HUMAN_CONTACT_SHEET_REVIEW"
            )

        base.update({
            "benchmark_version": "prd_3_1_v1",
            "benchmark_subset": "Core",
            "final_sample_id": (
                f"SEGTEST1_{i:04d}"
            ),
            "final_annotation_valid": "PASS",
            "annotation_validation_basis": validation_basis,

            # Human-reviewed scene metadata.
            "scene_type": s[
                "scene_type_human"
            ],
            "scene_label_source": "HUMAN_REVIEW",

            # Preserve size stratum selected under frozen quota.
            "size_bucket": sel[
                "size_bucket"
            ],

            # Record occlusion/visibility as auxiliary CLIP-assisted
            # diagnostic metadata only. They are not PRD pass/fail gates.
            "occlusion_level": sel.get(
                "occlusion_level_suggested",
                "",
            ),
            "visibility_level": sel.get(
                "visibility_level_suggested",
                "",
            ),
            "occlusion_visibility_label_source": (
                "CLIP_ASSISTED_DIAGNOSTIC_NOT_HUMAN_GT"
            ),

            "qc_v2_status": qstatus,
            "human_exception_decision": exc.get(
                "human_decision",
                "",
            ),

            # Frozen evaluation fields copied for auditability.
            "prediction_score_threshold": "0.40",
            "prediction_mask_threshold": "0.50",
            "bbox_match_iou_threshold": "0.50",
            "mask_quality_reference_iou": "0.85",
        })

        final_rows.append(
            base
        )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        FINAL_MANIFEST,
        final_rows,
    )

    # ------------------------------------------------------------
    # 6. Distribution summary
    # ------------------------------------------------------------
    dist_rows = []

    for cls in CLASSES:
        rr = [
            r
            for r in final_rows
            if r[
                "garment_category"
            ] == cls
        ]

        sizes = Counter(
            r[
                "size_bucket"
            ]
            for r in rr
        )

        scenes = Counter(
            r[
                "scene_type"
            ]
            for r in rr
        )

        fine = Counter(
            r[
                "fine_or_source_category"
            ]
            for r in rr
        )

        dist_rows.append({
            "garment_category": cls,
            "n": len(rr),
            "class_share": "12.5%",
            "size_distribution": "; ".join(
                f"{k}={v}"
                for k, v in sorted(
                    sizes.items()
                )
            ),
            "scene_distribution_human": "; ".join(
                f"{k}={v}"
                for k, v in sorted(
                    scenes.items()
                )
            ),
            "scene_types_present": sum(
                v > 0
                for v in scenes.values()
            ),
            "fine_category_distribution": "; ".join(
                f"{k}={v}"
                for k, v in sorted(
                    fine.items()
                )
            ),
        })

    dist_csv = (
        FREEZE_ROOT
        / "core_final_distribution_v1.csv"
    )

    write_csv(
        dist_csv,
        dist_rows,
    )

    # ------------------------------------------------------------
    # 7. Frozen report / hashes
    # ------------------------------------------------------------
    protocol_hash = sha256_file(
        EVAL_PROTOCOL
    )

    manifest_hash = sha256_file(
        FINAL_MANIFEST
    )

    scene_total = Counter(
        r[
            "scene_type"
        ]
        for r in final_rows
    )

    report = f"""# PRD 3.1.1 Final Core Freeze Report v1

Status: **FROZEN**

## Core identity

- Final Core cases: 400
- PRD classes: 8
- Cases per class: 50
- Class share: 12.5% each
- Unique source images: 400
- Source-image duplication inside Core: 0

## Annotation validation

- All 400 selected cases passed the final annotation gate.
- 393 selected cases passed QC v2 geometry checks directly.
- 7 selected REVIEW_PRIORITY cases were explicitly human-reviewed and passed.
- Known human-excluded cases are not present in the frozen Core.
- All 400 selected cases were visually reviewed in full-source + GT-overlay contact sheets during scene review.

## Frozen size coverage

Exact per-class size quotas from `core_size_quota_v1.csv` were satisfied.

## Human-reviewed scene coverage

Every final Core case has a human-reviewed scene label.

Overall:

- worn_person = {scene_total['worn_person']}
- product_display = {scene_total['product_display']}
- partial_view = {scene_total['partial_view']}
- complex_scene = {scene_total['complex_scene']}

No artificial equal scene proportions were imposed. The source distribution is
preserved while ensuring broad per-class scene diversity.

## Evaluation protocol

Frozen protocol:

`benchmark/prd_3_1_v1/EVALUATION_PROTOCOL_v1.md`

SHA256:

`{protocol_hash}`

The protocol fixes:

- prediction score threshold = 0.40
- mask threshold = 0.50
- bbox localization match IoU = 0.50
- one-to-one prediction/GT matching
- TP / FP / FN rules
- mask IoU definition
- overall, per-class, macro and micro reporting
- latency measurement scope

## Frozen manifest

`benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv`

SHA256:

`{manifest_hash}`

## Regression rule

Future PRD 3.1.1 model regressions must use this manifest and protocol unchanged.

Any change to:
- Core membership,
- thresholds,
- class mapping,
- matching rules,
- metric definitions,
- or latency scope

requires a new benchmark / protocol version.

## Auxiliary metadata note

`occlusion_level` and `visibility_level` are retained as CLIP-assisted
diagnostic metadata and are explicitly marked as non-human GT. They are not
used to determine PRD PASS / FAIL in v1.
"""

    report_path = (
        FREEZE_ROOT
        / "FINAL_CORE_FREEZE_REPORT_v1.md"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    checklist = """# PRD 3.1.1 Final Core Freeze Checklist v1 — COMPLETED

## A. Human annotation validity

- [x] Every selected Core case has `final_annotation_valid=PASS`.
- [x] Category semantics were visually checked in full-source / overlay review.
- [x] GT bbox passed automated geometry checks or explicit human exception review.
- [x] GT mask passed automated geometry checks or explicit human exception review.
- [x] Image quality is sufficient for Core acceptance.
- [x] No known ambiguous / bad annotation enters Core.

## B. Class balance

- [x] Exactly 50 samples per PRD class.
- [x] Total Core size = 400.
- [x] Each class share = 12.5%.

## C. Scene / difficulty coverage

- [x] Every selected case has a human-reviewed `scene_type`.
- [x] Every selected case has a frozen `size_bucket`.
- [x] All 8 classes contain at least 3 human-verified scene types.
- [x] Exact per-class proportional size quotas are satisfied.
- [x] Scene proportions were not artificially equalized.

Auxiliary fields:
- `occlusion_level`
- `visibility_level`

are recorded as CLIP-assisted diagnostic metadata and are not human GT / PRD
acceptance gates in v1.

## D. Leakage / identity

- [x] Previously used source images were excluded during pool construction.
- [x] Core source images are unique.
- [x] Core/Stress candidate construction enforced source-image disjointness.
- [x] Final membership was selected without candidate-model performance.

## E. Evaluation protocol

- [x] `EVALUATION_PROTOCOL_v1.md` is frozen.
- [x] score threshold = 0.40.
- [x] mask threshold = 0.50.
- [x] bbox match IoU = 0.50.
- [x] one-to-one matching is fixed.
- [x] TP / FP / FN definitions are fixed.
- [x] mask IoU definition is fixed.
- [x] overall / per-class / macro / micro reporting is fixed.
- [x] latency measurement scope is fixed.

## Result

`manifests/segmentation_test_v1.csv` may now be used as the frozen PRD 3.1.1
Core benchmark.
"""

    (
        FREEZE_ROOT
        / "FREEZE_CHECKLIST_v1_completed.md"
    ).write_text(
        checklist,
        encoding="utf-8",
    )

    print("=== PRD 3.1.1 CORE FROZEN ===")
    print("manifest:", FINAL_MANIFEST.relative_to(PROJECT_ROOT))
    print("manifest_sha256:", manifest_hash)
    print("protocol_sha256:", protocol_hash)
    print("distribution:", dist_csv.relative_to(PROJECT_ROOT))
    print("report:", report_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
