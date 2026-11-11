"""基准集管理：保留训练、开发、固定回归/审核划分，冻结后的标签不能随结果调整。

Prepare PRD 3.1.1 final Core-freeze review + freeze Evaluation Protocol v1.

Prerequisites
-------------
Place these files under:
benchmark/prd_3_1_v1/candidates/segmentation_core_stress_v1/
human_review_packet_v1/decisions/

- core_priority_human_review_decisions_v1.csv
- core_required_human_review_decisions_v1.csv
- core_exception_human_review_decisions_v1.csv
- stress_human_review_decisions_v1.csv

This script DOES NOT freeze the final 400 Core cases yet.
It:
1. merges current Core human decisions with QC v2;
2. prepares a final human-confirmation / coverage-annotation table;
3. summarizes class + size coverage;
4. writes the frozen evaluation protocol for future regression runs;
5. creates a freeze checklist.

Outputs
-------
benchmark/prd_3_1_v1/freeze_v1/
├── core_final_review_template_v1.csv
├── core_pre_freeze_summary_v1.csv
└── freeze_checklist_v1.md

benchmark/prd_3_1_v1/
└── evaluation_protocol_v1.md

Run
---
cd fashion_multimodal_analysis

python scripts/benchmarking/prepare_prd31_core_freeze_and_protocol_v1.py
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()

BENCH_ROOT = PROJECT_ROOT / "benchmark" / "prd_3_1_v1"
POOL_ROOT = BENCH_ROOT / "candidates" / "segmentation_core_stress_v1"

CORE_POOL = POOL_ROOT / "core_review_pool_v1.csv"
QC_V2 = POOL_ROOT / "qc" / "automated_gt_qc_v2.csv"

DECISION_DIR = POOL_ROOT / "human_review_packet_v1" / "decisions"

CORE_EXCEPTION = DECISION_DIR / "core_exception_human_review_decisions_v1.csv"

FREEZE_DIR = BENCH_ROOT / "freeze_v1"

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

SCENE_TYPES = [
    "worn_person",
    "product_display",
    "partial_view",
    "complex_scene",
]

OCCLUSION_LEVELS = [
    "none",
    "partial",
    "heavy",
]

VISIBILITY_LEVELS = [
    "full",
    "partial",
]


def read_csv(path: Path) -> Any:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
    """
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
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def size_bucket(area_ratio: float) -> str:
    """size bucket。

    Args:
        area_ratio: 面积 比例。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if area_ratio < 0.005:
        return "tiny"
    if area_ratio < 0.02:
        return "small"
    if area_ratio < 0.10:
        return "medium"
    return "large"


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)

    core_rows = read_csv(CORE_POOL)
    qc_rows = read_csv(QC_V2)
    decision_rows = read_csv(CORE_EXCEPTION)

    qc_idx = {
        r["review_sample_id"]: r for r in qc_rows if r.get("review_subset") == "core"
    }

    decision_idx = {r["review_sample_id"]: r for r in decision_rows}

    merged = []

    for row in core_rows:
        sid = row["review_sample_id"]
        qc = qc_idx.get(sid, {})
        dec = decision_idx.get(sid, {})

        human_decision = str(dec.get("human_decision", "")).strip()

        if human_decision == "EXCLUDE":
            pre_freeze_status = "EXCLUDED_BY_HUMAN_REVIEW"
        elif human_decision == "PASS":
            pre_freeze_status = "HUMAN_PASS"
        else:
            pre_freeze_status = "PENDING_FINAL_HUMAN_CONFIRMATION"

        area = float(row["bbox_area_ratio"])

        merged.append(
            {
                "review_sample_id": sid,
                "garment_category": row["garment_category"],
                "fine_or_source_category": row["fine_or_source_category"],
                "source_dataset": row["source_dataset"],
                "source_image": row["source_image"],
                "garment_id": row["garment_id"],
                "gt_mask_path": row["gt_mask_path"],
                "gt_overlay_path": row["gt_overlay_path"],
                "bbox_area_ratio": f"{area:.8f}",
                "size_bucket": size_bucket(area),
                "qc_v2_status": qc.get("qc_v2_status", ""),
                "qc_v2_reasons": qc.get("qc_v2_reasons", ""),
                "qc_v2_advisory": qc.get("qc_v2_advisory", ""),
                "human_exception_decision": human_decision,
                "human_exception_exclude_reason": dec.get("exclude_reason", ""),
                "pre_freeze_status": pre_freeze_status,
                # Human coverage annotation fields to complete before final freeze.
                "final_annotation_valid": (
                    "PASS"
                    if human_decision == "PASS"
                    else ("EXCLUDE" if human_decision == "EXCLUDE" else "")
                ),
                "scene_type": "",
                "occlusion_level": "",
                "visibility_level": "",
                "scene_review_note": "",
                # Filled by the final freeze script only after coverage review.
                "selected_for_core_v1": "",
                "final_sample_id": "",
            }
        )

    write_csv(
        FREEZE_DIR / "core_final_review_template_v1.csv",
        merged,
    )

    summary_rows = []

    for cls in CLASSES:
        rr = [r for r in merged if r["garment_category"] == cls]

        status_counts = Counter(r["pre_freeze_status"] for r in rr)

        size_counts = Counter(
            r["size_bucket"]
            for r in rr
            if r["pre_freeze_status"] != "EXCLUDED_BY_HUMAN_REVIEW"
        )

        eligible_after_known_exclusions = sum(
            1 for r in rr if r["pre_freeze_status"] != "EXCLUDED_BY_HUMAN_REVIEW"
        )

        summary_rows.append(
            {
                "garment_category": cls,
                "candidate_count": len(rr),
                "known_human_pass": status_counts["HUMAN_PASS"],
                "known_human_excluded": status_counts["EXCLUDED_BY_HUMAN_REVIEW"],
                "pending_final_confirmation": status_counts[
                    "PENDING_FINAL_HUMAN_CONFIRMATION"
                ],
                "eligible_after_known_exclusions": eligible_after_known_exclusions,
                "planned_final_count": 50,
                "planned_final_class_share": "12.5%",
                "tiny": size_counts["tiny"],
                "small": size_counts["small"],
                "medium": size_counts["medium"],
                "large": size_counts["large"],
                "coverage_status": (
                    "ENOUGH_FOR_50"
                    if eligible_after_known_exclusions >= 50
                    else "INSUFFICIENT_FOR_50"
                ),
            }
        )

    write_csv(
        FREEZE_DIR / "core_pre_freeze_summary_v1.csv",
        summary_rows,
    )

    protocol = """# PRD 3.1.1 Evaluation Protocol v1

Status: **FROZEN EVALUATION DEFINITION**

This protocol must be used unchanged for all PRD 3.1.1 model regressions
against benchmark `prd_3_1_v1`.

Any change to metric definitions, thresholds, matching, timing scope, class
mapping, or test-set composition requires a new benchmark/protocol version.

## 1. Benchmark subsets

### Core Acceptance

- 8 PRD classes:
  `top, pants, skirt, outerwear, dress, shoe, bag, accessory`.
- Final target: exactly 50 reviewed cases per class.
- Total target: 400 cases.
- Each class therefore contributes 12.5% of the Core benchmark.
- Core determines formal PRD PASS / FAIL.

Final sample selection must consider both class balance and scene coverage.
Within each class, record:
- size bucket;
- scene type;
- occlusion level;
- visibility level.

The final 50/class should avoid a single scene or difficulty stratum
dominating the class when the reviewed candidate pool permits broader
coverage.

### Stress

- Diagnostic only.
- Tiny/small, occluded, partial-view, low-resolution, and other difficult
  cases are intentionally preserved.
- Stress results must never be mixed into Core PRD PASS / FAIL.

## 2. Frozen class mapping

The PRD classes are exactly:

1. top
2. pants
3. skirt
4. outerwear
5. dress
6. shoe
7. bag
8. accessory

For Fashionpedia:

- `shoe` <- category id 23 `shoe`
- `bag` <- category id 24 `bag, wallet`
- `accessory` <- ids 13-22 and 25 according to
  `fashionpedia_to_prd_mapping_v1.json`

No keyword expansion or remapping is allowed inside this benchmark version.

## 3. Inference thresholds

Frozen development-selected defaults:

- prediction score threshold = `0.40`
- binary mask threshold = `0.50`
- bbox localization match threshold = IoU `>= 0.50`

The Core test set must not be used to retune these thresholds.

## 4. BBox IoU definition

For predicted box `P` and GT box `G`:

`BBox IoU = area(P intersection G) / area(P union G)`

Coordinate convention must be consistent inside an evaluation run.
Invalid/negative-area boxes are not valid matches.

## 5. Mask IoU definition

Predicted mask probabilities are binarized at threshold `0.50`.

For binary predicted mask `Mp` and GT mask `Mg`:

`Mask IoU = pixels(Mp AND Mg) / pixels(Mp OR Mg)`

Both masks must be represented in the same original-image coordinate space.

## 6. Prediction-to-GT one-to-one matching

Matching is performed independently within each image.

1. Discard predictions below score threshold `0.40`.
2. Sort retained predictions by confidence score descending.
3. For each prediction in that order:
   - compute bbox IoU against all currently unmatched GT instances;
   - select the unmatched GT with the highest bbox IoU;
   - create a localization match only when the best IoU is `>= 0.50`;
   - otherwise leave the prediction unmatched.
4. A prediction can match at most one GT.
5. A GT can match at most one prediction.

Class label is **not** used to establish the initial localization match.
This is intentional so localization and classification failures can be
measured separately.

## 7. Positive / negative decision rules

### Localization match

A pair is a localization match when:

- it is produced by the frozen one-to-one matching procedure; and
- bbox IoU `>= 0.50`.

### Localized class-correct pair

A localization match is class-correct when:

`predicted_category == GT_category`

### End-to-end True Positive (TP)

An end-to-end TP requires:

- one-to-one bbox localization match with IoU `>= 0.50`; and
- correct PRD category.

### False Negative (FN)

A GT instance is an FN when it has no end-to-end TP.

This includes:
- no prediction/localization match;
- matched spatial prediction but wrong category.

### False Positive (FP)

A prediction is an FP when it is not an end-to-end TP.

This includes:
- unmatched predictions;
- duplicate predictions that cannot obtain a unique GT match;
- localization-matched predictions with the wrong PRD category.

For diagnosis, unmatched, duplicate, and wrong-class FP counts must also be
reported separately.

## 8. Required segmentation metrics

### Localization

`BBox Recall@0.50 = localization-matched GT / all GT`

### Classification after localization

`Localized Category Accuracy =
class-correct localization matches / all localization matches`

### End-to-end detection/classification

`Precision = TP / (TP + FP)`

`Recall = TP / (TP + FN)`

`F1 = 2 * Precision * Recall / (Precision + Recall)`

Also report:

`End-to-End Recall = class-correct matched GT / all GT`

### Mask quality

Primary mask quality is computed on end-to-end TP pairs
(correct bbox localization + correct class), so mask quality is not credited
to a wrong-category detection.

Report:

- mean Mask IoU
- median Mask IoU
- count and rate of Mask IoU `>= 0.85`
- per-class versions of all mask metrics

The PRD v1 mask-quality gate is operationalized as:

`mean Mask IoU >= 0.85`

The `Mask IoU >= 0.85` sample pass rate is mandatory supporting information
and must always be reported.

## 9. Overall and per-class reporting

Every regression report must include:

- overall / micro metrics;
- all eight per-class metrics;
- macro average across the eight classes.

Definitions:

- Micro: aggregate instance counts before computing the metric.
- Macro: compute the metric separately for each PRD class, then take the
  unweighted mean over the eight classes.

A model must not be judged only by the overall average.

## 10. Latency protocol

Reference research hardware must be recorded with every run.

Frozen timing procedure:

- batch size = 1
- model loaded before timing
- inference/eval mode
- gradients disabled
- 10 warm-up inferences
- full Core benchmark timed
- 5 timed passes when feasible
- CUDA synchronize immediately before and after each timed inference
- include preprocessing
- include model forward inference
- include postprocessing
- exclude model loading
- exclude dependency loading/network download
- exclude report/CSV/image writing

Report:

- mean latency
- median latency
- p95 latency
- number of timed samples
- GPU / device
- precision mode
- software environment

Formal PRD segmentation-latency gate:

`mean latency <= 50 ms`

Median and p95 are mandatory diagnostics and must not be omitted.

## 11. Required failure taxonomy

At minimum:

- SEG_MISS
- SEG_FALSE_POSITIVE
- SEG_WRONG_CLASS
- SEG_BBOX_SHIFT
- SEG_MASK_UNDER
- SEG_MASK_OVER
- SEG_MASK_FRAGMENT
- SEG_DUPLICATE
- SEG_SMALL_OBJECT
- SEG_OCCLUSION
- SEG_OTHER

Every failed Core case should be attributable to at least one machine-readable
failure code during regression review.

## 12. Regression discipline

For every candidate model, compare against:

1. PRD thresholds;
2. frozen main baseline;
3. immediately previous accepted version.

Never:
- change Core membership after observing candidate model predictions;
- change a threshold using Core performance;
- remove a hard Core case because a model fails it;
- mix Stress results into Core PASS/FAIL.

If evaluation rules need to change, create `EVALUATION_PROTOCOL_v2` and a
corresponding benchmark version rather than silently changing v1.
"""

    (BENCH_ROOT / "evaluation_protocol_v1.md").write_text(
        protocol,
        encoding="utf-8",
    )

    checklist = """# PRD 3.1.1 Final Core Freeze Checklist v1

Before creating `segmentation_test_v1.csv`, confirm all items below.

## A. Human annotation validity

- [ ] Every selected Core case has `final_annotation_valid=PASS`.
- [ ] Category semantics have been human-confirmed.
- [ ] GT bbox is usable for frozen evaluation.
- [ ] GT mask is usable for frozen evaluation.
- [ ] Image quality is sufficient for Core acceptance.
- [ ] No ambiguous/known-bad annotation enters Core.

## B. Class balance

- [ ] Exactly 50 samples are selected for each PRD class.
- [ ] Total Core size is exactly 400.
- [ ] Each class share is 12.5%.

## C. Scene / difficulty coverage

Each selected case must have:

- [ ] `scene_type`
- [ ] `occlusion_level`
- [ ] `visibility_level`
- [ ] `size_bucket`

Allowed scene types:

- `worn_person`
- `product_display`
- `partial_view`
- `complex_scene`

Allowed occlusion levels:

- `none`
- `partial`
- `heavy`

Allowed visibility levels:

- `full`
- `partial`

Do not force artificial equal proportions when the source pool does not
support them. Instead, preserve broad coverage and document the actual
distribution per class.

## D. Leakage / identity

- [ ] No Core source image appeared in prior train/dev/pilot/ablation configs.
- [ ] Core source images are unique.
- [ ] Core and Stress source images are disjoint.
- [ ] Final membership was chosen without looking at candidate-model results.

## E. Evaluation protocol

- [ ] `evaluation_protocol_v1.md` is frozen.
- [ ] score threshold = 0.40
- [ ] mask threshold = 0.50
- [ ] bbox match IoU = 0.50
- [ ] one-to-one matching is fixed.
- [ ] TP / FP / FN definitions are fixed.
- [ ] mask IoU definition is fixed.
- [ ] overall / per-class / macro / micro reporting is fixed.
- [ ] latency measurement scope is fixed.

Only after all checks pass should a final freeze script write
`manifests/segmentation_test_v1.csv`.
"""

    (FREEZE_DIR / "freeze_checklist_v1.md").write_text(
        checklist,
        encoding="utf-8",
    )

    print("FINISHED")
    for p in [
        FREEZE_DIR / "core_final_review_template_v1.csv",
        FREEZE_DIR / "core_pre_freeze_summary_v1.csv",
        FREEZE_DIR / "freeze_checklist_v1.md",
        BENCH_ROOT / "evaluation_protocol_v1.md",
    ]:
        print(p.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
