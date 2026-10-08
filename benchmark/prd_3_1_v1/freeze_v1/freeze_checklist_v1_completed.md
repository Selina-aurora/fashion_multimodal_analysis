# PRD 3.1.1 Final Core Freeze Checklist v1 — COMPLETED

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
