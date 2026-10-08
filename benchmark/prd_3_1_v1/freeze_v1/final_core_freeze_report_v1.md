# PRD 3.1.1 Final Core Freeze Report v1

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

- worn_person = 192
- product_display = 8
- partial_view = 72
- complex_scene = 128

No artificial equal scene proportions were imposed. The source distribution is
preserved while ensuring broad per-class scene diversity.

## Evaluation protocol

Frozen protocol:

`benchmark/prd_3_1_v1/EVALUATION_PROTOCOL_v1.md`

SHA256:

`79706d4e9422e7b09aa29275f05be15c9bb1fd2a5de0bbec402b035b8a1fd248`

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

`c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89`

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
