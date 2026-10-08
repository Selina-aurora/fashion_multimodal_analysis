# PRD 3.1.1 Final Core Freeze Checklist v1

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

- [ ] `EVALUATION_PROTOCOL_v1.md` is frozen.
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
