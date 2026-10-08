# PRD 3.1.1 Provisional Core Selection v1

**NOT FINAL / NOT YET FROZEN**

## What this file does

- Selects exactly 50 cases per class (400 total).
- Enforces the frozen proportional size quota exactly.
- Tries to preserve fine-category and proposed scene-type proportions.
- Uses no candidate-model predictions or performance.

## Scene metadata review

Total selected cases requiring human scene review:
45

High-confidence spot-check cases:
16

Only `scene_type_suggested` is used for provisional scene-diversity balancing.
Low-confidence scene suggestions must be reviewed before final freeze.

Occlusion / visibility CLIP suggestions are auxiliary metadata only in this
provisional selection and are not used as a selection constraint.

## Human scene labels

Allowed final scene labels:

- worn_person
- product_display
- partial_view
- complex_scene

If the four-way taxonomy is genuinely ambiguous for a case, record
`scene_review_decision=AMBIGUOUS` and do not silently invent a label.

## Final freeze requirements

Do not create `segmentation_test_v1.csv` until:

1. all selected low-confidence scene cases have been human reviewed;
2. high-confidence spot checks are acceptable;
3. selected per-class scene distributions are recomputed using reviewed labels;
4. no class becomes scene-dominated when suitable alternatives exist;
5. Core stays exactly 50/class;
6. the freeze checklist is satisfied.
