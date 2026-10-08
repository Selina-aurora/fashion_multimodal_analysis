# Core part detector v1

## Purpose

Small-supervision feasibility pilot for part-level bbox localization.

## Classes

- collar
- cuff
- hem

## Model

`fasterrcnn_mobilenet_v3_large_fpn` pretrained on COCO.

## Training

- epochs: 5
- batch size: 2
- learning rate: 0.001
- weight decay: 0.0005
- score threshold: 0.25
- IoU threshold: 0.5
- seed: 20260914
- device: cpu

## Data policy

- Only manually labeled rows are used.
- Unusable rows are excluded.
- `hem_train_03` is conservatively excluded from v1 because its semantic
  target remains visually atypical/ambiguous.
- Validation is grouped by original image name to avoid image leakage.
- Exact duplicate image files across region annotations are merged at runtime.
- Horizontal flip is the only augmentation.

## Interpretation

This is a feasibility experiment. The frozen 15-case PRD diagnostic set is
not used for training or model selection and remains reserved for later
held-out evaluation.
