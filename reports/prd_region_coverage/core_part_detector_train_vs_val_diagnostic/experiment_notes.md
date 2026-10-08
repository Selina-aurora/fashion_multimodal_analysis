# Core part detector train-vs-val diagnostic

## Purpose

Determine whether the supervised collar / cuff / hem detector can fit the
training set and whether any learned localization transfers to validation.

## Checkpoint

- file: `best_loss_model.pt`
- epoch: 5
- train loss: 0.2866699161628882
- saved macro recall: 0.0
- saved image success: 0.0

## Model

`fasterrcnn_mobilenet_v3_large_fpn`

## Diagnostic settings

- device: cpu
- internal torchvision score threshold: 0.0
- detections per image: 300
- external score thresholds: [0.25, 0.05, 0.01, 0.001]
- IoU threshold: 0.5

## Interpretation guide

- TRAIN good, VAL poor:
  likely small-sample overfitting / insufficient generalization.
- TRAIN poor even at low thresholds:
  detector/training/localization setup has not learned the target boxes well.
- Metrics improve strongly only when score threshold is lowered:
  confidence calibration is a major issue.
- Same-class best IoU stays low across thresholds:
  localization quality itself is the main bottleneck.
