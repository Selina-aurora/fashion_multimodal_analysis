# Core part detector top-k ranking diagnostic

## Purpose

Test whether correct collar / cuff / hem boxes are already present among the
model's candidates but are buried below higher-scoring false positives.

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
- top-k values per class: [1, 3, 5]
- IoU threshold: 0.5
- no external score threshold is applied

## Interpretation

- High Top-1 / Top-3 recall:
  candidate ranking is usable and the previous low-threshold FP explosion can
  likely be controlled with ranking / NMS / priors.
- Large gain from Top-1 to Top-5:
  correct boxes exist, but their confidence ranking is weak.
- Low Top-5 recall despite high candidate-oracle recall:
  correct boxes exist but are buried deeper than rank 5.
- Low candidate-oracle recall:
  the detector is not generating a sufficiently accurate same-class box, so
  ranking alone cannot solve the problem.
