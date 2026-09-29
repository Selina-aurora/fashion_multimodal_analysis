# PRD 3.1.1 Threshold Sweep

Checkpoint:
outputs/prd_instance_segmentation/maskrcnn_8class_v3_b1_dataexp/checkpoint_last.pth

Results:

0.25:
bbox50 recall=0.7442

0.30:
bbox50 recall=0.7442

0.35:
bbox50 recall=0.6977

0.45:
bbox50 recall=0.5581

0.50:
bbox50 recall=0.4884

Selected operating point:
score_threshold=0.40

Reason:
Balanced trade-off between localization recall and prediction precision.
