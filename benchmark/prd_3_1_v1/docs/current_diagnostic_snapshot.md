# Current Diagnostic Snapshot — 2026-09-22

This file records current **diagnostic** results only. They are not yet the frozen product benchmark.

## Full-validation predicted-ROI propagation pilot

Matched ROI pairs: 22.

Matched category coverage:
- top 7
- pants 5
- skirt 0
- outerwear 1
- dress 5
- shoe 1
- bag 2
- accessory 1

GT-ROI vs predicted-ROI label agreement:
- primary_color: 15/18 = 0.8333
- pattern: 14/18 = 0.7778
- sleeve_length: 8/13 = 0.6154
- neckline: 7/13 = 0.5385
- silhouette_fit: 16/18 = 0.8889
- fashion_style: 11/18 = 0.6111

## ROI-quality diagnostic

- bbox IoU vs aggregate attribute stability Pearson r: -0.3506
- mask IoU vs aggregate attribute stability Pearson r: +0.4651

Small-n descriptive result only.

## BBox / mask isolation ablation

Reference A = GT bbox + GT mask.

Agreement with A:

- sleeve_length:
  - B Pred bbox + Pred mask: 8/13 = 0.6154
  - C GT bbox + Pred mask: 9/13 = 0.6923
  - D Pred bbox + GT mask: 12/13 = 0.9231
- neckline:
  - B: 7/13 = 0.5385
  - C: 7/13 = 0.5385
  - D: 10/13 = 0.7692
- silhouette_fit:
  - B: 16/18 = 0.8889
  - C: 16/18 = 0.8889
  - D: 15/18 = 0.8333
- fashion_style:
  - B: 11/18 = 0.6111
  - C: 11/18 = 0.6111
  - D: 14/18 = 0.7778

Current diagnostic interpretation:
For sleeve_length and neckline, predicted mask quality is a stronger propagation-error source than bbox crop quality on this matched set. This remains a controlled diagnostic, not human-GT attribute accuracy.

Expected source paths in the project:
- `reports/prd_integration/predicted_roi_attributes_fullval_v1/`
- `reports/prd_integration/roi_mask_isolation_v1/`
