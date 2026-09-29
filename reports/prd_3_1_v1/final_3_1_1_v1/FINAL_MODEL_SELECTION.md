# PRD 3.1.1 Final Instance Segmentation Report

## 1. Final decision

**Selected final model: V3-B1 Data Expansion.**

V2-balanced is retained as the reference baseline. V3-B1 is selected because it preserves approximately the same localization recall while improving localized category accuracy, end-to-end correct-class recall, mask quality, and the total number of correct Core-400 instances.

## 2. Frozen evaluation protocol

- Benchmark: Core-400
- Classes: top, pants, skirt, outerwear, dress, shoe, bag, accessory
- 50 GT cases per class; 400 total
- Score threshold: 0.40
- Prediction mask threshold: 0.50
- BBox match IoU: 0.50
- Mask reference threshold: IoU >= 0.85
- Benchmark remained unchanged during V3 ablations

## 3. Ablation experiments

| Experiment | Controlled change | BBox Recall | Category Acc. | E2E Recall | Mask IoU (correct) | Mask>=0.85 | Correct | Missed | Mean ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2-balanced | Reference baseline: class-balanced sampler | 0.3625 | 0.6621 | 0.2400 | 0.7391 | 0.0850 | 84 | 255 | 24.47 |
| V3-A1-highres | Higher input resolution: 800/1280 | 0.3300 | 0.6667 | 0.2200 | 0.7396 | 0.0625 | 79 | 268 | 27.00 |
| V3-A2-smallanchors | All RPN anchor scales shifted smaller | 0.3475 | 0.6763 | 0.2350 | 0.7576 | 0.0900 | 89 | 261 | 25.22 |
| V3-A2b-P2small | Only P2 anchor changed from 32 to 16 | 0.3525 | 0.6879 | 0.2425 | 0.7805 | 0.1025 | 93 | 259 | 24.21 |
| V3-A3-smallsampling | Small-object-aware training sampler | 0.2550 | 0.6176 | 0.1575 | 0.7592 | 0.0525 | 59 | 298 | 24.89 |
| V3-B1-dataexp | Unique-data expansion for underrepresented classes | 0.3600 | 0.6944 | 0.2500 | 0.7677 | 0.0975 | 94 | 256 | 24.71 |

## 4. V2 versus final V3-B1

- BBox50 recall: 0.3625 -> 0.3600
- Localized category accuracy: 0.6621 -> 0.6944
- End-to-end bbox50 + class recall: 0.2400 -> 0.2500
- Mean mask IoU on correct class: 0.7391 -> 0.7677
- Mask IoU >= 0.85 rate: 0.0850 -> 0.0975
- Correct cases: 84 -> 94
- Missed cases: 255 -> 256
- Mean inference: 24.47 ms -> 24.71 ms

## 5. Interpretation

The V3 ablations show that the main remaining limitation is missed detection rather than mask refinement alone. Increasing input resolution did not improve the primary detection metrics. Smaller RPN anchors produced only limited gains and introduced trade-offs across object-size groups. Small-object-aware oversampling degraded overall detection. In contrast, adding new, non-overlapping training examples for underrepresented classes improved classification and mask quality without a meaningful speed penalty.

V3-B1 does not materially improve pure localization recall over V2-balanced; the difference corresponds to approximately one localized GT instance on the 400-case benchmark. Its main benefit is better downstream correctness after localization.

## 6. Final limitation

**Missed detection remains the primary bottleneck.** Further work should therefore prioritize more diverse detection training data and/or a detector specifically optimized for small and weak-boundary garments, rather than continuing threshold, resolution, anchor, or sampler tuning on the frozen benchmark.

## 7. Final artifacts

- Final checkpoint: `outputs/prd_instance_segmentation/maskrcnn_8class_v3_b1_dataexp/checkpoint_last.pth`
- Final train manifest: `configs/prd_8class_train_v3.csv`
- Frozen Core benchmark: `benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv`
- Final report directory: `reports/prd_3_1_v1/final_3_1_1_v1`

## 8. Status

**PRD 3.1.1 instance segmentation: CLOSED for the current phase.**
