# Core Part Detector v1 — Final Closeout Summary

**Date:** 2026-09-15  
**Scope:** PRD 3.1.2 part-level localization feasibility pilot  
**Classes:** collar / cuff / hem  
**Model:** Faster R-CNN MobileNetV3-Large FPN, COCO-pretrained  
**Selected diagnostic checkpoint:** `best_loss_model.pt`, epoch 5, train loss 0.2866699162

## 1. Final conclusion

The supervised detector is **not a complete failure**. It learned meaningful localization signals for **collar** and **hem**, including transfer to the validation split. The original strict validation setting (`score >= 0.25`, `IoU >= 0.5`) masked this because correct boxes were generally assigned low confidence.

The remaining bottlenecks are:

- **Confidence under-calibration:** at score threshold 0.25, both TRAIN and VAL produced zero true positives.
- **Candidate explosion at low thresholds:** lowering the score threshold exposes many usable boxes, but also hundreds or thousands of false positives.
- **Cuff remains unresolved:** unlike collar and hem, cuff has weak candidate generation and very poor score ranking.

Therefore, the current supervised detector should be treated as a **feasibility result rather than a production-ready PRD acceptance model**.

## 2. Threshold diagnosis

On validation:

| External score threshold | TP | FP | FN | Macro Recall | Image Success |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0 | 0 | 10 | 0.000 | 0.000 |
| 0.05 | 2 | 39 | 8 | 0.222 | 0.250 |
| 0.01 | 4 | 632 | 6 | 0.556 | 0.500 |
| 0.001 | 5 | 2065 | 5 | 0.622 | 0.500 |

Interpretation: useful localization is present, but the detector's absolute confidence scale is too low for the original 0.25 threshold.

## 3. Top-k ranking diagnosis

Per-class ranking is much more useful than a naive low score threshold.

Validation Top-5 keeps the same overall TP/FN and macro recall as score >= 0.01:

- `score >= 0.01`: 4 TP / 632 FP / 6 FN, macro recall 0.556
- `Top-5 per class`: 4 TP / 116 FP / 6 FN, macro recall 0.556
- FP reduction: **81.6%**
- Maximum retained candidates: 15 per image

This confirms that a large part of the failure mode is **candidate overload**, not total absence of useful boxes.

## 4. Class-level findings

### Collar

- Validation candidate-oracle recall: **0.667** (2/3 GTs have at least one same-class IoU >= 0.5 candidate)
- Top-5 validation recall: **0.667**
- Mean first-good candidate rank on validation: **2.5**
- Practical conclusion: **positive feasibility signal**. Top-5 candidate ranking is sufficient to retain both detected validation positives.

### Hem

- Validation candidate-oracle recall: **1.000** (2/2)
- Top-3 validation recall: **1.000**
- Mean first-good candidate rank on validation: **2.0**
- Practical conclusion: **strongest class in this pilot**, but the validation support is only 2 GT boxes, so this is not a final performance claim.

### Cuff

- Validation candidate-oracle recall: **0.200** (1/5)
- Top-1 / Top-3 / Top-5 validation recall: **0.000**
- The only validation cuff with a valid same-class IoU >= 0.5 candidate appears at **score rank 30**
- Training cuff candidate-oracle recall: **0.412**
- Training median first-good rank: **14**
- Practical conclusion: **ranking alone cannot solve cuff**. The class requires additional supervision, stronger class-specific training, better spatial priors, or a different localization strategy.

## 5. Recommended 3.1.2 closeout decision

For the current small-supervision pilot:

- **collar:** retain **Top-5** candidate ranking as the feasible selection rule.
- **hem:** retain **Top-3** candidate ranking as the feasible selection rule.
- **cuff:** mark as **unresolved / not reliable in v1**.
- Do not continue brute-force threshold lowering or Top-10/20/30 expansion, because that reintroduces the original candidate explosion.
- Preserve the current results as diagnostic evidence and stop tuning on this small split.

## 6. Project interpretation

The experiment supports the following engineering conclusion:

> Fine-grained supervised localization is feasible for some structured garment parts, but performance is highly class-dependent under very limited supervision. Candidate ranking substantially improves usability for collar and hem, while cuff remains limited by candidate quality and class ranking rather than only confidence thresholding.

This is a **feasibility finding**, not a PRD acceptance result.

## 7. Next step

Close the current detector-tuning loop and proceed to **PRD 3.1.3 fine-grained attribute extraction**. Keep the detector outputs as optional local-region candidates rather than making all downstream attribute extraction depend on perfect part detection.

For future detector work, prioritize:
1. additional cuff annotations;
2. cuff-specific spatial/structural priors;
3. class-aware candidate suppression / NMS;
4. calibration only after candidate quality improves.

