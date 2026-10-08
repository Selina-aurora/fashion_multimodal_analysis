# PRD 3.1.1 Evaluation Protocol v1

Status: **FROZEN EVALUATION DEFINITION**

This protocol must be used unchanged for all PRD 3.1.1 model regressions
against benchmark `prd_3_1_v1`.

Any change to metric definitions, thresholds, matching, timing scope, class
mapping, or test-set composition requires a new benchmark/protocol version.

## 1. Benchmark subsets

### Core Acceptance

- 8 PRD classes:
  `top, pants, skirt, outerwear, dress, shoe, bag, accessory`.
- Final target: exactly 50 reviewed cases per class.
- Total target: 400 cases.
- Each class therefore contributes 12.5% of the Core benchmark.
- Core determines formal PRD PASS / FAIL.

Final sample selection must consider both class balance and scene coverage.
Within each class, record:
- size bucket;
- scene type;
- occlusion level;
- visibility level.

The final 50/class should avoid a single scene or difficulty stratum
dominating the class when the reviewed candidate pool permits broader
coverage.

### Stress

- Diagnostic only.
- Tiny/small, occluded, partial-view, low-resolution, and other difficult
  cases are intentionally preserved.
- Stress results must never be mixed into Core PRD PASS / FAIL.

## 2. Frozen class mapping

The PRD classes are exactly:

1. top
2. pants
3. skirt
4. outerwear
5. dress
6. shoe
7. bag
8. accessory

For Fashionpedia:

- `shoe` <- category id 23 `shoe`
- `bag` <- category id 24 `bag, wallet`
- `accessory` <- ids 13-22 and 25 according to
  `fashionpedia_to_prd_mapping_v1.json`

No keyword expansion or remapping is allowed inside this benchmark version.

## 3. Inference thresholds

Frozen development-selected defaults:

- prediction score threshold = `0.40`
- binary mask threshold = `0.50`
- bbox localization match threshold = IoU `>= 0.50`

The Core test set must not be used to retune these thresholds.

## 4. BBox IoU definition

For predicted box `P` and GT box `G`:

`BBox IoU = area(P intersection G) / area(P union G)`

Coordinate convention must be consistent inside an evaluation run.
Invalid/negative-area boxes are not valid matches.

## 5. Mask IoU definition

Predicted mask probabilities are binarized at threshold `0.50`.

For binary predicted mask `Mp` and GT mask `Mg`:

`Mask IoU = pixels(Mp AND Mg) / pixels(Mp OR Mg)`

Both masks must be represented in the same original-image coordinate space.

## 6. Prediction-to-GT one-to-one matching

Matching is performed independently within each image.

1. Discard predictions below score threshold `0.40`.
2. Sort retained predictions by confidence score descending.
3. For each prediction in that order:
   - compute bbox IoU against all currently unmatched GT instances;
   - select the unmatched GT with the highest bbox IoU;
   - create a localization match only when the best IoU is `>= 0.50`;
   - otherwise leave the prediction unmatched.
4. A prediction can match at most one GT.
5. A GT can match at most one prediction.

Class label is **not** used to establish the initial localization match.
This is intentional so localization and classification failures can be
measured separately.

## 7. Positive / negative decision rules

### Localization match

A pair is a localization match when:

- it is produced by the frozen one-to-one matching procedure; and
- bbox IoU `>= 0.50`.

### Localized class-correct pair

A localization match is class-correct when:

`predicted_category == GT_category`

### End-to-end True Positive (TP)

An end-to-end TP requires:

- one-to-one bbox localization match with IoU `>= 0.50`; and
- correct PRD category.

### False Negative (FN)

A GT instance is an FN when it has no end-to-end TP.

This includes:
- no prediction/localization match;
- matched spatial prediction but wrong category.

### False Positive (FP)

A prediction is an FP when it is not an end-to-end TP.

This includes:
- unmatched predictions;
- duplicate predictions that cannot obtain a unique GT match;
- localization-matched predictions with the wrong PRD category.

For diagnosis, unmatched, duplicate, and wrong-class FP counts must also be
reported separately.

## 8. Required segmentation metrics

### Localization

`BBox Recall@0.50 = localization-matched GT / all GT`

### Classification after localization

`Localized Category Accuracy =
class-correct localization matches / all localization matches`

### End-to-end detection/classification

`Precision = TP / (TP + FP)`

`Recall = TP / (TP + FN)`

`F1 = 2 * Precision * Recall / (Precision + Recall)`

Also report:

`End-to-End Recall = class-correct matched GT / all GT`

### Mask quality

Primary mask quality is computed on end-to-end TP pairs
(correct bbox localization + correct class), so mask quality is not credited
to a wrong-category detection.

Report:

- mean Mask IoU
- median Mask IoU
- count and rate of Mask IoU `>= 0.85`
- per-class versions of all mask metrics

The PRD v1 mask-quality gate is operationalized as:

`mean Mask IoU >= 0.85`

The `Mask IoU >= 0.85` sample pass rate is mandatory supporting information
and must always be reported.

## 9. Overall and per-class reporting

Every regression report must include:

- overall / micro metrics;
- all eight per-class metrics;
- macro average across the eight classes.

Definitions:

- Micro: aggregate instance counts before computing the metric.
- Macro: compute the metric separately for each PRD class, then take the
  unweighted mean over the eight classes.

A model must not be judged only by the overall average.

## 10. Latency protocol

Reference research hardware must be recorded with every run.

Frozen timing procedure:

- batch size = 1
- model loaded before timing
- inference/eval mode
- gradients disabled
- 10 warm-up inferences
- full Core benchmark timed
- 5 timed passes when feasible
- CUDA synchronize immediately before and after each timed inference
- include preprocessing
- include model forward inference
- include postprocessing
- exclude model loading
- exclude dependency loading/network download
- exclude report/CSV/image writing

Report:

- mean latency
- median latency
- p95 latency
- number of timed samples
- GPU / device
- precision mode
- software environment

Formal PRD segmentation-latency gate:

`mean latency <= 50 ms`

Median and p95 are mandatory diagnostics and must not be omitted.

## 11. Required failure taxonomy

At minimum:

- SEG_MISS
- SEG_FALSE_POSITIVE
- SEG_WRONG_CLASS
- SEG_BBOX_SHIFT
- SEG_MASK_UNDER
- SEG_MASK_OVER
- SEG_MASK_FRAGMENT
- SEG_DUPLICATE
- SEG_SMALL_OBJECT
- SEG_OCCLUSION
- SEG_OTHER

Every failed Core case should be attributable to at least one machine-readable
failure code during regression review.

## 12. Regression discipline

For every candidate model, compare against:

1. PRD thresholds;
2. frozen main baseline;
3. immediately previous accepted version.

Never:
- change Core membership after observing candidate model predictions;
- change a threshold using Core performance;
- remove a hard Core case because a model fails it;
- mix Stress results into Core PASS/FAIL.

If evaluation rules need to change, create `EVALUATION_PROTOCOL_v2` and a
corresponding benchmark version rather than silently changing v1.
