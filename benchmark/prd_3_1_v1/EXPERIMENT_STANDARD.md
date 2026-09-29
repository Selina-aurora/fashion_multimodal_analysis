# PRD 3.1 Experiment Standard v1

## 1. Versioning

Benchmark ID: `prd_3_1_v1`

Any modification to:
- benchmark samples,
- ground-truth labels,
- label definitions,
- metric definitions,
- acceptance thresholds,
- evaluation preprocessing,

must increment the benchmark version.

Model/checkpoint changes do **not** change the benchmark version; they create a new run ID.

## 2. Data split policy

Maintain three logically separate splits:

- `train`: model fitting only.
- `development`: threshold selection, prompt engineering, ablation, hyperparameter tuning.
- `benchmark_test`: frozen acceptance and regression testing only.

Once a sample enters `benchmark_test`, it may not be used to tune that system version.

## 3. Reproducibility record

Every benchmark run must save the fields in `templates/run_metadata_v1.json`, including:

- run ID and timestamp
- code commit / version
- model name and checkpoint
- benchmark version
- random seed
- device and software environment
- preprocessing configuration
- thresholds
- latency protocol
- output paths

Default benchmark seed for deterministic sampling utilities: `20260922`.

## 4. Common evaluation rules

### 4.1 Model state

- Load model once before timing.
- Use inference/eval mode.
- Disable gradient computation.
- No test-time augmentation unless a benchmark profile explicitly enables it.
- Do not alter thresholds after viewing benchmark-test results.

### 4.2 Latency protocol

Current research reference hardware should be recorded explicitly; current experiments use an RTX 4090 reference environment.

For comparable latency:
- batch size = 1
- model already loaded
- 10 warm-up inferences before timing
- evaluate the full frozen benchmark set
- repeat timed benchmark passes 5 times when feasible
- include preprocessing + model inference + postprocessing
- exclude model loading, dependency loading, network download, and report-file writing
- synchronize CUDA immediately before and after each timed inference
- report mean, median, p95, sample count, hardware and precision mode

Product release acceptance must also be repeated on the designated deployment hardware.

## 5. 3.1.1 Instance segmentation protocol

### Frozen label space

Eight categories:
`top, pants, skirt, outerwear, dress, shoe, bag, accessory`.

### Frozen matching / inference defaults for v1

- score threshold: `0.40`
- binary mask threshold: `0.50`
- bbox match threshold: IoU `>= 0.50`
- matching: greedy one-to-one within the same image
- class-correct metrics require both bbox match and category match

If a later model requires different thresholds, select them on the development set and register a new evaluation profile before running benchmark_test.

### Required metrics

Overall and by category:
- GT count
- prediction count
- bbox recall @ 0.50
- localized category accuracy
- class-correct end-to-end recall
- mean mask IoU
- median mask IoU
- mask IoU >= 0.85 count/rate
- mean / median / p95 latency
- false-positive count
- missed-GT count
- failure taxonomy counts

Also report by source dataset and object-size bucket.

## 6. 3.1.2 Language-guided local-region localization protocol

Frozen region label space:
`collar, cuff, hem, pocket, shoulder, waist, pattern, decoration`.

Each test row must contain a fixed natural-language query and human-reviewed target annotation.

### Primary strict metric

`strict_localization_success = 1` only when:
- the predicted target corresponds to the requested semantic region, and
- predicted bbox IoU with the human GT local-region bbox is `>= 0.50`.

If a local-region mask is annotated, also report mask IoU.

### Secondary diagnostic metric

`usable_localization_success` may be recorded for product-analysis purposes when the target is semantically correct but coarse.

This secondary label must **not** replace the strict acceptance metric.

### Required outputs

Per case:
- query
- expected region
- predicted region / status
- bbox IoU
- mask IoU if applicable
- strict success
- usable success
- latency
- error type
- reviewer status

## 7. 3.1.3 Fine-grained attribute protocol

Attribute evaluation must be applicability-aware.

A sample can be:
- `eligible`
- `not_applicable`
- `ambiguous`
- `excluded_with_reason`

Only `eligible` and non-ambiguous samples enter attribute accuracy denominators.

### Two mandatory tracks

**Track A — isolated module**
`human/GT ROI -> attribute extractor -> human GT attribute`

Measures intrinsic attribute-model ability.

**Track B — end-to-end**
`original image -> predicted garment ROI -> attribute extractor -> human GT attribute`

Measures product-chain performance.

### Additional diagnostic track

`GT-ROI model output vs Pred-ROI model output`

Measures propagation consistency. It is **not** human-GT accuracy.

### Required metrics

For each attribute:
- eligible count
- correct count
- accuracy
- ambiguous count
- not-applicable count
- confidence summary if available
- mean / median / p95 latency

Report micro and macro summaries separately.

## 8. 3.1 end-to-end diagnostic protocol

Maintain controlled bbox/mask isolation when investigating regressions:

- A = GT bbox + GT mask
- B = Pred bbox + Pred mask
- C = GT bbox + Pred mask
- D = Pred bbox + GT mask

Interpretation:
- C substantially below D -> mask error is the stronger contributor.
- D substantially below C -> bbox crop error is the stronger contributor.
- both degraded -> interaction or both components contribute.

This is a diagnostic experiment, not the primary product acceptance test.

## 9. Regression rules

Every candidate release is compared against:
1. the PRD acceptance threshold;
2. the currently frozen production/main baseline;
3. the immediately previous accepted release.

A regression report must list:
- improved metrics
- degraded metrics
- unchanged metrics
- newly failed cases
- newly fixed cases

Never approve a release based only on overall averages if a required category disappears from the test coverage.
