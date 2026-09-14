# 3.1.2 Language-guided Local Region Localization

## 1. Objective

The module accepts a fashion image plus a natural-language description and returns a candidate local region with a confidence score. Unlike fixed part segmentation, the target vocabulary is not limited to a predefined clothing-part list.

```text
Fashion image + text prompt
        ↓
Grounding DINO
        ↓
Candidate bounding box + score
        ↓
Optional ROI / local-window diagnostic
        ↓
Manual localization-quality review
```

Current model: `IDEA-Research/grounding-dino-tiny`.

## 2. Why Grounding DINO is used as the baseline

Grounding DINO directly supports open-vocabulary text-guided detection and can therefore test whether descriptions such as `sleeve`, `shirt collar`, or `front zipper closure` can be grounded without training a fixed fashion-part detector.

The baseline is intentionally diagnostic. A returned box does not prove that the requested fine-grained part is correctly localized.

## 3. Evaluation history

### 3.1 Pilot prompt and threshold experiments

Initial experiments compared category/part/detail/spatial prompts and detection thresholds from 0.2 to 0.5. These experiments established that prompt wording changes model output and that overly strict thresholds can increase misses.

Early prompt-improvement percentages are treated as **prediction coverage only**, not localization accuracy.

### 3.2 300-image repeated evaluation

Three non-overlapping groups of 100 DeepFashion2 images were used to compare baseline and clothing-context prompts for sleeve, collar, button and zipper.

Main finding: contextual prompts increase emitted predictions for sleeve/collar but also increase large/coarse boxes. Button and zipper remain rare and weak under random sampling.

See [`experiments/2026-09-10_grounding_localization_evaluation.md`](experiments/2026-09-10_grounding_localization_evaluation.md).

### 3.3 Verified-positive benchmark

Because random samples frequently did not actually contain the queried target, a verified-positive benchmark was built by manual screening:

| Target | Verified cases |
| --- | ---: |
| sleeve | 22 |
| collar | 22 |
| button | 22 |
| zipper | 22 |
| **Total** | **88** |

Three diagnostic conditions are compared:

- **FULL**: original image;
- **ROI**: annotation-assisted garment ROI (oracle diagnostic, not deployment ground truth);
- **LOCAL**: fixed target-conditioned local windows inside the ROI.

The LOCAL windows are deterministic spatial priors. They do not use a fine-grained ground-truth part bbox.

## 4. Frozen v3 local-window configuration

Ratios are relative to the garment ROI.

| Target | Relative windows `(x1, y1, x2, y2)` |
| --- | --- |
| sleeve | `(0.00, 0.02, 0.36, 0.75)`, `(0.64, 0.02, 1.00, 0.75)` |
| collar | `(0.20, 0.00, 0.80, 0.34)` |
| button | three overlapping vertical windows centered around the front torso |
| zipper | three overlapping vertical windows spanning most of garment height |

Exact values are recorded in [`../configs/grounding_v3.json`](../configs/grounding_v3.json) and `src/fashion_multimodal_analysis/grounding/config.py`.

The v3 window geometry was frozen after a small sanity check and was not repeatedly tuned against final 88-case results.

## 5. Final automatic 88-case results

| Target | FULL | ROI | LOCAL |
| --- | ---: | ---: | ---: |
| sleeve | 40.9% | 40.9% | 36.4% |
| collar | 68.2% | 68.2% | 45.5% |
| button | 18.2% | 18.2% | 9.1% |
| zipper | 4.5% | 4.5% | 9.1% |
| **Overall** | **33.0%** | **33.0%** | **25.0%** |

Mean top-box area ratio among detected cases:

- FULL: `0.963`
- ROI: `0.963`
- LOCAL: `0.219`

Interpretation:

1. FULL and ROI are effectively the same on this benchmark.
2. FULL/ROI detections are often extremely large and cannot be counted as precise part localization.
3. LOCAL makes detections spatially smaller but does not improve overall coverage.
4. Sleeve and collar are easier than button and zipper.
5. Tight cropping is retained as a mixed/negative diagnostic result.

## 6. Manual localization quality

Because the benchmark verifies target presence but has no fine-grained target bounding-box ground truth, localization quality is manually reviewed with four labels:

- `correct`: appropriate target localization;
- `coarse`: target is included, but the box is too broad;
- `wrong`: a prediction exists but localizes the wrong region;
- `missed`: no prediction.

Reporting definitions:

```text
Strict Accuracy = correct / N
Usable Rate     = (correct + coarse) / N
Missed Rate     = missed / N
```

The current audit under `reports/verified_positive_evaluation/final_analysis/` is still marked **preliminary** because low-confidence cases require final human confirmation. It must not be presented as a final localization-accuracy table until that review is complete.

## 7. Current failure modes

- garment-level or near-full-image boxes;
- head/face/hat confusion around collar prompts;
- background or non-clothing-object confusion for button;
- footwear/accessory confusion for zipper;
- misses caused by very small target scale;
- language-visual semantic association without precise boundary isolation.

## 8. Current follow-up requested by mentor

Before freezing 3.1.2:

1. finalize per-class manual localization metrics;
2. compare generic, garment-specific and explicit part-specific prompts on a small controlled subset;
3. test segmentation-first tight crop instead of repeatedly shrinking fixed windows;
4. inspect attention/response behavior where practical to identify input-side vs cross-modal alignment failure;
5. treat button/zipper as small-object targeted diagnostics rather than requiring immediate parity with sleeve/collar.

## 9. Main files

- `scripts/evaluate_verified_positive_benchmark_v3.py`
- `scripts/build_verified_positive_benchmark.py`
- `scripts/build_target_balanced_candidates.py`
- `scripts/analyze_grounding_failure_patterns.py`
- `reports/verified_positive_benchmark/`
- `reports/verified_positive_evaluation/`
- `configs/grounding_v3.json`
