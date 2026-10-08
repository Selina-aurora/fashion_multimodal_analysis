# Grounding Localization Evaluation - 2026-09-10

## 1. Objective

This experiment evaluates the current full-image Grounding DINO baseline for
PRD 3.1.2, language-guided local region localization. The goal is to determine
whether contextual clothing prompts improve fine-grained localization for
`sleeve`, `collar`, `button`, and `zipper`, and to identify failure modes before
the next model iteration.

## 2. Experimental Setup

- **Dataset:** DeepFashion2 training images
- **Evaluation pool:** 300 randomly sampled images
- **Sampling seed:** 42
- **Repeated groups:** 3 non-overlapping groups × 100 images
- **Model:** `IDEA-Research/grounding-dino-tiny`
- **Detection threshold:** 0.3
- **Targets:** `sleeve`, `collar`, `button`, `zipper`

### Prompt configurations

| Target | Baseline | Context |
| --- | --- | --- |
| sleeve | `sleeve` | `shirt sleeve` |
| collar | `collar` | `shirt collar` |
| button | `button` | `clothing button` |
| zipper | `zipper` | `clothing zipper` |

## 3. Evaluation Definitions

### 3.1 Target-label prediction rate

The percentage of images for which Grounding DINO emits at least one
target-related prediction. This metric measures **prediction coverage**, not
localization accuracy.

### 3.2 Bounding-box area diagnostics

For the highest-confidence target-related prediction in each image:

```text
bbox_area_ratio = bbox_area / full_image_area
```

The following diagnostics are reported:

- Mean bbox area ratio
- Median bbox area ratio
- P90 bbox area ratio
- Large Box Rate: bbox area ratio >= 0.50
- Very Large Box Rate: bbox area ratio >= 0.80

### 3.3 Manual localization audit

A stratified audit set was sampled from emitted predictions:

```text
4 targets × 2 prompt types × 5 cases = 40 cases
```

Each case was manually classified as:

- `correct`: bbox tightly localizes the intended clothing part
- `coarse`: bbox contains the intended part but includes substantial irrelevant area
- `wrong`: bbox localizes the wrong region or the target is absent

This audit evaluates sampled emitted predictions and **must not be interpreted
as full-dataset localization accuracy**.

## 4. Results

### 4.1 Prediction coverage across three groups

Values are mean ± sample standard deviation across three independent
100-image groups.

| Target | Baseline | Context |
| --- | ---: | ---: |
| sleeve | 62.7% ± 5.8% | 89.7% ± 4.0% |
| collar | 52.3% ± 4.0% | 81.0% ± 1.0% |
| button | 3.7% ± 0.6% | 3.3% ± 1.5% |
| zipper | 2.3% ± 0.6% | 2.7% ± 1.2% |

Context prompts substantially increase prediction coverage for `sleeve` and
`collar`, while `button` and `zipper` remain difficult.

### 4.2 Bounding-box quality diagnostics

| Target | Prompt | N | Mean area | Median area | Large box | Very large box |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sleeve | baseline | 188 | 0.282 | 0.205 | 19.1% | 3.2% |
| sleeve | context | 269 | 0.323 | 0.229 | 24.9% | 7.4% |
| collar | baseline | 157 | 0.315 | 0.207 | 28.0% | 10.2% |
| collar | context | 243 | 0.354 | 0.259 | 31.3% | 11.9% |
| button | baseline | 11 | 0.232 | 0.048 | 18.2% | 18.2% |
| button | context | 10 | 0.431 | 0.274 | 40.0% | 30.0% |
| zipper | baseline | 7 | 0.104 | 0.034 | 0.0% | 0.0% |
| zipper | context | 8 | 0.171 | 0.046 | 12.5% | 0.0% |

For `sleeve` and `collar`, contextual prompting increases coverage but also
increases the proportion of coarse, large-area predictions. The `button` and
`zipper` statistics should be interpreted cautiously because the number of
emitted predictions is small.

### 4.3 40-case manual localization audit

| Target | Prompt | Correct | Coarse | Wrong |
| --- | --- | ---: | ---: | ---: |
| sleeve | baseline | 0/5 | 2/5 | 3/5 |
| sleeve | context | 0/5 | 1/5 | 4/5 |
| collar | baseline | 0/5 | 0/5 | 5/5 |
| collar | context | 0/5 | 0/5 | 5/5 |
| button | baseline | 0/5 | 1/5 | 4/5 |
| button | context | 0/5 | 2/5 | 3/5 |
| zipper | baseline | 0/5 | 0/5 | 5/5 |
| zipper | context | 0/5 | 0/5 | 5/5 |

Overall audit result:

- Correct: 0 / 40
- Coarse: 6 / 40
- Wrong: 34 / 40

Observed failure modes include:

- whole-garment or whole-image boxes
- head / face / hat confusion for collar
- background and non-clothing-object confusion for button
- shoe, belt-buckle, and bag confusion for zipper

## 5. Key Findings

1. Contextual clothing prompts improve **prediction coverage** for sleeve and
   collar, but coverage improvement does not imply localization accuracy.
2. Additional context-induced detections contain a higher proportion of large
   or coarse boxes.
3. Full-image grounding is strongly affected by irrelevant regions such as
   head, footwear, accessories, and background.
4. Button and zipper require a target-balanced evaluation set because random
   sampling yields too few reliable positive cases.
5. A more constrained visual search space is required before expanding the
   evaluation.

## 6. Next Experiment: ROI-conditioned Grounding Ablation

The next iteration will connect PRD 3.1.1 and 3.1.2:

```text
Full image
    ↓
Garment ROI from 3.1.1
    ↓
Garment crop
    ↓
Grounding DINO
    ↓
Local part bbox
```

The first ablation will reuse the current 40 audit cases:

```text
Full-image Grounding
vs.
ROI-conditioned Grounding
```

Primary comparison metrics:

- Correct Localization Rate
- Coarse Localization Rate
- Wrong Localization Rate
- bbox area diagnostics
- inference latency

If ROI conditioning improves the audit results, the experiment will be
expanded to the 100-image and 300-image evaluation sets.

## 7. Related Outputs

- `reports/grounding_group_evaluation/bbox_quality_summary.csv`
- `reports/grounding_group_evaluation/bbox_quality_details.csv`
- `reports/grounding_group_evaluation/manual_localization_audit_reviewed.csv`
- `scripts/analyze_grounding_bbox_quality.py`
- `scripts/build_grounding_manual_audit.py`
