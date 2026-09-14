# Fashion Multimodal Analysis — Project Documentation

> **Status note (2026-09-14):** This repository is a current engineering snapshot, not a claim that final PRD acceptance thresholds have already been met. See `docs/prd_compliance_matrix.md` for the explicit gap analysis.


_Last updated: 2026-09-14_

## 1. Project goal

The project builds a fine-grained visual and multimodal analysis pipeline for e-commerce fashion images. The current milestone is PRD 3.1, which converts raw fashion images into structured visual evidence that can later support multimodal question answering and semantic generation.

The intended 3.1 pipeline is:

```text
Input fashion image
        ↓
3.1.1 Instance segmentation / garment or person ROI
        ↓
3.1.2 Language-guided local region localization
        ↓
3.1.3 Fine-grained attribute extraction
        ↓
Structured visual representation
        ↓
3.2 Multimodal semantic analysis
```

## 2. Repository organization

```text
fashion_multimodal_analysis/
├── configs/                  # Recorded experiment/baseline configurations
├── data/                     # Local-only smoke-test data placeholder
├── docs/                     # Maintained development and experiment documentation
├── reports/                  # Structured, version-controlled evaluation outputs
├── scripts/                  # Experiment / analysis entry points
├── src/fashion_multimodal_analysis/
│   ├── datasets/             # DeepFashion2 dataset utilities
│   └── grounding/            # Reusable 3.1.2 utilities
├── tests/                    # Lightweight unit tests
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

Large raw datasets, model weights, Hugging Face caches, generated image-heavy outputs, GPU transfer bundles, and virtual environments are excluded from Git.

## 3. Environment and dataset

Recommended Python version: 3.10 or later.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Current scripts expect DeepFashion2 in a sibling directory:

```text
parent_directory/
├── fashion_multimodal_analysis/
└── fashion_data/
    └── raw/
        └── train/
            └── train/
                ├── image/
                └── annos/
```

## 4. Module 3.1.1 — Instance segmentation and ROI

### 4.1 Baseline

The current baseline uses:

- `facebook/mask2former-swin-tiny-coco-instance`;
- DeepFashion2 annotations for dataset / target-pipeline validation;
- independent binary masks for multiple garment instances;
- person ROI as a practical upstream spatial constraint.

The COCO-pretrained model is treated as a workflow baseline, not as a final DeepFashion2-trained fashion-category segmentation model.

### 4.2 ROI development

Three ROI strategies were evaluated:

1. direct person mask;
2. person bbox + fixed margin;
3. person bbox + adaptive margin.

Direct person masks can remove skirts, shoes, bags, or garment extremities. Fixed 15% margins can become too wide for close-up people. The current adaptive rule is:

| Person bbox area ratio | Margin |
| --- | ---: |
| `< 0.30` | 20% |
| `0.30 - 0.60` | 10% |
| `>= 0.60` | 5% |

Current remaining edge cases include extreme close-ups and multi-person target selection.

Detailed development history: `docs/instance_segmentation_development.md`.

## 5. Module 3.1.2 — Language-guided local region localization

### 5.1 Baseline

Model: `IDEA-Research/grounding-dino-tiny`.

Input/output:

```text
Fashion image + natural-language prompt
        ↓
Grounding DINO
        ↓
Candidate bbox + confidence score
```

The module is intended to support open descriptions such as sleeve, collar, button, zipper, spatial part descriptions, and other future local concepts.

### 5.2 Experiment progression

The development sequence was:

```text
Grounding DINO baseline
        ↓
Prompt granularity + threshold pilot
        ↓
Prompt optimization / ablation
        ↓
300-image repeated evaluation
        ↓
40-case manual audit + bbox diagnostics
        ↓
ROI-conditioned grounding
        ↓
Target-balanced candidate generation
        ↓
88-case verified-positive benchmark
        ↓
FULL / ROI / LOCAL v3 controlled evaluation
```

### 5.3 Why a verified-positive benchmark was needed

Early random evaluation mixed true localization failures with images where the queried target was absent. A new benchmark was therefore manually screened so that every selected case contains a visible, usable target.

| Target | Positive cases |
| --- | ---: |
| sleeve | 22 |
| collar | 22 |
| button | 22 |
| zipper | 22 |
| **Total** | **88** |

### 5.4 Controlled conditions

- **FULL**: original image.
- **ROI**: annotation-assisted garment ROI; diagnostic/oracle condition.
- **LOCAL**: fixed target-conditioned tight windows inside the ROI.

The LOCAL windows are coarse spatial priors. They do not use fine-grained part ground truth.

### 5.5 Final 88-case automatic result

| Target | FULL | ROI | LOCAL |
| --- | ---: | ---: | ---: |
| sleeve | 40.9% | 40.9% | 36.4% |
| collar | 68.2% | 68.2% | 45.5% |
| button | 18.2% | 18.2% | 9.1% |
| zipper | 4.5% | 4.5% | 9.1% |
| **Overall** | **33.0%** | **33.0%** | **25.0%** |

Mean top-box area ratio among detected cases:

- FULL: 0.963
- ROI: 0.963
- LOCAL: 0.219

### 5.6 Interpretation

These automatic rates measure prediction coverage, not fine-grained localization accuracy.

The maintained conclusion is:

1. FULL and ROI are effectively identical on this benchmark.
2. Many FULL/ROI boxes are garment-level or near-full-image boxes.
3. LOCAL reduces predicted region size substantially but does not improve overall prediction coverage.
4. Sleeve and collar are easier than button and zipper.
5. Button and zipper are the main small-object failure modes.
6. Tight local cropping is a mixed/negative result and is not reported as a successful performance improvement.

### 5.7 Manual quality protocol

Manual labels:

- `correct`;
- `coarse`;
- `wrong`;
- `missed`.

Metrics:

```text
Strict Accuracy = correct / N
Usable Rate     = (correct + coarse) / N
Missed Rate     = missed / N
```

The current 88-case manual audit is still marked preliminary until low-confidence cases are finalized.

### 5.8 Current mentor-feedback diagnostics

Before 3.1.2 is frozen, the current plan is to:

- finalize per-class manual localization metrics;
- compare generic / garment-specific / explicit part-specific prompts on a small controlled subset;
- test segmentation-first tight cropping;
- inspect attention/response behavior where technically practical;
- run only targeted small-object checks for button / zipper instead of repeated tuning on the final benchmark.

Detailed maintained document: `docs/language_guided_localization.md`.

## 6. Module 3.1.3 — Fine-grained attribute extraction

Status: implementation is planned for the current development week and is not yet reported as complete.

Initial attributes:

- color;
- pattern / texture;
- sleeve length / sleeve type;
- collar / neckline;
- closure;
- confidence / uncertain state.

Target workflow:

```text
Image / garment ROI / localized region
        ↓
Attribute extraction baseline
        ↓
Structured JSON
        ↓
Post-processing and fallback
```

The first batch will use roughly 50-80 images, followed by parameter/prompt adjustments and a larger 100-150 sample evaluation if time permits.

## 7. Definition of the current 3.1 milestone

3.1 is considered consolidated when:

- 3.1.1, 3.1.2, and 3.1.3 can all run;
- baseline parameters are recorded and frozen;
- each module has structured evaluation and error analysis;
- interfaces are consistent;
- an end-to-end example is reproducible;
- source code, configs, reports, and Markdown documentation are synchronized.

## 8. Important files

### Configuration

- `configs/instance_segmentation_baseline.json`
- `configs/grounding_v3.json`

### Source

- `src/fashion_multimodal_analysis/datasets/`
- `src/fashion_multimodal_analysis/grounding/`

### Key scripts

- `scripts/run_mask2former_baseline.py`
- `scripts/evaluate_person_roi_adaptive_batch.py`
- `scripts/build_target_balanced_candidates.py`
- `scripts/build_verified_positive_benchmark.py`
- `scripts/evaluate_verified_positive_benchmark_v3.py`

### Final structured 3.1.2 result

- `reports/verified_positive_evaluation/case_results.csv`
- `reports/verified_positive_evaluation/target_summary.csv`
- `reports/verified_positive_evaluation/scale_group_summary.csv`
- `reports/verified_positive_evaluation/final_analysis/verified_positive_v3_analysis.md`

## 9. Reproducibility and repository policy

The project separates reproducible code/structured results from large generated artifacts:

- code, configs, reports, and maintained documentation are committed;
- `outputs/` is regenerated locally or on GPU and is ignored;
- raw DeepFashion2 data is stored outside the repository;
- GPU transfer bundles are temporary and ignored;
- virtual environments are never committed.

This keeps the repository reviewable while preserving the information required to reproduce the recorded experiments.
