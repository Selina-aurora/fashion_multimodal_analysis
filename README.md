# Fashion Multimodal Analysis

Multimodal fine-grained semantic understanding and intelligent analysis for e-commerce fashion images.

This repository contains the current implementation and evaluation artifacts for the PRD 3.1 vision foundation pipeline:

```text
3.1.1 Instance Segmentation / Garment ROI
        ↓
3.1.2 Language-guided Local Region Localization
        ↓
3.1.3 Fine-grained Attribute Extraction
        ↓
3.2 Multimodal Semantic Analysis (next stage)
```

## Current status

Status as of **2026-09-14**:

| Module | Status | Current state |
| --- | --- | --- |
| 3.1.1 Instance segmentation | In consolidation | Mask2Former baseline, DeepFashion2 dataset pipeline, binary masks, person/garment ROI and adaptive-margin diagnostics are implemented. Final parameter consolidation is scheduled in the current development week. |
| 3.1.2 Language-guided localization | Advanced baseline / final diagnostics | Grounding DINO baseline, prompt studies, 300-image evaluation, 40-case audit, 88-case verified-positive benchmark, and FULL/ROI/LOCAL controlled evaluation are completed. Manual per-class accuracy and mentor-requested diagnostic follow-ups are being finalized. |
| 3.1.3 Attribute extraction | Next implementation | Schema, baseline, parameter tuning and batch evaluation are planned for the current development week; no completed production baseline is claimed yet. |
| 3.2 Multimodal analysis | Not started | Starts after the 3.1 pipeline is consolidated. |

See [`docs/project_status.md`](docs/project_status.md) for the detailed progress record.

For a single consolidated overview, see [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md).

The repository intentionally separates **current progress** from **final PRD acceptance**. See [`docs/prd_compliance_matrix.md`](docs/prd_compliance_matrix.md) for requirement-by-requirement status and [`docs/coding_standard_compliance.md`](docs/coding_standard_compliance.md) for the coding-standard migration policy.

## Key 3.1.2 result

The final automatic 88-case verified-positive evaluation contains 22 positive cases for each target: `sleeve`, `collar`, `button`, and `zipper`.

| Target | FULL | ROI | LOCAL |
| --- | ---: | ---: | ---: |
| sleeve | 40.9% | 40.9% | 36.4% |
| collar | 68.2% | 68.2% | 45.5% |
| button | 18.2% | 18.2% | 9.1% |
| zipper | 4.5% | 4.5% | 9.1% |
| **Overall** | **33.0%** | **33.0%** | **25.0%** |

These values are **prediction/detection rates, not localization accuracy**. Visual review shows that many FULL/ROI boxes cover most of the garment or image. Mean detected-box area ratio is approximately 0.963 for FULL/ROI and 0.219 for LOCAL. Therefore the current conclusion is that tighter local cropping reduces box size but does not yet produce a stable overall localization improvement.

Structured results are under [`reports/verified_positive_evaluation/`](reports/verified_positive_evaluation/).

## Repository layout

```text
fashion_multimodal_analysis/
├── configs/                 # Frozen/recorded experiment configurations
├── docs/                    # Module notes, current status, experiment records
├── reports/                 # Structured CSV/JSON/Markdown evaluation outputs
├── scripts/                 # Reproducible experiment and analysis entry points
├── src/fashion_multimodal_analysis/
│   ├── datasets/            # DeepFashion2 dataset / collation utilities
│   └── grounding/           # Reusable grounding configuration and utilities
├── tests/                   # Lightweight unit tests
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

Large generated visualizations, raw datasets, model checkpoints, and virtual environments are intentionally excluded from version control.

## Environment

Recommended Python version: **3.10+**.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For a CUDA server, install the PyTorch build compatible with the server CUDA runtime before installing the remaining dependencies when necessary.

## Dataset layout

The current scripts expect the DeepFashion2 training split in a sibling directory:

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

The dataset itself is not included in this repository.

## Main experiment entry points

### 3.1.1: Mask2Former / ROI diagnostics

```bash
python scripts/run_mask2former_baseline.py
python scripts/evaluate_person_roi_adaptive_batch.py
```

### 3.1.2: verified-positive Grounding DINO evaluation

Four-target smoke test:

```bash
python scripts/evaluate_verified_positive_benchmark_v3.py --per-target-limit 1
```

Full benchmark:

```bash
python scripts/evaluate_verified_positive_benchmark_v3.py
```

The v3 configuration is recorded in [`configs/grounding_v3.json`](configs/grounding_v3.json).

## Evaluation policy

For language-guided localization, the project separates automatic prediction coverage from manual localization quality:

- `correct`: target is appropriately localized;
- `coarse`: target is covered, but the box is too broad;
- `wrong`: a box exists but localizes the wrong region;
- `missed`: no target prediction.

Reporting definitions:

```text
Strict Accuracy = correct / N
Usable Rate     = (correct + coarse) / N
Missed Rate     = missed / N
```

The current 88-case manual audit remains explicitly marked preliminary until low-confidence cases are confirmed.

## Documentation

Start with [`docs/README.md`](docs/README.md).

Important documents:

- [`docs/project_status.md`](docs/project_status.md)
- [`docs/instance_segmentation_development.md`](docs/instance_segmentation_development.md)
- [`docs/language_guided_localization.md`](docs/language_guided_localization.md)
- [`docs/attribute_extraction_plan.md`](docs/attribute_extraction_plan.md)
- [`docs/experiments/2026-09-10_grounding_localization_evaluation.md`](docs/experiments/2026-09-10_grounding_localization_evaluation.md)
- [`docs/experiments/2026-09-11_verified_positive_evaluation.md`](docs/experiments/2026-09-11_verified_positive_evaluation.md)

## Notes on reproducibility

- Final benchmark outputs are stored as structured CSV files under `reports/`.
- Large visualization folders are generated into `outputs/` and ignored by Git.
- The 88-case benchmark uses verified-positive samples, but it does not contain manually annotated fine-grained part bounding boxes; therefore true localization quality still requires manual audit.
- ROI in the 88-case experiment is annotation-assisted/oracle for diagnosis and must not be interpreted as a deployment-time ground-truth signal.
