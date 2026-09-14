# Script Index

The `scripts/` directory contains experiment and analysis entry points. Reusable primitives belong under `src/fashion_multimodal_analysis/`.

## 3.1.1 — instance segmentation / ROI

- `run_mask2former_baseline.py`: COCO-pretrained Mask2Former baseline inference.
- `check_deepfashion2_dataset.py`: DeepFashion2 dataset sample validation.
- `check_deepfashion2_dataloader.py`: DataLoader validation.
- `check_mask2former_forward.py`: one training forward-pass check.
- `check_mask2former_train_step.py`: one optimization-step check.
- `evaluate_person_roi_batch.py`: initial person-ROI batch experiment.
- `evaluate_person_roi_adaptive_batch.py`: adaptive-margin ROI experiment.
- `evaluate_person_roi_fallback_batch.py`: fallback behavior diagnostics.
- `evaluate_person_roi_quality_batch.py`: ROI-quality analysis.

## 3.1.2 — language-guided localization

### Baseline and prompt studies

- `run_grounding_dino_baseline.py`
- `run_grounding_dino_multi_test.py`
- `run_grounding_dino_prompt_optimization.py`
- `test_prompt_variants.py`
- `evaluate_prompt_variants.py`

### Larger evaluation and failure analysis

- `build_grounding_eval_samples.py`
- `run_grounding_group_evaluation.py`
- `analyze_grounding_bbox_quality.py`
- `build_grounding_manual_audit.py`
- `analyze_grounding_failure_patterns.py`
- `run_roi_conditioned_grounding.py`

### Verified-positive benchmark

- `build_target_balanced_candidates.py`
- `build_verified_positive_benchmark.py`
- `evaluate_verified_positive_benchmark.py`: initial FULL/ROI comparison.
- `evaluate_verified_positive_benchmark_v2.py`: FULL/ROI/LOCAL pilot.
- `evaluate_verified_positive_benchmark_v3.py`: frozen tight-window v3 evaluation used for the final 88-case automatic result.
- `build_gpu_bundle_v3.py`: minimal GPU transfer bundle builder.

## Utilities

- `visualize_annotation.py`
- `convert_annotation_to_masks.py`
- `visualize_grounding_group_results.py`
- `generate_grounding_contact_sheets.py`

Large generated files belong under `outputs/` and are ignored by Git. Structured experiment results belong under `reports/`.

## Coding-standard note

This directory preserves experiment/reproducibility entry points created during rapid iteration. Some historical scripts are intentionally longer than the coding-standard recommendation for reusable modules. New reusable logic should be extracted into `src/fashion_multimodal_analysis/`; the core package and tests are checked by CI. See [`../docs/coding_standard_compliance.md`](../docs/coding_standard_compliance.md).
