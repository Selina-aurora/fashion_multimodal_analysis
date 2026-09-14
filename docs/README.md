# Documentation Index

This folder contains the maintained project documentation. Historical pilot observations have been consolidated into dated experiment records so that GitHub preview remains readable and current conclusions are not mixed with early small-sample results.

## Current project documentation

- [`project_status.md`](project_status.md): current 3.1 progress and next actions.
- [`instance_segmentation_development.md`](instance_segmentation_development.md): detailed 3.1.1 development record.
- [`language_guided_localization.md`](language_guided_localization.md): maintained 3.1.2 technical document.
- [`attribute_extraction_plan.md`](attribute_extraction_plan.md): 3.1.3 implementation plan; not a completed-result document.
- [`operations/gpu_workflow.md`](operations/gpu_workflow.md): local/GPU execution workflow.

## Experiment records

- [`experiments/2026-09-08_grounding_baseline.md`](experiments/2026-09-08_grounding_baseline.md)
- [`experiments/2026-09-09_prompt_experiments.md`](experiments/2026-09-09_prompt_experiments.md)
- [`experiments/2026-09-10_grounding_localization_evaluation.md`](experiments/2026-09-10_grounding_localization_evaluation.md)
- [`experiments/2026-09-11_verified_positive_evaluation.md`](experiments/2026-09-11_verified_positive_evaluation.md)

## Reporting rule

Early pilot **detection rate** means emitted prediction coverage. It must not be reported as fine-grained localization accuracy. The maintained project conclusion is based on the later 300-image diagnostics, verified-positive benchmark, and manual localization review.

## Compliance and delivery

- [`prd_compliance_matrix.md`](prd_compliance_matrix.md): PRD requirements vs current status and remaining acceptance gaps.
- [`coding_standard_compliance.md`](coding_standard_compliance.md): coding-standard scope, CI checks and progressive migration policy.
