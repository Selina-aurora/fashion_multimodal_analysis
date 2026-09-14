# Reports

This directory stores structured experiment outputs that are small enough to keep in version control.

## Current authoritative 3.1.2 result

Use `verified_positive_evaluation/` for the final **88-case automatic** FULL/ROI/LOCAL result and `verified_positive_evaluation/final_analysis/` for the preliminary manual audit and interpretation.

Important: automatic detection/prediction coverage is **not** fine-grained localization accuracy.

## Historical / pilot reports

- `grounding_group_evaluation/`: 300-image repeated evaluation and 40-case audit.
- `grounding_failure_analysis/`: early failure taxonomy.
- `roi_conditioned_grounding/`: ROI diagnostic.
- `verified_positive_evaluation_pilot_v2/`: pre-v3 pilot.
- root-level `*_summary.md` files: early small-sample baseline summaries.

Historical pilot metrics are retained for traceability and must not replace the later verified-positive evaluation when reporting the current state.
