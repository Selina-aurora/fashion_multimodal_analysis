# Next Build Order

Do not move to PRD 3.2 until the PRD 3.1 benchmark is frozen and runnable.

## Phase 1 — 3.1.1 benchmark completion

- populate `segmentation_test_v1.csv`
- verify all 8 classes have sufficient frozen test coverage
- freeze mask GT
- generate overall / per-class / per-source / object-size metrics
- freeze balanced baseline as main reference and no-balance as ablation reference
- rerun latency using the common protocol

## Phase 2 — 3.1.2 human-GT benchmark

- create reviewed local-region GT
- use exact frozen queries
- annotate strict bbox/mask targets
- add ambiguous/excluded flags
- run strict accuracy + latency
- retain coarse/usable only as secondary diagnostic

## Phase 3 — 3.1.3 human-GT attribute benchmark

- freeze attribute label vocabularies
- annotate applicability
- annotate human GT
- run Track A and Track B
- report per-attribute + micro + macro

## Phase 4 — final regression / acceptance runner

Build one command that produces:
- acceptance_report.csv
- per_category_metrics.csv
- per_attribute_metrics.csv
- latency_summary.csv
- failure_cases.csv
- regression_comparison.csv
- machine-readable run metadata
- final markdown summary
