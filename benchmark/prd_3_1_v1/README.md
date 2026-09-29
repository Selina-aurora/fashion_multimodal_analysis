# PRD 3.1 Standard Benchmark & Acceptance Suite v1

This directory is the frozen evaluation framework for PRD 3.1.

## Purpose

It separates four questions that must not be mixed:

1. **Module capability** — can each module work when its input is controlled?
2. **End-to-end capability** — what happens when real predicted outputs are passed downstream?
3. **Acceptance** — does a version satisfy the PRD-defined metric and latency targets?
4. **Diagnosis** — if a version fails, which stage is responsible?

## Scope

- 3.1.1 Garment instance segmentation
- 3.1.2 Language-guided local-region localization
- 3.1.3 Fine-grained visual attribute extraction
- 3.1 end-to-end propagation / regression testing

## Required test artifacts

- frozen manifests under `manifests/`
- human-reviewed annotations under `annotations/`
- fixed schemas under `schemas/`
- run metadata for every benchmark execution
- per-instance predictions, aggregate metrics, failure cases, and latency statistics

## Benchmark rules

- The benchmark test split must not be used for model training, prompt tuning, threshold tuning, or label-set tuning.
- Any change to labels, thresholds, annotation rules, or test samples creates a new benchmark version.
- Every reported metric must be reproducible from a per-instance result table.
- Missing / unsupported / ambiguous cases must be explicit; they must never be silently counted as correct.
- Model loading and disk download time are excluded from latency. Preprocessing, forward inference, and postprocessing are included.
- GPU latency runs require synchronization before/after timed sections.
- Batch size is 1 unless a product deployment profile explicitly defines otherwise.

See:
- `EXPERIMENT_STANDARD.md`
- `ACCEPTANCE_CRITERIA.md`
- `docs/ANNOTATION_GUIDELINE.md`
- `docs/FAILURE_ANALYSIS_STANDARD.md`
