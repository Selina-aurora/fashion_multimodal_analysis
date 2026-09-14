# Repository Cleanup — 2026-09-14

- Added root `requirements.txt`, `requirements-dev.txt`, and `pyproject.toml`.
- Rebuilt the root README with current module status, dataset layout, commands and final 88-case automatic results.
- Added a documentation index and a current `project_status.md`.
- Replaced malformed/stale 3.1.2 Markdown notes with maintained GitHub-compatible documents.
- Preserved the detailed 3.1.1 development record and updated its current-stage plan.
- Added reusable 3.1.2 source utilities under `src/fashion_multimodal_analysis/grounding/`.
- Added small unit tests for grounding geometry and manual-quality metrics.
- Synced final 88-case structured CSV reports from the GPU result bundle.
- Replaced the old pilot `verified_positive_evaluation/evaluation_summary.md` with the final 88-case automatic summary.
- Excluded `.venv`, raw datasets, model caches, large `outputs/`, GPU transfer bundles and generated zips from Git tracking.

## v2 compliance pass

- Added a PRD compliance matrix separating current progress from final acceptance thresholds.
- Added coding-standard compliance documentation and GitHub Actions CI.
- Added Black, mypy and pylint to development tooling.
- Normalized reusable grounding utilities and tests to 88-character style, docstrings, type hints and absolute imports.
- Kept frozen/legacy experiment scripts as reproducibility artifacts and documented progressive refactoring expectations.
