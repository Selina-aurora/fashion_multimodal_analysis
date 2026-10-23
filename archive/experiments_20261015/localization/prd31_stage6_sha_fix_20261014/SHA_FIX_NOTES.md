# Stage 6 exact SHA correction (2026-10-14)

The original Stage 6 package had a **single-character transcription error** in the expected validation-manifest SHA256. It failed during `--dry-run` before loading any models.

- Input: `configs/prd_8class_val_v2_full_targets.csv`
- Erroneous SHA256: `39141f545ea5512d16ed074788b6f98474b33faff111d3b663ec0ffcc4c40b93`
- Correct SHA256: `39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93`
- Cross-checked against the user's successful Oct 14 GPU preflight, V5 Core400 regression and GPU chain run reports.
- All other experiment logic/data selection/thresholds untouched; offline tests add assertions for all three protected-input hashes.
- No training data, validation data, Core400, weights, or project source files are changed.

**This patch fixes a tool package expectation; it does not assert validation data passed on a fresh run.** Run dry-run first, then full run.
