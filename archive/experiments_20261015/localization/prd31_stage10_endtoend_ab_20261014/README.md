# Stage 10: Full 3.1.2 Multi-query A/B, October 14, 2026

**Purpose**: compare two *real GPU localization paths* on the same image, instance and query list.

- **A — baseline:** call the repository's `ModelBackends.ground(image, roi, region)` independently for each query. This is the currently implemented DINO+SAM route.
- **B — experimental:** one batch of DINO text queries on an identical isolated garment crop, then one shared SAM image encoding for all detected box prompts. Use the baseline's top-candidate selection, mask/garment intersection and original-image coordinate mapping.

Models, thresholds, images, checkpoint, original project files and frozen benchmarks are **unchanged**. Only temporary outputs under `/workspace/prd31_stage10_runs_20261014` are created.

The results are **multi-query throughput diagnostics**, not per-query latency acceptance, not a semantic accuracy test, and not authorization to change the production code. DINO is still processing multiple copies of the image. GPU memory usage can increase. Slight numerical differences can change proposal cropping and thus SAM masks; parity is explicitly checked and nonmatching outputs prohibit integration.

## GPU usage

Upload `prd31_stage10_endtoend_ab_20261014.zip` into `/workspace`:

```bash
cd /workspace
python -m zipfile -e prd31_stage10_endtoend_ab_20261014.zip /workspace/prd31_stage10_endtoend_ab_20261014
cd /workspace/fashion_multimodal_analysis
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage10_endtoend_ab_20261014/scripts/run_stage10.py --dry-run
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage10_endtoend_ab_20261014/scripts/run_stage10.py
```

If dry-run fails, do not run the last command. Both success and failure runs attempt to produce an `UPLOAD_THIS_ZIP` outside the project. Upload the ZIP for analysis.

## Measurement and parity protocol

- Four eligible predicted garment ROIs, up to four region queries per ROI; 3 warmups per variant and 3 timed passes per case. Alternate order for the paired measurements.
- Measure wall time with GPU synchronization at both boundaries. ROI selection, model loading and disk I/O excluded from timed path.
- Final outputs compared query-by-query: status, original-image bbox, SAM mask IoU, DINO score, SAM score, original text label. Strict mask-screen threshold IoU >=0.999, 1px bbox tolerance, 0.005 score tolerance and exact phrase equivalence.
- This is an exploratory *same-input* regression screen, with only 4 cases; even parity pass does not confer formal 3.1.2 acceptance or implementation clearance.
- If candidate semantic correctness is wrong, unchanged output is *still wrong*. Human reference annotations are necessary to assess accuracy.

## Files

- `scripts/run_stage10.py` main driver.
- `scripts/stage6_helpers.py` input SHA and checkpoint validators borrowed unchanged from previous stages.
- `scripts/stage9_dino_impl.py` DINO serial/batch helper borrowed from Stage 9.
- `scripts/stage8_sam_impl.py` SAM multi-prompt helper borrowed from Stage 8.
- `scripts/run_config.json` and `scripts/selected_images.json` fixed experimental settings.
- `scripts/test_stage10_offline.py` basic CPU-only parity tests.

## Expected result files

- `full_localization_timing.csv`: paired serial and batch timings.
- `final_region_parity.csv`: final region status/box/mask/score agreement.
- `full_localization_summary.json`: summary and integration safety gate.
- `input_audit.json`, `run_audit.json`, `selection.json`, and error trace if failed.
