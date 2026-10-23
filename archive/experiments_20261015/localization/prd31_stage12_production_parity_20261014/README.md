# Stage 12: Production-aligned DINO/SAM output parity (read-only)

This corrects Stage 10's invalid comparison: production `ground()` runs in strict mode with raw garment crop and `REGION_PROMPTS`, semantic `phrase_matches` filtering, maximum region crop-area gate, and `constrain_local_mask`. Stage 10's experimental arm did not preserve those rules.

The script compares the **live runtime `ModelBackends.ground()`** against a separate, production-aligned batched path. It also compares each native single DINO call with batched DINO on the **same exact image and prompt** before processing masks.

This is an exploratory development test, NOT formal 3.1.2 localization accuracy or 30ms acceptance. The strict parity script re-runs native DINO during the batched arm for safety, so the resulting group timings **cannot be used to claim any speedup**. If required runtime helpers are absent or any DINO outputs do not pass tolerance, the experiment stops and provides the failure report. No changes to model weights, project files, Core400, labels or official pipeline.

## GPU

Upload `prd31_stage12_production_parity_20261014.zip` to `/workspace`.

```bash
cd /workspace
python -m zipfile -e prd31_stage12_production_parity_20261014.zip /workspace/prd31_stage12_production_parity_20261014
cd /workspace/fashion_multimodal_analysis
export HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage12_production_parity_20261014/scripts/run_stage12.py --dry-run
# Only after DRY_RUN_PASS:
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage12_production_parity_20261014/scripts/run_stage12.py
```

Upload the `UPLOAD_THIS_ZIP` file regardless of success or failure.

Expected files: `strict_dino_parity.csv`, `final_region_parity.csv`, `strict_protocol_parity_summary.json`, `diagnostic_group_timing.csv`. These files are **not** for accuracy claims. If parity holds on 4 cases, the next task is wider independent sampling and separate optimized timing without serial verification overhead.
