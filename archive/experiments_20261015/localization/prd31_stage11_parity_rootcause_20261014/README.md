# Stage 11: Why Stage 10 final-region parity failed

**This is a diagnostic script, NOT a fix or production patch.** Stage10 4 cases / 16 queries showed 0/16 end-to-end parity despite 1.2768x multi-query throughput. Stage10's experimental path omits the application's later guards. Also DINO candidates diverged far more than numerical tolerance; this must be isolated before any optimization integration.

This package compares three DINO calls for the same predicted garment ROI and text query: (1) the repository's `run_grounding_dino`, (2) Stage9 helper serial, (3) Stage9 helper batch. It also records the actual application `backend.ground` status and saves the source used by the GPU runtime (plus function hashes and relevant keywords). No SAM A/B benchmark is rerun; the application call is used only for status tracing.

**GPU**: Upload ZIP to `/workspace`. Then:

```bash
cd /workspace
python -m zipfile -e prd31_stage11_parity_rootcause_20261014.zip /workspace/prd31_stage11_parity_rootcause_20261014
cd /workspace/fashion_multimodal_analysis
export HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage11_parity_rootcause_20261014/scripts/run_stage11.py --dry-run
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage11_parity_rootcause_20261014/scripts/run_stage11.py
```

Upload the exact path printed as `UPLOAD_THIS_ZIP` from the FULL (non-dry-run) command. Output outside repository, no training or changes to Core400/checkpoints. Failing experiments also attempt to produce an error ZIP.

**Interpretation**: If native serial vs helper batch matches but the production pipeline differs, prioritize restoring missing runtime postprocess/guard logic, not retuning DINO. If native vs batch diverges, check input preprocessing, DINO output-to-query mapping and candidate ranking before SAM. All diagnostics are small-sample, without human-reviewed part GT; they do not confer accuracy or latency acceptance.
