# Stage 8 — SAM same-image multi-box feasibility A/B (2026-10-14)

This is an **experimental GPU profiling package**, NOT an accepted algorithm optimization. It compares serial SAM calls with one SAM call that takes multiple box prompts for the **same garment crop**. The prompts are real Grounding DINO candidate boxes from the fixed eight-image development diagnostic. DINO inference is excluded from the A/B latency; the model and PRD threshold settings are unchanged.

Both sides use the same SamProcessor/SamModel, multimask selection by best IoU score, and original crop. Outputs are compared as binary-mask IoU per box and score difference. If this Transformers version does not support the multi-box output shape, it reports an error ZIP instead of guessing. It does not modify the project, checkpoints, GT, source images, or integration API.

## GPU commands

Upload this ZIP under `/workspace`.

```bash
cd /workspace
python -m zipfile -e prd31_stage8_sam_batch_20261014.zip /workspace/prd31_stage8_sam_batch_20261014
cd /workspace/fashion_multimodal_analysis
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage8_sam_batch_20261014/scripts/run_stage8.py --dry-run
```

Check `DRY_RUN_PASS` before running:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage8_sam_batch_20261014/scripts/run_stage8.py
```

Upload the new `UPLOAD_THIS_ZIP` even on failure. Results are saved under `/workspace/prd31_stage8_runs_20261014/`, outside the repository.

## Interpretation

- Stage 7 already measured Grounding DINO 124.76ms/call and SAM 107.35ms/invoked call. SAM was invoked only for 40/80 calls. This is not per-query latency acceptance.
- The Stage 8 experiment considers a *same-image multiple-box* setting only. Even an excellent batch speedup would NOT show that one independent text+image query is under 30ms, because Grounding DINO is still run separately for every query.
- Output parity requires 0.995+ mask IoU **per compared prompt** as an exploratory guard, not a semantic correctness metric. Even if passed, the backend should not be changed until all regions/GT acceptance and edge cases are separately evaluated.
- Some models or processor versions may not implement multi-box prompts compatibly. A failed feasibility experiment does not affect any earlier Core400, Stage 6 or Stage 7 results.
