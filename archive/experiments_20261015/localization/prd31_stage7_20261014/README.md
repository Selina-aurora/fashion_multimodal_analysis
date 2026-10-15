# PRD31 Stage 7: Grounding DINO / SAM Phase Profiling — 2026-10-14

This English-path package starts **after** successful Stage 6 GT-vs-pred-garment-ROI comparison (8 pairs, 21/32 identical attribute labels). It is not training and cannot certify PRD accuracy or latency.

## Existing results (read analysis/ and inputs/)

- Stage 6 matched 8 validation/development garment pairs selected from 63 eligible matches across 32 photos; class and garment bbox were correct for each selected pair.
- **Attribute label agreement 21/32** under GT garment ROI versus predicted garment ROI. This is conditional consistency, NOT accuracy.
- **Both-detected regional queries 11/16**, mean cross-input region bbox IoU 0.6181 on those 11 queries, not part-ground-truth IoU.
- Neckline in paired Stage 6 uses **same fixed top-center proxy** for both sides; NOT GT neckline/collar ROI.
- `analysis/paired_case_review_priority.csv` prioritizes 8 samples. `P0` is triage only, not automatic annotation.

## GPU usage

1. Upload this ZIP into `/workspace`.
2. Extract using Python:

```bash
cd /workspace
python -m zipfile -e prd31_stage7_dino_sam_profile_20261014.zip /workspace/prd31_stage7_20261014
cd /workspace/fashion_multimodal_analysis
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage7_20261014/scripts/run_stage7.py --dry-run
```

3. Only if dry-run says `DRY_RUN_PASS`:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage7_20261014/scripts/run_stage7.py
```

4. Upload the reported `UPLOAD_THIS_ZIP` back to ChatGPT. Stage 7 saves results under `/workspace/prd31_stage7_runs_20261014/`, never under project root.

## Measurements

For each grounded (predicted garment, region query), using the same 8 diagnostic photos as Stage 6, we measure:

- `total_ms`: existing `ModelBackends.ground` (crop + DINO + conditional SAM + post-processing).
- `dino_ms`: wrapped actual `run_grounding_dino` call.
- `sam_ms`: wrapped actual `run_sam` call; **0 if SAM was not called** (e.g. missed DINO proposals).
- `other_overhead_ms`: total minus measured phases, including extra CUDA sync.

10 warmup calls, 5 passes, up to 2 fixed pairs per each of 8 regions (up to 80 timed calls). **Instrumented CUDA synchronization adds overhead.** Therefore this is an internal *bottleneck isolation*, not a head-to-head latency score against the previous 180.72 ms. No model changes, no edits to data/GT/weights, no false accuracy claims.

In a future optimization experiment, change ONLY one independent variable at a time and require mask output to remain valid. The 30 ms PRD target remains unmet based on previous Stage 6 warmed diagnostic.
