# Stage 9: Grounding DINO multi-query batching diagnostic (2026-10-14)

This is a read-only A/B diagnostic following a successful Stage 8 SAM multi-box feasibility run.

## Stage 8 verified findings
- 2 garment ROI cases and 5 distinct DINO box proposals; serial and batch SAM produced mask IoU=1.0 on all 5 boxes.
- 10 timing rows: serial SAM group mean 273.8525 ms, batch group mean 117.1421 ms, mean speedup 2.3392x.
- Stage 8 did NOT include DINO timings, did NOT establish semantic correctness, and did NOT meet the 30 ms PRD target.

## Stage 9 scope
- Uses the **same isolated predicted garment crop** with independent DINO text queries.
- A: process queries one by one; B: DINO processor/model batch with duplicated images and individual text prompts.
- Compares the full candidate lists per query: candidate count, boxes, scores, labels.
- Times preprocessing + model forward + postprocessing for both methods, excluding garment segmentation, SAM, file IO and model loading.
- Batch form might be slower, show differences, or run out of GPU memory. Any of those are valid negative outcomes.
- No production pipeline edits, no new annotations, no model training, and no PRD accuracy/latency acceptance claim.

## Run on GPU (upload the ZIP to /workspace)
```bash
cd /workspace
python -m zipfile -e prd31_stage9_dino_batch_20261014.zip /workspace/prd31_stage9_dino_batch_20261014
cd /workspace/fashion_multimodal_analysis
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage9_dino_batch_20261014/scripts/run_stage9.py --dry-run
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage9_dino_batch_20261014/scripts/run_stage9.py
```
Do not proceed past `--dry-run` if it fails. Real run loads existing V5 Balanced150, DINO, SAM and CLIP as part of the established backend.

The runner always prints `UPLOAD_THIS_ZIP`, including for runtime failures. Upload the result ZIP to the chat for assessment.

**Important:** This is query batching, **not** a new shared DINO image-encoder implementation. No assumption of encoder reuse or throughput improvement is made.
