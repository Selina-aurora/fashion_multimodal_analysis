# PRD 3.1 Stage 13: timed production-matched batching diagnostic

**Purpose:** benchmark the *real* grouped throughput of full strict local-grounding pipeline against an experimental DINO multi-prompt + SAM multi-box scheduler. Stage 12 established parity on 4 garments and 16 region queries, but its experimental arm executed **extra native DINO serial checks inside the timer**. This Stage 13 moves those checks entirely **outside timing** and extends the diagnostic set to up to 8 pre-identified images, with 4 regions per garment when applicable.

**Important:** 1. This is a development A/B, *not* official PRD 3.1.2 92% localization accuracy or <=30ms latency acceptance. 2. Group/amortized throughput does not prove every individual region query meets the <=30ms PRD latency requirement. 3. The same previously used diagnostic images do not provide independent validation. 4. Strict final parity is tested at mask IoU>=0.999 and differences <=1px/0.005 score; non-identical pixels are **explicitly counted**, never advertised as exact sameness. 5. Do **not** deploy even if this passes; held-out samples + human semantics evaluation are required.

## Run on GPU

Upload `prd31_stage13_strict_batch_timing_20261014.zip` to `/workspace`, then:

```bash
cd /workspace
python -m zipfile -e prd31_stage13_strict_batch_timing_20261014.zip /workspace/prd31_stage13_strict_batch_timing_20261014
cd /workspace/fashion_multimodal_analysis
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage13_strict_batch_timing_20261014/scripts/run_stage13.py --dry-run
# only if DRY_RUN_PASS:
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage13_strict_batch_timing_20261014/scripts/run_stage13.py
```

Expected output: `/workspace/prd31_stage13_runs_20261014/prd31_stage13_run_..._UPLOAD_THIS.zip`. The script prints the exact path. **Upload the ZIP even if the GPU run fails**, because it includes logs.

## Controls
- Read-only project; no training; fixed V5 Balanced150 checkpoint; SHA256 checks train, val, frozen Core400 and source images before model initialization.
- 3 warmups per case/method, 5 alternating-order timed passes, up to 8 images/4 region queries per garment.
- `run_audit.json`, `stage13_summary.json`, `strict_dino_parity.csv`, `final_region_parity.csv`, `repeat_region_parity.csv`, `optimized_group_timing.csv`.
- Candidate parity checked independently, **not timed**. Both timed arms include DINO, SAM, and strict downstream processing.
- Reports automatically use new run directories outside `/workspace/fashion_multimodal_analysis`.
