# PRD31 Stage14 — additional validation-manifest images A/B

Stage13 completed a strict grouped A/B on 8 previously used diagnostic images:

- Production 4-query group 703.3137 ms; experimental batch 530.5088 ms.
- Ratio of mean group times 1.3257x; relative group latency reduction 24.57%.
- 32/32 final region records pass tolerance screen; 160/160 repeats pass.
- Of 16 detected results, 15 masks pixel-identical. Case_04/pocket IoU=0.9996300119.
- Amortized batch cost is 132.6272 ms per query, **not <=30 ms**.

**Next:** extend this A/B to up to 12 additional images from `configs/prd_8class_val_v2_full_targets.csv`.
The runner excludes exact resolved paths found in Stage13, the V5 training manifest,
and Core400, and saves a selection audit and SHA256 for every selected image.
These are validation-manifest images and NOT fresh blind independent samples. It does NOT
compute true 3.1.2 region localization accuracy or attribute accuracy. Actual formally
reviewed part GT is still missing. Do not deploy on success.

The script reuses the identical Stage13 strict A/B engine (DINO batching + SAM multi-box)
and model configuration. It does not alter project files. It writes run results outside repo.

## On GPU

Upload `prd31_stage14_valbatch_20261014.zip` to `/workspace`:

```bash
cd /workspace
python -m zipfile -e prd31_stage14_valbatch_20261014.zip /workspace/prd31_stage14_valbatch_20261014
cd /workspace/fashion_multimodal_analysis
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage14_valbatch_20261014/scripts/run_stage14.py --dry-run
# only when DRY_RUN_PASS, run:
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage14_valbatch_20261014/scripts/run_stage14.py
```

Default: up to 12 extra images, 4 queries per garment, 3 warmups, 5 alternating
full-pipeline timed passes. If you want a quicker pilot, run with `--max-cases 6` on both.

Upload the resulting `prd31_stage14_run_..._UPLOAD_THIS.zip` even if a failure occurs.
Any terminal status other than `STAGE14_VALIDATION_MANIFEST_AB_COMPLETED` means the
experiment did not finish; preserve logs and do not manually alter frozen files.

**Scope caveats:** GT in validation manifest is used only to identify a separate image
source and improve class coverage; no human-reviewed local-region GT or formal PRD metrics.
Training/Core overlap exclusion is exact image path based, not image-content deduplication.
The eight Stage13 images are excluded. Actual source selection and skips are saved.
