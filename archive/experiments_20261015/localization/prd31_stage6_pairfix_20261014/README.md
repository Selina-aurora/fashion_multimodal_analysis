# PRD31 Stage6 paired validation schema fix (October 14, 2026)

This **does not rerun latency**. Previous warmed latency outputs remain valid.

## Cause

The old validation ROI pair script expected `gt_bbox_x1`, `gt_bbox_y1`, `gt_bbox_x2`, `gt_bbox_y2`, and `gt_mask_path` in `configs/prd_8class_val_v2_full_targets.csv`. V5 full-targets manifests use `bbox_x1...bbox_y2` and `mask_path`. As a result, all 32 validation source images were skipped with `KeyError: 'gt_bbox_x1'` and zero paired comparisons were produced. This was a **schema failure**, NOT evidence the model matched zero garments.

## GPU steps

Upload `prd31_stage6_pairfix_20261014.zip` to `/workspace`. Use a NEW unpacking folder and output directory:

```bash
cd /workspace
python -m zipfile -e prd31_stage6_pairfix_20261014.zip /workspace/prd31_stage6_pairfix_20261014
cd /workspace/fashion_multimodal_analysis
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage6_pairfix_20261014/test_pairfix_offline.py
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage6_pairfix_20261014/run_stage6.py --dry-run --mode paired
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage6_pairfix_20261014/run_stage6.py --mode paired
```

**Only** `--mode paired` should be used for this rerun. It uses the same V5 Balanced150 15-epoch checkpoint and protected manifests, 32 validation images, up to 8 class-correct (bbox IoU >=0.5) garment-ROI pairs and 2 local queries per pair; no retraining. New output ZIP `UPLOAD_THIS_ZIP` prints at exit. Upload it even if errors occur. The script writes only to a new folder in `/workspace/prd31_stage6_20261014_runs/`, not into the project repository.

## Scope and limitations

This compares *GT garment* ROI input vs *predicted garment* ROI input, not GT part ROIs or verified attribute labels. Label agreement is input-stability, NOT recognition accuracy; part-box IoU is cross-input stability, NOT true localization accuracy. Results are conditional on successfully class-correct matched garments, and are development-only, not PRD acceptance. The absence of frozen human-reviewed V2 remains an open item. `validation_pair_schema.json` documents actual columns used. If all validation sources fail, the new run now emits a clear error rather than misleading zero pairs.
