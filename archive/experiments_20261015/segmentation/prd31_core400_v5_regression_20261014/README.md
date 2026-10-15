# 2026-10-14 Core400 V5 balanced regression

This is an **isolated, read-only-to-project regression runner**. It uses the existing `scripts/segmentation/evaluation/eval_prd31_core_regression_v2_balanced_v1.py` in the user's project rather than replacing it. The script checks all frozen SHA256 manifest values, checkpoint's train/val provenance, final epoch 15, 8-class map and CUDA availability **before** running inference. It does not train or edit checkpoint/GT/config/old reports.

Upload this zip to `/workspace` and run:

```bash
cd /workspace
python -m zipfile -e prd31_core400_v5_regression_20261014.zip /workspace/prd31_core400_v5_regression_20261014
cd /workspace/fashion_multimodal_analysis
python /workspace/prd31_core400_v5_regression_20261014/run_verified_core400.py
```

If necessary use `conda activate py312` first. The evaluator runs 400 images, 10 warmups and 5 timing passes and may take several minutes. The runner prints a `UPLOAD_THIS_ZIP:` path. Upload that exact ZIP to ChatGPT. The output is under `/workspace/prd31_regression_20261014/`, outside the repository.

**Important:** `STATUS: REGRESSION_COMPLETED` is necessary to treat a run as completed. Other statuses mean no valid new metric was produced. The output ZIP always contains a report explaining the error.

This measures **3.1.1 instance segmentation**, not total 3.1 end-to-end latency, 3.1.2 localization, or 3.1.3 attributes. Existing performance values must not be overwritten or asserted to be newly confirmed before examining the returned result.
