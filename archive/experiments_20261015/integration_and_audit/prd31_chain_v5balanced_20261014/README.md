# PRD 3.1 — V5 Balanced150 GPU full-chain interface rerun (2026-10-14)

This is an **8-image (or smaller, when legacy case set absent) GT-free integration smoke test**. It is NOT Core400 accuracy, PRD 3.1.2 region accuracy, 3.1.3 attribute accuracy, or full-chain latency acceptance.

It reuses Oct 9 `smoke_export.py`, `check_geometry.py` and `verify_chain_contracts.py`, keeping their historic files under `legacy/` for comparison. No source code, source manifests, frozen Core400, or checkpoint is modified. All new reports go to `/workspace/prd31_chain_v5balanced_20261014_runs/`.

## Run in existing (py312) GPU environment

Upload the English-named ZIP to `/workspace` then:

```bash
cd /workspace
python -m zipfile -e prd31_chain_v5balanced_20261014.zip /workspace/prd31_chain_v5balanced_20261014
cd /workspace/fashion_multimodal_analysis
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_chain_v5balanced_20261014/run_v5balanced_chain.py --dry-run
```

On `DRY_RUN_PASS`, run:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_chain_v5balanced_20261014/run_v5balanced_chain.py
```

Upload the ZIP printed after `UPLOAD_THIS_ZIP:` (even if script fails) so failed preconditions can be diagnosed safely. Run can take multiple minutes: loads existing segmentation, grounding, SAM and CLIP models. Offline Hugging Face settings are enabled to avoid unintended downloads.

## Validity checks
- SHA256 checks exact frozen Core400, V5 Balanced150 train CSV and V2 full-target validation CSV.
- Validates checkpoint epoch=15, train/val provenance, 8-class mapping, CUDA availability.
- Uses 10/9 8-image cases when both legacy `cases.json` and `run_config.json` exist. Otherwise picks one available validation image per PRD class, recording fallback explicitly; no GT is passed to the model.
- Geometry checks first, real model forward calls second, original-image coordinate, ROI, mask and attribute output contract checks last.
- Does not pretend `review/grounding_reviewed.csv`, `review/attribute_reviewed.csv` or `benchmark/prd_3_1_v2/frozen` exists.
- If semantic consistency patch is absent in the installed GPU code, reports that fact; no silent code modification.

A passing contract result means fields, calls and coordinate conventions held on this *small diagnostic set*. Semantic incorrectness and absent PRD test ground truth remain open issues.
