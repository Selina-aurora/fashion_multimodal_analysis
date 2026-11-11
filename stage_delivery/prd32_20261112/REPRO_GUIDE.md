# Reproducibility and file provenance

The original experiment files were packaged in multiple dated ZIPs on 2026-11-11. This update copies source scripts **byte for byte** and keeps smaller experiment reports/results without re-generating GPU data. See `PROVENANCE.json` for exact source archive, member path and SHA-256 per copied file.

## GPU dependency assumptions

The original scripts were run on an RTX 4090 environment with a V5 checkpoint and Qwen2.5-VL-7B-Instruct accessible on the GPU host. These model checkpoints, dataset images, custom runtime assets, and some environment-specific paths are **not bundled**. Do not assume this source snapshot alone can reproduce GPU results on another machine. The original pipeline also uses local guarded runtime modules and may require project-specific imports/PYTHONPATH; run help/tests in an isolated environment first.

```bash
python -m compileall -q stage_delivery/prd32_20261112/repro
# inspect the original README/inputs and paths before requesting GPU execution
```

For the **neutral red/cyan 5-pair** experiment, the missing files are the ten `repro/neutral_ab/images/{03..07}_{RED,CYAN}_control.png` files from the original `PRD32_ORACLE8_NEUTRAL_AB_V2_20261111_UPLOAD_THIS.zip` packet. Those 10 images should be staged **locally**, not committed to this public repository. With the images in place and a compatible GPU Python environment:

```bash
cd stage_delivery/prd32_20261112/repro/neutral_ab
python validate_inputs.py
python run_neutral_ab.py --model /workspace/models/Qwen2.5-VL-7B-Instruct --output outputs/neutral_results.jsonl
python build_review.py --results outputs/neutral_results.jsonl --output outputs/NEUTRAL_5_PAIRS_REVIEW.html
```

Until those commands have produced real GPU JSONL results, the test is **PENDING_GPU**, and empty/pending result placeholders are not evidence of completed inference.

## Source-result trust boundaries

- `evidence/` reports were copied without modification except for destination filename normalization.
- Development reviewer confirmations do not equal independent blind GT; some AI-prefilled review sheets remain unconfirmed and are labeled accordingly.
- Text answer equality is an exact-output stability check, **not** a correctness metric.
- `original_guard_invoked=false` in a candidate experiment is not an accepted Guard path.
- Do not copy any result from this snapshot into the public release decision without a new quality review and independent holdout.
