# PRD 3.1.3: October 12-13 sleeve continuous 8-case pilot (corrected)

This reuses the original Oct12 script and original eight annotated train-origin candidates. Changes: repaired full SHA256 of the pilot JSON; bundled three input files, added runtime Core400 hash guard and a --dry-run option; added a fixed nearest-anchor categorical agreement diagnostic. This is a **read-only eight-case TRAIN-ORIGIN exploratory experiment**, not an independent holdout, not six-class acceptance, and not sleeve-cuff midpoint error.

## On GPU

```bash
cd /workspace
python -m zipfile -e PRD313_PILOT8_READY.zip /workspace/prd313_pilot8_ready
cd /workspace/fashion_multimodal_analysis
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd313_pilot8_ready/run_pilot8_fixed.py --dry-run
# Then after DRY_RUN_PASS:
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd313_pilot8_ready/run_pilot8_fixed.py
```

Download the `UPLOAD_THIS_ZIP` reported by the second command. If the dry-run fails, send the full traceback instead of running the model.

## Interpretation

- Continuous scalar: CLIP semantic probability-weighted expected anchor in [0,1], NOT physical sleeve geometry.
- Human ratio: 2D sleeve coverage ratio from manual four-point labels.
- MAE/ordering is exploratory because these are distinct types of 0-1 quantities.
- Hard baselines: previous prompt and linear head on same eight; not retrained.
- Fixed six-anchor classification accuracy: pure diagnostic mapped from continuous CLIP scalar to nearest preset anchor. Anchors are **not tuned** on these eight examples; only cap/short/elbow represented.
- The source `pilot8_preflight_core400_report.json` is a PREVIOUS historical isolation check; runtime additionally checks the current Core400 manifest SHA, dataset manifest hashes, and eight source image hashes. Does not run a new exhaustive content-level duplicate check.
- Never overwrite frozen blind sleeve evaluation, train/val/Core400 manifests or checkpoints.
