# PRD31 Stage15 – Manual Region-GT Review Assets

Stage14 was a speed/output-parity experiment, NOT a semantic accuracy test. This package generates 48 human-review previews for Stage14's 12 development-validation predicted garment ROIs (top/dress only; collar, cuff, hem, pocket).

**Models and benchmark are read-only.** Output is saved outside the repository at `/workspace/prd31_stage15_review_exports_20261014`.

## Run

```bash
cd /workspace
python -m zipfile -e prd31_stage15_human_gt_prep_20261014.zip /workspace/prd31_stage15_20261014
cd /workspace/fashion_multimodal_analysis
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage15_20261014/scripts/run_stage15.py --dry-run
# If DRY_RUN_PASS:
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage15_20261014/scripts/run_stage15.py
```

Send the printed `UPLOAD_THIS_ZIP` to ChatGPT, containing 12 original-image previews, 12 unannotated garment crops, and 48 region overlays plus CSV for actual human judgment. ZIP may exceed 40MB depending on image sizes, though images are resized to reduce that risk.

## Ground truth policy

- **Human GT is blank by design**, never filled from predictions. The exported candidate mask is NOT a GT mask.
- Refer to `HOW_TO_REVIEW.md` in output ZIP. A green mask/red box is a model prediction, not human approval.
- If an item is hidden or ambiguous, mark uncertain. Do not invent true label or bbox.
- The script aborts when selected case class / crop dimensions / region list do not match Stage14's recorded selection.
- It records whether each independently replayed prediction status matches Stage14. If not, do not assume deterministic output.
- No training, no benchmark mutation, no deploy, no formal 92% accuracy or 30ms acceptance claim.
