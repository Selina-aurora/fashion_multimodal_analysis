# PRD 3.1 Stage 6 — warmed latency and paired garment-ROI perturbation experiment

Date: 2026-10-14. Runs read-only in the existing GPU project.

## What this adds (does not repeat the Oct 14 chain smoke test)

1. **Warmed module latency**: 10 model warmups, 5 timed passes for segmentation/image, Grounding DINO+SAM/query-instance, CLIP-based garment attribute set/instance. Timing is returned from the existing CUDA-synchronized `ModelBackends` methods. Selected 8 historical diagnostic images must match exact SHA256 values. Grounding coverage samples up to 2 query instances per region; source images are unchanged. Outputs `warmed_timing_raw.csv`, `grounding_latency_by_region.csv`, `warmed_latency_summary.json`.
2. **Paired validation-input perturbation**: the GPU segmentation predicts from `prd_8class_val_v2_full_targets.csv`, class-correct matched garment proposals (bbox IoU >= 0.5) are compared with the **GT garment** bbox/mask on the same image. Both ROI conditions use the same attribute routine with `controlled=True`, including the same fixed neckline-proxy convention. Reports category-conditional label agreement, NOT GT attribute accuracy. For sampled region phrases, it compares predicted grounding boxes when using GT garment ROI and predicted garment ROI. This is cross-input stability, **NOT region localization accuracy**. Outputs `paired_instances.csv`, `paired_attribute_agreement.csv`, `paired_region_stability.csv`, `paired_summary.json`.
3. Includes the 48-row previous automated review queue `priority_review_queue.csv` as a separate handoff; the queue is **not** human truth.

## GPU run steps (py312)

Upload `prd31_stage6_latency_paired_20261014.zip` to `/workspace`, then:

```bash
cd /workspace
python -m zipfile -e prd31_stage6_latency_paired_20261014.zip /workspace/prd31_stage6_latency_paired_20261014
cd /workspace/fashion_multimodal_analysis
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage6_latency_paired_20261014/run_stage6.py --dry-run
```

If `STATUS: DRY_RUN_PASS`, run:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python /workspace/prd31_stage6_latency_paired_20261014/run_stage6.py
```

Wait for `STATUS: STAGE6_DIAGNOSTIC_COMPLETED` and upload the ZIP printed after `UPLOAD_THIS_ZIP`. If it fails, upload the error ZIP and we will diagnose. Runs can take several minutes because of multiple model loads, Grounding DINO/SAM calls, and 10 warmups.

To run only one experiment: append `--mode latency` or `--mode paired`.

## Strict evaluation boundaries

- Existing frozen Core400, train/val CSV, model checkpoint, and project outputs are untouched. All new files are outside the repository in `/workspace/prd31_stage6_20261014_runs/`.
- The V5 Balanced150 checkpoint is validated with epoch/class mapping and training/validation SHA, plus Core400 SHA.
- **No artificial GT local boxes, region masks, or attribute labels are created.** The `val` masks are only GT of the parent garment.
- Two conditions are **GT garment ROI vs prediction garment ROI**. They are **not** GT *local-part* ROI vs predicted local-part ROI. Attribute scores here are agreement, not ground-truth accuracy.
- The comparison is conditioned on successful category-correct bbox matching; all unmatched validation instances are excluded. The selected subset is not a blind test, and the composition is recorded.
- The performance figures have two different units: region/query-instance, and full attribute set/garment. Do not compare aggregate set time naively with a per-attribute 20 ms target. Repeat samples on only 8 images cannot establish strict PRD latency acceptance.
- The previous 3.1.1 Core400 formal segmentation results remain the only confirmed accuracy regression. This is a **development diagnostic**, not full PRD 3.1 acceptance.

## Output summary meanings

- `warmed_latency_summary.json`: report module n, mean, median, P95, exceedances, scope.
- `paired_summary.json`: report exact number of matched validation pairs, available attribute comparisons and changes across garment ROI conditions.
- `paired_region_stability.csv`: output-box IoU across two input crops, **NOT intersection with human GT part box**.
- `paired_skipped_sources.csv`: errors reading source/GT mask for traceability.

Do not modify `benchmark/prd_3_1_v2/frozen/` or mark pending AI pre-review as signed-off GT.
