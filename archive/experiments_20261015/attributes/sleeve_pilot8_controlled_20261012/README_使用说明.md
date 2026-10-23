# Sleeve continuous pilot8 — confirmed disposition, 2026-10-12

## Status

**24 reviewed = 8 controlled pilot candidates + 16 isolated.** Of isolated records, 8 came from the visual recheck pool and 8 were previously excluded/nonmeasurable. The user accepted the QA handling of `v5_train_00286`, `v5_train_01719`, and `v5_train_01748`; these are *not* repaired labels and none enter pilot8.

- Pilot8 by class: cap=3, short=2, elbow=3. All have four normalized points and an apparent 2D sleeve coverage ratio.
- Existing train/val CSV were unmodified. Previous GPU check matched train hash `e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f` and val hash `39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93`.
- The 8 source images match the source SHA256 in the previous local export; they also have no hash or record-ID overlap with the 44-item previous sleeve pilot train/dev split index.
- Previous GPU audit compared against 32 validation images and 47 previously reviewed images, but **it does not constitute an exhaustive Core400 overlap verification**. Core400 freeze should remain untouched.
- No training, no model checkpoint changes and no claimed accuracy improvement. Candidate pilot labels must not be mistaken for a holdout evaluation set.

## Files

- `pilot8_candidates_FINAL.json`: exact 8 candidates, user-reviewed, source-visual QA approved for controlled train-side feasibility only.
- `pilot8_manifest.csv`: convenient inventory (UTF-8 BOM).
- `quarantine16_no_training.json`: all other 16 user records, original annotations retained for audit, plus clear disposition.
- `all24_dispositions.csv`: one-row-per-record routing summary.
- `original_user_review_UNCHANGED.json`: verbatim input provenance.
- `parent_split_index.json`: previous 32 train/12 dev, for overlap check.
- `gpu_audit_report_original.json`: previous GPU inspection evidence, not a current GPU run.
- `preflight_pilot8.py`: read-only GPU environment verification. Produces report; does not edit training/val data or run a model.

## Optional GPU preflight

Upload and unzip this package into `/workspace/sleeve_pilot8_controlled_20261012`, then run:

```bash
cd /workspace
python -m zipfile -e sleeve_pilot8_controlled_20261012.zip /workspace/sleeve_pilot8_controlled_20261012
python /workspace/sleeve_pilot8_controlled_20261012/preflight_pilot8.py --repo /workspace/fashion_multimodal_analysis --output /workspace/pilot8_preflight_report.json
```

If `PREFLIGHT_PASSED_CORE400_PENDING`, files and current split SHA look good but **Core400 hash comparison remains incomplete**. For a full Core400 isolation check supply the *real frozen* list of 400 distinct source-image SHA256 hashes via `--core400-hashes /path/to/core400_sha256.txt` (not predicted results). The script deliberately refuses a partial hash list. The result is not an automatic training approval.

## Experiment plan (proposal only)

1. Freeze baseline prediction code, model weights, evaluation inputs, metrics and random seed. Never modify Core400 labels or results in place.
2. Compare baseline vs a train-side pilot using exactly eight labels (3 cap, 2 short, 3 elbow), only if training code actually consumes the four-point/ratio fields. If there is no continuous sleeve model head / matching input pipeline, implement and verify that first; **eight labels are too few to claim a reliable performance gain**.
3. Use only an independently untouched, nonoverlapping evaluation set for test metrics. Track MAE (continuous ratio), per-class errors/confusion, sample count, dropped samples, and before/after inference latency. If the evaluated task remains categorical only, do not report continuous MAE as though it were evaluated.
4. Keep pilot artifacts isolated from production train CSV, all frozen dev/validation files, Core400 and v5/v8 checkpoint directories. Report changes only after actual experiments.

## QA decisions accepted

- `v5_train_00286`: cuff point likely wrong, requires redraw of C and same-arm S/E/W check before possible future use. Current ratio must not be trained on.
- `v5_train_01719`: crossed arms and contradictory sleeveless flag; exclude from this pilot. Raw annotation retained unchanged (do not silently convert into a new ground truth).
- `v5_train_01748`: asymmetric one-shoulder clothing; exclude from this pilot. Original whole-garment sleeveless=0 is not a trustworthy continuous label.

The other 5 recheck cases remain held, without inferred label corrections. All 8 already isolated records remain isolated.
