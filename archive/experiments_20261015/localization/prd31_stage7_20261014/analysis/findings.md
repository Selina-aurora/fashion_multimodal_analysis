# Stage 6 GT vs predicted garment ROI paired analysis (2026-10-14)

**Source**: user-uploaded validated GPU Stage 6 pair diagnostic. No new accuracy numbers have been inferred.

- Validation: 32 images scanned, 63 eligible candidate matched garments found, **8 selected**, across 8 classes. Source is validation/development, not blind/Core400.
- Paired attribute predictions: **21/32 = 65.625%** identical labels. This is ROI-perturbation consistency, **not** label correctness.
- Regional query comparisons: **11/16** detect region on both ROI inputs; average bbox IoU **0.6181** calculated ONLY on 11 both-detected pairs. It is neither localization recall nor GT IoU.
- Category mask IoU: top=0.750, pants=0.911, skirt=0.929, outerwear=0.887, dress=0.672, shoe=0.890, bag=0.726, accessory=0.486.
- Priorities: P0 1 cases, P1 5 cases, P2 2 cases. See CSV, these priority categories are a new **triage heuristic**, not human grading.
- The attribute comparison used `controlled=True` for both garment ROI inputs. **Both necklines use the same fixed top-center proxy**; it does not compare GT neckline to predicted collar.
- Only correctly class-matched garments (bbox IoU >=0.5) enter the paired cohort; the result is conditional, with selection bias and n=8.
- Do not overwrite Core400, retro-label review files, or claim formal acceptance. This package has no human-verified region GT masks or attribute labels.

## Investigate first

1. `valpair_05` dress: mask IoU 0.672, attributes agree only 1/6, shoulder/waist both disappear with predicted garment ROI.
2. `valpair_08` accessory: mask IoU 0.486; pattern output box IoU 0.117; decoration status switched.
3. `valpair_01` top: cuff localization box IoU 0.333, neckline differs using same proxy rule.
4. `valpair_07` bag: pattern+decoration localization box IoU 0.429, pattern label differs.

## Next GPU experiment

Profile the 3.1.2 pipeline **per stage** using the exact existing `ModelBackends.ground` call. We will time Grounding DINO and SAM under the same warmed diagnostic inputs, and record their fractions of total localization latency. No predictions, thresholds, or trained weights are changed. The timing wrappers introduce extra sync overhead; do **not** compare these component totals as a strict apples-to-apples rerun of Stage 6 latency.
