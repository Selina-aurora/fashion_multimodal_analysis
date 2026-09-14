# Verified-positive v3 Final Automatic Result Check

## Completeness
- 88 benchmark cases
- 22 cases each for sleeve, collar, button, zipper
- 264 condition rows = 88 × 3
- FULL and ROI predictions identical across all 88 cases: **True**

## Automatic detection
- FULL: 29/88 = 33.0%
- ROI: 29/88 = 33.0%
- LOCAL: 22/88 = 25.0%

Mean detected-box area ratio:
- FULL: 0.963
- ROI: 0.963
- LOCAL: 0.219

## Key interpretation
1. FULL and ROI are effectively the same condition on this benchmark; ROI did not change any prediction.
2. Raw detection rate is not localization accuracy. FULL/ROI detections are overwhelmingly image/garment-level boxes rather than fine-grained part boxes.
3. Tight local cropping makes boxes substantially smaller, but does not improve overall raw detection rate.
4. Small-object localization remains the main failure mode. Button and zipper remain especially weak.
5. The local-crop intervention should be reported as a negative/mixed result rather than as a successful improvement.

## Preliminary visual audit
A preliminary audit has been filled from the triplet contact sheets. It is intentionally conservative.
The row `zipper_019` is flagged low-confidence and should be manually confirmed before treating the audit as final.


> Manual localization labels remain preliminary until low-confidence cases are confirmed. Automatic detection statistics above are not localization accuracy.
