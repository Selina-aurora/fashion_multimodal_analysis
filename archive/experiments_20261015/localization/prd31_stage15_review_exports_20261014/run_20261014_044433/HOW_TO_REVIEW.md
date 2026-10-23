# Manual review guidelines

1. Open each `*_garment_plain.jpg` together with `*_original.jpg`. Use the prediction overlay only AFTER deciding which physical region should be present.
2. For each row of `review_manifest_FILL_MANUALLY.csv`, fill `human_region_presence_yes_no_uncertain`, then `human_localization_correct_coarse_wrong_missed_uncertain_not_applicable`, and notes. Do not auto-approve because the A/B model outputs agreed.
3. If a region is absent in reality, mark presence=no and localization=not_applicable, even if the model says detected; separately describe the false positive.
4. If the region is invisible, occluded, too small or semantically unclear, mark uncertain; do not guess coordinates.
5. Where confident, annotate the human region bbox in original image coordinates using `human_bbox_xyxy_if_drawn` and optionally a human mask file; this script DOES NOT produce that GT.
6. Only after human labels and case definitions have been reviewed should any localization accuracy metric be computed. This task is not representative of all eight garment classes or all eight local regions.
7. If `replay_status_matches_stage14` is not 48/48, inspect `replay_mismatches.json` before comparing against Stage14 results.
