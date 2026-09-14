# Grounding DINO Baseline Experiment — 2026-09-08

## Objective

Validate the basic language-guided localization pipeline and test how prompt granularity and threshold choice affect Grounding DINO outputs.

## Baseline

- Model: `IDEA-Research/grounding-dino-tiny`
- Input: fashion image + text prompt
- Output: candidate bounding box + confidence score

Prompts included:

- `shirt`
- `sleeve`
- `collar`
- `button`
- `sleeve cuff`
- `zipper`
- `blue denim shirt`
- `left sleeve`

## Threshold sensitivity

The same `sleeve` example was evaluated at thresholds `0.2`, `0.3`, `0.4`, and `0.5`. The prediction was retained from 0.2 to 0.4 and filtered at 0.5.

## Pilot interpretation

- Whole-garment and common-part prompts are easier to trigger than very small details.
- More specific prompts can change the predicted region, but specificity alone does not guarantee precise localization.
- A threshold that is too high can reduce recall.
- These pilot observations are not used as final accuracy claims.

Subsequent experiments expanded the sample size and introduced manual localization-quality review.
