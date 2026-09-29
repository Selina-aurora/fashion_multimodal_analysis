# Core-region SPATIAL_HR diagnostic

## Purpose

Test whether target-specific spatial priors plus a higher Grounding DINO
processor resolution can convert coarse garment-level predictions into tighter
region-level predictions for the three priority PRD regions:

- collar
- cuff
- hem

## Frozen variables

- Baseline cases: the same 15 cases from the 40-case PRD coverage pilot.
- Prompt: reused exactly from the frozen baseline.
- Threshold: 0.3.
- Model: `IDEA-Research/grounding-dino-tiny`.
- Baseline inference: reused from `formal_case_results.csv`.

## SPATIAL_HR settings

- collar: upper-center window `(0.15, 0.00, 0.85, 0.45)`
- cuff: two side windows; keep the window with the highest top score
  - left `(0.00, 0.00, 0.46, 1.00)`
  - right `(0.54, 0.00, 1.00, 1.00)`
- hem: lower band `(0.00, 0.55, 1.00, 1.00)`
- processor shortest edge: 1000
- processor longest edge: 1600

## Evaluation

Automatic:
- non-empty prediction rate
- top confidence
- effective top-box area ratio relative to the original garment crop
- inference time

Manual:
- correct
- coarse
- wrong
- missed

The desired improvement is mainly `coarse -> correct`, not merely
`missed -> detected`.
