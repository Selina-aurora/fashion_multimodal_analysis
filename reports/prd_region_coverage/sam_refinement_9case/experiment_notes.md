# SAM refinement 9-case pilot

## Purpose

Test whether SAM can refine coarse Grounding DINO / spatial proposals into
tighter local masks and bounding boxes for collar, cuff, and hem.

## Cases

- collar: collar_001, collar_010, collar_020
- cuff: cuff_006, cuff_008, cuff_019
- hem: hem_003, hem_008, hem_023

## SAM model

`facebook/sam-vit-base`

## Proposal priority

1. SPATIAL_HR prediction when available.
2. Frozen BASELINE Grounding DINO prediction.
3. Fixed target-specific spatial prior when neither detector returns a box.

## Evaluation

Automatic:
- proposal bbox area ratio
- SAM refined bbox area ratio
- SAM mask area ratio
- SAM predicted IoU score
- inference time

Manual:
- correct
- coarse
- wrong
- missed

This pilot is intended to determine whether a segmentation-refinement stage is
worth integrating into PRD 3.1.2. It is not a final acceptance result.
