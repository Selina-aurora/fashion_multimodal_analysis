# SAM refinement 9-case pilot — final manual audit

## Rubric

- **correct**: the SAM mask/refined bbox substantially isolates the requested local region.
- **coarse**: the requested region is included, but SAM segments a substantially larger garment region.
- **wrong**: SAM mainly segments another region/background.
- **missed**: no usable mask/bbox is returned.

## Result

No case reached strict `correct` localization. SAM generally followed the coarse box prompt and segmented the dominant garment object inside that box rather than recovering the semantic part boundary.

- **Collar:** 3/3 coarse.
- **Cuff:** 2/3 coarse, 1/3 wrong (`cuff_008`).
- **Hem:** 3/3 coarse.
- **Overall:** 0/9 correct, 8/9 coarse, 1/9 wrong.

## Automatic metrics

The average proposal-box area ratio decreased only modestly after SAM refinement:
- Collar: 0.2959 -> 0.2611
- Cuff: 0.2319 -> 0.2026
- Hem: 0.6811 -> 0.6297
- Overall: 0.4030 -> 0.3645

SAM predicted-IoU scores are high for several cases, but this score estimates mask quality relative to SAM's own prompted object hypothesis; it is **not** evidence that the mask matches the requested fashion part.

## Engineering conclusion

Box-prompted SAM is useful for turning a coarse proposal into a cleaner object mask, but it does not solve the semantic part-localization problem when the proposal itself is too broad. Therefore:

1. Do not continue tuning SAM alone.
2. Keep region-specific spatial priors as a proposal mechanism.
3. For PRD 3.1.2, move to a part-aware grounding/segmentation approach or obtain explicit part-level supervision.
4. Preserve Grounding DINO + SAM as a documented baseline/failure-analysis pipeline, not as the final acceptance implementation.
