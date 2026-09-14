# Grounding DINO Failure Pattern Analysis

## 1. Analysis Setup

- Source audit: `D:\projects\fashion_multimodal_analysis\reports\grounding_group_evaluation\manual_localization_audit_reviewed.csv`
- Reviewed cases: 40
- Larger-part group: `sleeve`, `collar`
- Small-object group: `button`, `zipper`

> Important: this 40-case audit was sampled from emitted predictions. The statistics below describe localization quality among audited prediction cases and must not be interpreted as full-dataset localization accuracy.

## 2. Localization Quality by Target

| Target | N | Correct | Coarse | Wrong | Wrong Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| sleeve | 10 | 0 | 3 | 7 | 70.00% |
| collar | 10 | 0 | 0 | 10 | 100.00% |
| button | 10 | 0 | 3 | 7 | 70.00% |
| zipper | 10 | 0 | 0 | 10 | 100.00% |

## 3. Failure Type Distribution

| Error Type | Count | Share | Sleeve | Collar | Button | Zipper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| target_absent | 15 | 37.50% | 5 | 5 | 2 | 3 |
| wrong_part | 5 | 12.50% | 2 | 1 | 0 | 2 |
| whole_garment | 4 | 10.00% | 3 | 0 | 1 | 0 |
| face_head | 4 | 10.00% | 0 | 4 | 0 | 0 |
| non_clothing_object | 4 | 10.00% | 0 | 0 | 3 | 1 |
| footwear | 4 | 10.00% | 0 | 0 | 0 | 4 |
| background_object | 2 | 5.00% | 0 | 0 | 2 | 0 |
| whole_image | 2 | 5.00% | 0 | 0 | 2 | 0 |

## 4. Small-Object Analysis

- Small-object group wrong rate: 85.00% (17/20).
- Larger-part group wrong rate: 85.00% (17/20).
- Most common small-object error: `target_absent`.
- Most common larger-part error: `target_absent`.

### Target-specific observations

- `sleeve`: 0 correct, 3 coarse, 7 wrong.
- `collar`: 0 correct, 0 coarse, 10 wrong.
- `button`: 0 correct, 3 coarse, 7 wrong.
- `zipper`: 0 correct, 0 coarse, 10 wrong.

## 5. Interpretation

The current failure distribution should be interpreted together with the ROI-conditioned diagnostic experiment.

For larger garment parts, failures frequently include whole-garment or coarse localization, indicating that Grounding DINO may recognize semantic association with the garment without precisely isolating the requested local region.

For small objects such as buttons and zippers, failures should be examined for target absence, non-clothing-object confusion, wrong-part activation, and low visual detail. These patterns are consistent with the hypothesis that small targets are more sensitive to limited pixel information and visual ambiguity.

The ROI-conditioned experiment should therefore be treated as a search-space reduction diagnostic rather than proof that cropping alone solves fine-grained localization.

## 6. Recommended Next Step

1. Preserve the current full-image baseline at `threshold = 0.3`.
2. Preserve ROI `threshold = 0.2` only as a diagnostic condition, because increased prediction coverage does not necessarily imply improved localization.
3. Build a target-balanced evaluation set for `button` and `zipper` with verified positive examples.
4. Separate `target absent` cases from true localization failures before reporting final localization accuracy.
5. Evaluate whether a stronger fine-grained localization or segmentation refinement method is required after the Grounding DINO baseline.
