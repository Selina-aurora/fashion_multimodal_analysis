# Project Status

_Last updated: 2026-09-14_

## 1. Scope

The current engineering milestone is to complete and consolidate PRD **3.1 Fine-grained Vision Foundation** before starting PRD 3.2.

```text
3.1.1 Instance Segmentation
        ↓
3.1.2 Language-guided Local Region Localization
        ↓
3.1.3 Fine-grained Attribute Extraction
        ↓
End-to-end 3.1 integration and parameter freeze
```

## 2. 3.1.1 Instance Segmentation

### Completed

- DeepFashion2 structure and annotation parsing.
- Mask2Former COCO-pretrained baseline inference.
- DeepFashion2 `Dataset` / `DataLoader` and independent binary instance-mask validation.
- Person ROI extraction and failure-case review.
- Fixed-margin ROI and adaptive-margin ROI experiments.
- Adaptive-margin rule currently recorded as:
  - bbox area ratio `< 0.30`: margin `0.20`;
  - `0.30 <= ratio < 0.60`: margin `0.10`;
  - ratio `>= 0.60`: margin `0.05`.

### Current interpretation

The COCO-pretrained Mask2Former baseline is used to validate the segmentation/ROI workflow. It is not treated as a final DeepFashion2-trained fashion-category segmenter. Person-mask-only cropping can remove garment extremities or accessories, while adaptive bbox margins are more robust for typical person scales. Extreme close-ups and multi-person target selection remain important edge cases.

### Current-week consolidation

- Re-check input resolution and confidence threshold.
- Re-check minimum-region filtering and post-processing.
- Confirm the final 3.1.1 baseline configuration used by the integrated 3.1 pipeline.

## 3. 3.1.2 Language-guided Localization

### Completed development

- Grounding DINO baseline: `IDEA-Research/grounding-dino-tiny`.
- Prompt-granularity and threshold pilot experiments.
- Prompt optimization / ablation code and automated summaries.
- 300-image evaluation using 3 non-overlapping groups of 100 images.
- 40-case manual localization audit and bbox-area diagnostics.
- ROI-conditioned grounding diagnostic.
- Target-balanced candidate generation and manual positive screening.
- 88-case verified-positive benchmark: 22 cases each for sleeve, collar, button, zipper.
- FULL / ROI / LOCAL controlled evaluation on GPU.
- `correct / coarse / wrong / missed` manual-quality scheme.

### Final automatic 88-case result

| Target | N | FULL | ROI | LOCAL |
| --- | ---: | ---: | ---: | ---: |
| sleeve | 22 | 40.9% | 40.9% | 36.4% |
| collar | 22 | 68.2% | 68.2% | 45.5% |
| button | 22 | 18.2% | 18.2% | 9.1% |
| zipper | 22 | 4.5% | 4.5% | 9.1% |
| **Overall** | **88** | **33.0%** | **33.0%** | **25.0%** |

Mean detected-box area ratio:

| Condition | Mean top-box area ratio |
| --- | ---: |
| FULL | 0.963 |
| ROI | 0.963 |
| LOCAL | 0.219 |

### Maintained conclusion

1. FULL and ROI predictions are effectively identical on the verified-positive benchmark; simple ROI cropping does not add value for this already garment-focused sample set.
2. Automatic detection/prediction coverage is not localization accuracy. Many FULL/ROI detections are garment-level or image-level boxes.
3. Tighter LOCAL windows produce much smaller boxes but do not improve overall prediction coverage.
4. Sleeve and collar are materially easier than button and zipper.
5. Button and zipper remain small-object failure modes and should not be forced to match larger-part performance in the current stage.
6. The local-crop result is mixed/negative and is retained as an error-analysis result rather than described as a successful optimization.

### Mentor-feedback follow-up before final freeze

- Finalize per-class manual localization metrics: strict accuracy, usable rate, missed rate.
- Re-check generic vs garment-specific vs part-specific language prompts on a small controlled subset.
- Test segmentation-first tight crop on a small diagnostic subset.
- Use attention/response visualization where technically practical to distinguish input-side crop issues from cross-modal alignment issues.
- Run only small targeted button/zipper checks (multi-scale / high-resolution local input / later oversampling or augmentation), rather than repeated tuning on the final 88-case benchmark.

## 4. 3.1.3 Fine-grained Attribute Extraction

Status: **planned for the current development week; not yet claimed as complete**.

Initial attribute schema:

- color;
- pattern / texture;
- sleeve length / sleeve type;
- collar / neckline;
- closure type;
- optional confidence / uncertain state.

Planned work:

1. define a structured JSON schema;
2. implement a first baseline from image / garment ROI / localized region to structured attributes;
3. run a 50-80 sample first pass and record field-level errors;
4. adjust prompts / candidate values / thresholds / fallback rules;
5. expand to roughly 100-150 samples if time permits;
6. integrate 3.1.1 -> 3.1.2 -> 3.1.3.

## 5. Current milestone definition of “3.1 complete”

The 3.1 milestone is considered complete when:

- all three modules can run;
- key parameters are recorded and frozen for the baseline;
- each module has an evaluation record and error analysis;
- interfaces between 3.1.1, 3.1.2 and 3.1.3 are consistent;
- an end-to-end example can be reproduced;
- code, configuration, reports and documentation are synchronized.

PRD 3.2 starts only after this consolidation.
