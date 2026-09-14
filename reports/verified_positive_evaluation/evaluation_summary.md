# Verified-positive Grounding Evaluation — Final 88-case Automatic Result

## Experimental setup

- Model: `IDEA-Research/grounding-dino-tiny`
- Detection threshold: 0.30
- Conditions: FULL, annotation-assisted garment ROI, target-conditioned LOCAL
- ROI margin ratio: 0.05
- Local enlargement factor: 1.0
- Evaluated cases: 88 (22 per target)

> ROI is an oracle diagnostic condition based on DeepFashion2 garment annotation. LOCAL uses fixed target-conditioned spatial priors and does not use fine-grained target ground truth.

## Automatic prediction coverage

| Target | N | FULL | ROI | LOCAL |
| --- | ---: | ---: | ---: | ---: |
| sleeve | 22 | 40.9% | 40.9% | 36.4% |
| collar | 22 | 68.2% | 68.2% | 45.5% |
| button | 22 | 18.2% | 18.2% | 9.1% |
| zipper | 22 | 4.5% | 4.5% | 9.1% |
| **Overall** | **88** | **33.0%** | **33.0%** | **25.0%** |

## Interpretation

Detection/prediction coverage is not localization accuracy. FULL/ROI predictions are often garment-level boxes. LOCAL substantially reduces predicted-box area but does not improve overall coverage. Manual `correct / coarse / wrong / missed` review is used for true localization-quality reporting and remains preliminary until low-confidence cases are finalized.
