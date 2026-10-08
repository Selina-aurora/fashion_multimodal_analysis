# Verified-Positive Grounding Evaluation

## Experimental Setup

- Model: `IDEA-Research/grounding-dino-tiny`
- Detection threshold: 0.30
- Conditions: full image, oracle garment ROI, target-conditioned local enlarged crop
- Garment ROI margin ratio: 0.050
- Local crop enlargement factor: 1.00x
- Evaluated cases: 8
- Larger-part group: sleeve + collar
- Small-object group: button + zipper

> The garment ROI uses DeepFashion2 annotation and is an oracle diagnostic condition.

> The local enlarged condition uses only fixed target-conditioned spatial priors inside the garment ROI. It does not use a fine-grained part ground-truth bbox.

## Detection Rate by Target

| Target | N | Full | ROI | Local | ROI-Full | Local-Full |
|---|---:|---:|---:|---:|---:|---:|
| sleeve | 2 | 50.0% | 50.0% | 50.0% | +0.0% | +0.0% |
| collar | 2 | 50.0% | 50.0% | 50.0% | +0.0% | +0.0% |
| button | 2 | 0.0% | 0.0% | 50.0% | +0.0% | +50.0% |
| zipper | 2 | 0.0% | 0.0% | 0.0% | +0.0% | +0.0% |

## Detection Rate by Scale Group

| Group | Full | ROI | Local | ROI-Full | Local-Full |
|---|---:|---:|---:|---:|---:|
| larger_part | 50.0% | 50.0% | 50.0% | +0.0% | +0.0% |
| small_object | 0.0% | 0.0% | 25.0% | +0.0% | +25.0% |

## Manual Localization Audit

Detection rate is not localization accuracy.

Review the triplet contact sheets and fill:

`D:\projects\fashion_multimodal_analysis\reports\verified_positive_evaluation\triplet_manual_audit.csv`

Allowed quality labels:

- `correct`: target localized appropriately
- `coarse`: target covered, but box is too broad
- `wrong`: detection exists but localization is wrong
- `missed`: no detection