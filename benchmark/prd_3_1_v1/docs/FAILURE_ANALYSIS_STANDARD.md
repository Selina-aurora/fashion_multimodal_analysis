# Failure Analysis Standard v1

Every failed benchmark case must receive a machine-readable failure code.

## 3.1.1 codes

- `SEG_MISS`
- `SEG_FALSE_POSITIVE`
- `SEG_WRONG_CLASS`
- `SEG_BBOX_SHIFT`
- `SEG_MASK_UNDER`
- `SEG_MASK_OVER`
- `SEG_MASK_FRAGMENT`
- `SEG_DUPLICATE`
- `SEG_SMALL_OBJECT`
- `SEG_OCCLUSION`
- `SEG_OTHER`

## 3.1.2 codes

- `GROUND_MISS`
- `GROUND_WRONG_REGION`
- `GROUND_COARSE`
- `GROUND_PARTIAL`
- `GROUND_MULTI_REGION`
- `GROUND_QUERY_MISMATCH`
- `GROUND_OCCLUSION`
- `GROUND_SMALL_REGION`
- `GROUND_OTHER`

## 3.1.3 codes

- `ATTR_WRONG_LABEL`
- `ATTR_AMBIGUOUS_VISUAL`
- `ATTR_NOT_APPLICABLE`
- `ATTR_ROI_PROPAGATION`
- `ATTR_MASK_PROPAGATION`
- `ATTR_BBOX_PROPAGATION`
- `ATTR_LOW_RESOLUTION`
- `ATTR_CLASSIFIER_INSTABILITY`
- `ATTR_OTHER`

## End-to-end stage attribution

Primary stage values:
- `3.1.1_detection`
- `3.1.1_classification`
- `3.1.1_mask`
- `3.1.2_grounding`
- `3.1.3_attribute`
- `interaction`
- `unknown`

Do not assign a causal stage unless supported by a controlled comparison or direct annotation evidence.
