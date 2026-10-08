# PRD 3.1.2 Attribute Grounding Error Analysis

## 1. Overview

This document summarizes the fine-grained attribute grounding evaluation and error analysis for PRD 3.1.2.

The objective is to analyze attribute-level localization limitations and identify optimization directions.

Evaluated attributes:
- sleeve
- collar
- button
- zipper

A manual localization audit was conducted to classify grounding failures.

## 2. Evaluation Summary

- Total manually audited cases: 40

## 3. Error Distribution

| Error Type | Count | Ratio |
|---|---:|---:|
| target_absent | 15 | 37.5% |
| wrong_part | 5 | 12.5% |
| whole_garment | 4 | 10.0% |
| face_head | 4 | 10.0% |
| non_clothing_object | 4 | 10.0% |
| background_object | 2 | 5.0% |
| whole_image | 2 | 5.0% |

## 4. Main Failure Patterns

### 4.1 Target Attribute Absence

The largest failure category is target_absent.

- Cases: 15/40
- Ratio: 37.5%

These failures occur when the requested attribute is not clearly visible or does not exist in the image.

Improving localization precision alone cannot solve these cases because the target attribute itself is unavailable.

### 4.2 Coarse Garment-Level Localization

Related errors:
- whole_garment: 4 cases
- wrong_part: 5 cases

Total:
- 9/40 cases
- 22.5%

The model can identify the garment region but fails to localize the fine-grained attribute.

### 4.3 Semantic Confusion

Related errors:
- face_head: 4 cases
- non_clothing_object: 4 cases
- background_object: 2 cases

Total:
- 10/40 cases
- 25%

These errors are caused by confusion between target attributes and visually similar regions.

## 5. Attribute Visibility Filtering Analysis

Manual analysis indicates that many grounding failures are caused by unavailable attributes rather than inaccurate localization.

A diagnostic filtering analysis was conducted by excluding samples labeled as target_absent.

| Metric | Value |
|---|---:|
| Total audited cases | 40 |
| Target absent cases | 15 |
| Potentially avoidable failures | 37.5% |

## 6. Optimization Direction

### 6.1 Attribute Visibility Verification

Introduce an attribute visibility check before grounding:

Image -> Attribute Visibility Check -> Grounding Model

If the attribute is not visible, the system can reject or skip unnecessary grounding.

### 6.2 Context-aware Localization

Future optimization directions:
- adaptive ROI selection
- context-preserving crop
- attribute-specific prompts

## 7. Conclusion

The PRD 3.1.2 evaluation shows that fine-grained attribute grounding is limited by both localization precision and attribute visibility.

A considerable proportion of failures originate from unavailable attributes, while remaining errors mainly involve coarse garment-level localization and semantic confusion.

Future optimization will focus on attribute awareness and context-aware grounding strategies.
