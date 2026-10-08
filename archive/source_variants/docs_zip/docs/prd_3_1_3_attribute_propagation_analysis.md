# PRD 3.1.3 Attribute Propagation Analysis

## 1. Objective

This stage evaluates whether predicted ROI regions can preserve attribute information during attribute propagation.

The goal is to analyze the consistency between attributes extracted from ground-truth ROI and predicted ROI.

Note:
This evaluation measures attribute propagation consistency, NOT final attribute accuracy against human ground truth.

---

## 2. Pipeline

The PRD 3.1.3 attribute propagation pipeline is:

```
Instance Segmentation
        |
        v
Predicted ROI Extraction
        |
        v
Attribute Prediction
        |
        v
Attribute Propagation Consistency Analysis
```

---

## 3. Evaluation Setting

Configuration:

- Manifest:
  `configs/prd_3_1_predicted_roi_matched_v2.csv`

- Matched ROI pairs:
  12

The evaluation compares matched GT ROI and predicted ROI pairs.

---

## 4. Attribute Agreement Results

| Attribute | Agreement |
|---|---:|
| primary_color | 1.0000 (12/12) |
| fashion_style | 0.7500 (9/12) |
| silhouette_fit | 0.7500 (9/12) |
| pattern | 0.5833 (7/12) |
| sleeve_length | 0.5714 (4/7) |
| neckline | 0.1429 (1/7) |

---

## 5. Analysis

### 5.1 Stable Attributes

Primary color shows strong propagation consistency:

- primary_color agreement: 12/12

This indicates that color-related attributes are relatively robust to ROI replacement.

---

### 5.2 Sensitive Attributes

Fine-grained structural attributes show lower consistency.

Examples:

- neckline
- sleeve_length

These attributes depend more strongly on precise ROI localization and visual details.

---

## 6. Limitations

The current evaluation is a pilot study.

Limitations:

- The matched ROI pilot set is small.
- Category coverage is incomplete.
- Results should not be interpreted as final 8-class acceptance.

---

## 7. Conclusion

The PRD 3.1.3 pilot evaluation demonstrates that predicted ROI regions can preserve several visual attributes after ROI replacement.

Color and high-level style attributes show stronger consistency, while fine-grained structural attributes remain sensitive to ROI quality.

Future optimization should focus on improving ROI localization quality and attribute-aware region selection.
