# PRD 3.1 Final Summary

## Overview

PRD 3.1 focuses on building and validating a fashion understanding pipeline from instance segmentation to fine-grained attribute analysis.

The pipeline contains three stages:

```
3.1.1 Instance Segmentation
          |
          v
3.1.2 Fine-grained Attribute Grounding
          |
          v
3.1.3 Attribute Propagation Analysis
```

---

# 1. PRD 3.1.1 Instance Segmentation

## Objective

Establish a stable garment instance segmentation benchmark.

## Completed Work

- Core benchmark construction
- Instance segmentation evaluation
- Error case analysis
- Benchmark documentation and freeze process

## Outcome

The segmentation stage provides garment-level ROI extraction for downstream attribute understanding.

---

# 2. PRD 3.1.2 Fine-grained Attribute Grounding

## Objective

Evaluate fine-grained localization capability for garment attributes.

Evaluated attributes include:

- sleeve
- collar
- button
- zipper

## Error Analysis

Total manually audited cases:

- 40

Failure distribution:

| Error Type | Count | Ratio |
|---|---:|---:|
| target_absent | 15 | 37.5% |
| wrong_part | 5 | 12.5% |
| whole_garment | 4 | 10.0% |
| face_head | 4 | 10.0% |
| non_clothing_object | 4 | 10.0% |
| background_object | 2 | 5.0% |
| whole_image | 2 | 5.0% |

## Main Findings

The largest limitation comes from attribute visibility.

A considerable proportion of failures occur because the target attribute is unavailable or unclear in the image.

A visibility filtering analysis shows:

- Total cases: 40
- Target absent cases: 15
- Potentially avoidable failures: 37.5%

Optimization direction:

- attribute visibility verification
- context-aware localization
- attribute-specific grounding strategies

---

# 3. PRD 3.1.3 Attribute Propagation Analysis

## Objective

Evaluate whether predicted ROI regions can maintain attribute information after replacement.

## Evaluation Setting

- Matched ROI pairs: 12

Evaluation metric:

- Attribute propagation consistency

Note:
This is consistency analysis, not human-ground-truth attribute accuracy.

## Results

| Attribute | Agreement |
|---|---:|
| primary_color | 1.0000 |
| fashion_style | 0.7500 |
| silhouette_fit | 0.7500 |
| pattern | 0.5833 |
| sleeve_length | 0.5714 |
| neckline | 0.1429 |

## Findings

- Color attributes are relatively stable.
- Fine-grained structural attributes are more sensitive to ROI quality.

---

# 4. Overall Conclusion

PRD 3.1 establishes a complete fashion understanding pipeline:

1. Garment instance segmentation
2. Fine-grained attribute grounding
3. Attribute propagation analysis

The current evaluation identifies major limitations and provides optimization directions for future improvements.

Future work will focus on improving attribute awareness, ROI quality, and fine-grained understanding capability.
