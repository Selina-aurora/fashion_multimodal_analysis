# PRD 3.1 Pipeline Progress Summary v1

## 1. Objective

This report summarizes the progress of the PRD 3.1 pipeline, including:

1.  PRD 3.1.1 instance segmentation
2.  PRD 3.1.2 fine-grained region localization and refinement
3.  PRD 3.1.3 predicted ROI attribute propagation

The goal is to validate the complete pipeline from instance-level ROI
generation to downstream attribute extraction, identify current
bottlenecks, and provide directions for future optimization.

------------------------------------------------------------------------

# 2. PRD 3.1.1 Instance Segmentation

## Experimental Setup

-   Model: Mask R-CNN v3-B1-dataexp
-   Checkpoint:
    `outputs/prd_instance_segmentation/maskrcnn_8class_v3_b1_dataexp/checkpoint_last.pth`
-   Device: CUDA

The model generates predicted garment ROIs including:

-   bounding boxes
-   masks
-   cropped regions

These predicted ROI outputs are used as inputs for downstream attribute
extraction.

## Status

PRD 3.1.1 benchmark and freeze process have been completed.

The predicted ROI outputs provide the required interface for subsequent
PRD 3.1.2 and PRD 3.1.3 analysis.

------------------------------------------------------------------------

# 3. PRD 3.1.2 Region Localization and Refinement

## 3.1.2 Baseline Analysis

A 40-case manual audit was conducted to evaluate fine-grained region
localization.

Evaluated regions included:

-   collar
-   cuff
-   hem
-   shoulder
-   waist
-   pattern
-   pocket
-   decoration

The analysis showed that the major failure mode was coarse localization.

The model could often return garment-related regions, but the predicted
regions were not sufficiently precise for fine-grained fashion parts.

------------------------------------------------------------------------

## 3.1.2 Refinement Experiments

Two refinement strategies were evaluated:

## A. Spatial High-Resolution Refinement

Purpose:

Evaluate whether local high-resolution crops improve fine-grained
localization.

Dataset:

-   15-case manual audit

Result:

-   The local crop reduced the search area.
-   However, strict semantic part localization was not significantly
    improved.

Conclusion:

Spatial high-resolution refinement is useful as an attention/proposal
mechanism, but it does not fully solve semantic part grounding.

------------------------------------------------------------------------

## B. SAM-based Refinement

Purpose:

Evaluate whether improved mask quality can recover fine-grained region
boundaries.

Dataset:

-   9-case manual audit

Results:

  Metric                   Result
  ---------------------- --------
  Correct localization        0/9
  Coarse localization         8/9
  Wrong                       1/9

Average proposal box area ratio:

  Region      Before   After SAM
  --------- -------- -----------
  Collar      0.2959      0.2611
  Cuff        0.2319      0.2026
  Hem         0.6811      0.6297
  Overall     0.4030      0.3645

Conclusion:

SAM improves mask compactness, but does not solve semantic fashion-part
localization.

------------------------------------------------------------------------

# 4. PRD 3.1.3 Predicted ROI Attribute Propagation

## Objective

This experiment evaluates whether predicted ROIs from PRD 3.1.1 can
directly support downstream attribute extraction.

The evaluation measures propagation consistency between GT ROI and
predicted ROI outputs.

It is not an absolute attribute accuracy evaluation.

------------------------------------------------------------------------

## Experimental Setup

ROI matching:

-   one-to-one matching
-   bbox IoU \>= 0.5

GT information is used only as comparison keys:

-   source_image
-   garment_id

Attribute extraction input:

-   predicted crop
-   predicted mask

------------------------------------------------------------------------

## Full-validation Results

Matched ROI pairs:

-   22

Category distribution:

  Category      Count
  ----------- -------
  Top               7
  Pants             5
  Skirt             0
  Outerwear         1
  Dress             5
  Shoe              1
  Bag               2
  Accessory         1

------------------------------------------------------------------------

## Attribute Propagation Agreement

  Attribute              Agreement
  ---------------- ---------------
  Primary color      15/18 (83.3%)
  Pattern            14/18 (77.8%)
  Sleeve length       8/13 (61.5%)
  Neckline            7/13 (53.8%)
  Silhouette fit     16/18 (88.9%)
  Fashion style      11/18 (61.1%)

------------------------------------------------------------------------

# 5. Overall Findings

The PRD 3.1 pipeline has been successfully validated from ROI generation
to downstream attribute extraction.

Main findings:

1.  Instance-level ROI generation provides usable inputs for downstream
    modules.
2.  Fine-grained fashion-part localization remains the primary
    bottleneck.
3.  Spatial HR refinement improves input focus but does not fully solve
    semantic grounding.
4.  SAM refinement improves mask compactness but cannot recover missing
    semantic part understanding.
5.  Global attributes such as color and silhouette are relatively robust
    to ROI variation.
6.  Local attributes such as neckline and sleeve length are more
    sensitive to ROI quality.

------------------------------------------------------------------------

# 6. Future Optimization Direction

Based on current experiments, future improvements should focus on:

1.  Part-aware grounding methods.
2.  Explicit fine-grained region supervision.
3.  Region-specific localization strategies.
4.  Larger category-balanced validation sets.

Current refinement methods are retained as documented baselines and
failure-analysis references.
