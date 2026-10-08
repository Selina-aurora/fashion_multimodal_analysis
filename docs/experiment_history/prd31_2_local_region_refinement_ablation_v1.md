# PRD 3.1.2 Local Region Refinement Ablation v1

## 1. Objective

This experiment evaluates whether additional local refinement strategies
can improve fine-grained fashion region localization after the initial
PRD 3.1.2 region coverage analysis.

The purpose is to analyze whether the observed localization limitation
comes from insufficient spatial resolution or insufficient mask quality.

Two refinement strategies are evaluated:

1.  Spatial high-resolution refinement (SPATIAL_HR)
2.  SAM-based mask refinement

The evaluation focuses on region localization quality, not only mask
compactness.

------------------------------------------------------------------------

## 2. Baseline Observation

The original PRD 3.1.2 40-case manual audit showed that fine-grained
region localization remains challenging.

The dominant failure mode was coarse localization:

-   The returned box often covered the whole garment.
-   Fine-grained regions such as collar, cuff, hem, shoulder, and waist
    were difficult to isolate.
-   A returned prediction did not necessarily indicate accurate part
    localization.

Therefore, refinement experiments were conducted to investigate possible
improvements.

------------------------------------------------------------------------

# 3. Spatial High-Resolution Refinement

## Setup

Spatial high-resolution refinement evaluates whether a more focused
local crop can improve region localization.

Evaluated regions:

-   collar
-   cuff
-   hem

Cases:

-   15-case manual audit

------------------------------------------------------------------------

## Results

  Region   Baseline                 SPATIAL_HR
  -------- ------------------------ ------------------------
  Collar   5/5 coarse               4/5 coarse, 1/5 missed
  Cuff     4/5 coarse, 1/5 missed   5/5 coarse
  Hem      3/5 coarse, 2/5 missed   5/5 missed

Overall:

-   The effective search area was reduced.
-   However, no transition from coarse localization to strict correct
    localization was observed.

------------------------------------------------------------------------

## Interpretation

Spatial high-resolution cropping improves input focus, but does not
solve semantic part grounding.

The model still tends to identify the dominant garment area instead of
isolating the requested fashion part.

------------------------------------------------------------------------

# 4. SAM Refinement

## Setup

SAM refinement evaluates whether improved mask generation can recover
fine-grained region boundaries.

Evaluated regions:

-   collar
-   cuff
-   hem

Cases:

-   9-case manual audit

------------------------------------------------------------------------

## Results

Overall:

  Metric      Result
  --------- --------
  Correct        0/9
  Coarse         8/9
  Wrong          1/9

Region-level:

-   Collar: 3/3 coarse
-   Cuff: 2/3 coarse, 1/3 wrong
-   Hem: 3/3 coarse

------------------------------------------------------------------------

## Automatic Metrics

Average proposal box area ratio:

  Region      Before   After SAM
  --------- -------- -----------
  Collar      0.2959      0.2611
  Cuff        0.2319      0.2026
  Hem         0.6811      0.6297
  Overall     0.4030      0.3645

SAM improves mask compactness and boundary quality, but strict semantic
localization is not improved.

------------------------------------------------------------------------

# 5. Comparison and Analysis

Both refinement strategies provide partial improvements:

  -----------------------------------------------------------------------
  Method                  Improvement             Limitation
  ----------------------- ----------------------- -----------------------
  Spatial HR              More focused local      Cannot distinguish
                          input                   semantic parts

  SAM refinement          Cleaner object masks    Depends on initial
                                                  coarse prompt
  -----------------------------------------------------------------------

The main bottleneck is not only segmentation quality, but semantic
fine-grained part grounding.

------------------------------------------------------------------------

# 6. Engineering Conclusion

The refinement experiments show:

1.  Spatial cropping is useful as a region proposal or attention
    restriction mechanism.
2.  SAM is useful for mask refinement after a reasonable proposal is
    available.
3.  Neither method alone solves fine-grained fashion part localization.

Therefore:

-   Keep Grounding-based proposal and SAM refinement as documented
    baselines.
-   Do not continue tuning SAM alone.
-   Future improvements should focus on part-aware
    grounding/segmentation methods or explicit part-level supervision.

The current PRD 3.1.2 pipeline successfully identifies the localization
bottleneck and provides evidence for future optimization directions.
