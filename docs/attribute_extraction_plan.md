# 3.1.3 Fine-grained Attribute Extraction Plan

## Status

This module is the next implementation task in the current 3.1 milestone. The repository does **not** claim a completed 3.1.3 baseline yet.

## Initial schema

The first version will focus on high-frequency attributes that can be represented consistently:

```json
{
  "color": null,
  "pattern": null,
  "texture": null,
  "sleeve_length": null,
  "sleeve_type": null,
  "collar_type": null,
  "neckline": null,
  "closure": null,
  "confidence": null,
  "uncertain": false
}
```

## Planned baseline workflow

```text
Image / garment ROI / localized region
        ↓
Attribute extraction baseline
        ↓
Structured JSON
        ↓
Validation / post-processing
        ↓
Fallback for missing, conflicting or uncertain fields
```

## Current-week implementation targets

1. finalize field definitions and candidate values;
2. implement the first baseline;
3. build a 50-80 image first-pass set;
4. record field-level errors;
5. adjust prompt/candidate/threshold/post-processing settings;
6. expand to approximately 100-150 samples if time permits;
7. integrate with 3.1.1 and 3.1.2 for end-to-end testing.

This file should be converted from a planning document to a development/result record once the baseline is implemented.
