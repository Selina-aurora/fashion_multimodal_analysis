# Core part detector v1 diagnostic

## Purpose
Explain why validation macro_recall@0.5 remains zero.

## Checkpoint
- epoch: 1
- saved macro recall: 0.0
- saved image success: 0.0

## Diagnostic threshold
Predictions are retained down to score >= 0.001.

The original validation metric requires:
- correct class
- score >= 0.25
- IoU >= 0.50
