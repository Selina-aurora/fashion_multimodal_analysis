# PRD 3.1 Acceptance Criteria v1

This document distinguishes **PRD gates** from additional diagnostic metrics.

## 3.1.1 Garment instance segmentation

PRD targets:
- segmentation latency: `<= 50 ms`
- mask IoU: `>= 0.85`

v1 operationalization:
- latency gate uses **mean per-instance module latency** under the frozen latency protocol;
- additionally report median and p95;
- mask quality gate uses **mean IoU over correctly localized and correctly classified matched instances**;
- additionally report `IoU >= 0.85` pass rate and per-class mean IoU.

Important: the per-class and high-IoU pass-rate statistics are mandatory diagnostics even though the PRD gives a single IoU target.

## 3.1.2 Language-guided local-region localization

PRD targets:
- localization accuracy: `>= 92%`
- latency: `<= 30 ms`

v1 operationalization:
- primary accuracy = strict localization success rate;
- strict success requires semantic target correctness and bbox IoU `>= 0.50`;
- `usable/coarse` is reported separately and cannot satisfy the strict PRD gate;
- latency follows the common frozen timing protocol.

## 3.1.3 Fine-grained attribute extraction

PRD targets:
- attribute accuracy: `>= 88%`
- extraction latency: `<= 20 ms`

v1 operationalization:
- primary accuracy is calculated against **human-reviewed GT labels**;
- denominator = eligible, non-ambiguous cases only;
- report per-attribute accuracy plus overall micro/macro accuracy;
- module acceptance uses Track A (controlled/GT ROI) to isolate the attribute module;
- Track B end-to-end accuracy is mandatory for product-chain reporting but is reported separately so upstream errors are not hidden.

## End-to-end 3.1

No new PRD numerical threshold is invented in v1.

The end-to-end report must show:
- successful garment detection/classification coverage;
- attribute accuracy on the complete product chain;
- propagation consistency;
- failure stage attribution.

## Pass status values

Use exactly:
- `PASS`
- `FAIL`
- `NOT_EVALUATED`
- `INSUFFICIENT_COVERAGE`

A metric with missing required class/attribute coverage cannot be marked PASS.
