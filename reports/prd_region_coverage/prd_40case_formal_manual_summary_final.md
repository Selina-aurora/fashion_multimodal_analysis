# PRD 3.1.2 40-case coverage pilot — final manual audit

## Audit rubric

- **correct**: prediction substantially matches the requested region.
- **coarse**: requested region is included, but the box is much broader than the target.
- **wrong**: prediction lands on another region/object.
- **missed**: no prediction is returned.

The review is intentionally conservative. `pattern_019` is marked correct because the striped pattern spans essentially the whole trouser region, so a garment-wide box matches the target extent.

## Results

| Region | N | Correct | Coarse | Wrong | Missed | Strict Acc. | Usable Rate | Non-empty Pred. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| collar | 5 | 0 | 5 | 0 | 0 | 0.0% | 100.0% | 100.0% |
| cuff | 5 | 0 | 4 | 0 | 1 | 0.0% | 80.0% | 80.0% |
| hem | 5 | 0 | 3 | 0 | 2 | 0.0% | 60.0% | 60.0% |
| pocket | 5 | 0 | 0 | 0 | 5 | 0.0% | 0.0% | 0.0% |
| shoulder | 5 | 0 | 2 | 0 | 3 | 0.0% | 40.0% | 40.0% |
| waist | 5 | 0 | 3 | 0 | 2 | 0.0% | 60.0% | 60.0% |
| pattern | 5 | 1 | 2 | 0 | 2 | 20.0% | 60.0% | 60.0% |
| decoration | 5 | 0 | 0 | 0 | 5 | 0.0% | 0.0% | 0.0% |
| OVERALL | 40 | 1 | 19 | 0 | 20 | 2.5% | 50.0% | 50.0% |

## Main findings

- The overall non-empty prediction rate is 50.0%, but strict localization accuracy is much lower, confirming that a returned box is not equivalent to fine-grained localization.
- Collar has the strongest coverage response, but all five detections remain garment-level/coarse.
- Cuff, hem, shoulder, waist, and pattern show partial feasibility, but the dominant failure mode is still an oversized garment-level box.
- Pocket and decoration are complete misses in this pilot at threshold 0.3.
- `pattern_019` is the only strict-correct case under the current rubric because the pattern itself occupies nearly the full trouser region.
- The next optimization should therefore focus on target-specific spatial priors/local high-resolution crops for collar/cuff/hem, while pocket/decoration require a separate prompt/model or region-proposal strategy.