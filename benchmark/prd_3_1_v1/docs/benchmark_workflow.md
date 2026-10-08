# Benchmark Workflow v1

## Build once

1. Freeze schemas.
2. Annotate benchmark_test.
3. Review/adjudicate annotations.
4. Freeze manifest checksums.
5. Register acceptance profile.

## For every candidate model

1. Save run metadata.
2. Run 3.1.1 benchmark.
3. Run 3.1.2 benchmark.
4. Run 3.1.3 Track A.
5. Run 3.1.3 Track B.
6. Generate end-to-end failure attribution.
7. Compare with PRD gate + frozen baseline + previous release.
8. Review regression cases.
9. Mark PASS / FAIL / INSUFFICIENT_COVERAGE.
10. Archive all per-instance outputs.

## Never do on benchmark_test

- prompt tuning
- threshold tuning
- changing label definitions
- selecting only favorable examples
- manually correcting predictions
- excluding failures after seeing results
