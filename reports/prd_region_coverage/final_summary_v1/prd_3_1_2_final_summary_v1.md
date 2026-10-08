# PRD 3.1.2 Final Summary v1

## Evaluation rubric

- correct: requested region is localized tightly and semantically correctly
- coarse: requested region is included, but the box is substantially broader than the target
- wrong: prediction falls on another region/object
- missed: no usable prediction
- usable rate = (correct + coarse) / N

## A. Frozen 40-case baseline (comparable across all 8 regions)

|Region|N|Correct|Coarse|Wrong|Missed|Strict Acc.|Usable|Missed Rate|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|collar|5|0|5|0|0|0.0%|100.0%|0.0%|
|cuff|5|0|4|0|1|0.0%|80.0%|20.0%|
|hem|5|0|3|0|2|0.0%|60.0%|40.0%|
|pocket|5|0|0|0|5|0.0%|0.0%|100.0%|
|shoulder|5|0|2|0|3|0.0%|40.0%|60.0%|
|waist|5|0|3|0|2|0.0%|60.0%|40.0%|
|pattern|5|1|2|0|2|20.0%|60.0%|40.0%|
|decoration|5|0|0|0|5|0.0%|0.0%|100.0%|
|OVERALL|40|1|19|0|20|2.5%|50.0%|50.0%|

## B. Targeted follow-up diagnostics

> These follow-up conditions are region-specific diagnostics and should not be aggregated as a single acceptance metric.

|Region|Condition|N|Correct|Coarse|Wrong|Missed|Strict Acc.|Usable|Interpretation|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|collar|Spatial-HR|5|0|4|0|1|0.0%|80.0%|Spatial crop did not tighten localization; SAM pilot also remained coarse.|
|cuff|Spatial-HR|5|0|5|0|0|0.0%|100.0%|Response availability improved, but localization remained coarse; supervised/top-k diagnostics still showed unstable candidate quality/ranking.|
|hem|Spatial-HR|5|0|0|0|5|0.0%|0.0%|Fixed spatial prior was unstable for hem; later supervised candidate diagnostics showed some feasibility but not robust acceptance-level localization.|
|shoulder|Spatial refinement visual review|5|0|5|0|0|0.0%|100.0%|Misses were recovered, but boxes remained broad upper-body/shoulder-inclusive regions.|
|waist|Spatial refinement visual review|5|0|5|0|0|0.0%|100.0%|Misses were recovered, but boxes remained broad waist-inclusive regions.|
|pattern|Spatial refinement visual review|5|1|4|0|0|20.0%|100.0%|Misses were recovered; only the case where the pattern occupies nearly the whole garment is strict-correct.|
|pocket|Prompt-scale diagnostic visual review|5|0|5|0|0|0.0%|100.0%|Candidate availability recovered, but boxes generally covered the whole trousers/outerwear rather than a tight pocket.|
|decoration|Prompt-scale diagnostic visual review|5|1|4|0|0|20.0%|100.0%|Candidate availability recovered; one large graphic was localized reasonably, while most boxes remained coarse.|

## Final conclusion

PRD 3.1.2 has completed an 8-region Grounding-DINO baseline, manual error audit, and region-specific diagnostics. The main failure mode has shifted from candidate absence to localization granularity. Spatial cropping can recover missed shoulder/waist/pattern cases, and prompt/scale changes can recover pocket/decoration responses, but most outputs remain garment-level or coarse. Collar/cuff/hem diagnostics also show that local high-resolution crops, SAM refinement, and a small supervised detector do not consistently produce tight local-region boxes. The current results are therefore retained as a feasibility/diagnostic baseline rather than an acceptance-level localization result; further minor prompt/threshold/crop tuning is not recommended at this stage.
