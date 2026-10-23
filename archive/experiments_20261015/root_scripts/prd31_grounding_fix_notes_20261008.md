# PRD 3.1 small-chain review and diagnostic correction

The eight development images produced 34 predicted instances and no runtime/interface errors. That does not establish semantic correctness.

All 272 region queries were marked detected. 265 returned only garment/the garment. For 27 of 34 instances all eight region boxes were exactly identical. Shoes were assigned collar/cuff/shoulder/waist. These are clear failures of target-specific region grounding.

Visual review: case_02 contains pants, with an overlapping dress prediction; case_03 black skirt was predicted outerwear; case_04 has overlapping outerwear/dress predictions. Do not change the frozen labels or hide these predictions.

This patch enables strict_region_grounding only in the diagnostic smoke script. The default historical prompt path remains available for other configurations. It uses short part-specific queries, rejects proposals without a target word, and routes structural queries using the predicted garment category. Category routing is a prediction-side design choice, not a ground-truth absence label; wrong predicted categories can also cause wrong routing.

No mask-area threshold, score threshold, model weight, manifest or PRD target is changed. The checkpoint remains the balanced150 run, checked against workers=2 and the recorded train/val manifest hashes.

Without a detected predicted collar, the existing attribute path returns localization_missing for neckline. Phrase matching is only a guard against known wrong labels, not proof of correct region boundaries. Missed/rejected queries remain visible. Pattern/decoration prompts are diagnostic choices and require visual review for coverage.

The repeated run selects the same images and writes a fresh timestamp directory. Summary adds region and attribute status counts. Local verification covered syntax, query routing, phrase rejection before SAM, source-coordinate mapping, and propagation of missing localization. Real CUDA accuracy/speed has not been validated here.

ZIP includes only two integration modules and the smoke script. Back up these three paths before extraction. Reports/checkpoints/data are untouched.
