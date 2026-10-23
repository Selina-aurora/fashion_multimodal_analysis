# Findings grounded in Stage 11 GPU output

- `STAGE11_ROOTCAUSE_DIAGNOSTIC_COMPLETED`: 4 cases, 16 queries, no writes to repository.
- Three DINO comparisons (native single, helper single, helper batch) passed 16/16 **on the same isolated white-background crop and standard `PROMPTS`**. This does **not** compare against the strict production DINO input.
- Live `ModelBackends.ground()` obtains `strict_region_grounding` from config and, in strict mode, calls DINO on the **raw crop** with **`REGION_PROMPTS`**; it filters with `phrase_matches`, rejects overly large local region proposals using `max_local_region_crop_area_ratio=0.65`, and limits SAM masks via `constrain_local_mask`.
- Stage 10 experimental path instead uses **white-background isolated crop** and standard `PROMPTS` and omits parts of strict gating/refinement. In Stage 10, production statuses were 8 detected/3 missed/5 geometry-rejected while experimental batch returned 16 detected.
- Primary finding: Stage 10 is not an apples-to-apples parity experiment; never integrate its experimental path as currently implemented. Whether each individual difference contributes to every error is not yet isolated.
