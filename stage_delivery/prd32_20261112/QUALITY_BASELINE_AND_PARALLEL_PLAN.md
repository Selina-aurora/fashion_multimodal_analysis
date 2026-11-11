# Template quality baseline and multitask parallelism: staged decision plan

**Origin:** 2026-11-11 supervisor feedback. This is a plan, not a completed A/B or production change.

## A. Freeze one equivalent quality baseline

- **Reference**: original Tags + VQA + Qwen Copy path, with original outputs and original Guard decisions. Freeze model, weights, preprocessing, target IDs, image hashes, prompt/format contracts and generation settings.
- **Candidate**: Tags + deterministic copy template. Verify that its schema and scope match the intended business output; a template with no VQA is **not** a complete alternative to an endpoint that promises VQA.
- **Population**: use the same fixed DEV60 target IDs first; include all eight clothing categories, instances with V5-vs-Tag category conflict, occlusion, multiple garments, and ambiguous visible detail. Review an independent sealed holdout only after development and selection are frozen.
- **Human review**: reviewers should not be shown which arm generated the copy. Score object-binding correctness, category consistency, visible-attribute support, missing key details, unsubstantiated material/process claims, fluency/usefulness, style/length conformance, and refusal/uncertainty handling. Publish raw counts and denominators, not only averages.
- **Hard-fail cases**: wrong garment/target, hallucinated category or material, incorrectly asserted print/logo, reference to a red bounding box in merchant copy, unsupported specific claim, Guard bypass.
- **Acceptance**: establish a written non-inferiority margin with supervisor **before** judging results; no candidate promoted purely because it is faster. Existing 16-case architecture A/B found **9/16 template statements inherited wrong category** from Tags; the additional **16/16 developer-reviewed cases were judged not ready to replace original copy** in the user-confirmed assisted review (`evidence/architecture/AB16_RULE_COPY_REVIEW_USER_CONFIRMED.csv`). These are not independent blind GT and do not by themselves establish final failure rates.

## B. Evaluate parallelism after quality is stable

Compare on the **same end-to-end request contract**:

1. Original serial Tags → VQA → Copy and original Guard (repro baseline).
2. True concurrent Tags/VQA/Copy calls (if GPU scheduling and memory allow); preserve equivalent prompts, image, output and Guard inputs.
3. VQA on-demand routing, **only** for requests that explicitly need VQA; separately evaluate merchant-copy-only requests. Do not call omission of a task 'acceleration' for an identical full task.
4. Optional shared visual prefill/batching, if implementation supports it; record whether model outputs and quality change.

Record `n`, task mix, image/target association, cache/warm state, GPU memory and contention, per-task latency, complete E2E p50/p95, throughput, schema validity, refusal and error rates. Use repeated runs with randomized arm order; distinguish concurrency gain from shorter output. **No production Guard bypass**.

## C. Current explicit non-decisions

- No changes to `main`, runtime model weights, output caps or released Guard.
- No claimed PRD 100-ms single-product or 400-ms full-interaction acceptance from partial-task timings.
- No training/fine-tuning on Oracle8 or DEV60 labels without separating a new independent blind test.
