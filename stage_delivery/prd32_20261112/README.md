# PRD 3.2 · Development evidence and reproduction snapshot (2026-11-12)

> **Status: SHADOW_BLOCKED / research-only.** This folder collects the November 11 work **without changing any production code**, overriding `main`, merging unreviewed candidates, or claiming PRD acceptance. The current repo default `main` is an October 26 PRD 3.1 stage checkpoint. The October 27 PRD 3.2 evidence is preserved in the parent branch `stage/prd32-20261027-closeout`.

## What is included

| Track | Verified experimental scope | Result / limitation | Evidence |
|---|---|---|---|
| Performance root cause | DEV60: 60 images, 171 V5 predicted instances, 513 Qwen calls (171 Tags + 171 VQA + 171 Copy) | Three-task mean **2362.77 ms**, P95 **2759.17 ms**; Tags + Copy consume ~87.4% of host E2E | `evidence/performance/` |
| Decode probe | RTX 4090 real GPU probe | Instrumented decode measurements; do not treat as end-to-end or kernel-only service speed | `evidence/performance/DECODE_GPU_*` |
| Architecture A/B V1 | 16 targets, four pathways, 112 formal Qwen calls | Three-task baseline 2325.01 ms; VQA-only 284.93 ms; Tags + Copy 2098.51 ms; Tags + template 1316.24 ms (means). **Different functional coverage**; 9/16 template outputs inherited incorrect Tags category. In a developer-assisted, user-confirmed 16-case copy review, 16/16 were *not approved as replacements* (not blind GT). | `evidence/architecture/` |
| VQA-only V2 | 60 DEV images, 171 predicted instances, one color VQA each | Mean **298.529 ms**, P95 **547.447 ms**; 171/171 same answers as frozen VQA baseline, **not 171 correct answers**. Original three-task Guard not used. | `evidence/vqa_only/` |
| VQA concise V2 | 34 targeted DEV requests × 3 prompts = 102 Qwen calls | V2 mean 321.02 ms vs baseline 416.04 ms; V1 mean 307.31 ms; content/abstention problems remain. | `evidence/vqa_concise/` |
| VQA multi-type pilot | 16 DEV targets / 32 questions; four question types | Single-task Qwen mean **639.46 ms**, with visible cross-target leakage and unsupported details; reviews are exploratory. | `evidence/vqa_multitype/` |
| Oracle8 A/B | 7 pairs (2 geometry, 5 marker color) | User accepted **non-blind** visual reviews. Marker-color study has a prompt mismatch: cyan images were asked about a *red* box; no causal color conclusion. | `evidence/oracle8/` |
| Neutral 5-pair red/cyan A/B | 5 targets × 2 marker colors; same neutral prompt; 10 GPU answers | **GPU inference completed, 5/5 pairs user-confirmed in chat on 2026-11-12**. Red better 2, cyan better 1, both materially wrong 2. Qualitative DEV review only; **SHADOW_BLOCKED**, not blind GT. | `evidence/oracle8/neutral_user_confirmed/` and `repro/neutral_ab/` |

## Code layout

- `repro/` contains source-exact Python experiment scripts extracted from the original uploaded experiment packs; common dependencies are deduplicated, with unit tests under `repro/tests/`.
- `repro/data/` contains fixed DEV manifests and previously generated experiment metadata; the data are **development references**, not independent ground-truth labels.
- `repro/neutral_ab/` has its own standalone scripts and manifests; it requires the corresponding 10 marked PNGs from the separate original input pack (not published here).
- `evidence/` preserves source reports and small machine-readable results, not generated preview images or model weights.
- `PROVENANCE.json` maps every copied file to its exact original package path and SHA-256.

## Quality and release restrictions

1. All candidate routes remain shadow-only. Original three-task Guard was not validated for VQA-only / template paths.
2. A matching answer, a V5 frozen-mask IoU of 1.0, or an accepted developer visual note **does not establish truth accuracy**.
3. DEV60, hand-reviewed Oracle8, and targeted 34-case/16-case sets are not an independent blind test. **The separate holdout remains sealed**.
4. `main` is not modified here; deployment and PRD acceptance require frozen candidate, schema/quality review, representative DEV60 regression, and an independent holdout blind evaluation.
5. Experimental image previews, review HTML pages, Qwen/V5 checkpoints, and model caches are deliberately excluded from this public repository update.

## Next decision requested by supervisor

Do not repeatedly tune decode caps in isolation. **First quantify template-based output quality against the original multi-task baseline**: correct target identity/category, factuality of visible attributes, unsupported claims, omissions, copy specificity/utility, refusals and guard coverage. Then compare genuine multi-task parallelism or on-demand routing on the same request scope, without silently dropping required functions. Measure mean/P95 E2E latency and GPU contention only after quality non-inferiority is established. See `QUALITY_BASELINE_AND_PARALLEL_PLAN.md`.
