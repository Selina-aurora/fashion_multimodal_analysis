# GitHub branch status and proposed active development route (checked 2026-11-12)

Repository: `Selina-aurora/fashion_multimodal_analysis` (public). Default branch: `main`. **This document is a review snapshot, not a GitHub branch rename/delete operation.**

| Branch | Current role / checked divergence vs `main` | Action |
|---|---|---|
| `main` | Stable PRD 3.1 stage delivery checkpoint. Last observed commit `4a82e3d` dated 2026-10-26. README currently mainly covers PRD 3.1. | Keep as default; no experimental overwrite. |
| `stage/prd32-20261027-closeout` | **1 commit ahead**, 0 behind, 16 added evidence files, under `stage_delivery/prd32_20261027/` (Oct 27 diagnostics). | Parent of new PRD 3.2 work branch; preserve. |
| `update/prd31-stage-20261023` | 0 ahead, 3 behind; PRD 3.1 stage-delivery branch already subsumed by main. | Historical only. |
| `update/prd31-20261015` | 0 ahead, 14 behind; older PRD 3.1 update. | Historical only. |
| `organized-20261008` | 0 ahead, 18 behind; older organization snapshot. | Historical only. |
| `backup-before-overwrite-20261008-1248` | GitHub comparison returns **no common ancestor** with main. | Preserve as detached-history backup; do not casually merge/delete. |
| `work/prd32-20261112-audit` | **Proposed** branch created from `stage/prd32-20261027-closeout` when the prepared update package is pushed. Covers verified Nov 11 code and evidence. | Work-in-progress; open a **draft PR to `main`**, review before any merge. |

## What is under active development?

The latest **existing remote PRD 3.2** branch is `stage/prd32-20261027-closeout`. Nov 11 work was generated in GPU/local artifact packages, not yet visible in remote GitHub. Once the update package is pushed, use `work/prd32-20261112-audit` as the active development branch and keep `main` as the safe integration baseline.

## Policy

- `main`: reviewed/release candidate only; no raw exploratory model-output promotion.
- `work/prd32-*`: PRD 3.2 experimental implementations and stage evidence, always shadow-blocked until acceptance.
- `stage/*`: dated delivery snapshot; retain its exact provenance.
- `update/*`, `organized-*`: historical branches. Delete only after review with supervisor; current task deliberately deletes nothing.
- `backup-*`: archival backups, especially when histories diverge.

## How to check differences yourself

- Compare `main` vs `stage/prd32-20261027-closeout` in GitHub Compare.
- After pushing the proposed branch, compare `main` vs `work/prd32-20261112-audit` and open draft PR.
- Use `git log --oneline --decorate --graph --all -n 30` and `git branch -vv` locally; counts can change as teammates push new commits.

**Connection limitation encountered:** The ChatGPT GitHub integration could read the repository but denied both branch creation and file uploads with HTTP 403 (`Resource not accessible by integration`). This package is therefore prepared **locally only**; do not represent it as already pushed.


2026-11-12 update: Oracle8 neutral 5-pair GPU inference completed; all five qualitative review verdicts user-confirmed in chat. Details: `stage_delivery/prd32_20261112/GPU_RECONCILIATION_AND_REVIEW_STATUS.md`.
