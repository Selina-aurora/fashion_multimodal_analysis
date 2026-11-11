# PRD 3.2 · GPU reconciliation and review state (2026-11-12)

- First GPU inventory: 23,342 scanned files, 4,054 text files packed; first-pass collection excluded images, model weights and cache. No Git branch or SHA could be read from the GPU project directory (`N/A`).
- Second pass: 3,825 of the 3,841 capacity-skipped files recovered and verified. 16 were outstanding; earlier archive comparison found 10 exact-hash existing PRD3.2 historical scripts and 6 older PRD3.1 regional/sleeve scripts. This does **not** mean all GPU binary assets are backed up.
- Neutral Oracle8 A/B: actual GPU results for 10 calls, 5/5 DEV pairs reviewed and accepted by the user in chat on Nov 12. Red relatively better on 2, cyan on 1; both materially wrong on 2. Qualitative review only, no independent GT or blind holdout.
- Inference, Guard and production release remain **SHADOW_BLOCKED**. No implementation was promoted; `main` is unchanged.
- For the full file-inventory and SHA-256 audit, see the separate private GPU audit/second-pass reports. They are not duplicated in this public GitHub code archive.

## Public repository data boundary

Only small review CSV/Markdown, machine-readable outputs, source scripts and provenance are included here. Original fashion images, review HTML with embedded imagery, model weights, and private GPU project backups are **not** published.
