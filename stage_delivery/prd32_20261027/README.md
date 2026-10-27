# PRD 3.2 | 2026-10-27 Development Closeout

**Status:** Development diagnostics complete for the day. PRD 3.2 is NOT accepted.

## Verified daily milestones
- Frozen PRD 3.1 interface replay -> PRD 3.2 VQA/tags/copy on 12 requests: 12/12 GPU inference complete; offline replay only.
- Training-source five-class tag run: 40/40 normalized Schema valid, 0/40 native strict JSON. **No human-verified five-class or independent eight-class accuracy established.**
- Human attribution for 19 disagreements with source category hints: 8 target-binding, 7 model classification, 2 visual ambiguity, 1 upstream bbox, 1 source-label issue.
- Plain target ROI on 15 development errors: 3/15 human category matches.
- Historical GT-pilot Mask versus ROI paired on same 15: 4/15 vs 3/15; one additional correction, `train_dev_013` (`top` -> `pants`); 11 both wrong.
- Corrected ROI visual review: 13/15 crops contain additional garments; 15/15 targets visible, 14/15 visually classifiable.
- Human QA/copy development review: VQA 6/6 initially marked correct but some refusals and evidence require re-check; old copy 1/3 satisfactory.

## Design decision
Keep the frozen PRD 3.1 candidate. Do not promote ROI/Mask as default based on these 15 selected error cases. Prioritize target-binding and category-decision constraints; keep Schema and latency issues on the active backlog.

## Important limitations
- Training-source cases are not holdout tests, and source labels are not fully human-verified GT.
- Historical GT-pilot bbox/Mask does not represent frozen V5 predictions or end-to-end acceptance.
- All 15 A/B cases were selected because earlier model predictions failed, so the denominators must not be called overall accuracy.
- Service-normalized Schema pass is not native strict-JSON compliance; production P95 and QPS are not measured.
- The sealed holdout was not accessed.
- In original Mask output, `roi_mixed_others_human` was accidentally mapped from `roi_single_item` (opposite polarity). Original file preserved; corrected copy fixes metadata only.

## Evidence
See `evidence/*.json` / `evidence/*.jsonl`, `EVIDENCE_MANIFEST.csv` with SHA256 hashes. All human review and GPU outputs remain separate; do not overwrite raw evidence.

## Next day
1. Evaluate targeted target-binding and garment class disambiguation on held-out development cases.
2. Establish independent eight-class manual GT and controlled evaluation without touching sealed holdout prematurely.
3. Verify QA abstention rubric, copy V4/V4.1 judgments and measured inference latency.
