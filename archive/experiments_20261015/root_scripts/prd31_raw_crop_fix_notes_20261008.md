# Raw-crop diagnostic integration patch

Two development ROI input comparisons (18 DINO calls, fixed box/text thresholds 0.3/0.25) show that masked-white crops tend to detect the full garment as collar/cuff/hem. Raw crops yield more localized proposals for selected queries. This is evidence on two cases, not a general accuracy result.

In case_01 the raw collar proposal is still small and may not cover the complete collar. In case_05 raw neckline is more local; raw cuff is missed (a sleeveless dress), but missing output alone does not certify absence. Raw hem for case_05 still covers the full dress. Full-image input also yields cross-object proposals.

Change only the strict-region diagnostic path: Grounding DINO now receives the raw predicted garment crop. SAM receives the same raw crop as before, and its output remains intersected with the predicted garment mask. Attributes retain existing inputs. Queries, phrase guards, category routing, thresholds, manifests, weights and PRD targets are unchanged. The legacy path still uses the masked-white crop. The output records grounding_input_view for detected predictions.

Local tests verify input pixels in strict/legacy paths and preservation of the garment-mask constraint. Real SAM-refined semantic accuracy must be reviewed after the GPU rerun. No training or timing acceptance is claimed.
