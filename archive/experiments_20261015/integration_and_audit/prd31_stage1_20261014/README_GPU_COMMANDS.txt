PRD 3.1 Stage 1 (2026-10-14) - ASCII-only file paths

1. Upload prd31_stage1_20261014_gpu_preflight.zip into /workspace on the GPU.
2. Run commands:
   cd /workspace
   python -m zipfile -e prd31_stage1_20261014_gpu_preflight.zip /workspace/prd31_stage1_20261014
   python /workspace/prd31_stage1_20261014/scripts/gpu_preflight_readonly.py --root /workspace/fashion_multimodal_analysis --out /workspace/prd31_preflight_20261014
3. Send back /workspace/prd31_preflight_20261014_UPLOAD_THIS.zip
4. If CORE400_HASH_MATCH False, stop and share the result. Do not alter the frozen Core400 manifest.

This is an inventory and hash verification script; it will NOT run model inference, train models or benchmark accuracy.
Original Chinese README content is preserved under README_stage1_progress_and_next.md.
