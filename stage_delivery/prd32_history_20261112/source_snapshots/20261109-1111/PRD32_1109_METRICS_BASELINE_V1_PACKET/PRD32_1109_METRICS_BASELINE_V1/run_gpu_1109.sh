#!/usr/bin/env bash
set -euo pipefail
cd /workspace/PRD32_1109_METRICS_BASELINE_V1
python -m unittest discover -s tests -v
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
OUT="/workspace/PRD32_1109_METRICS_GPU_RESULTS_${RUN_ID}"
python run_baseline.py --out "$OUT"
cd /workspace
python -m zipfile -c "$(basename "$OUT").zip" "$(basename "$OUT")"
ls -lh "$(basename "$OUT").zip"
echo "UPLOAD_THIS: /workspace/$(basename "$OUT").zip"
