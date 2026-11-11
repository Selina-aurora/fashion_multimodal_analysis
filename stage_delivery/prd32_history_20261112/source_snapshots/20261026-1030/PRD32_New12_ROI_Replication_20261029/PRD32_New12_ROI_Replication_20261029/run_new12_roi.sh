#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:-}"
PY="${PRD32_PYTHON:-/workspace/prd32_venv/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREP=/workspace/PRD32_NEW12_ROI_PREP_20261029_run01
GPU=/workspace/PRD32_NEW12_ROI_GPU_20261029_run01
SCORE=/workspace/PRD32_NEW12_ROI_SCORE_20261029_run01
if [[ ! -x "$PY" ]]; then echo "Missing Python env: $PY" >&2;exit 1;fi
case "$STAGE" in
 preflight)
  "$PY" "$HERE/paired_roi_replication.py" selftest
  "$PY" "$HERE/paired_roi_replication.py" preflight --output "$PREP"
  echo "PREVIEW_REQUIRED $PREP/Review_Crop_Input_Before_GPU.html"
  ;;
 gpu)
  "$PY" -u "$HERE/paired_roi_replication.py" gpu --prepared "$PREP" --output "$GPU"
  ;;
 score)
  "$PY" "$HERE/paired_roi_replication.py" score --prepared "$PREP" --gpu-run "$GPU" --output "$SCORE"
  echo "NEW12_ROI_DONE $SCORE"
  ;;
 *)echo 'Usage: bash run_new12_roi.sh preflight|gpu|score' >&2;exit 2;;
esac
