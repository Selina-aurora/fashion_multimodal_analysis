#!/usr/bin/env bash
# Use: bash run_roi.sh preflight|gpu|score
set -euo pipefail
STAGE="${1:-}"
PY="${PRD32_PYTHON:-/workspace/prd32_venv/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREP="/workspace/PRD32_SKIRT5_ROI_PREP_20261029_run01"
GPU="/workspace/PRD32_SKIRT5_ROI_GPU_20261029_run01"
SCORE="/workspace/PRD32_SKIRT5_ROI_SCORE_20261029_run01"
if [[ ! -x "$PY" ]];then echo "Missing $PY" >&2;exit 1;fi
case "$STAGE" in
 preflight)
  "$PY" "$HERE/roi_ablation.py" selftest
  "$PY" "$HERE/roi_ablation.py" preflight --output "$PREP"
  echo "CROPS_READY inspect $PREP/cropped_marked_images before GPU"
  ;;
 gpu)
  "$PY" -u "$HERE/roi_ablation.py" gpu --prepared "$PREP" --output "$GPU"
  ;;
 score)
  "$PY" "$HERE/roi_ablation.py" score --prepared "$PREP" --roi-run "$GPU" --output "$SCORE"
  echo "ROI_ABLATION_SCORE_READY $SCORE"
  ;;
 *)echo 'Usage: bash run_roi.sh preflight|gpu|score' >&2;exit 2;;
esac
