#!/usr/bin/env bash
# Expand successful DEV2 V6 run to 12 V5 targets (10 images), then 4 new Qwen Baseline smoke targets.
# Run only inside unpacked V7 folder on the existing GPU /workspace environment.
set -euo pipefail
cd "$(dirname "$0")"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
PREFIX="/workspace/PRD32_V7_${RUN_ID}"
PREFLIGHT="${PREFIX}_V5_PREFLIGHT"
PRODUCER="${PREFIX}_V5_PRODUCER"
QPREFLIGHT="${PREFIX}_QWEN_PREFLIGHT"
QWEN="${PREFIX}_QWEN"
ZIP="/workspace/PRD32_V7_GPU_RESULTS_${RUN_ID}.zip"

package_if_possible() {
    local result=$?
    trap - EXIT
    if [ -d "$PRODUCER" ]; then
        if [ -d "$QWEN" ]; then
            python package_results_v7.py --producer-dir "$PRODUCER" --qwen-dir "$QWEN" --out "$ZIP" || true
        else
            python package_results_v7.py --producer-dir "$PRODUCER" --out "$ZIP" || true
        fi
        if [ -f "$ZIP" ]; then
            echo "UPLOAD_THIS: $ZIP"
        fi
    fi
    echo "V7_EXIT_CODE: $result"
    exit "$result"
}
trap package_if_possible EXIT

python -m unittest discover -s tests -v
python run_v5_runtime_producer_v7.py --mode preflight --out "$PREFLIGHT"
python run_v5_runtime_producer_v7.py --mode live --out "$PRODUCER"
python run_qwen_from_v5_producer_v7.py --mode preflight --producer-dir "$PRODUCER" --out "$QPREFLIGHT"
python run_qwen_from_v5_producer_v7.py --mode live --producer-dir "$PRODUCER" --out "$QWEN"
echo "V7_STAGED_SHADOW_FINISHED_NOT_RELEASED"
