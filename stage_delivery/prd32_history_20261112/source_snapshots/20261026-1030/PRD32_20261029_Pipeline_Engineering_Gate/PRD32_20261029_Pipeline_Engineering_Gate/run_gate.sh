#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODE="${1:-selftest}"
PYTHON="${PYTHON:-/workspace/prd32_venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
if [[ "$MODE" == 'selftest' ]]; then
 "$PYTHON" test_gate.py
elif [[ "$MODE" == 'replay_bundled' ]]; then
 OUT="${2:-/workspace/PRD32_ENGINEERING_GATE_20261029_run01}"
 "$PYTHON" bridge_engineering_gate.py \
  --service evidence/ServiceOutputs_12_20261027.jsonl \
  --envelopes evidence/ReadOnlyEnvelopes_12_20261027.jsonl \
  --output "$OUT"
elif [[ "$MODE" == 'replay_custom' ]]; then
 [[ $# -eq 4 ]] || { echo 'Usage: bash run_gate.sh replay_custom /path/service_outputs.jsonl /path/handoff_envelopes.jsonl /path/new_output'; exit 2; }
 "$PYTHON" bridge_engineering_gate.py --service "$2" --envelopes "$3" --output "$4"
else
 echo 'Usage: bash run_gate.sh selftest | replay_bundled [output] | replay_custom service.jsonl envelopes.jsonl output'
 exit 2
fi
