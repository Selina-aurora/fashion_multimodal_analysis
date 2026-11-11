#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/workspace/prd32_venv/bin/python
if [[ ! -x "$PY" ]]; then PY=python; fi
"$PY" next_batch.py "${1:?Use preflight, gpu or score}"
