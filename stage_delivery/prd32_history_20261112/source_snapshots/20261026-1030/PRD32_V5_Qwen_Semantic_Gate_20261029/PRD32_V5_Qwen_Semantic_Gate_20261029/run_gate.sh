#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x /workspace/prd32_venv/bin/python ]]; then P=/workspace/prd32_venv/bin/python; else P=python; fi
"$P" -m unittest discover -s tests -v
"$P" audit_3task.py "$@"
