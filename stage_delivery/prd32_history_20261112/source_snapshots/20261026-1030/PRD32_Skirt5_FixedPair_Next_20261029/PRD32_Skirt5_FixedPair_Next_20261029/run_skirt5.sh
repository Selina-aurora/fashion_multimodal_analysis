#!/usr/bin/env bash
# Frozen V2/V3 skirt-target diagnostic -- use: bash run_skirt5.sh preflight|gpu|score
set -euo pipefail
STAGE="${1:-}"
BASE="${SKIRT5_WORKSPACE:-/workspace}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SKIRT5_PYTHON:-${BASE}/prd32_venv/bin/python}"
GATE="${SKIRT5_GATE_DIR:-${BASE}/PRD32_20261029_Validation_Gate}"
PREP="${SKIRT5_PREP_DIR:-${BASE}/PRD32_SKIRT_TARGET_RECOVERY_20261029_run01}"
FROZEN="${BASE}/PRD32_SKIRT5_FROZEN_20261029_run01"
PREFLIGHT="${BASE}/PRD32_SKIRT5_PREFLIGHT_20261029_run01"
RUN="${BASE}/PRD32_SKIRT5_GPU_20261029_run01"
SCORE="${BASE}/PRD32_SKIRT5_SCORE_20261029_run01"
REVIEW_HTML="${BASE}/PRD32_SKIRT5_RootCause_Review_20261029_run01.html"
MODEL="${SKIRT5_MODEL:-${BASE}/models/Qwen2.5-VL-7B-Instruct}"
REVIEW="${HERE}/Skirt5_BlindReview_Reviewed.json"
if [[ ! -x "$PY" ]]; then echo "ERROR: missing Python interpreter: $PY" >&2;exit 1;fi
if [[ ! -f "$GATE/prepare_validation.py" || ! -f "$GATE/run_and_score.py" ]]; then echo "ERROR: original Validation_Gate folder not found" >&2;exit 1;fi
check(){ "$PY" "$HERE/check_review.py" --review "$REVIEW" --prepared "$PREP" --html "$PREP/Blind_Review_Skirt_Targets.html"; }
case "$STAGE" in
 preflight)
  check
  if [[ ! -e "$FROZEN" ]]; then
     "$PY" "$GATE/prepare_validation.py" freeze --prepared "$PREP" --review "$REVIEW" --output "$FROZEN"
  else
     "$PY" - "$FROZEN" "$REVIEW" <<'PYCODE'
import sys,json,hashlib,pathlib
p=pathlib.Path(sys.argv[1]);d=json.loads((p/'freeze.json').read_text()); sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
assert d['total']==5 and d['scoreable']==4 and sha(sys.argv[2])==d['blind_review_sha256'], 'Existing freeze was made from DIFFERENT review'
assert sha(p/'frozen_inference_plan.jsonl')==d['plan_sha256'] and sha(p/'frozen_human_reference.jsonl')==d['human_reference_sha256'],'Frozen files changed'
print('EXISTING_FREEZE_VERIFIED')
PYCODE
  fi
  if [[ ! -e "$PREFLIGHT" ]]; then
     "$PY" "$GATE/run_and_score.py" preflight --frozen "$FROZEN" --output "$PREFLIGHT"
  else
     "$PY" - "$PREFLIGHT/preflight_summary.json" <<'PYCODE'
import sys,json
s=json.load(open(sys.argv[1]));assert s['case_count']==5 and s['status']=='READY_NO_GPU'
print('EXISTING_PREFLIGHT_VERIFIED')
PYCODE
  fi
  echo "SKIRT5_READY_FOR_GPU"
 ;;
 gpu)
  check
  if [[ ! -f "$PREFLIGHT/preflight_summary.json" ]]; then echo 'ERROR: run preflight stage first' >&2;exit 1;fi
  if [[ -e "$RUN" ]]; then echo "ERROR: Existing GPU output directory $RUN ; never overwrite. Review results or use a different run ID manually." >&2;exit 1;fi
  "$PY" -u "$GATE/run_and_score.py" run --frozen "$FROZEN" --model "$MODEL" --output "$RUN"
 ;;
 score)
  check
  if [[ ! -f "$RUN/predictions_pair.jsonl" ]]; then echo 'ERROR: Missing paired predictions; run gpu first' >&2;exit 1;fi
  "$PY" - "$RUN/run_summary.json" <<'PYCODE'
import sys,json
s=json.load(open(sys.argv[1]));assert s['count']==5 and s['complete_pairs']==5, 'Not all 5 model pairs completed successfully'
print('SKIRT5_INFERENCE_PAIRS_VERIFIED',s['complete_pairs'])
PYCODE
  if [[ ! -e "$SCORE" ]]; then
    "$PY" "$GATE/run_and_score.py" score --frozen "$FROZEN" --predictions "$RUN/predictions_pair.jsonl" --output "$SCORE"
  else
    echo 'EXISTING_SCORE_FOUND; will validate with sensitivity script'
  fi
  "$PY" "$HERE/sensitivity.py" --comparison "$SCORE/paired_comparison.csv" --summary "$SCORE/comparison_summary.json" --output "$SCORE/skirt5_case002_sensitivity.json"
  if [[ ! -e "$REVIEW_HTML" ]]; then
    "$PY" "$GATE/run_and_score.py" rootcause-review --frozen "$FROZEN" --predictions "$RUN/predictions_pair.jsonl" --output "$REVIEW_HTML"
  fi
  echo "SKIRT5_SCORE_READY $SCORE"
 ;;
 *)
  echo 'Usage: bash run_skirt5.sh preflight|gpu|score' >&2
  exit 2
 ;;
esac
