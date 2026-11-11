#!/usr/bin/env python3
"""VQA-only offline shadow policy (NOT production Guard).

Every outcome is explicitly non-publishable. This module does not import or
change the original three-task Guard. Used to triage already-captured DEV60
responses, never to authorize output to customers.
"""
from __future__ import annotations
import hashlib
import re
from typing import Mapping

SCHEMA = 'prd32_vqa_only_shadow_policy_v1'
APPROVED_QUESTION_TYPES = frozenset({'color'})  # only type tested in captured records
ABSTAIN_MARKERS = ('无法确定','不确定','看不清','无法判断','不知道','不能确定','无法识别','不清楚')
DEBUG_MARKERS = ('红框','边框','框内','目标服饰','标注','矩形框','框选','图中圈出')


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def evaluate(row: Mapping, dev_review: Mapping, frozen: Mapping, *, question_type='color') -> dict:
    """Return fully traceable, fail-closed SHADOW result.

    Checked aspects deliberately kept separate from actual accuracy/approval.
    Raises on missing malformed identity: never silently match wrong rows.
    """
    case = str(row.get('case_id', ''))
    inst = str(row.get('instance_id', ''))
    key = f'{case}/{inst}'
    if not case or not inst or key == '/':
        raise ValueError('Missing case/instance; cannot correlate source')
    if dev_review.get('case_id') != case or dev_review.get('instance_id') != inst:
        raise ValueError('Different review instance; refuse cross-target attribution')
    reasons = []
    answer = row.get('vqa_answer', '')
    if not isinstance(answer, str) or not answer.strip():
        reasons.append('MISSING_VQA_OUTPUT')
        answer = ''
    # Frozen SHA's role: identity/reproducibility, not factual accuracy.
    if not isinstance(frozen, Mapping) or not frozen:
        reasons.append('MISSING_FROZEN_BASELINE')
    else:
        if str(row.get('vqa_response_sha256','')) != sha256_text(answer):
            reasons.append('MEASURED_ANSWER_SHA_MISMATCH')
        if frozen.get('vqa_response_sha256') != sha256_text(answer):
            reasons.append('FROZEN_BASELINE_OUTPUT_DRIFT')
    if row.get('vqa_matches_frozen_baseline') is not True:
        reasons.append('BASELINE_MATCH_FLAG_FALSE')
    try:
        mask_iou = float(row.get('frozen_mask_iou'))
        if not 0.9999 <= mask_iou <= 1.000001:
            reasons.append('FROZEN_MASK_LINEAGE_MISMATCH')
    except (ValueError, TypeError):
        reasons.append('INVALID_MASK_LINEAGE_FIELD')
    if row.get('production_release') is not False or row.get('original_guard_invoked') is not False:
        reasons.append('UNEXPECTED_RELEASE_OR_ORIGINAL_GUARD_CLAIM')
    if row.get('holdout_used') is not False:
        reasons.append('HOLDOUT_SCOPE_MISMATCH')
    if row.get('route') != 'VQA_ONLY_SHADOW':
        reasons.append('ROUTE_SCOPE_MISMATCH')
    if question_type not in APPROVED_QUESTION_TYPES:
        reasons.append('QUESTION_TYPE_NOT_VALIDATED')
    # The upstream evidence is user-confirmed AI-assisted DEV audit, NOT independent GT.
    target_status = dev_review.get('ai_prefill_identity', '')
    mask_status = dev_review.get('ai_prefill_mask', '')
    if target_status != '匹配':
        reasons.append('TARGET_IDENTITY_NOT_FULLY_MATCHED')
    if mask_status != '正确':
        reasons.append('TARGET_MASK_NOT_FULLY_CORRECT')
    if dev_review.get('category_reference_status') != 'DETERMINATE':
        reasons.append('CATEGORY_REFERENCE_UNRESOLVED')
    qa_status = dev_review.get('ai_prefill_vqa','')
    if qa_status != '正确':
        reasons.append('DEV_COLOR_QA_NOT_CONFIRMED_CORRECT')
    stripped = answer.strip()
    if any(x in stripped for x in ABSTAIN_MARKERS):
        reasons.append('MODEL_ABSTAINED')
    if any(x in stripped for x in DEBUG_MARKERS):
        reasons.append('DEBUG_OR_TARGET_MARKER_TEXT')
    if len(stripped) > 32:
        reasons.append('COLOR_ANSWER_FORMAT_REVIEW')
    if re.search(r'(?:https?://|<\/?script|```)', stripped, re.I):
        reasons.append('SUSPICIOUS_OUTPUT_FORMAT')
    # No production approval exists. Even a clean candidate is SHADOW ONLY.
    if reasons:
        structural = {'MISSING_VQA_OUTPUT','MISSING_FROZEN_BASELINE','MEASURED_ANSWER_SHA_MISMATCH',
                      'FROZEN_BASELINE_OUTPUT_DRIFT','BASELINE_MATCH_FLAG_FALSE',
                      'FROZEN_MASK_LINEAGE_MISMATCH','INVALID_MASK_LINEAGE_FIELD',
                      'UNEXPECTED_RELEASE_OR_ORIGINAL_GUARD_CLAIM','HOLDOUT_SCOPE_MISMATCH',
                      'ROUTE_SCOPE_MISMATCH'}
        primary = ('BLOCKED_INTEGRITY' if structural.intersection(reasons)
                   else 'BLOCKED_TARGET_EVIDENCE' if any(r in reasons for r in
                        ('TARGET_IDENTITY_NOT_FULLY_MATCHED','TARGET_MASK_NOT_FULLY_CORRECT',
                         'CATEGORY_REFERENCE_UNRESOLVED'))
                   else 'BLOCKED_ANSWER_REVIEW')
    else:
        primary='SHADOW_REVIEW_CANDIDATE'
    return {
        'schema':SCHEMA, 'review_key':key, 'question_type':question_type,
        'primary_status':primary,'risk_reasons':reasons,'risk_reason_count':len(reasons),
        'answer':answer,'vqa_answer_sha256':sha256_text(answer),
        'target_identity_review':target_status,'mask_review':mask_status,
        'vqa_quality_review':qa_status,'category_reference_status':dev_review.get('category_reference_status',''),
        'dev_review_type':'USER_CONFIRMED_AI_ASSISTED_NOT_INDEPENDENT_GT',
        'latency_total_ms':row.get('vqa_only_total_ms'),
        'generated_tokens':row.get('generated_tokens'),
        'publish_allowed':False,'eligible_for_production':False,
        'original_three_task_guard_executed_here':False,
        'original_three_task_guard_approved':False,
        'independent_holdout_evaluated':False,
        'vqa_semantic_accuracy_proven':False,
    }
