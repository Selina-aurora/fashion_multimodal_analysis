"""Fail-closed Guard V5 *offline sidecar* contract.

Nothing here approves category publication, alias fusion, inference changes,
or independent holdout use. Review evidence is advisory to any proposed runtime
integration, which must be tested separately against a real inference trace.
"""
from __future__ import annotations
from guard_v4_contract import validate_candidate_binding

LEGAL_GT = frozenset({'top','pants','skirt','outerwear','dress','shoe','bag','accessory'})
LEGAL_RELATIONS = frozenset({'同物重复','整体与局部','不同对象混淆','独立对象'})


def evaluate_v5_candidate(source: dict, candidate: dict, review: dict | None = None) -> dict:
    """Check explicit identity metadata and *flag*, never auto-correct, outputs.

    Identity fields are REQUIRED in the candidate and sourced from an immutable
    prediction key, not inferred from caption text or bounding box overlap.
    A confirmed user review verifies just the stated observation, not a unique
    alias partner, a new label for undecidable samples, or Qwen runtime binding.
    """
    base = validate_candidate_binding(source, candidate)
    issues = list(base['violations'])
    flags = []
    if review:
        if review.get('case_id') != source.get('case_id') or review.get('instance_id') != source.get('instance_id'):
            issues.append('REVIEW_IDENTITY_MISMATCH')
        if review.get('review_status') != 'USER_CONFIRMED_IN_CHAT':
            issues.append('REVIEW_NOT_USER_CONFIRMED')
        if review.get('image_sha256') != source.get('image_sha256'):
            issues.append('REVIEW_IMAGE_SHA_MISMATCH')
        relation = review.get('instance_relation')
        if relation not in LEGAL_RELATIONS:
            issues.append('REVIEW_UNKNOWN_RELATION')
        if relation in {'同物重复', '整体与局部'}:
            flags.append('DUPLICATE_OR_PARENT_PART_PENDING_EXPLICIT_IDENTITY_PAIR')
        elif relation == '不同对象混淆':
            flags.append('MULTI_OBJECT_AMBIGUITY')
        if review.get('target_box') != '正确对象':
            flags.append('PARTIAL_OR_MIXED_TARGET_BOX')
        label = review.get('category_gt')
        if label not in LEGAL_GT:
            if label != '无法判断/不适用':
                issues.append('INVALID_REVIEW_CATEGORY')
            flags.append('GT_CATEGORY_UNRESOLVED')
        elif candidate.get('candidate_category') != label:
            flags.append('CANDIDATE_CATEGORY_DISAGREES_WITH_REVIEW')
        if review.get('qwen_semantic') == '混合/无法判断':
            flags.append('CROSS_TASK_SEMANTIC_REVIEW_FLAG')
    if candidate.get('shadow_alias_candidate') in (True, 'YES'):
        flags.append('ALIAS_QUARANTINE_REQUIRED')
    if issues:
        decision = 'BLOCK_IDENTITY_OR_INTEGRITY'
    elif flags:
        decision = 'BLOCK_REVIEW_OR_ALIAS_RISK'
    else:
        decision = 'SHADOW_ONLY_NOT_PRODUCTION_APPROVED'
    return {
        'decision': decision,
        'identity_contract_pass': not issues,
        'identity_violations': issues,
        'review_risk_flags': flags,
        'category_publication_approved': False,
        'alias_merge_approved': False,
        'production_approved': False,
        'independent_holdout_authorized': False,
    }
