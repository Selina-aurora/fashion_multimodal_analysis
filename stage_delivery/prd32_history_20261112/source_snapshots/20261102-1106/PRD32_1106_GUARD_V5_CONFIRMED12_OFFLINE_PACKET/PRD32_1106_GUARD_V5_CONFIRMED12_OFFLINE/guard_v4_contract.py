"""PRD 3.2 V4 *offline* metadata binding contract.

Checks explicit instance identifiers only. Passing does not prove visual identity,
category correctness, segmentation quality, or production eligibility.
"""
from __future__ import annotations

ALLOWED_CATEGORIES = frozenset({
    'top', 'pants', 'skirt', 'outerwear', 'dress', 'shoe', 'bag', 'accessory'
})


def validate_candidate_binding(source: dict, candidate: dict) -> dict:
    """Fail closed on missing/mismatched identities, including cross-instance attributes.

    Returns a decision. No outcome authorizes category publishing or deployment.
    """
    required = ('case_id', 'instance_id', 'image_sha256', 'query_instance_id',
                'roi_instance_id', 'mask_instance_id', 'response_instance_id')
    expected = {
        'case_id': source.get('case_id'),
        'instance_id': source.get('instance_id'),
        'image_sha256': source.get('image_sha256'),
        'query_instance_id': source.get('instance_id'),
        'roi_instance_id': source.get('instance_id'),
        'mask_instance_id': source.get('instance_id'),
        'response_instance_id': source.get('instance_id'),
    }
    problems = []
    if not source.get('case_id') or not source.get('instance_id') or not source.get('image_sha256'):
        problems.append('BAD_SOURCE_IDENTITY')
    for key in required:
        if key not in candidate or candidate[key] in ('', None):
            problems.append('MISSING_'+key.upper())
        elif candidate[key] != expected[key]:
            problems.append('MISMATCH_'+key.upper())
    if candidate.get('alias_target_instance_id') not in (None, ''):
        problems.append('ALIAS_NOT_AUTHORIZED')
    if candidate.get('candidate_category') not in ALLOWED_CATEGORIES:
        problems.append('CATEGORY_MISSING_OR_OUT_OF_TAXONOMY')
    if candidate.get('category_published') not in (None, False, 'NO'):
        problems.append('UNAUTHORIZED_CATEGORY_PUBLICATION')
    attrs = candidate.get('attributes', [])
    if not isinstance(attrs, list):
        problems.append('ATTRIBUTES_NOT_LIST')
    else:
        for idx, attr in enumerate(attrs):
            if not isinstance(attr, dict) or attr.get('source_instance_id') != source.get('instance_id'):
                problems.append(f'ATTR_INSTANCE_CROSSOVER_{idx}')
    return {
        'decision': 'BLOCK_BINDING_VIOLATION' if problems else 'SHADOW_ONLY_NEEDS_INDEPENDENT_VISUAL_GT',
        'violations': problems,
        'mainline_approved': False,
        'category_publication_approved': False,
        'visual_identity_verified': False,
        'category_gt_verified': False,
    }
