"""Provisional VQA-only request boundary; deliberately NOT a production Guard.

Only records completeness checks and imposes a hard release prohibition.  The
original Guard is a full three-task contract and is NOT called on partial output.
Passing this check never grants publishing or product approval.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib

STATUS='SHADOW_BLOCKED_PARTIAL_GUARD_CONTRACT_NOT_VALIDATED'
REQUIRED=('case_id','instance_id','vqa_answer','v5_mask_iou','source_image_sha256')


def validate_shadow_vqa(output: dict) -> dict:
    if not isinstance(output,dict): raise ValueError('candidate output must be an object')
    for field in REQUIRED:
        if field not in output: raise ValueError('missing VQA route field '+field)
    if not isinstance(output['vqa_answer'],str) or not output['vqa_answer'].strip():
        raise ValueError('empty VQA answer; fail closed')
    if float(output['v5_mask_iou']) < 0.9999:
        raise ValueError('failed frozen V5 reproducibility check; fail closed')
    if not isinstance(output['source_image_sha256'],str) or len(output['source_image_sha256'])!=64:
        raise ValueError('missing source image SHA256')
    if output.get('publish_allowed',False) is not False:
        raise ValueError('shadow route attempted to enable release')
    return {'status':STATUS,'task_scope':'vqa_only',
      'required_fields_verified':True,'response_sha256':hashlib.sha256(output['vqa_answer'].encode('utf-8')).hexdigest(),
      'quality_validated':False,'original_three_task_guard_invoked':False,
      'independent_gt':False,'publish_allowed':False,
      'requires_new_policy_approval':True}
