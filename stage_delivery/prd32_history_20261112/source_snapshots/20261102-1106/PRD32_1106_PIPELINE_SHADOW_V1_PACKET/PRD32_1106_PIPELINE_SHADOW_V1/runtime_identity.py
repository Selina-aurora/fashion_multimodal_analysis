"""Immutable binding envelopes for future PRD3.2 runtime integration.

These objects are *new hooks*, not installed in the live GPU pipeline.
The host must attach instance IDs; a language-model string is never proof
of target identity. The sidecar never changes class predictions.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
import re

SHA_RE = re.compile(r'^[0-9a-f]{64}$')

@dataclass(frozen=True)
class InstanceKey:
    case_id: str
    instance_id: str
    image_sha256: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.instance_id or not SHA_RE.fullmatch(self.image_sha256):
            raise ValueError('Invalid immutable instance key')

@dataclass(frozen=True)
class QwenRequestEnvelope:
    key: InstanceKey
    task: str
    variant: str
    request_id: str
    roi_instance_id: str
    mask_instance_id: str
    marked_image_sha256: str
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if self.task not in {'tags','vqa','copy'}:
            raise ValueError('Unsupported task')
        if self.variant not in {'baseline','focus_v1'}:
            raise ValueError('Unknown model variant')
        if self.request_id != f'{self.key.case_id}__{self.key.instance_id}__{self.task}':
            raise ValueError('Request ID does not belong to target instance/task')
        if self.roi_instance_id != self.key.instance_id or self.mask_instance_id != self.key.instance_id:
            raise ValueError('Cross-instance ROI/Mask forbidden')
        if not SHA_RE.fullmatch(self.marked_image_sha256):
            raise ValueError('Invalid marked-image digest')
        x1,y1,x2,y2=self.bbox_xyxy
        if not all(0<=x<1e7 for x in (x1,y1,x2,y2)) or x1>=x2 or y1>=y2:
            raise ValueError('Invalid bbox')

def make_request(key:InstanceKey, task:str,variant:str,roi_instance_id:str,mask_instance_id:str,marked_image_bytes:bytes,bbox_xyxy:tuple[float,float,float,float]) -> QwenRequestEnvelope:
    """Call only at the actual request-creation boundary, before Qwen inference."""
    return QwenRequestEnvelope(key,task,variant,f'{key.case_id}__{key.instance_id}__{task}',roi_instance_id,mask_instance_id,sha256(marked_image_bytes).hexdigest(),bbox_xyxy)

def attach_host_response(envelope:QwenRequestEnvelope, *, completed_request_id:str, raw_text:str, response_origin_instance_id:str|None = None)->dict[str,Any]:
    """Host verifies completed request correlation; model text cannot authenticate itself.

    response_origin_instance_id must be set from the request scheduler, never
    parsed from the model's prose. The returned flag does not approve publication.
    """
    if completed_request_id != envelope.request_id:
        raise ValueError('Completed request ID mismatch')
    if response_origin_instance_id is None:
        raise ValueError('Host response origin is not evidenced')
    if response_origin_instance_id != envelope.key.instance_id:
        raise ValueError('Cross-instance response source')
    return {
        'case_id':envelope.key.case_id,
        'target_instance_id':envelope.key.instance_id,
        'image_sha256':envelope.key.image_sha256,
        'request_id':envelope.request_id,
        'task':envelope.task,
        'model_variant':envelope.variant,
        'roi_instance_id':envelope.roi_instance_id,
        'mask_instance_id':envelope.mask_instance_id,
        'response_instance_id':response_origin_instance_id,
        'marked_image_sha256':envelope.marked_image_sha256,
        'raw_text':raw_text,
        'guard_status':'HOST_IDENTITY_CORRELATED_SHADOW_ONLY',
        'publish_allowed':False,
    }
