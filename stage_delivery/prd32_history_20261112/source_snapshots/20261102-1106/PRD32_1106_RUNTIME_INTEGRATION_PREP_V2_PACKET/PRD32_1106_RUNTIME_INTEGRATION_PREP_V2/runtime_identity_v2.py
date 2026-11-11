"""Host-side request/response identity contract for PRD3.2 (shadow-only).

Integration candidate only. Never mutates model outputs, merges instances, publishes
categories, or contacts model services. Source IDs must be recorded by each host
stage independently; copying the expected ID into every field proves nothing.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping
import re

HEX64 = re.compile(r'^[0-9a-f]{64}$')
SAFE_ID = re.compile(r'^[A-Za-z0-9_.-]{1,100}$')
TASKS = frozenset({'tags', 'vqa', 'copy'})
VARIANTS = frozenset({'baseline', 'focus_v1'})


def _checked_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(f'invalid {label}')
    return value


def _checked_hash(value: str, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f'invalid {label}')
    return value


@dataclass(frozen=True)
class SegmentedTarget:
    case_id: str
    instance_id: str
    image_sha256: str
    bbox: tuple[float, float, float, float]

    def __post_init__(self):
        _checked_id(self.case_id, 'case_id')
        _checked_id(self.instance_id, 'instance_id')
        _checked_hash(self.image_sha256, 'image_sha256')
        if len(self.bbox) != 4:
            raise ValueError('bbox must be xyxy')
        import math
        x1,y1,x2,y2=self.bbox
        if not all(isinstance(n,(int,float)) and math.isfinite(n) and 0<=n<1e7 for n in self.bbox) or x1>=x2 or y1>=y2:
            raise ValueError('bbox invalid')


@dataclass(frozen=True)
class ObservedRegion:
    """Independent ROI and mask source references from actual host transformations."""
    roi_source_instance_id: str
    mask_source_instance_id: str
    mask_sha256: str

    def __post_init__(self):
        _checked_id(self.roi_source_instance_id, 'roi_source_instance_id')
        _checked_id(self.mask_source_instance_id, 'mask_source_instance_id')
        _checked_hash(self.mask_sha256, 'mask_sha256')


@dataclass(frozen=True)
class DispatchEnvelope:
    target: SegmentedTarget
    region: ObservedRegion
    job_id: str
    request_id: str
    task: str
    variant: str
    attempt_id: str
    marked_sha256: str


class HostRequestLedger:
    """Records actual dispatched requests; completion must correlate with the host job.

    Distinct task, variant, and attempt make the correlation key unique. The model
    text is never accepted as evidence of source identity.
    """
    def __init__(self):
        self._pending: dict[str, DispatchEnvelope] = {}
        self._used_job_ids: set[str] = set()
        self._used_request_ids: set[str] = set()

    def dispatch(self, *, target: SegmentedTarget, region: ObservedRegion,
                 task: str, variant: str, job_id: str, attempt_id: str,
                 marked_image_bytes: bytes) -> DispatchEnvelope:
        if task not in TASKS or variant not in VARIANTS:
            raise ValueError('unsupported task or variant')
        _checked_id(job_id, 'job_id')
        _checked_id(attempt_id, 'attempt_id')
        if not isinstance(marked_image_bytes, bytes) or not marked_image_bytes:
            raise ValueError('actual marked-image bytes required')
        if region.roi_source_instance_id != target.instance_id:
            raise ValueError('ROI origin differs from segmented target')
        if region.mask_source_instance_id != target.instance_id:
            raise ValueError('mask origin differs from segmented target')
        request_id = '__'.join((target.case_id,target.instance_id,task,variant,attempt_id))
        if job_id in self._used_job_ids or request_id in self._used_request_ids:
            raise ValueError('duplicate job or request correlation ID')
        envelope = DispatchEnvelope(target,region,job_id,request_id,task,variant,attempt_id,
                                    sha256(marked_image_bytes).hexdigest())
        self._pending[job_id] = envelope
        self._used_job_ids.add(job_id)
        self._used_request_ids.add(request_id)
        return envelope

    def complete(self, *, job_id: str, completed_request_id: str,
                 raw_text: str, response_job_id: str) -> dict:
        """response_job_id must come from host scheduler's job handle, not LLM text."""
        e = self._pending.get(job_id)
        if e is None:
            raise ValueError('unknown or already completed host job')
        if response_job_id != job_id or completed_request_id != e.request_id:
            raise ValueError('host response did not correlate with dispatched job')
        if not isinstance(raw_text,str):
            raise ValueError('model response must be text')
        del self._pending[job_id]
        return {
            'case_id':e.target.case_id,
            'instance_id':e.target.instance_id,
            'image_sha256':e.target.image_sha256,
            'mask_sha256':e.region.mask_sha256,
            'roi_source_instance_id':e.region.roi_source_instance_id,
            'mask_source_instance_id':e.region.mask_source_instance_id,
            'marked_image_sha256':e.marked_sha256,
            'request_id':e.request_id,
            'host_job_id':e.job_id,
            'task':e.task,'variant':e.variant,'attempt_id':e.attempt_id,
            'raw_response':raw_text,
            'status':'SHADOW_RUNTIME_IDENTITY_CORRELATED_NOT_GT',
            'category_publication_approved':False,
            'alias_merge_approved':False,
            'production_approved':False,
        }
