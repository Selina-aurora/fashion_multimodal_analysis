"""Opt-in PRD3.2 host-side identity bridge; DEV-only, shadow-only.

Never loads a model, reads GT, changes V5/Qwen predictions, or releases outputs.
The *host* must independently populate source/ROI/mask/job observations.
Historical logs cannot be upgraded to proven runtime identity by this module.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Any, Callable
import re

from runtime_identity_v2 import HostRequestLedger, ObservedRegion, SegmentedTarget

DEV_ID = re.compile(r"^fresh_dev_\d{3,}$")
INSTANCE_ID = re.compile(r"^pred_\d{4,}$")
TASK_ORDER = ("tags", "vqa", "copy")


class RuntimeBindingError(ValueError):
    """Fail-closed mismatch in host-observed identity/provenance."""


def _digest(data: bytes, label: str) -> str:
    if not isinstance(data, bytes) or not data:
        raise RuntimeBindingError(f"actual {label} bytes required")
    return sha256(data).hexdigest()


class RuntimeGuardBridge:
    """Correlate actual dispatches/completions, then reuse the GPU guard.

    `guard_outputs_fn` must be the existing GPU `guard_outputs` implementation.
    Every task/variant/attempt gets its own request ID; the legacy guard's
    item-level `request_id` is intentionally OMITTED, while task-specific IDs
    are independently verified by the HostRequestLedger.
    """

    def __init__(self, guard_outputs_fn: Callable[..., dict[str, Any]]):
        if not callable(guard_outputs_fn):
            raise TypeError("guard_outputs_fn must be callable")
        self._guard = guard_outputs_fn
        self.ledger = HostRequestLedger()
        self._pending_meta: dict[str, dict[str, Any]] = {}
        self._finished: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self._target_facts: dict[tuple[str, str], dict[str, Any]] = {}
        self._job_to_variant: dict[str, str] = {}

    def dispatch(
        self, *, case_id: str, instance_id: str,
        source_image_bytes: bytes, expected_image_sha256: str,
        bbox_xyxy: tuple[float, float, float, float], v5_category: str,
        roi_source_instance_id: str, mask_source_instance_id: str,
        roi_crop_bytes: bytes, expected_roi_sha256: str,
        mask_bytes: bytes, expected_mask_sha256: str,
        marked_image_bytes: bytes,
        task: str, variant: str, job_id: str, attempt_id: str,
        vqa_question: str | None = None,
        duplicate_candidate: bool = False,
    ) -> str:
        """Register a *real host dispatch*, before calling Qwen.

        Arguments called `*_source_instance_id` must come from the crop/mask
        producer's own provenance, not from the expected target ID copied by
        a downstream serializer. All bytes must be those actually processed.
        """
        if not isinstance(case_id, str) or not DEV_ID.fullmatch(case_id):
            raise RuntimeBindingError("DEV-only case_id required")
        if not isinstance(instance_id, str) or not INSTANCE_ID.fullmatch(instance_id):
            raise RuntimeBindingError("DEV prediction instance_id required")
        if not isinstance(v5_category, str) or not v5_category:
            raise RuntimeBindingError("nonempty V5 category required")
        if not isinstance(duplicate_candidate, bool):
            raise RuntimeBindingError("duplicate_candidate must be boolean")
        if task == "vqa" and (not isinstance(vqa_question, str) or not vqa_question.strip()):
            raise RuntimeBindingError("actual VQA question required")
        if task != "vqa" and vqa_question is not None:
            raise RuntimeBindingError("vqa_question only allowed for vqa")
        if _digest(source_image_bytes, "source image") != expected_image_sha256:
            raise RuntimeBindingError("original image SHA256 differs from actual bytes")
        if _digest(roi_crop_bytes, "ROI crop") != expected_roi_sha256:
            raise RuntimeBindingError("ROI crop SHA256 differs from actual bytes")
        if _digest(mask_bytes, "mask") != expected_mask_sha256:
            raise RuntimeBindingError("mask SHA256 differs from actual bytes")
        _digest(marked_image_bytes, "marked image")
        try:
            target = SegmentedTarget(case_id, instance_id, expected_image_sha256, tuple(bbox_xyxy))
            region = ObservedRegion(roi_source_instance_id, mask_source_instance_id, expected_mask_sha256)
        except (ValueError, TypeError) as exc:
            raise RuntimeBindingError(str(exc)) from exc
        key = (case_id, instance_id)
        fact = {"case_id": case_id, "instance_id": instance_id,
                "image_sha256": expected_image_sha256, "bbox_xyxy": tuple(bbox_xyxy),
                "v5_category": v5_category, "roi_sha256": expected_roi_sha256,
                "mask_sha256": expected_mask_sha256,
                "duplicate_candidate": duplicate_candidate}
        if key in self._target_facts and self._target_facts[key] != fact:
            raise RuntimeBindingError("target/mask/bbox/category facts changed across tasks or variants")
        if job_id in self._pending_meta:
            raise RuntimeBindingError("duplicate pending host job")
        try:
            e = self.ledger.dispatch(target=target, region=region, task=task,
                                     variant=variant, job_id=job_id,
                                     attempt_id=attempt_id, marked_image_bytes=marked_image_bytes)
        except (ValueError, TypeError) as exc:
            raise RuntimeBindingError(str(exc)) from exc
        self._target_facts.setdefault(key, fact)
        self._job_to_variant[job_id] = variant
        self._pending_meta[job_id] = {"key": (case_id, instance_id, variant, task),
                                      "request_id": e.request_id,
                                      "vqa_question": vqa_question,
                                      "marked_sha256": e.marked_sha256}
        return e.request_id

    def complete(self, *, host_job_id: str, response_host_job_id: str,
                 completed_request_id: str, raw_text: str) -> dict[str, Any]:
        """Call on completion using the scheduler job handle (never model prose)."""
        meta = self._pending_meta.get(host_job_id)
        if meta is None:
            raise RuntimeBindingError("unknown or already completed host job")
        if meta["key"] in self._finished:
            raise RuntimeBindingError("task result already completed; retry must use new attempt and explicit policy")
        try:
            correlated = self.ledger.complete(job_id=host_job_id,
                                               completed_request_id=completed_request_id,
                                               response_job_id=response_host_job_id,
                                               raw_text=raw_text)
        except (ValueError, TypeError) as exc:
            raise RuntimeBindingError(str(exc)) from exc
        self._finished[meta["key"]] = {
            "raw_text": raw_text,
            "record": correlated,
            "vqa_question": meta["vqa_question"],
            "marked_sha256": meta["marked_sha256"],
        }
        del self._pending_meta[host_job_id]
        return {"status": "CORRELATED_SHADOW_ONLY", "request_id": completed_request_id,
                "host_job_id": host_job_id, "response_sha256": _digest(raw_text.encode("utf-8"), "response")
                if raw_text else sha256(b"").hexdigest(),
                "production_approved": False}

    def finalize_variant(self, variant: str) -> dict[str, Any]:
        """Validate all completed target/task triplets and call existing guard.

        The existing guard is called ONCE for the variant's full target batch,
        preserving same-image box-overlap cues. Different variants are separate.
        """
        if variant not in ("baseline", "focus_v1"):
            raise RuntimeBindingError("unsupported variant")
        if any(meta["key"][2] == variant for meta in self._pending_meta.values()):
            raise RuntimeBindingError("variant contains unfinished host jobs")
        groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for (case_id, inst, v, task), row in self._finished.items():
            if v == variant:
                groups[case_id, inst][task] = row
        if not groups:
            raise RuntimeBindingError("no completed targets for variant")
        rows, lineage = [], []
        for key in sorted(groups):
            tasks = groups[key]
            if set(tasks) != set(TASK_ORDER):
                raise RuntimeBindingError(f"incomplete task triplet for {key}: {sorted(tasks)}")
            common = self._target_facts[key]
            if len({tasks[t]["marked_sha256"] for t in TASK_ORDER}) != 1:
                raise RuntimeBindingError(f"marked image changed between tasks for {key}")
            if len({tasks[t]["record"]["mask_sha256"] for t in TASK_ORDER}) != 1:
                raise RuntimeBindingError(f"mask changed between tasks for {key}")
            envelopes = {}
            ids = {}
            for task in TASK_ORDER:
                record = tasks[task]["record"]
                if (record["case_id"], record["instance_id"]) != key:
                    raise RuntimeBindingError("host output identity mismatch")
                envelopes[task] = {"case_id": key[0], "target_instance_id": key[1],
                                   "request_id": record["request_id"],
                                   "payload": tasks[task]["raw_text"]}
                ids[task] = record["request_id"]
            if len(set(ids.values())) != 3:
                raise RuntimeBindingError("reused request_id across tasks")
            rows.append({"case_id": key[0], "instance_id": key[1],
                         "v5_category": common["v5_category"],
                         "bbox_xyxy": list(common["bbox_xyxy"]),
                         "duplicate_candidate": common["duplicate_candidate"],
                         "vqa_question": tasks["vqa"]["vqa_question"],
                         **envelopes})
            lineage.append({"case_id": key[0], "instance_id": key[1],
                            "task_request_ids": ids,
                            "image_sha256": common["image_sha256"],
                            "mask_sha256": common["mask_sha256"],
                            "roi_crop_sha256": common["roi_sha256"],
                            "marked_sha256": tasks["tags"]["marked_sha256"]})
        guard_report = self._guard(rows)
        if guard_report.get("instance_count") != len(rows):
            raise RuntimeBindingError("GPU guard instance count mismatch")
        return {"schema": "prd32_runtime_guard_bridge_v3",
                "status": "SHADOW_RUNTIME_CORRELATION_ONLY_NOT_VISUAL_GT",
                "variant": variant,
                "instance_count": len(rows),
                "host_job_correlated": len(rows)*3,
                "task_level_request_id_verified": True,
                "source_bytes_sha256_verified_on_dispatch": True,
                "mask_bytes_sha256_verified_on_dispatch": True,
                "roi_bytes_sha256_verified_on_dispatch": True,
                "guard_service_status": guard_report.get("service_status"),
                "guard_flags": guard_report.get("flag_counts"),
                "guard_geometric_overlap_pairs": guard_report.get("geometric_overlap_pairs", []),
                "guard_instances": guard_report.get("instances", []),
                "lineage": lineage,
                "historical_dev60_runtime_binding_proven": False,
                "actual_model_inference_performed_by_bridge": False,
                "ground_truth_used": False,
                "holdout_opened": False,
                "release_allowed": False,
                "production_approved": False}
