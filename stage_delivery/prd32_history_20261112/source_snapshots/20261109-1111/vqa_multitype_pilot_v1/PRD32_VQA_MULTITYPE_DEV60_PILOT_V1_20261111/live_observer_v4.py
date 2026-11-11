"""GT-free synchronous live Qwen call observer, shadow-only.

Observes bytes and the actual in-process call boundaries. It does NOT observe
independent ROI/mask producers or prove visual attention. Never publishes model
labels, merges instances, or reads independent holdout.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable

VALID_CASE = re.compile(r"^fresh_dev_\d{3}$")
VALID_INSTANCE = re.compile(r"^pred_\d{4}$")
TASKS = ("tags", "vqa", "copy")
VARIANTS = {"baseline", "focus_v1"}
HEX = re.compile(r"^[0-9a-f]{64}$")


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _no_holdout(path: Path) -> None:
    forbidden = {"holdout", "sealed", "blind", "valid", "validation", "test", "eval", "devtest"}
    if any(part.lower() in forbidden for part in Path(path).parts):
        raise ValueError("refuse path containing restricted data partition")


def _bbox(box: Any) -> tuple[float, float, float, float]:
    if not isinstance(box, (tuple, list)) or len(box) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in box):
        raise ValueError("invalid bounding box")
    x1, y1, x2, y2 = map(float, box)
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError("invalid bbox coordinates")
    return (x1, y1, x2, y2)


class LiveShadowObserver:
    """One variant, new DEV-only output; task-level correlation from synchronous host call."""

    def __init__(self, variant: str):
        if variant not in VARIANTS:
            raise ValueError("unsupported variant")
        self.variant = variant
        self.pending: dict[str, dict[str, Any]] = {}
        self.completed: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.seen_request_ids: set[str] = set()
        self.used_job_ids: set[str] = set()

    def before_model(self, *, rec: dict, qrec: dict, task: str, marked: Path,
                     prompt: str, prompt_sha: str, job_id: str) -> str:
        case, iid = rec["case_id"], rec["target_instance_id"]
        if not VALID_CASE.fullmatch(case) or not VALID_INSTANCE.fullmatch(iid):
            raise ValueError("not a DEV predicted instance")
        if task not in TASKS:
            raise ValueError("unsupported task")
        if task == "vqa" and (not isinstance(qrec.get("query"), str) or not qrec["query"].strip()):
            raise ValueError("VQA question missing")
        if (qrec.get("target_instance_id"), qrec.get("task")) != (iid, task):
            raise ValueError("runner target/task mismatched the validated snapshot")
        if _bbox(qrec.get("target_bbox_xyxy")) != _bbox(rec.get("target_bbox_xyxy")):
            raise ValueError("runtime bbox differs from V5 snapshot")
        img = Path(qrec.get("image_path", ""))
        _no_holdout(img)
        if str(img) != rec["image_path"]:
            raise ValueError("original image path mismatch")
        exp_sha = rec["image_sha256"]
        if not isinstance(exp_sha, str) or not HEX.fullmatch(exp_sha):
            raise ValueError("expected image hash invalid")
        if qrec.get("image_sha256") != exp_sha or sha_file(img) != exp_sha:
            raise ValueError("actual input image SHA-256 mismatch")
        if not isinstance(prompt, str) or not prompt.strip() or digest(prompt.encode("utf-8")) != prompt_sha:
            raise ValueError("prompt hash changed")
        marked = Path(marked)
        _no_holdout(marked)
        if marked.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"} or not marked.is_file():
            raise ValueError("marked image is missing")
        marked_hash = sha_file(marked)
        key = (case, iid, task)
        host_request_id = "__".join((case, iid, task, self.variant, "attempt_01"))
        if key in self.completed or job_id in self.used_job_ids or host_request_id in self.seen_request_ids:
            raise ValueError("duplicate host task/job/request identity")
        self.pending[job_id] = {"case_id": case, "instance_id": iid, "task": task,
                                "variant": self.variant, "host_request_id": host_request_id,
                                "job_id": job_id, "image_sha256": exp_sha,
                                "bbox_xyxy": list(_bbox(rec["target_bbox_xyxy"])),
                                "marked_sha256": marked_hash,
                                "prompt_sha256": prompt_sha,
                                "v5_category": rec["predicted_category_v5_unverified"],
                                "vqa_question": qrec.get("query") if task == "vqa" else None}
        self.seen_request_ids.add(host_request_id)
        self.used_job_ids.add(job_id)
        return host_request_id

    def after_model(self, *, job_id: str, response_job_id: str,
                    completed_host_request_id: str, raw_text: str) -> dict:
        pending = self.pending.get(job_id)
        if pending is None:
            raise ValueError("no matching pending model job")
        if pending["job_id"] != response_job_id or pending["host_request_id"] != completed_host_request_id:
            raise ValueError("host response did not match dispatched job")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("missing model response text")
        key = pending["case_id"], pending["instance_id"], pending["task"]
        if key in self.completed:
            raise ValueError("duplicate response")
        ev = {**pending, "response_sha256": digest(raw_text.encode("utf-8")),
              "raw_text": raw_text, "status": "HOST_SYNCHRONOUS_CALL_CORRELATED_SHADOW_ONLY",
              "roi_producer_identity_verified": False,
              "mask_producer_identity_verified": False,
              "visual_target_binding_verified": False,
              "release_allowed": False}
        self.completed[key] = ev
        del self.pending[job_id]
        return {k: v for k, v in ev.items() if k != "raw_text"}

    def finalize(self, guard: Callable[[list[dict]], dict]) -> dict:
        if self.pending:
            raise ValueError("unfinished model calls")
        groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
        for (case, iid, task), ev in self.completed.items():
            groups[case, iid][task] = ev
        if not groups:
            raise ValueError("no model calls to finalize")
        items, traces = [], []
        for (case, iid), tasks in sorted(groups.items()):
            if set(tasks) != set(TASKS):
                raise ValueError("incomplete target task triplet")
            common = tasks["tags"]
            if any(e["image_sha256"] != common["image_sha256"] or e["bbox_xyxy"] != common["bbox_xyxy"] or e["marked_sha256"] != common["marked_sha256"] or e["v5_category"] != common["v5_category"] for e in tasks.values()):
                raise ValueError("source/marked/target changed across tasks")
            entries = {}
            for task in TASKS:
                e = tasks[task]
                entries[task] = {"case_id": case, "target_instance_id": iid,
                                 "request_id": e["host_request_id"], "payload": e["raw_text"]}
            items.append({"case_id": case, "instance_id": iid, "v5_category": common["v5_category"],
                          "bbox_xyxy": common["bbox_xyxy"], "vqa_question": tasks["vqa"]["vqa_question"],
                          **entries})
            traces.append({"case_id": case, "instance_id": iid,
                           "request_ids": {t: tasks[t]["host_request_id"] for t in TASKS},
                           "image_sha256": common["image_sha256"], "marked_sha256": common["marked_sha256"],
                           "roi_producer_identity_verified": False,
                           "mask_producer_identity_verified": False})
        report = guard(items)
        if report.get("instance_count") != len(items):
            raise ValueError("guard instance count mismatch")
        return {"schema": "prd32_live_qwen_shadow_v4", "variant": self.variant,
                "status": "LIVE_HOST_CALLS_CORRELATED_BUT_ROI_MASK_VISUAL_UNVERIFIED",
                "instances": len(items), "tasks_correlated": len(self.completed),
                "original_image_bytes_checked_at_call_time": True,
                "marked_image_bytes_checked_at_call_time": True,
                "task_specific_request_ids_checked": True,
                "host_correlation_scope": "SYNCHRONOUS_IN_PROCESS_ONLY_NOT_ASYNC_SCHEDULER",
                "roi_producer_identity_verified": False,
                "mask_producer_identity_verified": False,
                "gt_used": False, "holdout_accessed": False,
                "release_allowed": False, "production_approved": False,
                "guard_service_status": report.get("service_status"),
                "guard_flags": report.get("flag_counts"),
                "guard_instances": report.get("instances"),
                "lineage": traces}
