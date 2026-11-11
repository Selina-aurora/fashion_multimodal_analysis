"""Strict DEV60-only V5 producer-to-Qwen evidence contract (no GT, no release)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

V5_RAW_SHA256 = "688cb801daebd0484e8740c877c021eca86c83f362857d4fc982d02fb50f2710"
V5_CKPT_SHA256 = "c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0"
TARGETS = (('fresh_dev_015', 'pred_0000'), ('fresh_dev_004', 'pred_0002'), ('fresh_dev_011', 'pred_0003'), ('fresh_dev_054', 'pred_0001'), ('fresh_dev_001', 'pred_0000'), ('fresh_dev_003', 'pred_0001'), ('fresh_dev_005', 'pred_0001'), ('fresh_dev_010', 'pred_0001'), ('fresh_dev_014', 'pred_0000'), ('fresh_dev_014', 'pred_0006'), ('fresh_dev_008', 'pred_0000'), ('fresh_dev_008', 'pred_0002'))
QWEN_SMOKE_TARGETS = (('fresh_dev_015', 'pred_0000'), ('fresh_dev_011', 'pred_0003'), ('fresh_dev_003', 'pred_0001'), ('fresh_dev_014', 'pred_0006'))
FORBIDDEN = {"holdout", "sealed", "blind", "independent_test", "testset", "fullval", "validation"}
CATEGORIES = {"top", "pants", "skirt", "outerwear", "dress", "shoe", "bag", "accessory"}
# Same V5 source bytes as the successful 2026-11-06 two-target live run.
EXPECTED_MODULE_HASHES = {
    "src/fashion_multimodal_analysis/evaluation/runtime.py": "ff85ca020695498ab287d73e8afd915b44345ecd2168b38eb24410fb8ddddfb8",
    "src/fashion_multimodal_analysis/segmentation/modeling.py": "e17c526cfae507e12d79f83ac1662aae6334497ee20be5547d54642da44d2454",
    "src/fashion_multimodal_analysis/common/schema.py": "86916cc6a565c682320618e890fdf61e4ec7e8881a971a5cd9178502baee4f0c",
}



def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(obj: dict) -> str:
    return digest_bytes(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def refuse_restricted(path: Path):
    if any(any(bad in p.lower() for bad in FORBIDDEN) for p in Path(path).parts):
        raise ValueError("restricted path: " + str(path))


def verified_train_image(path: Path, root: Path, expected_sha: str) -> Path:
    refuse_restricted(path)
    candidate = Path(path).resolve(strict=True)
    approved = Path(root).resolve(strict=True)
    if not candidate.is_relative_to(approved) or "train" not in [x.lower() for x in candidate.parts]:
        raise ValueError("image outside approved training-image root")
    if candidate.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("unexpected image type")
    if digest_file(candidate) != expected_sha:
        raise ValueError("source image hash mismatch")
    return candidate


def pinned_frozen_targets(raw_dir: Path, image_root: Path) -> dict:
    raw_file = raw_dir / "v5_raw_predictions.jsonl"
    if digest_file(raw_file) != V5_RAW_SHA256:
        raise ValueError("frozen V5 source hash changed")
    summary = read_json(raw_dir / "v5_real_summary.json")
    run = read_json(raw_dir / "run_manifest.json")
    if (summary.get("n_images"), summary.get("n_predicted_instances")) != (60, 171):
        raise ValueError("frozen DEV60 counts differ")
    if run.get("checkpoint_sha256") != V5_CKPT_SHA256 or summary.get("checkpoint_sha256") != V5_CKPT_SHA256:
        raise ValueError("frozen checkpoint SHA changed")
    if run.get("score_threshold") != .4 or run.get("mask_threshold") != .5 or run.get("human_gt_used_for_inference") is not False:
        raise ValueError("frozen inference settings differ")
    records = read_jsonl(raw_file)
    if len(records) != 60:
        raise ValueError("expected 60 frozen images")
    found = {}
    for idx, rec in enumerate(records, 1):
        cid = f"fresh_dev_{idx:03d}"
        if rec.get("case_id") != cid:
            raise ValueError("frozen case order changed")
        ids = {(cid, x) for x in [iid for case, iid in TARGETS if case == cid]}
        if not ids:
            continue
        img = verified_train_image(Path(rec["source_image_path"]), image_root, rec["image_sha256"])
        for j, ins in enumerate(rec["instances"]):
            key = (cid, f"pred_{j:04d}")
            if key not in ids:
                continue
            if ins.get("instance_id") != key[1] or ins.get("predicted_category") not in CATEGORIES:
                raise ValueError("frozen target ID/category invalid")
            found[key] = {"case_id": cid, "instance_id": key[1], "image_path": str(img),
                          "image_sha256": rec["image_sha256"], "image_size": rec["image_size"],
                          "expected_category": ins["predicted_category"],
                          "expected_bbox": ins["bbox_xyxy"], "expected_score": ins["score"],
                          "frozen_mask_path": str(raw_dir / f"{cid}_{key[1]}_mask.png"),
                          "expected_instance_count": len(rec["instances"])}
    if set(found) != set(TARGETS):
        raise ValueError("missing frozen target records")
    return found


def bbox_iou(a, b):
    if len(a) != 4 or len(b) != 4 or any(not math.isfinite(float(x)) for x in list(a)+list(b)):
        raise ValueError("invalid bbox")
    x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))
    inter = max(0., x2-x1) * max(0., y2-y1)
    area_a = max(0., float(a[2])-float(a[0])) * max(0., float(a[3])-float(a[1]))
    area_b = max(0., float(b[2])-float(b[0])) * max(0., float(b[3])-float(b[1]))
    if area_a <= 0 or area_b <= 0:
        raise ValueError("degenerate bbox")
    return inter/(area_a+area_b-inter)


def mask_iou(produced, reference):
    import numpy as np
    a = np.asarray(produced) > 0
    b = np.asarray(reference) > 0
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("incompatible masks")
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        raise ValueError("empty mask")
    return int(np.logical_and(a, b).sum()) / union


def canonical_mask_digest(mask) -> str:
    import numpy as np
    a = np.asarray(mask)
    if a.ndim != 2 or not a.size or not a.any():
        raise ValueError("empty/non-image live mask")
    packed = np.packbits((a > 0).astype(np.uint8)).tobytes()
    return digest_bytes(f"{a.shape[0]}:{a.shape[1]}:".encode() + packed)


def check_producer_match(*, target: dict, ins: dict, reference_mask, produced_mask,
                         min_bbox_iou=.995, min_mask_iou=.98):
    if ins["instance_id"] != target["instance_id"] or ins["category"] != target["expected_category"]:
        raise ValueError("producer instance ID or class does not match the frozen target")
    b = bbox_iou(ins["bbox"], target["expected_bbox"])
    m = mask_iou(produced_mask, reference_mask)
    if b < min_bbox_iou or m < min_mask_iou or abs(float(ins["score"])-float(target["expected_score"])) > .02:
        raise ValueError(f"live V5 target differs from frozen result: bbox_iou={b:.5f} mask_iou={m:.5f}")
    return {"bbox_iou": round(b, 6), "mask_iou": round(m, 6),
            "score_abs_diff": round(abs(float(ins["score"])-float(target["expected_score"])), 6)}


def verified_event(event: dict, mask_png: Path, frozen_targets: dict):
    required = ("producer_event_sha256", "case_id", "instance_id", "source_image_sha256",
                "producer_mask_sha256", "mask_png_sha256", "producer_bbox_xyxy",
                "producer_category", "v5_checkpoint_sha256", "status")
    if any(key not in event for key in required):
        raise ValueError("producer event incomplete")
    e = {k: v for k, v in event.items() if k != "producer_event_sha256"}
    if canonical_sha(e) != event["producer_event_sha256"]:
        raise ValueError("producer event content changed")
    key = (event["case_id"], event["instance_id"])
    ref = frozen_targets.get(key)
    if ref is None or event["source_image_sha256"] != ref["image_sha256"]:
        raise ValueError("producer event source mismatch")
    if event["v5_checkpoint_sha256"] != V5_CKPT_SHA256 or event["status"] != "V5_RUNTIME_PRODUCER_CAPTURED_SHADOW_NOT_RELEASED":
        raise ValueError("producer checkpoint/status mismatch")
    if event["producer_category"] != ref["expected_category"]:
        raise ValueError("producer class changed")
    if digest_file(mask_png) != event["mask_png_sha256"]:
        raise ValueError("producer mask PNG changed")
    from PIL import Image
    import numpy as np
    with Image.open(mask_png) as f:
        a = np.asarray(f.convert("L"))
    if canonical_mask_digest(a) != event["producer_mask_sha256"]:
        raise ValueError("producer mask pixels changed")
    return ref
