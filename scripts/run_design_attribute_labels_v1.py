from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "openai/clip-vit-base-patch32"
REPORT_DIR = PROJECT_ROOT / "reports" / "prd_attribute_extraction" / "design_labels_v1"
UPPER_CATEGORIES = {"top", "outerwear", "dress"}
SUPPORTED_CATEGORIES = {"top", "outerwear", "dress", "pants", "skirt"}

SLEEVE_PROMPTS: Dict[str, str] = {
    "sleeveless": "a sleeveless fashion garment with no sleeves and open armholes",
    "cap": "a fashion garment with tiny cap sleeves covering only the shoulder edge",
    "short": "a fashion garment with short sleeves ending on the upper arm",
    "elbow": "a fashion garment with sleeves ending around the elbow",
    "three_quarter": "a fashion garment with three-quarter sleeves ending below the elbow",
    "long": "a fashion garment with long sleeves reaching the wrist",
}

NECKLINE_PROMPTS: Dict[str, str] = {
    "crew": "a fashion garment with a close crew neck neckline",
    "v_neck": "a fashion garment with a clearly V-shaped neckline",
    "round_scoop": "a fashion garment with a round or scoop neckline",
    "square": "a fashion garment with a square neckline",
    "boat": "a fashion garment with a wide boat neckline across the upper chest",
    "halter": "a fashion garment with a halter neckline",
    "off_shoulder": "an off-shoulder fashion garment exposing the shoulders",
    "one_shoulder": "a one-shoulder fashion garment with an asymmetric neckline",
    "turtleneck": "a fashion garment with a tall turtleneck covering the neck",
    "mock_neck": "a fashion garment with a short raised mock neck",
    "polo": "a fashion garment with a polo-style collar",
    "shirt_collar": "a fashion garment with a visible shirt-style collar",
    "hooded": "a fashion garment with a hood around the neckline",
    "other": "a fashion garment with another clearly visible neckline type",
}

FIT_PROMPTS_BY_CATEGORY: Dict[str, Dict[str, str]] = {
    "top": {
        "slim_fitted": "a fitted top with a narrow close-to-body silhouette",
        "regular_straight": "a regular-fit top with a straight balanced silhouette",
        "loose_oversized": "a loose or oversized top with a wide relaxed silhouette",
    },
    "outerwear": {
        "slim_fitted": "fitted outerwear with a narrow close-to-body silhouette",
        "regular_straight": "regular-fit outerwear with a straight balanced silhouette",
        "loose_oversized": "loose or oversized outerwear with a wide relaxed silhouette",
    },
    "pants": {
        "slim_tapered": "slim or tapered pants narrowing toward the ankle",
        "straight": "straight-leg pants with similar width from thigh to hem",
        "wide_leg": "wide-leg pants with a broad loose leg silhouette",
        "flared": "flared pants that widen clearly toward the hem",
    },
    "skirt": {
        "straight": "a straight skirt with a narrow vertical silhouette",
        "a_line": "an A-line skirt that gradually widens from waist to hem",
        "flared": "a strongly flared skirt with a wide expanded hem",
    },
    "dress": {
        "fitted": "a fitted dress with a close-to-body silhouette",
        "straight": "a straight dress with a mostly vertical silhouette",
        "a_line": "an A-line dress gradually widening toward the hem",
        "flared": "a flared dress with a clearly expanded skirt silhouette",
    },
}

STYLE_PROMPTS: Dict[str, str] = {
    "casual": "a casual everyday fashion garment with a relaxed practical style",
    "formal": "a formal polished fashion garment suitable for business or formal occasions",
    "sporty": "a sporty athletic fashion garment with an activewear-inspired style",
    "streetwear": "a streetwear fashion garment with an urban contemporary style",
    "elegant": "an elegant refined fashion garment with a sophisticated appearance",
    "minimalist": "a minimalist fashion garment with clean simple lines and restrained details",
    "vintage": "a vintage or retro-inspired fashion garment",
    "romantic": "a romantic feminine fashion garment with soft decorative styling",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="configs/garment_instances_gt_pilot_100.csv")
    p.add_argument("--style-json", default="reports/prd_attribute_extraction/style_v1/style_feature_vector_v1.json")
    p.add_argument("--audit-size", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def resolve_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    for c in [p, Path.cwd() / p, PROJECT_ROOT / p]:
        if c.exists():
            return c.resolve()
    matches = [x.resolve() for x in PROJECT_ROOT.rglob(p.name) if x.is_file()]
    if len(matches) == 1:
        print(f"[path fallback] {raw} -> {matches[0].relative_to(PROJECT_ROOT)}")
        return matches[0]
    raise FileNotFoundError(raw)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"source_image", "garment_id", "garment_category", "crop_path", "mask_path"}
    if not rows:
        raise ValueError("Manifest is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return rows


def read_style_json(path: Path) -> dict[tuple[str, str], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    garments = payload.get("garments", payload.get("records", []))
    idx = {}
    for item in garments:
        key = (str(item.get("source_image", "")).replace("\\", "/").strip(), str(item.get("garment_id", "")).strip())
        idx[key] = item
    return idx


def load_crop_and_mask(row: dict[str, str]) -> tuple[Image.Image, Image.Image | None]:
    crop = Image.open(resolve_path(row["crop_path"])).convert("RGB")
    mask_raw = row.get("mask_path", "").strip()
    if not mask_raw:
        return crop, None
    mask = Image.open(resolve_path(mask_raw)).convert("L")
    if mask.size != crop.size:
        mask = mask.resize(crop.size, Image.Resampling.NEAREST)
    return crop, mask


def masked_white(crop: Image.Image, mask: Image.Image | None) -> Image.Image:
    if mask is None:
        return crop
    return Image.composite(crop, Image.new("RGB", crop.size, "white"), mask)


def tight_bbox(mask: Image.Image | None, size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    if mask is None:
        return (0, 0, w, h)
    arr = np.asarray(mask, dtype=np.uint8) > 127
    if not arr.any():
        return (0, 0, w, h)
    ys, xs = np.nonzero(arr)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def neckline_crop(crop: Image.Image, mask: Image.Image | None) -> Image.Image:
    x1, y1, x2, y2 = tight_bbox(mask, crop.size)
    gw, gh = max(1, x2 - x1), max(1, y2 - y1)
    lx1 = max(0, x1 + round(gw * 0.08))
    lx2 = min(crop.width, x1 + round(gw * 0.92))
    ly1 = max(0, y1)
    ly2 = min(crop.height, y1 + round(gh * 0.48))
    local = crop.crop((lx1, ly1, lx2, ly2))
    if mask is None:
        return local
    local_mask = mask.crop((lx1, ly1, lx2, ly2))
    return Image.composite(local, Image.new("RGB", local.size, "white"), local_mask)


def norm_entropy(probs: torch.Tensor) -> float:
    p = probs.detach().float().cpu().clamp_min(1e-12)
    return float((-torch.sum(p * torch.log(p)).item()) / math.log(len(p))) if len(p) > 1 else 0.0


def classify(image: Image.Image, prompts: Dict[str, str], model, processor, device) -> dict:
    labels: List[str] = list(prompts)
    inputs = processor(text=[prompts[x] for x in labels], images=image, return_tensors="pt", padding=True)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.inference_mode():
        probs = model(**inputs).logits_per_image[0].softmax(dim=0).cpu()
    order = torch.argsort(probs, descending=True)
    i1 = int(order[0]); i2 = int(order[1]) if len(order) > 1 else i1
    top3 = [{"label": labels[int(i)], "score": float(probs[int(i)])} for i in order[:3]]
    s1, s2 = float(probs[i1]), float(probs[i2])
    return {
        "label": labels[i1], "confidence": s1,
        "top2_label": labels[i2], "top2_score": s2,
        "margin": s1 - s2, "entropy": norm_entropy(probs), "top3": top3,
        "confidence_note": "candidate-relative CLIP softmax; not calibrated probability",
    }


def geometry_gate(style_item: dict) -> dict:
    q = style_item.get("quality", style_item.get("style_quality", {}))
    uq = q.get("upper_geometry_quality")
    tq = q.get("trouser_geometry_quality")
    return {
        "upper_geometry_quality": uq,
        "upper_geometry_valid_for_downstream": uq in {"good", "usable"} if uq is not None else None,
        "trouser_geometry_quality": tq,
        "trouser_geometry_valid_for_downstream": tq in {"good", "usable"} if tq is not None else None,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def make_audit(records: list[dict], n: int, seed: int) -> list[dict]:
    rows = []
    for r in records:
        for attr, item in r["design_attributes"].items():
            if item is None:
                continue
            rows.append({
                "source_image": r["source_image"], "garment_id": r["garment_id"],
                "garment_category": r["garment_category"], "attribute": attr,
                "predicted_label": item["label"], "top1_score": item["confidence"],
                "top2_label": item["top2_label"], "top2_score": item["top2_score"],
                "margin": item["margin"], "entropy": item["entropy"],
                "manual_label": "", "manual_correct": "", "ambiguous": "",
                "error_type": "", "notes": "", "reviewer": "",
            })
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows if n <= 0 or n >= len(rows) else rows[:n]


def main() -> None:
    args = parse_args()
    manifest = read_manifest(resolve_path(args.manifest))
    style_idx = read_style_json(resolve_path(args.style_json))

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    records = []
    for i, row in enumerate(manifest, start=1):
        cat = row["garment_category"].strip()
        if cat not in SUPPORTED_CATEGORIES:
            continue
        src = row["source_image"].replace("\\", "/").strip()
        gid = row["garment_id"].strip()
        style_item = style_idx.get((src, gid), {})
        features = style_item.get("style_features", {})
        quality = style_item.get("quality", style_item.get("style_quality", {}))

        crop, mask = load_crop_and_mask(row)
        whole = masked_white(crop, mask)
        attrs = {
            "sleeve_length": None,
            "neckline": None,
            "silhouette_fit": classify(whole, FIT_PROMPTS_BY_CATEGORY[cat], model, processor, device),
            "fashion_style": classify(whole, STYLE_PROMPTS, model, processor, device),
        }
        if cat in UPPER_CATEGORIES:
            attrs["sleeve_length"] = classify(whole, SLEEVE_PROMPTS, model, processor, device)
            attrs["neckline"] = classify(neckline_crop(crop, mask), NECKLINE_PROMPTS, model, processor, device)

        records.append({
            "source_image": src, "garment_id": gid, "garment_category": cat,
            "design_attributes": attrs,
            "continuous_style_features": features,
            "style_quality": quality,
            "geometry_gate": geometry_gate(style_item),
        })
        if i % 20 == 0 or i == len(manifest):
            print(f"processed {i}/{len(manifest)}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "3.1.3-design-labels-v1",
        "model": MODEL_NAME,
        "confidence_note": "candidate-relative CLIP softmax; not calibrated probability",
        "geometry_policy": "weak geometry is flagged invalid for trusted downstream geometry use; raw values are preserved",
        "records": records,
    }
    (REPORT_DIR / "design_attribute_predictions_v1.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    flat = []
    for r in records:
        x = {"source_image": r["source_image"], "garment_id": r["garment_id"], "garment_category": r["garment_category"]}
        for attr in ["sleeve_length", "neckline", "silhouette_fit", "fashion_style"]:
            item = r["design_attributes"][attr]
            for field in ["label", "confidence", "top2_label", "margin", "entropy"]:
                x[f"{attr}_{field}"] = "" if item is None else item[field]
        for field in ["sleeve_length_ratio", "neckline_depth_score", "neckline_width_score", "collar_height_score"]:
            x[field] = r["continuous_style_features"].get(field, "")
        x.update(r["geometry_gate"])
        flat.append(x)
    write_csv(REPORT_DIR / "design_attribute_predictions_v1.csv", flat)

    audit = make_audit(records, args.audit_size, args.seed)
    write_csv(REPORT_DIR / "manual_audit_holdout_v1.csv", audit)

    lines = [
        "PRD 3.1.3 design attribute categorical baseline v1",
        "================================================",
        f"model={MODEL_NAME}", f"records={len(records)}", f"audit_rows={len(audit)}", f"audit_seed={args.seed}", "",
        "Label distributions", "-------------------",
    ]
    for attr in ["sleeve_length", "neckline", "silhouette_fit", "fashion_style"]:
        c = Counter(r["design_attributes"][attr]["label"] for r in records if r["design_attributes"][attr] is not None)
        lines.append(attr + ":")
        lines.extend(f"  {k}={v}" for k, v in c.most_common())
    lines += ["", "Important", "---------",
              "- baseline only; not PRD acceptance",
              "- CLIP confidence is candidate-relative, not calibrated",
              "- do not tune on the manual holdout and reuse it as final validation",
              "- weak geometry is gated from trusted downstream geometry use",
              "- replace neckline proxy with real 3.1.2 ROI in final end-to-end integration"]
    (REPORT_DIR / "summary_v1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nFINISHED")
    print("reports/prd_attribute_extraction/design_labels_v1/")


if __name__ == "__main__":
    main()
