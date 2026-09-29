"""
Freeze the Fashionpedia -> PRD mapping and build a LEAKAGE-SAFE
3.1.1 segmentation benchmark REVIEW POOL.

Important
---------
This does NOT freeze benchmark_test yet.

It deliberately excludes entire source images that have already appeared
in train / validation / pilot / matched / ablation configs. Excluding only
(source_image, garment_id) is not enough for a blind image benchmark because
another garment from the same image would leak visual context.

Default review-pool design
--------------------------
- 150 candidates per PRD class = 1200 review candidates.
- Later, after human review, freeze 100 per class = 800 benchmark instances.
- Sampling is source-image unique within each PRD class and diversified over
  fine/source categories where possible.
- seed = 20260922.

Sources
-------
DeepFashion2:
    top, pants, skirt, outerwear, dress

Fashionpedia:
    shoe <- id 23 "shoe"
    bag <- id 24 "bag, wallet"
    accessory <- ids 13-22 and 25:
        glasses
        hat
        headband, head covering, hair accessory
        tie
        glove
        watch
        belt
        leg warmer
        tights, stockings
        sock
        scarf

Outputs
-------
benchmark/prd_3_1_v1/
├── schemas/fashionpedia_to_prd_mapping_v1.json
└── candidates/
    └── segmentation_review_pool_v1/
        ├── segmentation_review_pool_v1.csv
        ├── segmentation_review_checklist_v1.csv
        ├── candidate_pool_summary.csv
        ├── contact_sheets/
        │   ├── top.jpg
        │   ├── pants.jpg
        │   ├── ...
        │   └── accessory.jpg
        └── summary.txt

Run
---
cd /workspace/fashion_multimodal_analysis

python scripts/build_prd31_segmentation_review_pool_v1.py

Optional
--------
python scripts/build_prd31_segmentation_review_pool_v1.py \
  --per-class 150 \
  --contact-sheet-items 48 \
  --seed 20260922
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "fashion_data"

BENCH_ROOT = PROJECT_ROOT / "benchmark" / "prd_3_1_v1"
USED_REGISTRY = BENCH_ROOT / "audit" / "used_sample_registry.csv"

OUT_DIR = BENCH_ROOT / "candidates" / "segmentation_review_pool_v1"
CONTACT_DIR = OUT_DIR / "contact_sheets"

MAPPING_OUT = (
    BENCH_ROOT
    / "schemas"
    / "fashionpedia_to_prd_mapping_v1.json"
)

DF2_ANNO_DIR = DATA_ROOT / "raw" / "train" / "train" / "annos"
DF2_IMAGE_DIR = DATA_ROOT / "raw" / "train" / "train" / "image"

FP_ROOT = DATA_ROOT / "raw" / "fashionpedia"
FP_ANN = FP_ROOT / "annotations" / "instances_attributes_val2020.json"

CLASSES = [
    "top",
    "pants",
    "skirt",
    "outerwear",
    "dress",
    "shoe",
    "bag",
    "accessory",
]

DF2_FINE = {
    1: "short_sleeve_top",
    2: "long_sleeve_top",
    3: "short_sleeve_outwear",
    4: "long_sleeve_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeve_dress",
    11: "long_sleeve_dress",
    12: "vest_dress",
    13: "sling_dress",
}

DF2_TO_PRD = {
    1: "top",
    2: "top",
    3: "outerwear",
    4: "outerwear",
    5: "top",
    6: "top",
    7: "pants",
    8: "pants",
    9: "skirt",
    10: "dress",
    11: "dress",
    12: "dress",
    13: "dress",
}

FP_ID_TO_PRD = {
    13: "accessory",  # glasses
    14: "accessory",  # hat
    15: "accessory",  # headband, head covering, hair accessory
    16: "accessory",  # tie
    17: "accessory",  # glove
    18: "accessory",  # watch
    19: "accessory",  # belt
    20: "accessory",  # leg warmer
    21: "accessory",  # tights, stockings
    22: "accessory",  # sock
    23: "shoe",       # shoe
    24: "bag",        # bag, wallet
    25: "accessory",  # scarf
}

EXPECTED_FP_NAMES = {
    13: "glasses",
    14: "hat",
    15: "headband, head covering, hair accessory",
    16: "tie",
    17: "glove",
    18: "watch",
    19: "belt",
    20: "leg warmer",
    21: "tights, stockings",
    22: "sock",
    23: "shoe",
    24: "bag, wallet",
    25: "scarf",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=150)
    p.add_argument("--contact-sheet-items", type=int, default=48)
    p.add_argument("--seed", type=int, default=20260922)
    return p.parse_args()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def norm(v: Any) -> str:
    return str(v if v is not None else "").replace("\\", "/").strip()


def rel_data(path: Path) -> str:
    return "../fashion_data/" + str(
        path.resolve().relative_to(DATA_ROOT.resolve())
    ).replace("\\", "/")


def valid_bbox_xyxy(box):
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = map(float, box)
    except Exception:
        return False
    return x2 > x1 and y2 > y1


def bbox_area_xyxy(box):
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def object_size_bucket(area_ratio: float) -> str:
    # Internal benchmark stratification, not an external standard.
    if area_ratio < 0.02:
        return "small"
    if area_ratio < 0.10:
        return "medium"
    return "large"


def used_image_registry():
    rows = read_csv(USED_REGISTRY)

    df2_names = set()
    fp_names = set()
    all_paths = set()

    for r in rows:
        p = norm(r.get("source_image", ""))
        if not p:
            continue

        all_paths.add(p)
        name = Path(p).name

        if "fashionpedia" in p.lower():
            fp_names.add(name)
        elif "/raw/train/train/" in p.lower():
            df2_names.add(name)

    return df2_names, fp_names, all_paths


def sample_diverse(
    candidates: list[dict[str, Any]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Round-robin over fine/source categories while preferring unique source images.
    """
    by_fine = defaultdict(list)
    for row in candidates:
        by_fine[row["fine_or_source_category"]].append(row)

    for items in by_fine.values():
        rng.shuffle(items)

    fine_names = list(by_fine.keys())
    rng.shuffle(fine_names)

    selected = []
    selected_images = set()
    pointers = {k: 0 for k in fine_names}

    while len(selected) < n:
        progressed = False

        for fine in fine_names:
            items = by_fine[fine]
            i = pointers[fine]

            while i < len(items):
                item = items[i]
                i += 1
                pointers[fine] = i

                img = item["source_image"]
                if img in selected_images:
                    continue

                selected.append(item)
                selected_images.add(img)
                progressed = True
                break

            if len(selected) >= n:
                break

        if not progressed:
            break

    return selected


def scan_df2(excluded_image_names: set[str]):
    by_class = defaultdict(list)

    ann_files = sorted(DF2_ANNO_DIR.glob("*.json"))

    for idx, ann_path in enumerate(ann_files, start=1):
        image_path = DF2_IMAGE_DIR / f"{ann_path.stem}.jpg"
        if not image_path.is_file():
            alternatives = list(DF2_IMAGE_DIR.glob(f"{ann_path.stem}.*"))
            if not alternatives:
                continue
            image_path = alternatives[0]

        if image_path.name in excluded_image_names:
            continue

        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        try:
            with Image.open(image_path) as im:
                width, height = im.size
        except Exception:
            continue

        for item_key, item in data.items():
            if not str(item_key).startswith("item") or not isinstance(item, dict):
                continue

            try:
                cid = int(item.get("category_id"))
            except Exception:
                continue

            prd = DF2_TO_PRD.get(cid)
            fine = DF2_FINE.get(cid)

            if not prd or not fine:
                continue

            bbox = item.get("bounding_box")
            seg = item.get("segmentation")

            if not valid_bbox_xyxy(bbox) or not seg:
                continue

            area = bbox_area_xyxy(bbox)
            ratio = area / max(1, width * height)

            by_class[prd].append({
                "benchmark_version": "prd_3_1_v1",
                "pool_status": "REVIEW_CANDIDATE",
                "source_dataset": "DeepFashion2",
                "source_image": rel_data(image_path),
                "source_image_name": image_path.name,
                "garment_id": f"{item_key}_{fine}",
                "garment_category": prd,
                "fine_or_source_category": fine,
                "source_category_id": cid,
                "annotation_id": "",
                "annotation_path": rel_data(ann_path),
                "image_width": width,
                "image_height": height,
                "gt_bbox_x1": bbox[0],
                "gt_bbox_y1": bbox[1],
                "gt_bbox_x2": bbox[2],
                "gt_bbox_y2": bbox[3],
                "bbox_area_ratio": f"{ratio:.8f}",
                "object_size_bucket": object_size_bucket(ratio),
                "has_gt_segmentation": 1,
            })

        if idx % 20000 == 0:
            print(f"DeepFashion2 scanned {idx}/{len(ann_files)}...")

    return by_class


def discover_fp_image(file_name: str) -> Path | None:
    direct_candidates = [
        FP_ROOT / "images" / "test" / file_name,
        FP_ROOT / "images" / "val" / file_name,
        FP_ROOT / file_name,
    ]

    for p in direct_candidates:
        if p.is_file():
            return p

    hits = list(FP_ROOT.rglob(file_name))
    return hits[0] if hits else None


def scan_fashionpedia(excluded_image_names: set[str]):
    payload = json.loads(FP_ANN.read_text(encoding="utf-8"))

    categories = {
        int(c["id"]): str(c.get("name", "")).strip()
        for c in payload["categories"]
    }

    # Hard consistency check against the historically used mapping.
    mismatches = []
    for cid, expected in EXPECTED_FP_NAMES.items():
        actual = categories.get(cid)
        if actual != expected:
            mismatches.append((cid, expected, actual))

    if mismatches:
        raise RuntimeError(
            "Fashionpedia category inventory does not match frozen mapping: "
            + repr(mismatches)
        )

    images = {
        int(i["id"]): i
        for i in payload["images"]
    }

    by_class = defaultdict(list)
    image_cache = {}

    for ann in payload["annotations"]:
        try:
            cid = int(ann["category_id"])
        except Exception:
            continue

        prd = FP_ID_TO_PRD.get(cid)
        if not prd:
            continue

        image_info = images.get(int(ann["image_id"]))
        if not image_info:
            continue

        file_name = str(image_info["file_name"])

        if file_name in excluded_image_names:
            continue

        if file_name not in image_cache:
            image_cache[file_name] = discover_fp_image(file_name)

        image_path = image_cache[file_name]
        if image_path is None:
            continue

        bbox_xywh = ann.get("bbox")
        seg = ann.get("segmentation")

        if (
            not isinstance(bbox_xywh, list)
            or len(bbox_xywh) != 4
            or not seg
        ):
            continue

        try:
            x, y, w, h = map(float, bbox_xywh)
        except Exception:
            continue

        if w <= 0 or h <= 0:
            continue

        width = int(image_info.get("width", 0))
        height = int(image_info.get("height", 0))

        if width <= 0 or height <= 0:
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except Exception:
                continue

        x1, y1, x2, y2 = x, y, x + w, y + h
        ratio = (w * h) / max(1, width * height)

        ann_id = int(ann["id"])

        by_class[prd].append({
            "benchmark_version": "prd_3_1_v1",
            "pool_status": "REVIEW_CANDIDATE",
            "source_dataset": "Fashionpedia",
            "source_image": rel_data(image_path),
            "source_image_name": image_path.name,
            "garment_id": f"fashionpedia_ann_{ann_id}",
            "garment_category": prd,
            "fine_or_source_category": categories[cid],
            "source_category_id": cid,
            "annotation_id": ann_id,
            "annotation_path": rel_data(FP_ANN),
            "image_width": width,
            "image_height": height,
            "gt_bbox_x1": f"{x1:.4f}",
            "gt_bbox_y1": f"{y1:.4f}",
            "gt_bbox_x2": f"{x2:.4f}",
            "gt_bbox_y2": f"{y2:.4f}",
            "bbox_area_ratio": f"{ratio:.8f}",
            "object_size_bucket": object_size_bucket(ratio),
            "has_gt_segmentation": 1,
        })

    return by_class, categories


def make_contact_sheet(
    rows: list[dict[str, Any]],
    out_path: Path,
    max_items: int,
):
    rows = rows[:max_items]
    if not rows:
        return

    cols = 6
    tile_w = 220
    tile_h = 250
    rows_n = (len(rows) + cols - 1) // cols

    sheet = Image.new(
        "RGB",
        (cols * tile_w, rows_n * tile_h),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for i, r in enumerate(rows):
        rr = i // cols
        cc = i % cols
        x0 = cc * tile_w
        y0 = rr * tile_h

        image_path = PROJECT_ROOT / r["source_image"]
        if not image_path.exists():
            image_path = (PROJECT_ROOT / r["source_image"]).resolve()

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        x1 = max(0, int(round(float(r["gt_bbox_x1"]))))
        y1 = max(0, int(round(float(r["gt_bbox_y1"]))))
        x2 = min(image.width, int(round(float(r["gt_bbox_x2"]))))
        y2 = min(image.height, int(round(float(r["gt_bbox_y2"]))))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))
        crop.thumbnail((200, 185))

        px = x0 + (tile_w - crop.width) // 2
        py = y0 + 5
        sheet.paste(crop, (px, py))

        label = (
            f"{r['garment_category']}\n"
            f"{r['fine_or_source_category'][:26]}\n"
            f"{r['source_dataset']}\n"
            f"{r['source_image_name']}"
        )
        draw.multiline_text(
            (x0 + 5, y0 + 195),
            label,
            fill="black",
            font=font,
            spacing=2,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_OUT.parent.mkdir(parents=True, exist_ok=True)

    df2_used_images, fp_used_images, _ = used_image_registry()

    print("Used source-image exclusion:")
    print("  DeepFashion2 images:", len(df2_used_images))
    print("  Fashionpedia images:", len(fp_used_images))

    # Freeze mapping semantics now that code + empirical audit agree.
    mapping_payload = {
        "schema_version": "fashionpedia_to_prd_mapping_v1",
        "source": "Fashionpedia instances_attributes_val2020.json",
        "policy": (
            "Exact source category IDs inherited from existing project "
            "training/validation mapping; no keyword expansion in benchmark."
        ),
        "mapping": {
            str(cid): {
                "source_name": EXPECTED_FP_NAMES[cid],
                "prd_class": FP_ID_TO_PRD[cid],
            }
            for cid in sorted(FP_ID_TO_PRD)
        },
    }

    MAPPING_OUT.write_text(
        json.dumps(mapping_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Scanning untouched DeepFashion2 pool...")
    df2 = scan_df2(df2_used_images)

    print("Scanning untouched Fashionpedia pool...")
    fp, fp_categories = scan_fashionpedia(fp_used_images)

    all_candidates = defaultdict(list)

    for cls in ["top", "pants", "skirt", "outerwear", "dress"]:
        all_candidates[cls].extend(df2.get(cls, []))

    for cls in ["shoe", "bag", "accessory"]:
        all_candidates[cls].extend(fp.get(cls, []))

    selected = []
    summary_rows = []

    for cls in CLASSES:
        pool = all_candidates[cls]
        class_rng = random.Random(
            args.seed + CLASSES.index(cls) * 1009
        )

        sample = sample_diverse(
            pool,
            args.per_class,
            class_rng,
        )

        if len(sample) < args.per_class:
            raise RuntimeError(
                f"{cls}: only {len(sample)} leakage-safe unique-image "
                f"candidates available; requested {args.per_class}"
            )

        selected.extend(sample)

        fine_counts = Counter(
            r["fine_or_source_category"]
            for r in sample
        )
        size_counts = Counter(
            r["object_size_bucket"]
            for r in sample
        )

        summary_rows.append({
            "garment_category": cls,
            "eligible_raw_candidates": len(pool),
            "selected_review_candidates": len(sample),
            "unique_selected_images": len(
                {r["source_image"] for r in sample}
            ),
            "fine_or_source_distribution": "; ".join(
                f"{k}={v}"
                for k, v in sorted(fine_counts.items())
            ),
            "size_distribution": "; ".join(
                f"{k}={v}"
                for k, v in sorted(size_counts.items())
            ),
        })

        make_contact_sheet(
            sample,
            CONTACT_DIR / f"{cls}.jpg",
            args.contact_sheet_items,
        )

    # Stable sample IDs.
    selected.sort(
        key=lambda r: (
            CLASSES.index(r["garment_category"]),
            r["source_dataset"],
            r["source_image"],
            r["garment_id"],
        )
    )

    for i, row in enumerate(selected, start=1):
        row["review_pool_sample_id"] = f"SEGREV1_{i:04d}"
        row["random_seed"] = args.seed
        row["leakage_check"] = "PASS_SOURCE_IMAGE_UNSEEN"

    write_csv(
        OUT_DIR / "segmentation_review_pool_v1.csv",
        selected,
    )

    checklist = []
    for r in selected:
        checklist.append({
            "review_pool_sample_id": r["review_pool_sample_id"],
            "source_dataset": r["source_dataset"],
            "source_image": r["source_image"],
            "garment_id": r["garment_id"],
            "garment_category": r["garment_category"],
            "fine_or_source_category": r["fine_or_source_category"],
            "bbox_annotation_valid": "",
            "mask_annotation_valid": "",
            "category_label_valid": "",
            "image_quality_valid": "",
            "ambiguous": "",
            "review_status": "unreviewed",
            "reviewer_note": "",
            "exclude_reason": "",
        })

    write_csv(
        OUT_DIR / "segmentation_review_checklist_v1.csv",
        checklist,
    )

    write_csv(
        OUT_DIR / "candidate_pool_summary.csv",
        summary_rows,
    )

    lines = [
        "PRD 3.1.1 Segmentation Benchmark Review Pool v1",
        "================================================",
        "",
        f"seed={args.seed}",
        f"requested_per_class={args.per_class}",
        f"selected_total={len(selected)}",
        "",
        "Leakage policy",
        "--------------",
        "- Entire source images already seen in project configs are excluded.",
        "- Review-pool rows are NOT benchmark_test yet.",
        "- Do not use this review pool for model/prompt/threshold tuning.",
        "",
        "Planned freeze",
        "--------------",
        "- Human-review bbox/mask/category/image quality.",
        "- Freeze 100 reviewed PASS cases per class where possible.",
        "- Final target = 800 segmentation benchmark instances.",
        "",
        "Class summary",
        "-------------",
    ]

    for r in summary_rows:
        lines.append(
            f"{r['garment_category']}: "
            f"eligible={r['eligible_raw_candidates']} "
            f"review_selected={r['selected_review_candidates']} "
            f"size=({r['size_distribution']})"
        )

    lines += [
        "",
        "Frozen Fashionpedia mapping",
        "---------------------------",
        "shoe <- id 23: shoe",
        "bag <- id 24: bag, wallet",
        "accessory <- ids 13,14,15,16,17,18,19,20,21,22,25",
        "",
        "Next",
        "----",
        "- Review contact sheets first.",
        "- Then complete segmentation_review_checklist_v1.csv.",
        "- Only after review will a separate script freeze segmentation_test_v1.csv.",
    ]

    (OUT_DIR / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("FINISHED")
    for p in [
        MAPPING_OUT,
        OUT_DIR / "summary.txt",
        OUT_DIR / "candidate_pool_summary.csv",
        OUT_DIR / "segmentation_review_pool_v1.csv",
        OUT_DIR / "segmentation_review_checklist_v1.csv",
    ]:
        print(p.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
