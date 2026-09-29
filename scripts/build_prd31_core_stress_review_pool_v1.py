"""
Build PRD 3.1.1 difficulty-controlled Core + Stress review pools.

Design
------
CORE (used for PRD acceptance after human review)
- planned freeze: 50 cases / class = 400 total
- review candidates: target 55 / class
- visibility floor:
    bbox area ratio >= 0.005 (>= 0.5% of image)
    bbox minimum side >= 50 px
- source image must be unseen in all previous project configs
- source image is globally unique across selected Core/Stress candidates
- valid bbox + non-empty GT segmentation required

STRESS (diagnostic only; never mixed into PRD PASS/FAIL)
- target: 10 cases / class = 80 total
- selected from the lowest-area valid tail remaining after Core
- min bbox side >= 10 px so cases remain inspectable
- records whether it satisfies absolute small-object criteria:
    area ratio < 0.02 OR min side < 50 px
- source image is globally unique and disjoint from Core

Important
---------
The script uses ONLY source annotations/geometry to build the review sets.
It does not run or inspect model predictions.

It also materializes GT masks and GT-mask overlay crops for human annotation QC.

Inputs
------
benchmark/prd_3_1_v1/audit/used_sample_registry.csv
../fashion_data/raw/train/train/annos/*.json
../fashion_data/raw/train/train/image/*
../fashion_data/raw/fashionpedia/annotations/instances_attributes_val2020.json
../fashion_data/raw/fashionpedia/images/...

Outputs
-------
benchmark/prd_3_1_v1/candidates/segmentation_core_stress_v1/
├── core_review_pool_v1.csv
├── stress_review_pool_v1.csv
├── review_checklist_v1.csv
├── pool_summary.csv
├── summary.txt
├── masks/
├── overlays/
├── contact_sheets_core/
└── contact_sheets_stress/

Run
---
cd /workspace/fashion_multimodal_analysis

python scripts/build_prd31_core_stress_review_pool_v1.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "fashion_data"

BENCH_ROOT = PROJECT_ROOT / "benchmark" / "prd_3_1_v1"

USED_REGISTRY = (
    BENCH_ROOT
    / "audit"
    / "used_sample_registry.csv"
)

OUT_DIR = (
    BENCH_ROOT
    / "candidates"
    / "segmentation_core_stress_v1"
)

MASK_DIR = OUT_DIR / "masks"
OVERLAY_DIR = OUT_DIR / "overlays"
CORE_SHEET_DIR = OUT_DIR / "contact_sheets_core"
STRESS_SHEET_DIR = OUT_DIR / "contact_sheets_stress"

DF2_ANNO_DIR = DATA_ROOT / "raw" / "train" / "train" / "annos"
DF2_IMAGE_DIR = DATA_ROOT / "raw" / "train" / "train" / "image"

FP_ROOT = DATA_ROOT / "raw" / "fashionpedia"
FP_ANN = (
    FP_ROOT
    / "annotations"
    / "instances_attributes_val2020.json"
)

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

# Scarce classes first prevents another class from consuming a shared
# Fashionpedia source image before bag gets a chance to use it.
SELECTION_ORDER = [
    "bag",
    "accessory",
    "shoe",
    "outerwear",
    "skirt",
    "dress",
    "pants",
    "top",
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

# Exact mapping frozen from the project's existing training/validation code.
FP_ID_TO_PRD = {
    13: "accessory",
    14: "accessory",
    15: "accessory",
    16: "accessory",
    17: "accessory",
    18: "accessory",
    19: "accessory",
    20: "accessory",
    21: "accessory",
    22: "accessory",
    23: "shoe",
    24: "bag",
    25: "accessory",
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

    p.add_argument(
        "--core-review-per-class",
        type=int,
        default=55,
    )
    p.add_argument(
        "--core-freeze-per-class",
        type=int,
        default=50,
    )
    p.add_argument(
        "--stress-per-class",
        type=int,
        default=10,
    )
    p.add_argument(
        "--core-min-area-ratio",
        type=float,
        default=0.005,
    )
    p.add_argument(
        "--core-min-side-px",
        type=float,
        default=50.0,
    )
    p.add_argument(
        "--stress-min-side-px",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260922,
    )
    p.add_argument(
        "--core-sheet-items",
        type=int,
        default=55,
    )
    p.add_argument(
        "--stress-sheet-items",
        type=int,
        default=10,
    )

    return p.parse_args()


def read_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def norm(v: Any) -> str:
    return (
        str(v if v is not None else "")
        .replace("\\", "/")
        .strip()
    )


def rel_data(path: Path) -> str:
    rel = path.resolve().relative_to(
        DATA_ROOT.resolve()
    )
    return (
        "../fashion_data/"
        + str(rel).replace("\\", "/")
    )


def rel_project(path: Path) -> str:
    return str(
        path.resolve().relative_to(
            PROJECT_ROOT.resolve()
        )
    ).replace("\\", "/")


def used_image_names():
    rows = read_csv(USED_REGISTRY)

    df2 = set()
    fp = set()

    for r in rows:
        p = norm(r.get("source_image", ""))

        if not p:
            continue

        name = Path(p).name

        if "fashionpedia" in p.lower():
            fp.add(name)
        elif "/raw/train/train/" in p.lower():
            df2.add(name)

    return df2, fp


def area_bin(ratio: float) -> str:
    if ratio < 0.005:
        return "tiny_<0.5%"
    if ratio < 0.02:
        return "small_0.5-2%"
    if ratio < 0.10:
        return "medium_2-10%"
    return "large_>=10%"


def minside_bin(v: float) -> str:
    if v < 20:
        return "<20"
    if v < 50:
        return "20-49"
    if v < 100:
        return "50-99"
    return ">=100"


def valid_bbox_xyxy(box):
    if (
        not isinstance(box, list)
        or len(box) != 4
    ):
        return False

    try:
        x1, y1, x2, y2 = map(
            float,
            box,
        )
    except Exception:
        return False

    return x2 > x1 and y2 > y1


def segmentation_nonempty(seg) -> bool:
    if isinstance(seg, dict):
        return bool(seg)

    if isinstance(seg, list):
        if not seg:
            return False

        if all(
            isinstance(x, (int, float))
            for x in seg
        ):
            return len(seg) >= 6

        return any(
            isinstance(poly, list)
            and len(poly) >= 6
            for poly in seg
        )

    return False


def decode_segmentation(
    segmentation,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Decode polygons or COCO RLE to uint8 {0,1}.
    """
    if isinstance(segmentation, list):
        polygons = segmentation

        if polygons and all(
            isinstance(v, (int, float))
            for v in polygons
        ):
            polygons = [polygons]

        canvas = Image.new(
            "L",
            (width, height),
            0,
        )
        draw = ImageDraw.Draw(canvas)

        for poly in polygons:
            if (
                isinstance(poly, list)
                and len(poly) >= 6
            ):
                coords = [
                    (
                        float(poly[i]),
                        float(poly[i + 1]),
                    )
                    for i in range(
                        0,
                        len(poly) - 1,
                        2,
                    )
                ]

                draw.polygon(
                    coords,
                    fill=1,
                )

        return np.asarray(
            canvas,
            dtype=np.uint8,
        )

    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
        except Exception as exc:
            raise RuntimeError(
                "Encountered COCO RLE segmentation but "
                "pycocotools is unavailable. Install pycocotools "
                "or keep polygon-only cases."
            ) from exc

        rle = segmentation

        if isinstance(
            segmentation.get("counts"),
            list,
        ):
            rle = mask_utils.frPyObjects(
                segmentation,
                height,
                width,
            )

        decoded = mask_utils.decode(rle)

        if decoded.ndim == 3:
            decoded = np.any(
                decoded > 0,
                axis=2,
            )

        return (
            decoded > 0
        ).astype(np.uint8)

    raise TypeError(
        f"Unsupported segmentation type: "
        f"{type(segmentation).__name__}"
    )


def scan_df2(
    excluded_names: set[str],
):
    by_class = defaultdict(list)

    files = sorted(
        DF2_ANNO_DIR.glob("*.json")
    )

    for i, ann_path in enumerate(
        files,
        start=1,
    ):
        image_path = (
            DF2_IMAGE_DIR
            / f"{ann_path.stem}.jpg"
        )

        if not image_path.is_file():
            hits = list(
                DF2_IMAGE_DIR.glob(
                    f"{ann_path.stem}.*"
                )
            )

            if not hits:
                continue

            image_path = hits[0]

        if image_path.name in excluded_names:
            continue

        try:
            data = json.loads(
                ann_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        try:
            with Image.open(image_path) as im:
                width, height = im.size
        except Exception:
            continue

        for item_key, item in data.items():
            if (
                not str(item_key).startswith(
                    "item"
                )
                or not isinstance(
                    item,
                    dict,
                )
            ):
                continue

            try:
                cid = int(
                    item.get(
                        "category_id"
                    )
                )
            except Exception:
                continue

            cls = DF2_TO_PRD.get(cid)
            fine = DF2_FINE.get(cid)

            if not cls or not fine:
                continue

            bbox = item.get(
                "bounding_box"
            )
            seg = item.get(
                "segmentation"
            )

            if (
                not valid_bbox_xyxy(bbox)
                or not segmentation_nonempty(seg)
            ):
                continue

            x1, y1, x2, y2 = map(
                float,
                bbox,
            )

            bw = x2 - x1
            bh = y2 - y1
            area_ratio = (
                bw
                * bh
                / max(
                    1.0,
                    width * height,
                )
            )
            min_side = min(bw, bh)

            by_class[cls].append(
                {
                    "source_dataset": "DeepFashion2",
                    "source_image": rel_data(
                        image_path
                    ),
                    "source_image_name": image_path.name,
                    "annotation_path": rel_data(
                        ann_path
                    ),
                    "annotation_id": "",
                    "garment_id": (
                        f"{item_key}_{fine}"
                    ),
                    "garment_category": cls,
                    "fine_or_source_category": fine,
                    "source_category_id": cid,
                    "image_width": width,
                    "image_height": height,
                    "gt_bbox_x1": x1,
                    "gt_bbox_y1": y1,
                    "gt_bbox_x2": x2,
                    "gt_bbox_y2": y2,
                    "bbox_area_ratio": area_ratio,
                    "bbox_min_side_px": min_side,
                    "segmentation": seg,
                }
            )

        if i % 20000 == 0:
            print(
                f"DeepFashion2 scanned "
                f"{i}/{len(files)}..."
            )

    return by_class


def find_fp_image(
    file_name: str,
    cache: dict[str, Path | None],
):
    if file_name in cache:
        return cache[file_name]

    direct = [
        FP_ROOT
        / "images"
        / "test"
        / file_name,
        FP_ROOT
        / "images"
        / "val"
        / file_name,
    ]

    for p in direct:
        if p.is_file():
            cache[file_name] = p
            return p

    hits = list(
        FP_ROOT.rglob(file_name)
    )

    out = hits[0] if hits else None
    cache[file_name] = out

    return out


def scan_fp(
    excluded_names: set[str],
):
    payload = json.loads(
        FP_ANN.read_text(
            encoding="utf-8"
        )
    )

    images = {
        int(x["id"]): x
        for x in payload["images"]
    }

    categories = {
        int(x["id"]): str(
            x.get("name", "")
        ).strip()
        for x in payload["categories"]
    }

    for cid, expected in (
        EXPECTED_FP_NAMES.items()
    ):
        actual = categories.get(cid)

        if actual != expected:
            raise RuntimeError(
                "Frozen Fashionpedia mapping "
                f"mismatch: id={cid} "
                f"expected={expected!r} "
                f"actual={actual!r}"
            )

    by_class = defaultdict(list)
    image_cache = {}

    for ann in payload["annotations"]:
        try:
            cid = int(
                ann["category_id"]
            )
        except Exception:
            continue

        cls = FP_ID_TO_PRD.get(cid)

        if cls is None:
            continue

        info = images.get(
            int(ann["image_id"])
        )

        if not info:
            continue

        file_name = str(
            info.get(
                "file_name",
                "",
            )
        ).strip()

        if (
            not file_name
            or file_name
            in excluded_names
        ):
            continue

        image_path = find_fp_image(
            file_name,
            image_cache,
        )

        if image_path is None:
            continue

        bbox = ann.get("bbox")
        seg = ann.get(
            "segmentation"
        )

        if (
            not isinstance(
                bbox,
                list,
            )
            or len(bbox) != 4
            or not segmentation_nonempty(
                seg
            )
        ):
            continue

        try:
            x, y, bw, bh = map(
                float,
                bbox,
            )
        except Exception:
            continue

        if bw <= 0 or bh <= 0:
            continue

        width = int(
            info.get("width", 0)
        )
        height = int(
            info.get("height", 0)
        )

        if width <= 0 or height <= 0:
            try:
                with Image.open(
                    image_path
                ) as im:
                    width, height = im.size
            except Exception:
                continue

        x1 = x
        y1 = y
        x2 = x + bw
        y2 = y + bh

        area_ratio = (
            bw
            * bh
            / max(
                1.0,
                width * height,
            )
        )
        min_side = min(
            bw,
            bh,
        )
        ann_id = int(
            ann["id"]
        )

        by_class[cls].append(
            {
                "source_dataset": "Fashionpedia",
                "source_image": rel_data(
                    image_path
                ),
                "source_image_name": image_path.name,
                "annotation_path": rel_data(
                    FP_ANN
                ),
                "annotation_id": ann_id,
                "garment_id": (
                    "fashionpedia_ann_"
                    f"{ann_id}"
                ),
                "garment_category": cls,
                "fine_or_source_category": (
                    categories[cid]
                ),
                "source_category_id": cid,
                "image_width": width,
                "image_height": height,
                "gt_bbox_x1": x1,
                "gt_bbox_y1": y1,
                "gt_bbox_x2": x2,
                "gt_bbox_y2": y2,
                "bbox_area_ratio": area_ratio,
                "bbox_min_side_px": min_side,
                "segmentation": seg,
            }
        )

    return by_class


def stratum_key(row):
    return (
        row["fine_or_source_category"],
        area_bin(
            float(
                row["bbox_area_ratio"]
            )
        ),
    )


def select_core(
    pool,
    n,
    used_images,
    min_area,
    min_side,
    rng,
):
    eligible = [
        x
        for x in pool
        if float(
            x["bbox_area_ratio"]
        )
        >= min_area
        and float(
            x["bbox_min_side_px"]
        )
        >= min_side
        and x[
            "source_image"
        ]
        not in used_images
    ]

    groups = defaultdict(list)

    for x in eligible:
        groups[
            stratum_key(x)
        ].append(x)

    for vals in groups.values():
        rng.shuffle(vals)

    group_keys = list(
        groups.keys()
    )
    rng.shuffle(group_keys)

    selected = []
    pointers = {
        k: 0
        for k in group_keys
    }

    while len(selected) < n:
        progressed = False

        for key in group_keys:
            vals = groups[key]
            idx = pointers[key]

            while idx < len(vals):
                x = vals[idx]
                idx += 1
                pointers[key] = idx

                if (
                    x["source_image"]
                    in used_images
                ):
                    continue

                selected.append(x)
                used_images.add(
                    x["source_image"]
                )
                progressed = True
                break

            if len(selected) >= n:
                break

        if not progressed:
            break

    return selected, len(eligible)


def select_stress(
    pool,
    n,
    used_images,
    min_side,
):
    eligible = [
        x
        for x in pool
        if float(
            x["bbox_min_side_px"]
        )
        >= min_side
        and x[
            "source_image"
        ]
        not in used_images
    ]

    # Absolute small cases first, then relative low-area tail.
    eligible.sort(
        key=lambda x: (
            0
            if (
                float(
                    x[
                        "bbox_area_ratio"
                    ]
                )
                < 0.02
                or float(
                    x[
                        "bbox_min_side_px"
                    ]
                )
                < 50
            )
            else 1,
            float(
                x[
                    "bbox_area_ratio"
                ]
            ),
            float(
                x[
                    "bbox_min_side_px"
                ]
            ),
        )
    )

    selected = []

    for x in eligible:
        if (
            x["source_image"]
            in used_images
        ):
            continue

        selected.append(x)
        used_images.add(
            x["source_image"]
        )

        if len(selected) >= n:
            break

    return selected, len(eligible)


def materialize_case(
    row,
    subset,
    sample_id,
):
    image_path = (
        PROJECT_ROOT
        / row["source_image"]
    ).resolve()

    image = Image.open(
        image_path
    ).convert("RGB")

    mask = decode_segmentation(
        row["segmentation"],
        image.width,
        image.height,
    )

    if not np.any(mask):
        raise RuntimeError(
            f"Decoded empty mask: "
            f"{row['source_image']} "
            f"{row['garment_id']}"
        )

    class_name = row[
        "garment_category"
    ]

    mask_path = (
        MASK_DIR
        / subset
        / class_name
        / f"{sample_id}.png"
    )

    overlay_path = (
        OVERLAY_DIR
        / subset
        / class_name
        / f"{sample_id}.jpg"
    )

    mask_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    overlay_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        mask.astype(
            np.uint8
        )
        * 255
    ).save(mask_path)

    x1 = max(
        0,
        int(
            math.floor(
                float(
                    row[
                        "gt_bbox_x1"
                    ]
                )
            )
        ),
    )
    y1 = max(
        0,
        int(
            math.floor(
                float(
                    row[
                        "gt_bbox_y1"
                    ]
                )
            )
        ),
    )
    x2 = min(
        image.width,
        int(
            math.ceil(
                float(
                    row[
                        "gt_bbox_x2"
                    ]
                )
            )
        ),
    )
    y2 = min(
        image.height,
        int(
            math.ceil(
                float(
                    row[
                        "gt_bbox_y2"
                    ]
                )
            )
        ),
    )

    # Small padding helps reviewers see boundaries.
    pad_x = max(
        6,
        int(
            0.08
            * max(
                1,
                x2 - x1,
            )
        ),
    )
    pad_y = max(
        6,
        int(
            0.08
            * max(
                1,
                y2 - y1,
            )
        ),
    )

    px1 = max(
        0,
        x1 - pad_x,
    )
    py1 = max(
        0,
        y1 - pad_y,
    )
    px2 = min(
        image.width,
        x2 + pad_x,
    )
    py2 = min(
        image.height,
        y2 + pad_y,
    )

    arr = np.asarray(
        image,
        dtype=np.uint8,
    ).copy()

    # Blend green onto GT mask; no model output is shown.
    tint = np.zeros_like(arr)
    tint[..., 1] = 255

    alpha = 0.38
    inside = mask.astype(bool)

    arr[inside] = (
        (
            1.0 - alpha
        )
        * arr[inside]
        + alpha
        * tint[inside]
    ).astype(np.uint8)

    overlay = Image.fromarray(
        arr
    ).crop(
        (
            px1,
            py1,
            px2,
            py2,
        )
    )

    d = ImageDraw.Draw(overlay)

    d.rectangle(
        (
            x1 - px1,
            y1 - py1,
            x2 - px1,
            y2 - py1,
        ),
        outline="white",
        width=2,
    )

    overlay.save(
        overlay_path,
        quality=92,
    )

    out = dict(row)
    out.pop(
        "segmentation",
        None,
    )

    out.update(
        {
            "benchmark_version": (
                "prd_3_1_v1"
            ),
            "review_subset": subset,
            "review_sample_id": sample_id,
            "gt_mask_path": rel_project(
                mask_path
            ),
            "gt_overlay_path": rel_project(
                overlay_path
            ),
            "area_bin": area_bin(
                float(
                    row[
                        "bbox_area_ratio"
                    ]
                )
            ),
            "minside_bin": minside_bin(
                float(
                    row[
                        "bbox_min_side_px"
                    ]
                )
            ),
            "absolute_small_object": int(
                float(
                    row[
                        "bbox_area_ratio"
                    ]
                )
                < 0.02
                or float(
                    row[
                        "bbox_min_side_px"
                    ]
                )
                < 50
            ),
            "human_review_status": (
                "unreviewed"
            ),
        }
    )

    return out


def make_contact_sheet(
    rows,
    out_path,
    max_items,
):
    items = rows[:max_items]

    if not items:
        return

    cols = 5
    tile_w = 250
    tile_h = 290
    nrows = (
        len(items)
        + cols
        - 1
    ) // cols

    sheet = Image.new(
        "RGB",
        (
            cols * tile_w,
            nrows * tile_h,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        sheet
    )
    font = ImageFont.load_default()

    for i, row in enumerate(
        items
    ):
        rr = i // cols
        cc = i % cols
        x0 = cc * tile_w
        y0 = rr * tile_h

        overlay_path = (
            PROJECT_ROOT
            / row[
                "gt_overlay_path"
            ]
        ).resolve()

        try:
            img = Image.open(
                overlay_path
            ).convert("RGB")
        except Exception:
            continue

        img.thumbnail(
            (230, 210)
        )

        px = (
            x0
            + (
                tile_w
                - img.width
            )
            // 2
        )

        sheet.paste(
            img,
            (
                px,
                y0 + 5,
            ),
        )

        label = (
            f"{row['review_sample_id']} "
            f"{row['garment_category']}\n"
            f"{row['fine_or_source_category'][:28]}\n"
            f"area={100*float(row['bbox_area_ratio']):.2f}% "
            f"min={float(row['bbox_min_side_px']):.0f}px\n"
            f"{row['source_dataset']}"
        )

        draw.multiline_text(
            (
                x0 + 5,
                y0 + 220,
            ),
            label,
            fill="black",
            font=font,
            spacing=2,
        )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    sheet.save(
        out_path,
        quality=92,
    )


def main():
    args = parse_args()

    if (
        args.core_review_per_class
        < args.core_freeze_per_class
    ):
        raise ValueError(
            "--core-review-per-class "
            "must be >= "
            "--core-freeze-per-class"
        )

    for d in [
        OUT_DIR,
        MASK_DIR,
        OVERLAY_DIR,
        CORE_SHEET_DIR,
        STRESS_SHEET_DIR,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    df2_used, fp_used = (
        used_image_names()
    )

    print(
        "Previously used source images:"
    )
    print(
        "  DeepFashion2:",
        len(df2_used),
    )
    print(
        "  Fashionpedia:",
        len(fp_used),
    )

    print(
        "Scanning DeepFashion2..."
    )
    df2 = scan_df2(
        df2_used
    )

    print(
        "Scanning Fashionpedia..."
    )
    fp = scan_fp(
        fp_used
    )

    pools = defaultdict(list)

    for cls in [
        "top",
        "pants",
        "skirt",
        "outerwear",
        "dress",
    ]:
        pools[cls].extend(
            df2.get(
                cls,
                [],
            )
        )

    for cls in [
        "shoe",
        "bag",
        "accessory",
    ]:
        pools[cls].extend(
            fp.get(
                cls,
                [],
            )
        )

    globally_selected_images = set()

    core_raw = {}
    core_eligible_counts = {}

    print(
        "Selecting Core review candidates..."
    )

    for cls in SELECTION_ORDER:
        rng = random.Random(
            args.seed
            + CLASSES.index(
                cls
            )
            * 1009
        )

        selected, eligible_n = (
            select_core(
                pools[cls],
                args.core_review_per_class,
                globally_selected_images,
                args.core_min_area_ratio,
                args.core_min_side_px,
                rng,
            )
        )

        core_raw[cls] = selected
        core_eligible_counts[
            cls
        ] = eligible_n

        if len(
            selected
        ) < args.core_freeze_per_class:
            raise RuntimeError(
                f"{cls}: only "
                f"{len(selected)} "
                "globally unique Core review "
                "candidates available; "
                f"need at least "
                f"{args.core_freeze_per_class}"
            )

        print(
            f"  {cls}: "
            f"eligible={eligible_n}, "
            f"selected={len(selected)}"
        )

    stress_raw = {}
    stress_eligible_counts = {}

    print(
        "Selecting Stress diagnostic candidates..."
    )

    for cls in SELECTION_ORDER:
        selected, eligible_n = (
            select_stress(
                pools[cls],
                args.stress_per_class,
                globally_selected_images,
                args.stress_min_side_px,
            )
        )

        stress_raw[
            cls
        ] = selected
        stress_eligible_counts[
            cls
        ] = eligible_n

        print(
            f"  {cls}: "
            f"eligible_remaining="
            f"{eligible_n}, "
            f"selected={len(selected)}"
        )

    # Materialize masks + overlays.
    core_rows = []
    stress_rows = []

    core_counter = 0
    stress_counter = 0

    print(
        "Materializing Core GT masks/overlays..."
    )

    for cls in CLASSES:
        for row in core_raw[
            cls
        ]:
            core_counter += 1
            sid = (
                f"SEGCORE1_"
                f"{core_counter:04d}"
            )

            core_rows.append(
                materialize_case(
                    row,
                    "core",
                    sid,
                )
            )

    print(
        "Materializing Stress GT masks/overlays..."
    )

    for cls in CLASSES:
        for row in stress_raw[
            cls
        ]:
            stress_counter += 1
            sid = (
                f"SEGSTR1_"
                f"{stress_counter:04d}"
            )

            stress_rows.append(
                materialize_case(
                    row,
                    "stress",
                    sid,
                )
            )

    write_csv(
        OUT_DIR
        / "core_review_pool_v1.csv",
        core_rows,
    )

    write_csv(
        OUT_DIR
        / "stress_review_pool_v1.csv",
        stress_rows,
    )

    checklist = []

    for row in (
        core_rows
        + stress_rows
    ):
        checklist.append(
            {
                "review_sample_id": row[
                    "review_sample_id"
                ],
                "review_subset": row[
                    "review_subset"
                ],
                "garment_category": row[
                    "garment_category"
                ],
                "fine_or_source_category": (
                    row[
                        "fine_or_source_category"
                    ]
                ),
                "source_dataset": row[
                    "source_dataset"
                ],
                "source_image": row[
                    "source_image"
                ],
                "garment_id": row[
                    "garment_id"
                ],
                "gt_overlay_path": row[
                    "gt_overlay_path"
                ],
                "category_valid": "",
                "bbox_valid": "",
                "mask_valid": "",
                "image_quality_valid": "",
                "ambiguous": "",
                "review_status": (
                    "unreviewed"
                ),
                "reviewer_note": "",
                "exclude_reason": "",
            }
        )

    write_csv(
        OUT_DIR
        / "review_checklist_v1.csv",
        checklist,
    )

    summary_rows = []

    for cls in CLASSES:
        crows = [
            x
            for x in core_rows
            if x[
                "garment_category"
            ]
            == cls
        ]
        srows = [
            x
            for x in stress_rows
            if x[
                "garment_category"
            ]
            == cls
        ]

        core_bins = Counter(
            x["area_bin"]
            for x in crows
        )
        stress_bins = Counter(
            x["area_bin"]
            for x in srows
        )

        summary_rows.append(
            {
                "garment_category": cls,
                "core_raw_pool": len(
                    pools[cls]
                ),
                "core_eligible_visibility_floor": (
                    core_eligible_counts[
                        cls
                    ]
                ),
                "core_review_selected": len(
                    crows
                ),
                "core_planned_freeze": (
                    args.core_freeze_per_class
                ),
                "core_area_bins": "; ".join(
                    f"{k}={v}"
                    for k, v in sorted(
                        core_bins.items()
                    )
                ),
                "stress_eligible_remaining": (
                    stress_eligible_counts[
                        cls
                    ]
                ),
                "stress_selected": len(
                    srows
                ),
                "stress_area_bins": "; ".join(
                    f"{k}={v}"
                    for k, v in sorted(
                        stress_bins.items()
                    )
                ),
            }
        )

        make_contact_sheet(
            crows,
            CORE_SHEET_DIR
            / f"{cls}.jpg",
            args.core_sheet_items,
        )

        make_contact_sheet(
            srows,
            STRESS_SHEET_DIR
            / f"{cls}.jpg",
            args.stress_sheet_items,
        )

    write_csv(
        OUT_DIR
        / "pool_summary.csv",
        summary_rows,
    )

    all_selected = (
        core_rows
        + stress_rows
    )

    unique_images = {
        x["source_image"]
        for x in all_selected
    }

    if len(
        unique_images
    ) != len(
        all_selected
    ):
        raise RuntimeError(
            "Global source-image uniqueness "
            "check failed."
        )

    lines = [
        (
            "PRD 3.1.1 Core + Stress "
            "Segmentation Review Pool v1"
        ),
        (
            "================================"
            "===================="
        ),
        "",
        f"seed={args.seed}",
        (
            "core_visibility_floor="
            f"area_ratio>="
            f"{args.core_min_area_ratio}, "
            f"min_side_px>="
            f"{args.core_min_side_px}"
        ),
        (
            "core_review_target_per_class="
            f"{args.core_review_per_class}"
        ),
        (
            "core_freeze_target_per_class="
            f"{args.core_freeze_per_class}"
        ),
        (
            "stress_target_per_class="
            f"{args.stress_per_class}"
        ),
        (
            "core_review_selected_total="
            f"{len(core_rows)}"
        ),
        (
            "stress_selected_total="
            f"{len(stress_rows)}"
        ),
        (
            "selected_source_images_unique="
            f"{len(unique_images)}"
        ),
        "",
        "Protocol",
        "--------",
        (
            "- Core is the only subset intended "
            "for PRD acceptance PASS/FAIL."
        ),
        (
            "- Stress is diagnostic and must be "
            "reported separately."
        ),
        (
            "- Previously used source images are "
            "excluded."
        ),
        (
            "- Source images are globally unique "
            "across Core and Stress."
        ),
        (
            "- Selection uses source annotation "
            "geometry only; no model predictions "
            "are inspected."
        ),
        (
            "- GT masks and overlays are for "
            "human annotation QC."
        ),
        "",
        "Human review requirement",
        "------------------------",
        (
            "- Every Core case must pass "
            "category, bbox, mask and image-quality "
            "review before final freeze."
        ),
        (
            "- Freeze exactly "
            f"{args.core_freeze_per_class} "
            "reviewed PASS cases/class if coverage "
            "permits."
        ),
        (
            "- Do not replace failed cases using "
            "model performance."
        ),
        (
            "- Stress cases should also be reviewed "
            "before reporting diagnostic metrics."
        ),
        "",
        "Class summary",
        "-------------",
    ]

    for r in summary_rows:
        lines.append(
            f"{r['garment_category']}: "
            f"core_eligible="
            f"{r['core_eligible_visibility_floor']} "
            f"core_review="
            f"{r['core_review_selected']} "
            f"stress="
            f"{r['stress_selected']}"
        )

    lines += [
        "",
        "Next",
        "----",
        (
            "- Review Core contact sheets with "
            "GT mask overlays."
        ),
        (
            "- Complete review_checklist_v1.csv."
        ),
        (
            "- Then freeze final Core "
            "segmentation_test_v1.csv."
        ),
        (
            "- Preserve Stress as "
            "segmentation_stress_test_v1.csv."
        ),
    ]

    (
        OUT_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=== CORE + STRESS REVIEW POOL READY ==="
    )

    for p in [
        OUT_DIR / "summary.txt",
        OUT_DIR / "pool_summary.csv",
        OUT_DIR / "core_review_pool_v1.csv",
        OUT_DIR / "stress_review_pool_v1.csv",
        OUT_DIR / "review_checklist_v1.csv",
    ]:
        print(
            p.relative_to(
                PROJECT_ROOT
            )
        )


if __name__ == "__main__":
    main()
