"""
Build a PROVISIONAL 400-case Core selection for PRD 3.1.1.

This is not the final benchmark freeze.

Selection constraints
---------------------
- exactly 50 cases per PRD class;
- exact size-bucket quotas from core_size_quota_v1.csv;
- preserve fine-category proportions as closely as possible;
- preserve proposed scene-type proportions as closely as possible;
- do not use candidate-model performance;
- known human-excluded cases have already been removed upstream.

Human-review reduction
----------------------
CLIP produced 191 total low-confidence metadata cases, but only scene_type is
used here for Core scene-coverage selection. Therefore this script creates a
focused packet containing:
1) selected Core cases with LOW scene confidence/margin;
2) a deterministic high-confidence spot-check sample (2 per class).

Occlusion / visibility suggestions are carried as auxiliary metadata only and
are NOT used to decide final Core membership in this provisional selector.

Inputs
------
benchmark/prd_3_1_v1/freeze_v1/coverage_review_v1/
    core_size_quota_v1.csv
    scene_suggestions_v1/core_coverage_review_with_suggestions_v1.csv

Outputs
-------
benchmark/prd_3_1_v1/freeze_v1/provisional_selection_v1/
├── provisional_core_400_v1.csv
├── provisional_selection_summary_v1.csv
├── selected_scene_review_v1.csv
├── high_confidence_spotcheck_v1.csv
├── PROVISIONAL_SELECTION_NOTES.md
└── review_contact_sheets/
    ├── scene_low_conf_*.jpg
    └── spotcheck_*.jpg

Run
---
cd /workspace/fashion_multimodal_analysis

python scripts/build_prd31_provisional_core_selection_v1.py
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCH_ROOT = PROJECT_ROOT / "benchmark" / "prd_3_1_v1"
COVERAGE_ROOT = BENCH_ROOT / "freeze_v1" / "coverage_review_v1"

INPUT_CSV = (
    COVERAGE_ROOT
    / "scene_suggestions_v1"
    / "core_coverage_review_with_suggestions_v1.csv"
)

SIZE_QUOTA_CSV = COVERAGE_ROOT / "core_size_quota_v1.csv"

OUT_DIR = (
    BENCH_ROOT
    / "freeze_v1"
    / "provisional_selection_v1"
)

SHEET_DIR = OUT_DIR / "review_contact_sheets"

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

SCENE_TYPES = [
    "worn_person",
    "product_display",
    "partial_view",
    "complex_scene",
]


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)

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
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def largest_remainder_quota(
    counts: Counter,
    final_n: int,
):
    total = sum(counts.values())

    if total <= 0:
        return {}

    raw = {
        k: counts[k] * final_n / total
        for k in counts
    }

    quota = {
        k: int(math.floor(v))
        for k, v in raw.items()
    }

    remaining = final_n - sum(quota.values())

    order = sorted(
        counts,
        key=lambda k: (
            raw[k] - quota[k],
            counts[k],
            str(k),
        ),
        reverse=True,
    )

    for key in order[:remaining]:
        quota[key] += 1

    return quota


def load_size_quotas():
    rows = read_csv(SIZE_QUOTA_CSV)

    out = defaultdict(dict)

    for r in rows:
        out[r["garment_category"]][r["size_bucket"]] = int(
            r["recommended_final_count"]
        )

    return out


def scene_low(row):
    return (
        float(row["scene_confidence"]) < 0.40
        or float(row["scene_margin"]) < 0.08
    )


def greedy_select_class(
    rows,
    size_targets,
):
    """
    Exact size quota; approximately proportional fine-category and scene quotas.
    """
    final_n = sum(size_targets.values())

    fine_counts = Counter(
        r["fine_or_source_category"]
        for r in rows
    )

    scene_counts = Counter(
        r["scene_type_suggested"]
        for r in rows
    )

    fine_targets = largest_remainder_quota(
        fine_counts,
        final_n,
    )

    scene_targets = largest_remainder_quota(
        scene_counts,
        final_n,
    )

    sel_size = Counter()
    sel_fine = Counter()
    sel_scene = Counter()

    remaining = list(rows)
    selected = []

    while len(selected) < final_n:
        candidates = [
            r
            for r in remaining
            if sel_size[r["size_bucket"]]
            < size_targets.get(r["size_bucket"], 0)
        ]

        if not candidates:
            raise RuntimeError(
                "Cannot satisfy exact size quota. "
                f"selected={len(selected)}/{final_n} "
                f"size_targets={dict(size_targets)} "
                f"selected_size={dict(sel_size)}"
            )

        def score(r):
            size = r["size_bucket"]
            fine = r["fine_or_source_category"]
            scene = r["scene_type_suggested"]

            size_def = (
                size_targets.get(size, 0)
                - sel_size[size]
            ) / max(1, size_targets.get(size, 0))

            fine_def = (
                fine_targets.get(fine, 0)
                - sel_fine[fine]
            ) / max(1, fine_targets.get(fine, 0))

            scene_def = (
                scene_targets.get(scene, 0)
                - sel_scene[scene]
            ) / max(1, scene_targets.get(scene, 0))

            # Exact size is the hard structure; fine-category and scene
            # proportions are soft diversity targets.
            return (
                5.0 * size_def
                + 2.5 * fine_def
                + 1.5 * scene_def
            )

        # Deterministic: score descending, sample id ascending.
        candidates.sort(
            key=lambda r: (
                -score(r),
                r["review_sample_id"],
            )
        )

        chosen = candidates[0]
        selected.append(chosen)
        remaining.remove(chosen)

        sel_size[chosen["size_bucket"]] += 1
        sel_fine[chosen["fine_or_source_category"]] += 1
        sel_scene[chosen["scene_type_suggested"]] += 1

    return (
        selected,
        remaining,
        fine_targets,
        scene_targets,
    )


def resolve_project(raw: str):
    p = Path(str(raw).strip())
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def make_review_sheets(
    rows,
    prefix,
    out_dir,
):
    if not rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    page_size = 8

    for start in range(0, len(rows), page_size):
        page_rows = rows[start:start + page_size]
        page_no = start // page_size + 1

        cols = 2
        rows_n = 4
        tile_w = 760
        tile_h = 330

        canvas = Image.new(
            "RGB",
            (cols * tile_w, rows_n * tile_h),
            "white",
        )

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        for i, row in enumerate(page_rows):
            rr = i // cols
            cc = i % cols
            x0 = cc * tile_w
            y0 = rr * tile_h

            src = resolve_project(row["source_image"])
            overlay = resolve_project(row["gt_overlay_path"])

            try:
                full = Image.open(src).convert("RGB")
                ov = Image.open(overlay).convert("RGB")
            except Exception:
                continue

            full.thumbnail((340, 220))
            ov.thumbnail((340, 220))

            canvas.paste(
                full,
                (
                    x0 + 10 + (340 - full.width) // 2,
                    y0 + 70 + (220 - full.height) // 2,
                ),
            )
            canvas.paste(
                ov,
                (
                    x0 + 390 + (340 - ov.width) // 2,
                    y0 + 70 + (220 - ov.height) // 2,
                ),
            )

            title = (
                f"{row['review_sample_id']} | "
                f"{row['garment_category']} | "
                f"{row['fine_or_source_category']}"
            )

            meta = (
                f"size={row['size_bucket']} | "
                f"scene?={row['scene_type_suggested']} | "
                f"p={float(row['scene_confidence']):.2f} | "
                f"margin={float(row['scene_margin']):.2f}"
            )

            draw.text(
                (x0 + 8, y0 + 8),
                title,
                fill="black",
                font=font,
            )
            draw.text(
                (x0 + 8, y0 + 28),
                meta,
                fill="black",
                font=font,
            )
            draw.text(
                (x0 + 8, y0 + 48),
                "Human scene label: ____________________",
                fill="black",
                font=font,
            )

            draw.rectangle(
                (
                    x0 + 2,
                    y0 + 2,
                    x0 + tile_w - 3,
                    y0 + tile_h - 3,
                ),
                outline="gray",
                width=1,
            )

        out = out_dir / f"{prefix}_{page_no:02d}.jpg"
        canvas.save(out, quality=92)


def main():
    rows = read_csv(INPUT_CSV)
    size_quotas = load_size_quotas()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHEET_DIR.mkdir(parents=True, exist_ok=True)

    selected_all = []
    excluded_all = []
    summary_rows = []

    for cls in CLASSES:
        rr = [
            r
            for r in rows
            if r["garment_category"] == cls
        ]

        selected, excluded, fine_targets, scene_targets = greedy_select_class(
            rr,
            size_quotas[cls],
        )

        for r in selected:
            out = dict(r)
            out["provisional_selected"] = "1"
            out["scene_human_review_required"] = (
                "1" if scene_low(r) else "0"
            )
            out["scene_type_final"] = ""
            out["scene_review_decision"] = ""
            selected_all.append(out)

        for r in excluded:
            out = dict(r)
            out["provisional_selected"] = "0"
            excluded_all.append(out)

        selected_size = Counter(
            r["size_bucket"]
            for r in selected
        )
        selected_fine = Counter(
            r["fine_or_source_category"]
            for r in selected
        )
        selected_scene = Counter(
            r["scene_type_suggested"]
            for r in selected
        )

        summary_rows.append({
            "garment_category": cls,
            "eligible": len(rr),
            "selected": len(selected),
            "excluded": len(excluded),
            "selected_class_share": "12.5%",
            "size_target": "; ".join(
                f"{k}={v}"
                for k, v in sorted(size_quotas[cls].items())
            ),
            "size_selected": "; ".join(
                f"{k}={v}"
                for k, v in sorted(selected_size.items())
            ),
            "fine_target_approx": "; ".join(
                f"{k}={v}"
                for k, v in sorted(fine_targets.items())
            ),
            "fine_selected": "; ".join(
                f"{k}={v}"
                for k, v in sorted(selected_fine.items())
            ),
            "scene_target_approx": "; ".join(
                f"{k}={v}"
                for k, v in sorted(scene_targets.items())
            ),
            "scene_selected_suggested": "; ".join(
                f"{k}={v}"
                for k, v in sorted(selected_scene.items())
            ),
            "selected_low_scene_confidence": sum(
                scene_low(r)
                for r in selected
            ),
        })

    # Stable class/sample ordering.
    selected_all.sort(
        key=lambda r: (
            CLASSES.index(r["garment_category"]),
            r["review_sample_id"],
        )
    )

    excluded_all.sort(
        key=lambda r: (
            CLASSES.index(r["garment_category"]),
            r["review_sample_id"],
        )
    )

    # Assign provisional Core IDs but do NOT call them final IDs.
    for i, r in enumerate(selected_all, start=1):
        r["provisional_core_id"] = f"COREPROV1_{i:04d}"

    write_csv(
        OUT_DIR / "provisional_core_400_v1.csv",
        selected_all,
    )

    write_csv(
        OUT_DIR / "provisional_not_selected_v1.csv",
        excluded_all,
    )

    write_csv(
        OUT_DIR / "provisional_selection_summary_v1.csv",
        summary_rows,
    )

    scene_review = [
        r for r in selected_all
        if r["scene_human_review_required"] == "1"
    ]

    write_csv(
        OUT_DIR / "selected_scene_review_v1.csv",
        scene_review,
    )

    # Deterministic high-confidence spot check: first two per class by ID.
    spotcheck = []

    for cls in CLASSES:
        rr = [
            r for r in selected_all
            if r["garment_category"] == cls
            and r["scene_human_review_required"] == "0"
        ]

        rr.sort(
            key=lambda r: r["review_sample_id"]
        )
        spotcheck.extend(rr[:2])

    write_csv(
        OUT_DIR / "high_confidence_spotcheck_v1.csv",
        spotcheck,
    )

    make_review_sheets(
        scene_review,
        "scene_low_conf",
        SHEET_DIR,
    )

    make_review_sheets(
        spotcheck,
        "spotcheck",
        SHEET_DIR,
    )

    notes = f"""# PRD 3.1.1 Provisional Core Selection v1

**NOT FINAL / NOT YET FROZEN**

## What this file does

- Selects exactly 50 cases per class (400 total).
- Enforces the frozen proportional size quota exactly.
- Tries to preserve fine-category and proposed scene-type proportions.
- Uses no candidate-model predictions or performance.

## Scene metadata review

Total selected cases requiring human scene review:
{len(scene_review)}

High-confidence spot-check cases:
{len(spotcheck)}

Only `scene_type_suggested` is used for provisional scene-diversity balancing.
Low-confidence scene suggestions must be reviewed before final freeze.

Occlusion / visibility CLIP suggestions are auxiliary metadata only in this
provisional selection and are not used as a selection constraint.

## Human scene labels

Allowed final scene labels:

- worn_person
- product_display
- partial_view
- complex_scene

If the four-way taxonomy is genuinely ambiguous for a case, record
`scene_review_decision=AMBIGUOUS` and do not silently invent a label.

## Final freeze requirements

Do not create `segmentation_test_v1.csv` until:

1. all selected low-confidence scene cases have been human reviewed;
2. high-confidence spot checks are acceptable;
3. selected per-class scene distributions are recomputed using reviewed labels;
4. no class becomes scene-dominated when suitable alternatives exist;
5. Core stays exactly 50/class;
6. the freeze checklist is satisfied.
"""

    (
        OUT_DIR / "PROVISIONAL_SELECTION_NOTES.md"
    ).write_text(
        notes,
        encoding="utf-8",
    )

    print("FINISHED")
    print("selected_total=", len(selected_all))
    print("scene_review_cases=", len(scene_review))
    print("spotcheck_cases=", len(spotcheck))

    for p in [
        OUT_DIR / "provisional_core_400_v1.csv",
        OUT_DIR / "provisional_selection_summary_v1.csv",
        OUT_DIR / "selected_scene_review_v1.csv",
        OUT_DIR / "high_confidence_spotcheck_v1.csv",
        OUT_DIR / "PROVISIONAL_SELECTION_NOTES.md",
    ]:
        print(p.relative_to(PROJECT_ROOT))

    print(
        "contact sheets:",
        SHEET_DIR.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
