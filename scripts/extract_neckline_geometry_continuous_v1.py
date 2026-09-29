"""PRD 3.1.3 continuous neckline geometry pilot v1.

Goal
----
Represent neckline as continuous geometric dimensions instead of forcing
everything into one hard neckline label.

Why 2 dimensions?
-----------------
Neckline is not naturally one scalar. A useful first representation is:

1. neckline_depth_ratio
   How far the center neckline opening descends relative to the garment height.

2. neckline_width_ratio
   How wide the upper-center opening is relative to garment width.

Additional outputs:
- neckline_open_area_ratio
- shoulder_baseline_ratio
- geometry_reliable

This is a geometry pilot using garment-instance masks only. There is no
DeepFashion2 neckline ground truth, so the main validation is qualitative:
inspect contact sheets sorted by depth and width.

Example
-------
python scripts/extract_neckline_geometry_continuous_v1.py \
    --manifest configs/garment_instances_gt_pilot_100.csv

Outputs
-------
reports/prd_attribute_extraction/neckline_continuous_v1/
    neckline_geometry_features.csv
    neckline_summary.txt
    run_info.txt

outputs/prd_attribute_extraction/neckline_continuous_v1/
    per_instance/
    contact_sheet_sorted_by_depth.jpg
    contact_sheet_sorted_by_width.jpg
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "neckline_continuous_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_attribute_extraction"
    / "neckline_continuous_v1"
)

UPPER_BODY_CATEGORIES = {
    "top",
    "outerwear",
    "dress",
}


def resolve_path(raw: str) -> Path:
    p = Path(raw).expanduser()

    if p.is_absolute() and p.exists():
        return p.resolve()

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = (PROJECT_ROOT / p).resolve()
    if project_candidate.exists():
        return project_candidate

    raise FileNotFoundError(
        "Path not found.\n"
        f"typed path       : {raw}\n"
        f"cwd candidate    : {cwd_candidate}\n"
        f"project candidate: {project_candidate}"
    )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {
        "source_image",
        "garment_id",
        "garment_category",
        "crop_path",
        "mask_path",
    }

    if not rows:
        raise ValueError("Manifest has no rows.")

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Manifest missing required columns: {sorted(missing)}"
        )

    return rows


def load_crop_and_mask(
    row: dict[str, str],
) -> tuple[Image.Image, np.ndarray]:
    crop_path = resolve_path(row["crop_path"])
    crop = Image.open(crop_path).convert("RGB")

    mask_raw = row.get("mask_path", "").strip()

    if not mask_raw:
        mask = np.ones(
            (crop.height, crop.width),
            dtype=bool,
        )
        return crop, mask

    mask_path = resolve_path(mask_raw)
    mask_img = Image.open(mask_path).convert("L")

    if mask_img.size != crop.size:
        mask_img = mask_img.resize(
            crop.size,
            Image.Resampling.NEAREST,
        )

    mask = np.asarray(mask_img, dtype=np.uint8) > 127

    if not mask.any():
        raise ValueError(
            f"Empty mask for {row['garment_id']}"
        )

    return crop, mask


def tight_bbox(
    mask: np.ndarray,
) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def tight_crop(
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = tight_bbox(mask)

    return (
        mask[y1:y2, x1:x2],
        (x1, y1, x2, y2),
    )


def top_boundary(
    mask: np.ndarray,
) -> np.ndarray:
    """First foreground y for every x; NaN if column has no foreground."""
    h, w = mask.shape

    boundary = np.full(
        w,
        np.nan,
        dtype=np.float64,
    )

    for x in range(w):
        ys = np.flatnonzero(
            mask[:, x]
        )

        if len(ys):
            boundary[x] = float(
                ys.min()
            )

    return boundary


def safe_nanmedian(
    arr: np.ndarray,
) -> float:
    valid = arr[
        np.isfinite(arr)
    ]

    if len(valid) == 0:
        return float("nan")

    return float(
        np.median(valid)
    )


def band_values(
    boundary: np.ndarray,
    start_frac: float,
    end_frac: float,
) -> np.ndarray:
    w = len(boundary)

    x1 = max(
        0,
        min(
            w - 1,
            int(round(
                w * start_frac
            )),
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            w,
            int(round(
                w * end_frac
            )),
        ),
    )

    return boundary[x1:x2]


def extract_features(
    mask: np.ndarray,
) -> dict[str, float | bool]:
    tight, _ = tight_crop(mask)
    h, w = tight.shape

    boundary = top_boundary(
        tight
    )

    # Shoulder-side bands: close enough to the neck to reflect garment top
    # structure, but not restricted to the center opening itself.
    left_band = band_values(
        boundary,
        0.18,
        0.38,
    )

    right_band = band_values(
        boundary,
        0.62,
        0.82,
    )

    center_band = band_values(
        boundary,
        0.36,
        0.64,
    )

    left_y = safe_nanmedian(
        left_band
    )

    right_y = safe_nanmedian(
        right_band
    )

    center_y = safe_nanmedian(
        center_band
    )

    shoulder_candidates = [
        value
        for value in [
            left_y,
            right_y,
        ]
        if np.isfinite(value)
    ]

    geometry_reliable = (
        len(shoulder_candidates) == 2
        and np.isfinite(center_y)
        and h >= 12
        and w >= 12
    )

    if shoulder_candidates:
        shoulder_y = float(
            np.median(
                np.asarray(
                    shoulder_candidates,
                    dtype=np.float64,
                )
            )
        )
    else:
        shoulder_y = 0.0

    if not np.isfinite(center_y):
        center_y = shoulder_y

    depth_raw = max(
        0.0,
        center_y - shoulder_y,
    )

    neckline_depth_ratio = (
        depth_raw
        / max(
            float(h),
            1.0,
        )
    )

    # Width of columns around the upper center whose garment top starts
    # meaningfully below the shoulder baseline.
    opening_threshold = (
        shoulder_y
        + max(
            2.0,
            0.035 * h,
        )
    )

    x1 = int(
        round(
            w * 0.18
        )
    )

    x2 = int(
        round(
            w * 0.82
        )
    )

    x1 = max(
        0,
        min(
            w - 1,
            x1,
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            w,
            x2,
        ),
    )

    central_boundary = (
        boundary[x1:x2]
    )

    opening_columns = (
        ~np.isfinite(
            central_boundary
        )
        | (
            central_boundary
            > opening_threshold
        )
    )

    neckline_width_ratio = (
        float(
            opening_columns.sum()
        )
        / max(
            float(w),
            1.0,
        )
    )

    # Upper-center open/background area.
    roi_x1 = int(
        round(
            w * 0.25
        )
    )

    roi_x2 = int(
        round(
            w * 0.75
        )
    )

    roi_y1 = max(
        0,
        int(
            round(
                shoulder_y
            )
        ),
    )

    roi_y2 = min(
        h,
        max(
            roi_y1 + 1,
            int(
                round(
                    shoulder_y
                    + 0.32 * h
                )
            ),
        ),
    )

    roi = tight[
        roi_y1:roi_y2,
        roi_x1:roi_x2,
    ]

    if roi.size:
        neckline_open_area_ratio = float(
            (~roi).sum()
            / roi.size
        )
    else:
        neckline_open_area_ratio = 0.0

    return {
        "shoulder_baseline_ratio": (
            shoulder_y
            / max(
                float(h),
                1.0,
            )
        ),
        "center_top_ratio": (
            center_y
            / max(
                float(h),
                1.0,
            )
        ),
        "neckline_depth_ratio": (
            neckline_depth_ratio
        ),
        "neckline_width_ratio": (
            neckline_width_ratio
        ),
        "neckline_open_area_ratio": (
            neckline_open_area_ratio
        ),
        "geometry_reliable": (
            bool(
                geometry_reliable
            )
        ),
    }


def save_visual(
    crop: Image.Image,
    mask: np.ndarray,
    record: dict,
    save_path: Path,
) -> None:
    white = Image.new(
        "RGB",
        crop.size,
        "white",
    )

    mask_img = Image.fromarray(
        (
            mask.astype(
                np.uint8
            )
            * 255
        ),
        mode="L",
    )

    foreground = Image.composite(
        crop,
        white,
        mask_img,
    )

    panel_h = 205

    canvas = Image.new(
        "RGB",
        (
            foreground.width,
            foreground.height
            + panel_h,
        ),
        "white",
    )

    canvas.paste(
        foreground,
        (0, panel_h),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    font = ImageFont.load_default()

    lines = [
        record["garment_id"],
        (
            "depth_ratio: "
            f"{record['neckline_depth_ratio']:.3f}"
        ),
        (
            "width_ratio: "
            f"{record['neckline_width_ratio']:.3f}"
        ),
        (
            "open_area: "
            f"{record['neckline_open_area_ratio']:.3f}"
        ),
        (
            "shoulder_base: "
            f"{record['shoulder_baseline_ratio']:.3f}"
        ),
        (
            "reliable: "
            f"{record['geometry_reliable']}"
        ),
    ]

    y = 10

    for line in lines:
        draw.text(
            (10, y),
            line,
            fill="black",
            font=font,
        )
        y += 30

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        save_path,
        quality=92,
    )


def make_contact_sheet(
    records: list[
        tuple[float, Path]
    ],
    output_path: Path,
) -> None:
    if not records:
        return

    records = sorted(
        records,
        key=lambda x: x[0],
    )

    tiles = []

    for _, path in records:
        with Image.open(
            path
        ) as img:
            img = img.convert(
                "RGB"
            )

            img.thumbnail(
                (390, 520)
            )

            tile = Image.new(
                "RGB",
                (400, 530),
                "white",
            )

            x = (
                400 - img.width
            ) // 2

            y = (
                530 - img.height
            ) // 2

            tile.paste(
                img,
                (x, y),
            )

            tiles.append(
                tile
            )

    cols = 4

    rows = (
        len(tiles)
        + cols - 1
    ) // cols

    sheet = Image.new(
        "RGB",
        (
            cols * 400,
            rows * 530,
        ),
        "white",
    )

    for idx, tile in enumerate(
        tiles
    ):
        x = (
            idx % cols
        ) * 400

        y = (
            idx // cols
        ) * 530

        sheet.paste(
            tile,
            (x, y),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        quality=92,
    )


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    args = parser.parse_args()

    manifest_path = resolve_path(
        args.manifest
    )

    all_rows = read_manifest(
        manifest_path
    )

    rows = [
        row
        for row in all_rows
        if (
            row.get(
                "garment_category",
                "",
            )
            .strip()
            .lower()
            in UPPER_BODY_CATEGORIES
        )
    ]

    if not rows:
        raise ValueError(
            "No upper-body garment instances found."
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_instance_dir = (
        OUTPUT_DIR
        / "per_instance"
    )

    per_instance_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"manifest       : {args.manifest}"
    )

    print(
        f"all instances  : {len(all_rows)}"
    )

    print(
        f"neckline cases : {len(rows)}"
    )

    print(
        "feature        : neckline depth + width continuous geometry"
    )

    output_rows = []

    depth_visuals = []
    width_visuals = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        crop, mask = load_crop_and_mask(
            row
        )

        features = extract_features(
            mask
        )

        record = {
            "source_image": (
                row[
                    "source_image"
                ]
            ),
            "garment_id": (
                row[
                    "garment_id"
                ]
            ),
            "garment_category": (
                row[
                    "garment_category"
                ]
            ),
            **features,
        }

        output_rows.append(
            record
        )

        print(
            f"[{index:03d}/{len(rows):03d}] "
            f"{record['garment_id']:<35} "
            f"depth={record['neckline_depth_ratio']:.3f} "
            f"width={record['neckline_width_ratio']:.3f} "
            f"reliable={record['geometry_reliable']}"
        )

        visual_path = (
            per_instance_dir
            / (
                f"{index:03d}_"
                f"{record['garment_id']}.jpg"
            )
        )

        save_visual(
            crop,
            mask,
            record,
            visual_path,
        )

        if record[
            "geometry_reliable"
        ]:
            depth_visuals.append(
                (
                    record[
                        "neckline_depth_ratio"
                    ],
                    visual_path,
                )
            )

            width_visuals.append(
                (
                    record[
                        "neckline_width_ratio"
                    ],
                    visual_path,
                )
            )

    write_csv(
        REPORT_DIR
        / "neckline_geometry_features.csv",
        output_rows,
        [
            "source_image",
            "garment_id",
            "garment_category",
            "shoulder_baseline_ratio",
            "center_top_ratio",
            "neckline_depth_ratio",
            "neckline_width_ratio",
            "neckline_open_area_ratio",
            "geometry_reliable",
        ],
    )

    reliable = [
        row
        for row in output_rows
        if row[
            "geometry_reliable"
        ]
    ]

    depths = [
        row[
            "neckline_depth_ratio"
        ]
        for row in reliable
    ]

    widths = [
        row[
            "neckline_width_ratio"
        ]
        for row in reliable
    ]

    areas = [
        row[
            "neckline_open_area_ratio"
        ]
        for row in reliable
    ]

    summary = (
        "PRD 3.1.3 neckline continuous geometry pilot v1\n"
        "================================================\n"
        f"upper_body_instances={len(output_rows)}\n"
        f"geometry_reliable_instances={len(reliable)}\n"
        "gt_available=false\n"
        "hard_classification=false\n"
        "\n"
        "neckline_depth_ratio\n"
        "--------------------\n"
        f"mean={float(np.mean(depths)):.4f}\n"
        f"median={float(np.median(depths)):.4f}\n"
        f"min={float(np.min(depths)):.4f}\n"
        f"max={float(np.max(depths)):.4f}\n"
        "\n"
        "neckline_width_ratio\n"
        "--------------------\n"
        f"mean={float(np.mean(widths)):.4f}\n"
        f"median={float(np.median(widths)):.4f}\n"
        f"min={float(np.min(widths)):.4f}\n"
        f"max={float(np.max(widths)):.4f}\n"
        "\n"
        "neckline_open_area_ratio\n"
        "------------------------\n"
        f"mean={float(np.mean(areas)):.4f}\n"
        f"median={float(np.median(areas)):.4f}\n"
        f"min={float(np.min(areas)):.4f}\n"
        f"max={float(np.max(areas)):.4f}\n"
        "\n"
        "Interpretation\n"
        "--------------\n"
        "- Neckline is represented as multiple continuous dimensions, not one scalar.\n"
        "- Larger depth ratio means the center opening extends farther downward.\n"
        "- Larger width ratio means the upper-center opening spans more garment width.\n"
        "- DeepFashion2 provides no neckline geometry GT, so inspect sorted contact sheets qualitatively.\n"
        "- Do not define business thresholds from this pilot alone.\n"
    )

    (
        REPORT_DIR
        / "neckline_summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 neckline continuous geometry pilot v1\n"
        f"manifest={args.manifest}\n"
        f"upper_body_instances={len(output_rows)}\n"
        f"geometry_reliable_instances={len(reliable)}\n"
        "input=garment_instance_mask\n"
        "output_features=neckline_depth_ratio,neckline_width_ratio,neckline_open_area_ratio\n"
        "hard_classification=false\n"
        "gt_available=false\n"
        "material_and_craftsmanship=excluded\n"
    )

    (
        REPORT_DIR
        / "run_info.txt"
    ).write_text(
        run_info,
        encoding="utf-8",
    )

    make_contact_sheet(
        depth_visuals,
        OUTPUT_DIR
        / "contact_sheet_sorted_by_depth.jpg",
    )

    make_contact_sheet(
        width_visuals,
        OUTPUT_DIR
        / "contact_sheet_sorted_by_width.jpg",
    )

    print(
        "\n=== FINISHED ==="
    )

    print(
        "FEATURES : "
        "reports/prd_attribute_extraction/neckline_continuous_v1/"
        "neckline_geometry_features.csv"
    )

    print(
        "SUMMARY  : "
        "reports/prd_attribute_extraction/neckline_continuous_v1/"
        "neckline_summary.txt"
    )

    print(
        "DEPTH    : "
        "outputs/prd_attribute_extraction/neckline_continuous_v1/"
        "contact_sheet_sorted_by_depth.jpg"
    )

    print(
        "WIDTH    : "
        "outputs/prd_attribute_extraction/neckline_continuous_v1/"
        "contact_sheet_sorted_by_width.jpg"
    )

    print(
        "================"
    )


if __name__ == "__main__":
    main()
