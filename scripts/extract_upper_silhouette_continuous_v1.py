"""PRD 3.1.3 upper-garment silhouette continuous features v1.

Purpose
-------
Build the next part of the mentor-requested "continuous style feature" system:
upper-garment silhouette / fit descriptors.

This version intentionally uses geometry for attributes that are genuinely
about the OUTER garment silhouette.

Evaluated garment types
-----------------------
top
outerwear

Main continuous descriptors
---------------------------
1. garment_length_proxy in [0,1]
   = bbox_height / (bbox_height + bbox_width)

   Larger -> vertically longer/narrower silhouette.
   IMPORTANT: this is an image-space proxy, NOT true body-relative garment
   length. A future body/torso reference can replace it.

2. waist_to_chest_ratio
   = median waist-band silhouette width / median chest-band width

   < 1 -> waist is narrower than chest (more waist suppression)
   ~ 1 -> straighter silhouette
   > 1 -> waist is as wide/wider than chest

3. hem_to_chest_ratio
   = median lower-hem silhouette width / median chest-band width

   < 1 -> hem narrows relative to chest
   ~ 1 -> straight lower silhouette
   > 1 -> hem expands relative to chest

Additional diagnostics
----------------------
chest_width_ratio
waist_width_ratio
hem_width_ratio
lower_taper_slope
width_profile_cv
geometry_quality

No hard labels or business thresholds are defined here.

Example
-------
python scripts/extract_upper_silhouette_continuous_v1.py \
    --manifest configs/garment_instances_gt_pilot_100.csv

Outputs
-------
reports/prd_attribute_extraction/upper_silhouette_v1/
    upper_silhouette_features.csv
    upper_silhouette_summary.txt
    run_info.txt

outputs/prd_attribute_extraction/upper_silhouette_v1/
    per_instance/
    contact_sheet_sorted_by_length_proxy.jpg
    contact_sheet_sorted_by_waist_to_chest.jpg
    contact_sheet_sorted_by_hem_to_chest.jpg
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
    / "upper_silhouette_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_attribute_extraction"
    / "upper_silhouette_v1"
)

UPPER_CATEGORIES = {
    "top",
    "outerwear",
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

    mask = (
        np.asarray(
            mask_img,
            dtype=np.uint8,
        )
        > 127
    )

    if not mask.any():
        raise ValueError(
            f"Empty mask: {row['garment_id']}"
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


def row_width(
    mask_row: np.ndarray,
) -> float | None:
    xs = np.flatnonzero(mask_row)

    if len(xs) == 0:
        return None

    return float(
        xs.max()
        - xs.min()
        + 1
    )


def width_profile(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape

    ys = []
    widths = []

    for y in range(h):
        width = row_width(mask[y])

        if width is None:
            continue

        ys.append(
            y / max(
                h - 1,
                1,
            )
        )

        widths.append(
            width / max(
                w,
                1,
            )
        )

    return (
        np.asarray(
            ys,
            dtype=np.float64,
        ),
        np.asarray(
            widths,
            dtype=np.float64,
        ),
    )


def band_widths(
    mask: np.ndarray,
    start_frac: float,
    end_frac: float,
) -> list[float]:
    h, w = mask.shape

    y1 = max(
        0,
        min(
            h - 1,
            int(round(
                h * start_frac
            )),
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            h,
            int(round(
                h * end_frac
            )),
        ),
    )

    values = []

    for y in range(y1, y2):
        width = row_width(mask[y])

        if width is not None:
            values.append(
                width / max(
                    w,
                    1,
                )
            )

    return values


def robust_median(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        np.median(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def extract_features(
    mask: np.ndarray,
) -> dict[str, float | str]:
    tight, _ = tight_crop(mask)

    h, w = tight.shape

    garment_length_proxy = (
        h / max(
            h + w,
            1,
        )
    )

    # Use bands rather than one row so mild pose/mask noise has less effect.
    chest_width_ratio = robust_median(
        band_widths(
            tight,
            0.22,
            0.38,
        )
    )

    waist_width_ratio = robust_median(
        band_widths(
            tight,
            0.50,
            0.64,
        )
    )

    hem_width_ratio = robust_median(
        band_widths(
            tight,
            0.82,
            0.95,
        )
    )

    waist_to_chest_ratio = (
        waist_width_ratio
        / max(
            chest_width_ratio,
            1e-6,
        )
    )

    hem_to_chest_ratio = (
        hem_width_ratio
        / max(
            chest_width_ratio,
            1e-6,
        )
    )

    ys, widths = width_profile(
        tight
    )

    # Fit only lower torso: waist -> hem.
    lower_mask = (
        ys >= 0.45
    )

    if lower_mask.sum() >= 3:
        lower_taper_slope = float(
            np.polyfit(
                ys[
                    lower_mask
                ],
                widths[
                    lower_mask
                ],
                deg=1,
            )[0]
        )
    else:
        lower_taper_slope = 0.0

    mean_width = float(
        widths.mean()
    )

    width_profile_cv = float(
        widths.std()
        / max(
            mean_width,
            1e-6,
        )
    )

    valid_row_fraction = (
        len(widths)
        / max(
            h,
            1,
        )
    )

    # This is only a diagnostic flag, not a correctness label.
    if (
        valid_row_fraction >= 0.85
        and width_profile_cv <= 0.45
        and chest_width_ratio > 0.10
    ):
        geometry_quality = "good"
    elif (
        valid_row_fraction >= 0.65
        and chest_width_ratio > 0.05
    ):
        geometry_quality = "usable"
    else:
        geometry_quality = "weak"

    return {
        "garment_length_proxy": float(
            garment_length_proxy
        ),
        "chest_width_ratio": float(
            chest_width_ratio
        ),
        "waist_width_ratio": float(
            waist_width_ratio
        ),
        "hem_width_ratio": float(
            hem_width_ratio
        ),
        "waist_to_chest_ratio": float(
            waist_to_chest_ratio
        ),
        "hem_to_chest_ratio": float(
            hem_to_chest_ratio
        ),
        "lower_taper_slope": float(
            lower_taper_slope
        ),
        "width_profile_cv": float(
            width_profile_cv
        ),
        "valid_row_fraction": float(
            valid_row_fraction
        ),
        "geometry_quality": (
            geometry_quality
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

    # Draw band guides on the actual crop using the tight mask bbox.
    draw_fg = ImageDraw.Draw(
        foreground
    )

    x1, y1, x2, y2 = tight_bbox(
        mask
    )

    bbox_h = max(
        1,
        y2 - y1,
    )

    band_info = [
        (
            0.30,
            "chest",
        ),
        (
            0.57,
            "waist",
        ),
        (
            0.885,
            "hem",
        ),
    ]

    for frac, label in band_info:
        y = (
            y1
            + int(round(
                bbox_h * frac
            ))
        )

        draw_fg.line(
            (
                x1,
                y,
                x2,
                y,
            ),
            fill="black",
            width=1,
        )

        draw_fg.text(
            (
                x1 + 2,
                max(
                    0,
                    y - 12,
                ),
            ),
            label,
            fill="black",
        )

    panel_h = 255

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
        (
            f"{record['garment_id']} | "
            f"{record['garment_category']}"
        ),
        (
            "length_proxy: "
            f"{record['garment_length_proxy']:.3f}"
        ),
        (
            "waist/chest: "
            f"{record['waist_to_chest_ratio']:.3f}"
        ),
        (
            "hem/chest: "
            f"{record['hem_to_chest_ratio']:.3f}"
        ),
        (
            "lower_slope: "
            f"{record['lower_taper_slope']:.3f}"
        ),
        (
            "profile_cv: "
            f"{record['width_profile_cv']:.3f}"
        ),
        (
            "geometry_quality: "
            f"{record['geometry_quality']}"
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
        y += 33

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        save_path,
        quality=92,
    )


def make_contact_sheet(
    visual_records: list[
        tuple[float, Path]
    ],
    output_path: Path,
) -> None:
    if not visual_records:
        return

    visual_records = sorted(
        visual_records,
        key=lambda item: item[0],
    )

    tiles = []

    for _, path in visual_records:
        with Image.open(path) as img:
            img = img.convert(
                "RGB"
            )

            img.thumbnail(
                (390, 540)
            )

            tile = Image.new(
                "RGB",
                (400, 550),
                "white",
            )

            x = (
                400 - img.width
            ) // 2

            y = (
                550 - img.height
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
            rows * 550,
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
        ) * 550

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


def summarize_feature(
    rows: list[dict],
    feature: str,
) -> str:
    values = np.asarray(
        [
            float(
                row[
                    feature
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    return (
        f"{feature}: "
        f"mean={values.mean():.4f}, "
        f"median={np.median(values):.4f}, "
        f"min={values.min():.4f}, "
        f"max={values.max():.4f}"
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
            in UPPER_CATEGORIES
        )
    ]

    if not rows:
        raise ValueError(
            "No top/outerwear instances found."
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
        f"manifest        : {args.manifest}"
    )

    print(
        f"all instances   : {len(all_rows)}"
    )

    print(
        f"upper cases     : {len(rows)}"
    )

    print(
        "features        : length proxy + waist/chest + hem/chest"
    )

    output_rows = []

    length_visuals = []
    waist_visuals = []
    hem_visuals = []

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
                .strip()
                .lower()
            ),
            **features,
        }

        output_rows.append(
            record
        )

        print(
            f"[{index:03d}/{len(rows):03d}] "
            f"{record['garment_id']:<35} "
            f"len={record['garment_length_proxy']:.3f} "
            f"waist/chest={record['waist_to_chest_ratio']:.3f} "
            f"hem/chest={record['hem_to_chest_ratio']:.3f} "
            f"q={record['geometry_quality']}"
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

        if (
            record[
                "geometry_quality"
            ]
            != "weak"
        ):
            length_visuals.append(
                (
                    record[
                        "garment_length_proxy"
                    ],
                    visual_path,
                )
            )

            waist_visuals.append(
                (
                    record[
                        "waist_to_chest_ratio"
                    ],
                    visual_path,
                )
            )

            hem_visuals.append(
                (
                    record[
                        "hem_to_chest_ratio"
                    ],
                    visual_path,
                )
            )

    fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "garment_length_proxy",
        "chest_width_ratio",
        "waist_width_ratio",
        "hem_width_ratio",
        "waist_to_chest_ratio",
        "hem_to_chest_ratio",
        "lower_taper_slope",
        "width_profile_cv",
        "valid_row_fraction",
        "geometry_quality",
    ]

    write_csv(
        REPORT_DIR
        / "upper_silhouette_features.csv",
        output_rows,
        fields,
    )

    quality_counts = {
        "good": 0,
        "usable": 0,
        "weak": 0,
    }

    for row in output_rows:
        quality_counts[
            row[
                "geometry_quality"
            ]
        ] += 1

    summary_lines = [
        "PRD 3.1.3 upper silhouette continuous features v1",
        "================================================",
        f"instances={len(output_rows)}",
        "hard_classification=false",
        "gt_available=false",
        "",
        "Geometry quality",
        "----------------",
        f"good={quality_counts['good']}",
        f"usable={quality_counts['usable']}",
        f"weak={quality_counts['weak']}",
        "",
        "Feature distributions",
        "---------------------",
        summarize_feature(
            output_rows,
            "garment_length_proxy",
        ),
        summarize_feature(
            output_rows,
            "waist_to_chest_ratio",
        ),
        summarize_feature(
            output_rows,
            "hem_to_chest_ratio",
        ),
        summarize_feature(
            output_rows,
            "lower_taper_slope",
        ),
        "",
        "Interpretation",
        "--------------",
        "- garment_length_proxy is an image-space silhouette proxy, not body-relative physical garment length.",
        "- waist_to_chest_ratio < 1 indicates waist narrowing relative to chest.",
        "- hem_to_chest_ratio > 1 indicates lower-hem expansion relative to chest.",
        "- These features are intended as continuous style descriptors; no business thresholds are set.",
        "- DeepFashion2 has no direct GT for these fit/shape dimensions here, so validate qualitatively on sorted contact sheets.",
        "- Material/fabric and craftsmanship remain excluded.",
    ]

    (
        REPORT_DIR
        / "upper_silhouette_summary.txt"
    ).write_text(
        "\n".join(
            summary_lines
        )
        + "\n",
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 upper silhouette continuous features v1\n"
        f"manifest={args.manifest}\n"
        f"instances={len(output_rows)}\n"
        "garment_categories=top,outerwear\n"
        "input=garment_instance_mask\n"
        "output_features=garment_length_proxy,waist_to_chest_ratio,hem_to_chest_ratio,lower_taper_slope\n"
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
        length_visuals,
        OUTPUT_DIR
        / "contact_sheet_sorted_by_length_proxy.jpg",
    )

    make_contact_sheet(
        waist_visuals,
        OUTPUT_DIR
        / "contact_sheet_sorted_by_waist_to_chest.jpg",
    )

    make_contact_sheet(
        hem_visuals,
        OUTPUT_DIR
        / "contact_sheet_sorted_by_hem_to_chest.jpg",
    )

    print(
        "\n=== FINISHED ==="
    )

    print(
        "FEATURES : "
        "reports/prd_attribute_extraction/upper_silhouette_v1/"
        "upper_silhouette_features.csv"
    )

    print(
        "SUMMARY  : "
        "reports/prd_attribute_extraction/upper_silhouette_v1/"
        "upper_silhouette_summary.txt"
    )

    print(
        "LENGTH   : "
        "outputs/prd_attribute_extraction/upper_silhouette_v1/"
        "contact_sheet_sorted_by_length_proxy.jpg"
    )

    print(
        "WAIST    : "
        "outputs/prd_attribute_extraction/upper_silhouette_v1/"
        "contact_sheet_sorted_by_waist_to_chest.jpg"
    )

    print(
        "HEM      : "
        "outputs/prd_attribute_extraction/upper_silhouette_v1/"
        "contact_sheet_sorted_by_hem_to_chest.jpg"
    )

    print(
        "================"
    )


if __name__ == "__main__":
    main()
