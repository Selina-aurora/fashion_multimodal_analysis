"""PRD 3.1.3 long-trouser fit continuous geometry v2.

Why v2
------
The first pants-silhouette pilot mixed shorts and trousers. That is not ideal:
shorts naturally have a wide lower opening and do not contain a meaningful
knee-to-hem leg profile. Pants LENGTH and pants FIT should therefore be
separated.

Pipeline
--------
1. pants_length_ratio handles shorts <-> trousers on all pants instances.
2. This script evaluates FIT only on long-trouser instances.

Continuous fit descriptors
--------------------------
thigh_width_ratio
knee_width_ratio
hem_width_ratio

knee_to_thigh_ratio
    knee width / thigh width

hem_to_knee_ratio
    hem width / knee width

hem_to_thigh_ratio
    hem width / thigh width

lower_leg_fullness
    mean actual foreground width in the lower half

lower_width_slope
    width trend from upper leg to hem

Interpretation (descriptive only)
---------------------------------
Smaller hem_to_thigh_ratio -> stronger taper toward the ankle.
Near 1 -> straighter width profile.
Larger values -> relatively wider hem.

No skinny/straight/wide-leg hard labels are produced and no business
thresholds are set.

Example
-------
python scripts/extract_trouser_fit_continuous_v2.py \
    --manifest configs/garment_instances_gt_pilot_100.csv
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
    / "trouser_fit_v2"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_attribute_extraction"
    / "trouser_fit_v2"
)


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


def is_long_trouser(row: dict[str, str]) -> bool:
    gid = row.get("garment_id", "").strip().lower()
    return "trousers" in gid


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
            f"Empty mask: {row['garment_id']}"
        )

    return crop, mask


def tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def tight_crop(mask: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = tight_bbox(mask)
    return mask[y1:y2, x1:x2]


def contiguous_segments(
    row: np.ndarray,
) -> list[tuple[int, int]]:
    xs = np.flatnonzero(row)

    if len(xs) == 0:
        return []

    segments = []
    start = int(xs[0])
    prev = int(xs[0])

    for value in xs[1:]:
        value = int(value)

        if value != prev + 1:
            segments.append((start, prev + 1))
            start = value

        prev = value

    segments.append((start, prev + 1))
    return segments


def row_actual_width(
    row: np.ndarray,
) -> tuple[float, int]:
    segments = contiguous_segments(row)

    if not segments:
        return 0.0, 0

    width = float(
        sum(
            x2 - x1
            for x1, x2 in segments
        )
    )

    return width, len(segments)


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
            int(round(h * start_frac)),
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            h,
            int(round(h * end_frac)),
        ),
    )

    values = []

    for y in range(y1, y2):
        width, n_segments = row_actual_width(mask[y])

        if n_segments == 0:
            continue

        values.append(
            width / max(w, 1)
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


def lower_profile(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = mask.shape

    ys = []
    widths = []
    split_counts = []

    start_y = int(round(h * 0.28))

    for y in range(start_y, h):
        width, n_segments = row_actual_width(mask[y])

        if n_segments == 0:
            continue

        ys.append(
            y / max(h - 1, 1)
        )
        widths.append(
            width / max(w, 1)
        )
        split_counts.append(n_segments)

    return (
        np.asarray(ys, dtype=np.float64),
        np.asarray(widths, dtype=np.float64),
        np.asarray(split_counts, dtype=np.int64),
    )


def extract_features(
    mask: np.ndarray,
) -> dict[str, float | str]:
    tight = tight_crop(mask)

    h, _ = tight.shape

    thigh = robust_median(
        band_widths(
            tight,
            0.28,
            0.42,
        )
    )

    knee = robust_median(
        band_widths(
            tight,
            0.52,
            0.66,
        )
    )

    hem = robust_median(
        band_widths(
            tight,
            0.84,
            0.96,
        )
    )

    knee_to_thigh = (
        knee / max(thigh, 1e-6)
    )

    hem_to_knee = (
        hem / max(knee, 1e-6)
    )

    hem_to_thigh = (
        hem / max(thigh, 1e-6)
    )

    ys, widths, segments = lower_profile(
        tight
    )

    if len(widths) >= 3:
        lower_width_slope = float(
            np.polyfit(
                ys,
                widths,
                deg=1,
            )[0]
        )
    else:
        lower_width_slope = 0.0

    lower_mask = ys >= 0.50

    if lower_mask.any():
        lower_leg_fullness = float(
            widths[lower_mask].mean()
        )
    else:
        lower_leg_fullness = 0.0

    split_leg_fraction = (
        float(
            (segments >= 2).mean()
        )
        if len(segments)
        else 0.0
    )

    valid_row_fraction = (
        len(widths)
        / max(
            h - int(round(h * 0.28)),
            1,
        )
    )

    width_cv = (
        float(
            widths.std()
            / max(widths.mean(), 1e-6)
        )
        if len(widths)
        else 0.0
    )

    # QC is intentionally stricter than v1. "good" here still means
    # numerically usable, not manually verified style correctness.
    if (
        valid_row_fraction >= 0.88
        and thigh > 0.08
        and knee > 0.05
        and hem > 0.03
        and width_cv <= 0.55
    ):
        geometry_quality = "good"
    elif (
        valid_row_fraction >= 0.68
        and thigh > 0.05
        and hem > 0.02
    ):
        geometry_quality = "usable"
    else:
        geometry_quality = "weak"

    return {
        "thigh_width_ratio": float(thigh),
        "knee_width_ratio": float(knee),
        "hem_width_ratio": float(hem),
        "knee_to_thigh_ratio": float(knee_to_thigh),
        "hem_to_knee_ratio": float(hem_to_knee),
        "hem_to_thigh_ratio": float(hem_to_thigh),
        "lower_width_slope": float(lower_width_slope),
        "lower_leg_fullness": float(lower_leg_fullness),
        "split_leg_fraction": float(split_leg_fraction),
        "valid_row_fraction": float(valid_row_fraction),
        "width_profile_cv": float(width_cv),
        "geometry_quality": geometry_quality,
    }


def save_visual(
    crop: Image.Image,
    mask: np.ndarray,
    record: dict,
    save_path: Path,
) -> None:
    white = Image.new("RGB", crop.size, "white")

    mask_img = Image.fromarray(
        mask.astype(np.uint8) * 255,
        mode="L",
    )

    foreground = Image.composite(
        crop,
        white,
        mask_img,
    )

    draw_fg = ImageDraw.Draw(foreground)

    x1, y1, x2, y2 = tight_bbox(mask)
    bbox_h = max(1, y2 - y1)

    for frac, label in [
        (0.35, "thigh"),
        (0.59, "knee"),
        (0.90, "hem"),
    ]:
        y = y1 + int(round(bbox_h * frac))

        draw_fg.line(
            (x1, y, x2, y),
            fill="black",
            width=1,
        )

        draw_fg.text(
            (x1 + 2, max(0, y - 12)),
            label,
            fill="black",
        )

    panel_h = 255

    canvas = Image.new(
        "RGB",
        (
            foreground.width,
            foreground.height + panel_h,
        ),
        "white",
    )

    canvas.paste(
        foreground,
        (0, panel_h),
    )

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    lines = [
        record["garment_id"],
        f"thigh: {record['thigh_width_ratio']:.3f}",
        f"knee: {record['knee_width_ratio']:.3f}",
        f"hem: {record['hem_width_ratio']:.3f}",
        (
            "hem/thigh: "
            f"{record['hem_to_thigh_ratio']:.3f}"
        ),
        (
            "lower_fullness: "
            f"{record['lower_leg_fullness']:.3f}"
        ),
        (
            "quality: "
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
    records: list[tuple[float, Path]],
    output_path: Path,
) -> None:
    if not records:
        return

    records = sorted(
        records,
        key=lambda item: item[0],
    )

    tiles = []

    for _, path in records:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((390, 530))

            tile = Image.new(
                "RGB",
                (400, 540),
                "white",
            )

            x = (400 - img.width) // 2
            y = (540 - img.height) // 2

            tile.paste(img, (x, y))
            tiles.append(tile)

    cols = 4
    rows = (
        len(tiles) + cols - 1
    ) // cols

    sheet = Image.new(
        "RGB",
        (
            cols * 400,
            rows * 540,
        ),
        "white",
    )

    for idx, tile in enumerate(tiles):
        x = (idx % cols) * 400
        y = (idx // cols) * 540
        sheet.paste(tile, (x, y))

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
        writer.writerows(rows)


def summarize_feature(
    rows: list[dict],
    feature: str,
) -> str:
    values = np.asarray(
        [float(row[feature]) for row in rows],
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
        if is_long_trouser(row)
    ]

    if not rows:
        raise ValueError(
            "No long-trouser instances found. "
            "Expected garment_id containing 'trousers'."
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_instance_dir = OUTPUT_DIR / "per_instance"

    per_instance_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"manifest       : {args.manifest}")
    print(f"all instances  : {len(all_rows)}")
    print(f"trouser cases  : {len(rows)}")
    print(
        "mode           : long-trouser fit only; shorts excluded"
    )

    output_rows = []

    taper_visuals = []
    fullness_visuals = []
    hem_visuals = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        crop, mask = load_crop_and_mask(row)
        features = extract_features(mask)

        record = {
            "source_image": row["source_image"],
            "garment_id": row["garment_id"],
            "garment_category": row["garment_category"],
            **features,
        }

        output_rows.append(record)

        print(
            f"[{index:02d}/{len(rows):02d}] "
            f"{record['garment_id']:<28} "
            f"hem/thigh={record['hem_to_thigh_ratio']:.3f} "
            f"fullness={record['lower_leg_fullness']:.3f} "
            f"q={record['geometry_quality']}"
        )

        visual_path = (
            per_instance_dir
            / f"{index:02d}_{record['garment_id']}.jpg"
        )

        save_visual(
            crop,
            mask,
            record,
            visual_path,
        )

        if record["geometry_quality"] != "weak":
            taper_visuals.append(
                (
                    record[
                        "hem_to_thigh_ratio"
                    ],
                    visual_path,
                )
            )

            fullness_visuals.append(
                (
                    record[
                        "lower_leg_fullness"
                    ],
                    visual_path,
                )
            )

            hem_visuals.append(
                (
                    record[
                        "hem_width_ratio"
                    ],
                    visual_path,
                )
            )

    fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "thigh_width_ratio",
        "knee_width_ratio",
        "hem_width_ratio",
        "knee_to_thigh_ratio",
        "hem_to_knee_ratio",
        "hem_to_thigh_ratio",
        "lower_width_slope",
        "lower_leg_fullness",
        "split_leg_fraction",
        "valid_row_fraction",
        "width_profile_cv",
        "geometry_quality",
    ]

    write_csv(
        REPORT_DIR / "trouser_fit_features_v2.csv",
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
            row["geometry_quality"]
        ] += 1

    summary_lines = [
        "PRD 3.1.3 long-trouser fit continuous geometry v2",
        "=================================================",
        f"instances={len(output_rows)}",
        "shorts_excluded=true",
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
            "thigh_width_ratio",
        ),
        summarize_feature(
            output_rows,
            "knee_width_ratio",
        ),
        summarize_feature(
            output_rows,
            "hem_width_ratio",
        ),
        summarize_feature(
            output_rows,
            "hem_to_thigh_ratio",
        ),
        summarize_feature(
            output_rows,
            "lower_leg_fullness",
        ),
        summarize_feature(
            output_rows,
            "split_leg_fraction",
        ),
        "",
        "Interpretation",
        "--------------",
        "- Pants length and pants fit are intentionally separated.",
        "- shorts are excluded because they do not provide a meaningful long-leg taper profile.",
        "- hem_to_thigh_ratio is a continuous taper descriptor, not a hard style class.",
        "- lower_leg_fullness is an additional continuous width/fullness descriptor.",
        "- No business thresholds are set.",
        "- No direct DeepFashion2 fit GT is available; validate qualitatively.",
        "- Material/fabric and craftsmanship remain excluded.",
    ]

    (
        REPORT_DIR / "trouser_fit_summary_v2.txt"
    ).write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 long-trouser fit continuous geometry v2\n"
        f"manifest={args.manifest}\n"
        f"instances={len(output_rows)}\n"
        "selection=garment_id_contains_trousers\n"
        "shorts_excluded=true\n"
        "input=garment_instance_mask\n"
        "output_features=thigh_width_ratio,knee_width_ratio,hem_width_ratio,hem_to_thigh_ratio,lower_leg_fullness\n"
        "hard_classification=false\n"
        "gt_available=false\n"
        "material_and_craftsmanship=excluded\n"
    )

    (
        REPORT_DIR / "run_info.txt"
    ).write_text(
        run_info,
        encoding="utf-8",
    )

    make_contact_sheet(
        taper_visuals,
        OUTPUT_DIR
        / "contact_sheet_sorted_by_hem_to_thigh.jpg",
    )

    make_contact_sheet(
        fullness_visuals,
        OUTPUT_DIR
        / "contact_sheet_sorted_by_lower_fullness.jpg",
    )

    make_contact_sheet(
        hem_visuals,
        OUTPUT_DIR
        / "contact_sheet_sorted_by_hem_width.jpg",
    )

    print("\n=== FINISHED ===")

    print(
        "FEATURES : "
        "reports/prd_attribute_extraction/trouser_fit_v2/"
        "trouser_fit_features_v2.csv"
    )

    print(
        "SUMMARY  : "
        "reports/prd_attribute_extraction/trouser_fit_v2/"
        "trouser_fit_summary_v2.txt"
    )

    print(
        "TAPER    : "
        "outputs/prd_attribute_extraction/trouser_fit_v2/"
        "contact_sheet_sorted_by_hem_to_thigh.jpg"
    )

    print(
        "FULLNESS : "
        "outputs/prd_attribute_extraction/trouser_fit_v2/"
        "contact_sheet_sorted_by_lower_fullness.jpg"
    )

    print("================")


if __name__ == "__main__":
    main()
