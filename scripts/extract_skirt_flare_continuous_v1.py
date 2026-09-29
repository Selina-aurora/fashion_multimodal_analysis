"""PRD 3.1.3 continuous skirt-flare geometry feature pilot v1.

Goal
----
Represent skirt silhouette / hem style as continuous geometry rather than
hard labels such as straight / A-line / flared.

Primary outputs
---------------
hem_to_waist_ratio
    Raw geometry ratio. Larger generally means a more expanded/flared hem.

skirt_flare_score
    Bounded [0,1] diagnostic mapping of hem_to_waist_ratio:
        ratio <= 0.9  -> 0
        ratio >= 2.5  -> 1
    Values between are linearly mapped.
    These anchors are provisional normalization anchors, NOT business thresholds.

Additional continuous descriptors
---------------------------------
waist_width_ratio
mid_width_ratio
hem_width_ratio
hem_to_mid_ratio
width_expansion_slope
lower_area_ratio

Important
---------
- Uses garment-instance mask only.
- No hard skirt-style class is produced.
- DeepFashion2 has no skirt-flare GT here, so this is a qualitative/geometry
  pilot. The contact sheet sorted by flare score is the main sanity check.
- Material/fabric and craftsmanship remain excluded.

Example
-------
python scripts/extract_skirt_flare_continuous_v1.py \
    --manifest configs/garment_instances_gt_pilot_100.csv

Outputs
-------
reports/prd_attribute_extraction/skirt_flare_v1/
    skirt_flare_features.csv
    skirt_flare_summary.txt
    run_info.txt

outputs/prd_attribute_extraction/skirt_flare_v1/
    per_instance/
    contact_sheet_sorted_by_flare.jpg
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
    / "skirt_flare_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_attribute_extraction"
    / "skirt_flare_v1"
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


def tight_crop(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    return mask[y1:y2, x1:x2]


def row_width(mask_row: np.ndarray) -> float | None:
    xs = np.flatnonzero(mask_row)

    if len(xs) == 0:
        return None

    return float(
        xs.max() - xs.min() + 1
    )


def median_band_width(
    mask: np.ndarray,
    start_frac: float,
    end_frac: float,
) -> float:
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

    widths = []

    for y in range(y1, y2):
        width = row_width(mask[y])

        if width is not None:
            widths.append(
                width / max(w, 1)
            )

    if not widths:
        return 0.0

    return float(
        np.median(
            np.asarray(
                widths,
                dtype=np.float64,
            )
        )
    )


def width_expansion_slope(
    mask: np.ndarray,
) -> float:
    """Linear slope of normalized silhouette width from top to bottom."""
    h, w = mask.shape

    xs = []
    ys = []

    for y in range(h):
        width = row_width(mask[y])

        if width is None:
            continue

        ys.append(
            y / max(h - 1, 1)
        )

        xs.append(
            width / max(w, 1)
        )

    if len(xs) < 3:
        return 0.0

    slope = np.polyfit(
        np.asarray(ys),
        np.asarray(xs),
        deg=1,
    )[0]

    return float(slope)


def extract_features(
    mask: np.ndarray,
) -> dict[str, float]:
    tight = tight_crop(mask)

    waist_width_ratio = median_band_width(
        tight,
        0.08,
        0.22,
    )

    mid_width_ratio = median_band_width(
        tight,
        0.42,
        0.60,
    )

    hem_width_ratio = median_band_width(
        tight,
        0.82,
        0.96,
    )

    hem_to_waist_ratio = (
        hem_width_ratio
        / max(
            waist_width_ratio,
            1e-6,
        )
    )

    hem_to_mid_ratio = (
        hem_width_ratio
        / max(
            mid_width_ratio,
            1e-6,
        )
    )

    slope = width_expansion_slope(
        tight
    )

    lower_start = int(
        round(
            tight.shape[0] * 0.50
        )
    )

    lower_area_ratio = (
        float(
            tight[
                lower_start:
            ].sum()
        )
        / max(
            float(
                tight.sum()
            ),
            1.0,
        )
    )

    # Provisional normalization only.
    # 0.9 roughly corresponds to non-expanding/straight silhouette,
    # 2.5+ to strongly expanded hem in this geometry scale.
    skirt_flare_score = (
        hem_to_waist_ratio
        - 0.9
    ) / (
        2.5 - 0.9
    )

    skirt_flare_score = float(
        np.clip(
            skirt_flare_score,
            0.0,
            1.0,
        )
    )

    return {
        "waist_width_ratio": (
            waist_width_ratio
        ),
        "mid_width_ratio": (
            mid_width_ratio
        ),
        "hem_width_ratio": (
            hem_width_ratio
        ),
        "hem_to_waist_ratio": (
            hem_to_waist_ratio
        ),
        "hem_to_mid_ratio": (
            hem_to_mid_ratio
        ),
        "width_expansion_slope": (
            slope
        ),
        "lower_area_ratio": (
            lower_area_ratio
        ),
        "skirt_flare_score": (
            skirt_flare_score
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

    font = (
        ImageFont.load_default()
    )

    lines = [
        record["garment_id"],
        (
            "flare_score: "
            f"{record['skirt_flare_score']:.3f}"
        ),
        (
            "hem/waist: "
            f"{record['hem_to_waist_ratio']:.3f}"
        ),
        (
            "hem/mid: "
            f"{record['hem_to_mid_ratio']:.3f}"
        ),
        (
            "expansion_slope: "
            f"{record['width_expansion_slope']:.3f}"
        ),
        (
            "lower_area: "
            f"{record['lower_area_ratio']:.3f}"
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
    visual_records: list[
        tuple[float, Path]
    ],
    output_path: Path,
) -> None:
    if not visual_records:
        return

    visual_records = sorted(
        visual_records,
        key=lambda x: x[0],
    )

    tiles = []

    for _, path in visual_records:
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
        writer.writerows(rows)


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
            == "skirt"
        )
    ]

    if not rows:
        raise ValueError(
            "No skirt instances found in manifest."
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
        f"manifest      : {args.manifest}"
    )
    print(
        f"all instances : {len(all_rows)}"
    )
    print(
        f"skirt cases   : {len(rows)}"
    )
    print(
        "feature       : continuous skirt flare geometry"
    )

    records = []
    visual_records = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        crop, mask = (
            load_crop_and_mask(
                row
            )
        )

        features = (
            extract_features(
                mask
            )
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
            **features,
        }

        records.append(
            record
        )

        print(
            f"[{index:02d}/{len(rows):02d}] "
            f"{record['garment_id']:<24} "
            f"flare={record['skirt_flare_score']:.3f} "
            f"hem/waist={record['hem_to_waist_ratio']:.3f}"
        )

        visual_path = (
            per_instance_dir
            / (
                f"{index:02d}_"
                f"{record['garment_id']}.jpg"
            )
        )

        save_visual(
            crop,
            mask,
            record,
            visual_path,
        )

        visual_records.append(
            (
                record[
                    "skirt_flare_score"
                ],
                visual_path,
            )
        )

    write_csv(
        REPORT_DIR
        / "skirt_flare_features.csv",
        records,
        [
            "source_image",
            "garment_id",
            "waist_width_ratio",
            "mid_width_ratio",
            "hem_width_ratio",
            "hem_to_waist_ratio",
            "hem_to_mid_ratio",
            "width_expansion_slope",
            "lower_area_ratio",
            "skirt_flare_score",
        ],
    )

    scores = [
        row[
            "skirt_flare_score"
        ]
        for row in records
    ]

    raw_ratios = [
        row[
            "hem_to_waist_ratio"
        ]
        for row in records
    ]

    summary = (
        "PRD 3.1.3 skirt flare continuous geometry v1\n"
        "=============================================\n"
        f"skirt_instances={len(records)}\n"
        "hard_classification=false\n"
        "gt_available=false\n"
        "\n"
        "flare_score summary\n"
        "-------------------\n"
        f"mean={float(np.mean(scores)):.4f}\n"
        f"median={float(np.median(scores)):.4f}\n"
        f"min={float(np.min(scores)):.4f}\n"
        f"max={float(np.max(scores)):.4f}\n"
        "\n"
        "hem_to_waist_ratio summary\n"
        "---------------------------\n"
        f"mean={float(np.mean(raw_ratios)):.4f}\n"
        f"median={float(np.median(raw_ratios)):.4f}\n"
        f"min={float(np.min(raw_ratios)):.4f}\n"
        f"max={float(np.max(raw_ratios)):.4f}\n"
        "\n"
        "Interpretation\n"
        "--------------\n"
        "- Larger hem_to_waist_ratio means stronger hem expansion geometrically.\n"
        "- flare_score is only a normalized continuous descriptor, not a skirt-style label.\n"
        "- DeepFashion2 provides no skirt-flare GT here, so validate qualitatively on the sorted contact sheet.\n"
        "- Do not define business thresholds from this pilot alone.\n"
    )

    (
        REPORT_DIR
        / "skirt_flare_summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 skirt flare continuous geometry pilot v1\n"
        f"manifest={args.manifest}\n"
        f"skirt_instances={len(records)}\n"
        "input=garment_instance_mask\n"
        "output_feature=skirt_flare_score_[0,1]\n"
        "primary_raw_feature=hem_to_waist_ratio\n"
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
        visual_records,
        OUTPUT_DIR
        / "contact_sheet_sorted_by_flare.jpg",
    )

    print(
        "\n=== FINISHED ==="
    )

    print(
        "FEATURES : "
        "reports/prd_attribute_extraction/skirt_flare_v1/"
        "skirt_flare_features.csv"
    )

    print(
        "SUMMARY  : "
        "reports/prd_attribute_extraction/skirt_flare_v1/"
        "skirt_flare_summary.txt"
    )

    print(
        "CONTACT  : "
        "outputs/prd_attribute_extraction/skirt_flare_v1/"
        "contact_sheet_sorted_by_flare.jpg"
    )

    print(
        "================"
    )


if __name__ == "__main__":
    main()
