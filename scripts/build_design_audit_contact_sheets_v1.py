"""Build visual contact sheets for the 40-row design-attribute manual audit.

Run from project root:
    python scripts/build_design_audit_contact_sheets_v1.py

Inputs:
    reports/prd_attribute_extraction/design_labels_v1_fixed/
        manual_audit_holdout_v1_fixed.csv
    configs/garment_instances_gt_pilot_100.csv

Outputs:
    outputs/prd_attribute_extraction/design_labels_v1_fixed/audit_contact_sheets/
        sleeve_length.jpg
        neckline.jpg
        silhouette_fit.jpg
        fashion_style.jpg

Each tile shows the garment crop plus:
- CSV row number
- garment_id
- predicted label
- top-2 label
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "design_labels_v1_fixed"
    / "manual_audit_holdout_v1_fixed.csv"
)

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "configs"
    / "garment_instances_gt_pilot_100.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_attribute_extraction"
    / "design_labels_v1_fixed"
    / "audit_contact_sheets"
)


def resolve_path(raw: str) -> Path:
    p = Path(raw).expanduser()

    for candidate in (p, Path.cwd() / p, PROJECT_ROOT / p):
        if candidate.is_file():
            return candidate.resolve()

    matches = [
        x.resolve()
        for x in PROJECT_ROOT.rglob(p.name)
        if x.is_file()
    ]
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(f"Cannot resolve path: {raw}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def manifest_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out = {}
    for row in rows:
        out[
            (
                normalize_path(row["source_image"]),
                row["garment_id"].strip(),
            )
        ] = row
    return out


def load_garment_crop(row: dict[str, str]) -> Image.Image:
    crop = Image.open(resolve_path(row["crop_path"])).convert("RGB")

    mask_raw = row.get("mask_path", "").strip()
    if not mask_raw:
        return crop

    mask = Image.open(resolve_path(mask_raw)).convert("L")
    if mask.size != crop.size:
        mask = mask.resize(crop.size, Image.Resampling.NEAREST)

    bg = Image.new("RGB", crop.size, "white")
    return Image.composite(crop, bg, mask)


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    img = image.copy()
    img.thumbnail((width, height), Image.Resampling.LANCZOS)

    x = (width - img.width) // 2
    y = (height - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_sheet(attribute: str, entries: list[tuple[int, dict[str, str], Image.Image]]) -> Image.Image:
    cols = 2
    tile_w = 720
    tile_h = 540
    img_w = 420
    img_h = 440
    margin = 18

    rows_n = (len(entries) + cols - 1) // cols
    sheet = Image.new(
        "RGB",
        (cols * tile_w, rows_n * tile_h + 70),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    draw.text(
        (20, 20),
        f"Design attribute manual audit — {attribute}",
        fill="black",
        font=font,
    )

    for idx, (csv_row, audit_row, image) in enumerate(entries):
        r = idx // cols
        c = idx % cols
        x0 = c * tile_w
        y0 = 70 + r * tile_h

        fitted = fit_image(image, img_w, img_h)
        sheet.paste(fitted, (x0 + margin, y0 + margin))

        text_x = x0 + img_w + 2 * margin
        text_y = y0 + margin

        lines = [
            f"CSV row: {csv_row}",
            f"image: {Path(audit_row['source_image']).name}",
            f"garment: {audit_row['garment_id']}",
            f"category: {audit_row['garment_category']}",
            "",
            f"pred: {audit_row['predicted_label']}",
            f"top1: {float(audit_row['top1_score']):.3f}",
            f"top2: {audit_row['top2_label']}",
            f"top2 score: {float(audit_row['top2_score']):.3f}",
            f"margin: {float(audit_row['margin']):.3f}",
        ]

        for line in lines:
            draw.text((text_x, text_y), line, fill="black", font=font)
            text_y += 25

        draw.rectangle(
            (x0 + 4, y0 + 4, x0 + tile_w - 4, y0 + tile_h - 4),
            outline="gray",
            width=1,
        )

    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        default=str(DEFAULT_AUDIT.relative_to(PROJECT_ROOT)),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST.relative_to(PROJECT_ROOT)),
    )
    args = parser.parse_args()

    audit_path = resolve_path(args.audit)
    manifest_path = resolve_path(args.manifest)

    audit_rows = read_csv(audit_path)
    manifest_rows = read_csv(manifest_path)
    index = manifest_index(manifest_rows)

    grouped = defaultdict(list)
    missing = []

    # CSV header is row 1, first data row is row 2.
    for zero_idx, audit_row in enumerate(audit_rows):
        csv_row = zero_idx + 2
        key = (
            normalize_path(audit_row["source_image"]),
            audit_row["garment_id"].strip(),
        )

        manifest_row = index.get(key)
        if manifest_row is None:
            missing.append((csv_row, key))
            continue

        image = load_garment_crop(manifest_row)
        grouped[audit_row["attribute"]].append(
            (csv_row, audit_row, image)
        )

    if missing:
        preview = "\n".join(
            f"row {row}: {key[0]} | {key[1]}"
            for row, key in missing[:10]
        )
        raise RuntimeError(
            f"{len(missing)} audit rows could not be joined to manifest.\n{preview}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for attribute, entries in grouped.items():
        entries.sort(key=lambda x: x[0])
        sheet = make_sheet(attribute, entries)
        out = OUTPUT_DIR / f"{attribute}.jpg"
        sheet.save(out, quality=94)
        print(f"{attribute}: {len(entries)} -> {out.relative_to(PROJECT_ROOT)}")

    print("\nFinished. Upload the four JPG contact sheets to ChatGPT.")


if __name__ == "__main__":
    main()
