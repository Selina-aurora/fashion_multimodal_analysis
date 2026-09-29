
from __future__ import annotations

import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CASES = (
    PROJECT_ROOT
    / "reports"
    / "prd_region_coverage"
    / "remaining5_diagnostic_v1"
    / "cases.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_region_coverage"
    / "shoulder_waist_pattern_refinement_v1"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_region_coverage"
    / "shoulder_waist_pattern_refinement_v1"
)

MANIFEST_OUT = REPORT_DIR / "refinement_manifest.csv"
CONTACT_OUT = REPORT_DIR / "refinement_windows_contact_sheet.jpg"

TARGET_REGIONS = {"shoulder", "waist", "pattern"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_project_path(raw: str) -> Path:
    p = Path(str(raw).strip().replace("\\", "/"))
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def windows_for(region: str, w: int, h: int):
    if region == "shoulder":
        # Two upper-side windows. This is a single spatial-prior ablation:
        # same prompt/threshold later, only input window changes.
        return [
            ("upper_left", 0.00, 0.00, 0.62, 0.48),
            ("upper_right", 0.38, 0.00, 1.00, 0.48),
        ]

    if region == "waist":
        return [
            ("middle_band", 0.05, 0.28, 0.95, 0.72),
        ]

    if region == "pattern":
        return [
            ("garment_center", 0.06, 0.06, 0.94, 0.94),
        ]

    raise ValueError(region)


def main():
    rows = [
        r for r in read_csv(INPUT_CASES)
        if r.get("region", "").strip().lower() in TARGET_REGIONS
    ]

    if len(rows) != 15:
        print(f"WARNING: expected 15 source cases, found {len(rows)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    out_rows = []
    preview_tiles = []

    for row in rows:
        region = row["region"].strip().lower()
        src = resolve_project_path(row["candidate_crop_path"])

        if not src.is_file():
            raise FileNotFoundError(f"Candidate crop missing: {src}")

        image = Image.open(src).convert("RGB")
        w, h = image.size

        for window_name, x1r, y1r, x2r, y2r in windows_for(region, w, h):
            x1 = max(0, min(w - 1, int(round(x1r * w))))
            y1 = max(0, min(h - 1, int(round(y1r * h))))
            x2 = max(x1 + 1, min(w, int(round(x2r * w))))
            y2 = max(y1 + 1, min(h, int(round(y2r * h))))

            crop = image.crop((x1, y1, x2, y2))

            # Upscale small windows only for saved diagnostic input.
            # Grounding DINO may resize internally later, but keeping larger
            # saved crops also makes manual inspection easier.
            scale = max(1.0, 900.0 / max(crop.size))
            if scale > 1.0:
                crop = crop.resize(
                    (
                        int(round(crop.width * scale)),
                        int(round(crop.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )

            out_path = (
                OUTPUT_DIR
                / region
                / f"{row['candidate_id']}_{window_name}.jpg"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out_path, quality=95)

            out_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "region": region,
                    "image_name": row.get("image_name", ""),
                    "item_id": row.get("item_id", ""),
                    "baseline_quality": row.get("localization_quality", ""),
                    "baseline_detected": row.get("detected", ""),
                    "baseline_top_score": row.get("top_score", ""),
                    "baseline_top_box_area_ratio": row.get("top_box_area_ratio", ""),
                    "prompt": row.get("prompt", ""),
                    "threshold": row.get("threshold", ""),
                    "window_name": window_name,
                    "parent_crop_path": row["candidate_crop_path"],
                    "window_x1": x1,
                    "window_y1": y1,
                    "window_x2": x2,
                    "window_y2": y2,
                    "window_area_ratio_parent": round(
                        ((x2 - x1) * (y2 - y1)) / float(w * h), 6
                    ),
                    "refinement_crop_path": rel(out_path),
                    "refinement_detected": "",
                    "refinement_top_score": "",
                    "refinement_top_box_area_ratio_parent": "",
                    "refinement_quality": "",
                    "review_note": "",
                }
            )

            # Preview tile with window drawn on parent image
            p = image.copy()
            d = ImageDraw.Draw(p)
            d.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=4)
            font = ImageFont.load_default()
            d.text(
                (5, 5),
                f"{row['candidate_id']} | {region} | {window_name}",
                fill=(255, 0, 0),
                font=font,
            )
            p.thumbnail((380, 330), Image.Resampling.LANCZOS)
            preview_tiles.append(p)

    write_csv(MANIFEST_OUT, out_rows)

    # Build one compact contact sheet.
    cols = 4
    tile_w, tile_h = 400, 360
    rows_n = (len(preview_tiles) + cols - 1) // cols

    sheet = Image.new("RGB", (cols * tile_w, rows_n * tile_h), "white")

    for i, tile in enumerate(preview_tiles):
        x = (i % cols) * tile_w + (tile_w - tile.width) // 2
        y = (i // cols) * tile_h + 10
        sheet.paste(tile, (x, y))

    sheet.save(CONTACT_OUT, quality=92)

    print("=== REFINEMENT INPUTS BUILT ===")
    print(f"Source cases: {len(rows)}")
    print(f"Refinement crops: {len(out_rows)}")
    print("Regions:")
    print("  shoulder: 5 cases x 2 side windows")
    print("  waist:    5 cases x 1 middle band")
    print("  pattern:  5 cases x 1 center window")
    print()
    print("Manifest:", MANIFEST_OUT.relative_to(PROJECT_ROOT))
    print("Contact :", CONTACT_OUT.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
