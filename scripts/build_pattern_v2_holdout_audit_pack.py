"""Build a row-numbered image pack for the NEW pattern-v2 holdout audit.

Place at:
    scripts/build_pattern_v2_holdout_audit_pack.py

Run from project root:
    python scripts/build_pattern_v2_holdout_audit_pack.py

Inputs
------
reports/prd_attribute_extraction/pattern_v2_hierarchical/
    pattern_v2_predictions.csv
    pattern_v2_holdout_audit_template.csv

outputs/prd_attribute_extraction/pattern_v2_hierarchical/
    per_instance/

Outputs
-------
outputs/prd_attribute_extraction/pattern_v2_hierarchical/
    holdout_audit_v1/
        01_<garment_id>.jpg
        ...
    holdout_audit_v1_contact_sheet.jpg

The ROW number on each tile matches the row order in
pattern_v2_holdout_audit_template.csv.
"""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "pattern_v2_hierarchical"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_attribute_extraction"
    / "pattern_v2_hierarchical"
)

PREDICTIONS_CSV = REPORT_DIR / "pattern_v2_predictions.csv"
HOLDOUT_CSV = REPORT_DIR / "pattern_v2_holdout_audit_template.csv"
PER_INSTANCE_DIR = OUTPUT_DIR / "per_instance"

PACK_DIR = OUTPUT_DIR / "holdout_audit_v1"
CONTACT_SHEET = OUTPUT_DIR / "holdout_audit_v1_contact_sheet.jpg"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_name(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def main() -> None:
    predictions = read_csv(PREDICTIONS_CSV)
    holdout = read_csv(HOLDOUT_CSV)

    if not PER_INSTANCE_DIR.exists():
        raise FileNotFoundError(PER_INSTANCE_DIR)

    pred_index = {}

    for idx, row in enumerate(predictions, start=1):
        key = (
            row["source_image"].strip(),
            row["garment_id"].strip(),
        )
        pred_index[key] = idx

    PACK_DIR.mkdir(parents=True, exist_ok=True)

    for old in PACK_DIR.glob("*.jpg"):
        old.unlink()

    panels = []
    missing = []

    for holdout_idx, row in enumerate(holdout, start=1):
        key = (
            row["source_image"].strip(),
            row["garment_id"].strip(),
        )

        pred_idx = pred_index.get(key)

        if pred_idx is None:
            missing.append(
                (holdout_idx, row["garment_id"], "not in predictions")
            )
            continue

        garment_id = row["garment_id"].strip()

        src = (
            PER_INSTANCE_DIR
            / f"{pred_idx:03d}_{garment_id}.jpg"
        )

        if not src.exists():
            missing.append(
                (holdout_idx, garment_id, str(src))
            )
            continue

        dst = (
            PACK_DIR
            / f"{holdout_idx:02d}_{safe_name(garment_id)}.jpg"
        )

        shutil.copy2(src, dst)

        with Image.open(dst) as img:
            img = img.convert("RGB")
            img.thumbnail((360, 450))

            panel = Image.new(
                "RGB",
                (400, 560),
                "white",
            )

            draw = ImageDraw.Draw(panel)
            font = ImageFont.load_default()

            title_lines = [
                f"ROW {holdout_idx:02d} | {garment_id}",
                f"final: {row['final_pattern']}",
                (
                    f"gate: {row['stage1_top1']} "
                    f"({float(row['stage1_top1_score']):.3f})"
                ),
                (
                    f"gate margin/H: "
                    f"{float(row['stage1_margin']):.3f} / "
                    f"{float(row['stage1_entropy']):.3f}"
                ),
            ]

            if row.get("stage2_top1", "").strip():
                title_lines.append(
                    (
                        f"subtype: {row['stage2_top1']} "
                        f"({float(row['stage2_top1_score']):.3f})"
                    )
                )
                title_lines.append(
                    (
                        f"subtype margin/H: "
                        f"{float(row['stage2_margin']):.3f} / "
                        f"{float(row['stage2_entropy']):.3f}"
                    )
                )

            y = 8

            for line in title_lines:
                draw.text(
                    (10, y),
                    line,
                    fill="black",
                    font=font,
                )
                y += 19

            image_y = 125
            x = (400 - img.width) // 2

            panel.paste(
                img,
                (x, image_y),
            )

            panels.append(panel)

    if not panels:
        raise RuntimeError("No holdout images were created.")

    cols = 4
    rows = math.ceil(len(panels) / cols)

    sheet = Image.new(
        "RGB",
        (cols * 400, rows * 560),
        "white",
    )

    for idx, panel in enumerate(panels):
        x = (idx % cols) * 400
        y = (idx // cols) * 560
        sheet.paste(panel, (x, y))

    sheet.save(
        CONTACT_SHEET,
        quality=92,
    )

    print("=== PATTERN V2 HOLDOUT AUDIT PACK ===")
    print(f"holdout rows     : {len(holdout)}")
    print(f"images created   : {len(panels)}")
    print(f"missing          : {len(missing)}")
    print(
        "folder           : outputs/prd_attribute_extraction/"
        "pattern_v2_hierarchical/holdout_audit_v1/"
    )
    print(
        "contact sheet    : outputs/prd_attribute_extraction/"
        "pattern_v2_hierarchical/holdout_audit_v1_contact_sheet.jpg"
    )

    if missing:
        print("")
        print("Missing rows:")
        for item in missing:
            print(item)

    print("=====================================")


if __name__ == "__main__":
    main()
