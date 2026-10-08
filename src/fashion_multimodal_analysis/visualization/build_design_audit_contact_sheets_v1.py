"""可视化与审核：图片用于发现共性错误，人工判断需要回填到审核记录。

Build visual contact sheets for the 40-row design-attribute manual audit.

Run from project root:
    python scripts/visualization/build_design_audit_contact_sheets_v1.py

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
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()

DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "design_labels_v1_fixed"
    / "manual_audit_holdout_v1_fixed.csv"
)

DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "garment_instances_gt_pilot_100.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_attribute_extraction"
    / "design_labels_v1_fixed"
    / "audit_contact_sheets"
)


def resolve_path(raw: Any) -> Path:
    """将清单或配置中的相对路径解析到当前项目/数据目录。

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(raw)


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_path(value: str) -> str:
    """规范化 路径。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return value.replace("\\", "/").strip()


def manifest_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    """清单 index。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        返回 out，由函数体中同名变量的计算/收集过程得到。
    """
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
    """加载 garment 裁剪。

    Args:
        row: 记录字段，使用 crop_path。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。
    """
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
    """fit 图像。

    Args:
        image: 本步骤处理的图像对象。
        width: 图像或目标表示的宽度。
        height: 图像或目标表示的高度。

    Returns:
        返回 canvas，由函数体中同名变量的计算/收集过程得到。
    """
    canvas = Image.new("RGB", (width, height), "white")
    img = image.copy()
    img.thumbnail((width, height), Image.Resampling.LANCZOS)

    x = (width - img.width) // 2
    y = (height - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_sheet(
    attribute: str, entries: list[tuple[int, dict[str, str], Image.Image]]
) -> Image.Image:
    """生成 表。

    Args:
        attribute: 属性。
        entries: entries。

    Returns:
        返回 sheet，由函数体中同名变量的计算/收集过程得到。
    """
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
    """解析运行参数并执行本模块的实验入口。

    Raises:
        RuntimeError: 当前运行条件不满足实验要求。
    """
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
        grouped[audit_row["attribute"]].append((csv_row, audit_row, image))

    if missing:
        preview = "\n".join(
            f"row {row}: {key[0]} | {key[1]}" for row, key in missing[:10]
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
