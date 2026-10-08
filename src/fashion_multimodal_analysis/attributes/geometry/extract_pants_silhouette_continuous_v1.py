"""3.1.3 可视属性提取：关注颜色、图案和服装设计/几何属性；一致性不等于人工真值准确率。

PRD 3.1.3 pants silhouette continuous geometry v1.

Purpose
-------
Extend the continuous "style / fit" representation for pants beyond length.

This script does NOT predict hard labels such as skinny / straight / tapered /
wide-leg. Instead it extracts continuous silhouette descriptors from the pants
instance mask.

Why geometry is appropriate here
--------------------------------
For pants, leg width and taper are outer-silhouette properties. They are more
directly measurable from a garment mask than internal neckline structure.

Main continuous descriptors
---------------------------
upper_leg_width_ratio
    Median total foreground width in the upper-leg band, normalized by bbox
    width. "Total foreground width" sums separate leg segments instead of
    measuring the empty gap between them.

knee_width_ratio
    Median total foreground width around the knee region.

hem_width_ratio
    Median total foreground width near the lower hem.

taper_ratio
    hem_width_ratio / upper_leg_width_ratio

    < 1  -> lower legs narrower than upper legs
    ~ 1  -> relatively straight width profile
    > 1  -> lower silhouette expands

lower_width_slope
    Linear slope of normalized total foreground width from upper leg to hem.

lower_leg_fullness
    Mean normalized foreground width across the lower half.

split_leg_fraction
    Fraction of evaluated rows containing two or more foreground segments.
    This is a diagnostic of visible leg separation / pose, not a style label.

Important
---------
- No hard style thresholds are defined.
- No direct GT for leg fit/taper is available in this DeepFashion2 pilot.
- Validate by sorted contact sheets.
- Material/fabric and craftsmanship remain excluded.

Example
-------
python scripts/attributes/geometry/extract_pants_silhouette_continuous_v1.py
--manifest configs/garment_instances_gt_pilot_100.csv

Outputs
-------
reports/prd_attribute_extraction/pants_silhouette_v1/
    pants_silhouette_features.csv
    pants_silhouette_summary.txt
    run_info.txt

outputs/prd_attribute_extraction/pants_silhouette_v1/
    per_instance/
    contact_sheet_sorted_by_taper_ratio.jpg
    contact_sheet_sorted_by_hem_width.jpg
    contact_sheet_sorted_by_lower_fullness.jpg
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()

REPORT_DIR = (
    PROJECT_ROOT / "reports" / "prd_attribute_extraction" / "pants_silhouette_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "prd_attribute_extraction" / "pants_silhouette_v1"
)

PANTS_CATEGORIES = {
    "pants",
    "trousers",
    "jeans",
    "shorts",
}


def resolve_path(raw: Any) -> Path:
    """将清单或配置中的相对路径解析到当前项目/数据目录。

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(raw)


def read_manifest(path: Path) -> list[dict[str, str]]:
    """读取实例清单，不在读取阶段改变样本划分。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
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
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    return rows


def load_crop_and_mask(
    row: dict[str, str],
) -> tuple[Image.Image, np.ndarray]:
    """读取服饰裁剪和对应掩码，保持二者尺寸一致。

    Args:
        row: 记录字段，使用 crop_path, garment_id。

    Returns:
        按顺序返回 crop, mask 等结果。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
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
        raise ValueError(f"Empty mask: {row['garment_id']}")

    return crop, mask


def tight_bbox(
    mask: np.ndarray,
) -> tuple[int, int, int, int]:
    """取得目标前景的紧边界框。

    Args:
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    ys, xs = np.nonzero(mask)

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def tight_crop(mask: np.ndarray) -> np.ndarray:
    """按目标前景的紧边界裁剪图像。

    Args:
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    x1, y1, x2, y2 = tight_bbox(mask)
    return mask[y1:y2, x1:x2]


def contiguous_segments(
    row: np.ndarray,
) -> list[tuple[int, int]]:
    """contiguous segments。

    Args:
        row: 一条实例、预测或审核记录。

    Returns:
        返回 segments，由函数体中同名变量的计算/收集过程得到。
    """
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


def row_total_foreground_width(
    row: np.ndarray,
) -> tuple[float, int]:
    """记录 total foreground 宽度。

    Args:
        row: 一条实例、预测或审核记录。

    Returns:
        按顺序返回 total 等结果。
    """
    segments = contiguous_segments(row)

    if not segments:
        return 0.0, 0

    total = float(sum(x2 - x1 for x1, x2 in segments))

    return total, len(segments)


def band_stats(
    mask: np.ndarray,
    start_frac: float,
    end_frac: float,
) -> tuple[list[float], list[int]]:
    """band stats。

    Args:
        mask: 掩码。
        start_frac: start frac。
        end_frac: end frac。

    Returns:
        按顺序返回 widths, segments_per_row 等结果。
    """
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
    segments_per_row = []

    for y in range(y1, y2):
        total_width, n_segments = row_total_foreground_width(mask[y])

        if n_segments == 0:
            continue

        widths.append(
            total_width
            / max(
                w,
                1,
            )
        )

        segments_per_row.append(n_segments)

    return widths, segments_per_row


def robust_median(
    values: list[float],
) -> float:
    """计算对异常点较稳健的中位数统计。

    Args:
        values: 本步骤处理的数值或附加字段。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
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
    """lower profile。

    Args:
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    h, w = mask.shape

    ys = []
    widths = []
    segment_counts = []

    start_y = int(round(h * 0.30))

    for y in range(start_y, h):
        total_width, n_segments = row_total_foreground_width(mask[y])

        if n_segments == 0:
            continue

        ys.append(
            y
            / max(
                h - 1,
                1,
            )
        )

        widths.append(
            total_width
            / max(
                w,
                1,
            )
        )

        segment_counts.append(n_segments)

    return (
        np.asarray(
            ys,
            dtype=np.float64,
        ),
        np.asarray(
            widths,
            dtype=np.float64,
        ),
        np.asarray(
            segment_counts,
            dtype=np.int64,
        ),
    )


def extract_features(
    mask: np.ndarray,
) -> dict[str, float | str]:
    """按本实验的定义提取服饰区域特征。

    Args:
        mask: 掩码。

    Returns:
        结果字典，主要字段为 upper_leg_width_ratio, knee_width_ratio, hem_width_ratio,
        taper_ratio, lower_width_slope, lower_leg_fullness, split_leg_fraction,
        valid_row_fraction, width_profile_cv, geometry_quality。
    """
    tight = tight_crop(mask)

    h, w = tight.shape

    upper_widths, upper_segments = band_stats(
        tight,
        0.30,
        0.45,
    )

    knee_widths, knee_segments = band_stats(
        tight,
        0.55,
        0.70,
    )

    hem_widths, hem_segments = band_stats(
        tight,
        0.82,
        0.96,
    )

    upper_leg_width_ratio = robust_median(upper_widths)

    knee_width_ratio = robust_median(knee_widths)

    hem_width_ratio = robust_median(hem_widths)

    taper_ratio = hem_width_ratio / max(
        upper_leg_width_ratio,
        1e-6,
    )

    ys, widths, segment_counts = lower_profile(tight)

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

    lower_half_mask = ys >= 0.50

    if lower_half_mask.any():
        lower_leg_fullness = float(widths[lower_half_mask].mean())
    else:
        lower_leg_fullness = 0.0

    if len(segment_counts):
        split_leg_fraction = float((segment_counts >= 2).mean())
    else:
        split_leg_fraction = 0.0

    valid_row_fraction = len(widths) / max(
        h - int(round(h * 0.30)),
        1,
    )

    width_cv = (
        float(
            widths.std()
            / max(
                widths.mean(),
                1e-6,
            )
        )
        if len(widths)
        else 0.0
    )

    if (
        valid_row_fraction >= 0.85
        and upper_leg_width_ratio > 0.08
        and hem_width_ratio > 0.03
        and width_cv <= 0.65
    ):
        geometry_quality = "good"
    elif valid_row_fraction >= 0.65 and upper_leg_width_ratio > 0.05:
        geometry_quality = "usable"
    else:
        geometry_quality = "weak"

    return {
        "upper_leg_width_ratio": float(upper_leg_width_ratio),
        "knee_width_ratio": float(knee_width_ratio),
        "hem_width_ratio": float(hem_width_ratio),
        "taper_ratio": float(taper_ratio),
        "lower_width_slope": float(lower_width_slope),
        "lower_leg_fullness": float(lower_leg_fullness),
        "split_leg_fraction": float(split_leg_fraction),
        "valid_row_fraction": float(valid_row_fraction),
        "width_profile_cv": float(width_cv),
        "geometry_quality": (geometry_quality),
    }


def save_visual(
    crop: Image.Image,
    mask: np.ndarray,
    record: dict,
    save_path: Path,
) -> None:
    """保存样本可视化，供错误案例复核。

    Args:
        crop: 裁剪。
        mask: 掩码。
        record: 记录字段，使用 garment_id, garment_category, upper_leg_width_ratio,
        knee_width_ratio, hem_width_ratio, taper_ratio, lower_leg_fullness。
        save_path: 对应文件的相对路径或当前解析后的路径。
    """
    white = Image.new(
        "RGB",
        crop.size,
        "white",
    )

    mask_img = Image.fromarray(
        (mask.astype(np.uint8) * 255),
        mode="L",
    )

    foreground = Image.composite(
        crop,
        white,
        mask_img,
    )

    draw_fg = ImageDraw.Draw(foreground)

    x1, y1, x2, y2 = tight_bbox(mask)

    bbox_h = max(
        1,
        y2 - y1,
    )

    band_info = [
        (0.375, "upper"),
        (0.625, "knee"),
        (0.89, "hem"),
    ]

    for frac, label in band_info:
        y = y1 + int(round(bbox_h * frac))

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

    panel_h = 245

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
        (f"{record['garment_id']} | " + f"{record['garment_category']}"),
        ("upper_width: " + f"{record['upper_leg_width_ratio']:.3f}"),
        ("knee_width: " + f"{record['knee_width_ratio']:.3f}"),
        ("hem_width: " + f"{record['hem_width_ratio']:.3f}"),
        ("taper_ratio: " + f"{record['taper_ratio']:.3f}"),
        ("lower_fullness: " + f"{record['lower_leg_fullness']:.3f}"),
        ("quality: " + f"{record['geometry_quality']}"),
    ]

    y = 10

    for line in lines:
        draw.text(
            (10, y),
            line,
            fill="black",
            font=font,
        )
        y += 32

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
    """把样本图与说明拼成审核联系表，便于逐例比较。

    Args:
        records: records。
        output_path: 对应文件的相对路径或当前解析后的路径。
    """
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

            tile.paste(
                img,
                (x, y),
            )

            tiles.append(tile)

    cols = 4
    rows = (len(tiles) + cols - 1) // cols

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
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
        fieldnames: fieldnames。
    """
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
    """汇总一个属性/特征的有效样本与结果。

    Args:
        rows: 待处理的逐行记录。
        feature: 特征。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    values = np.asarray(
        [float(row[feature]) for row in rows],
        dtype=np.float64,
    )

    return (
        f"{feature}: "
        + f"mean={values.mean():.4f}, "
        + f"median={np.median(values):.4f}, "
        + f"min={values.min():.4f}, "
        + f"max={values.max():.4f}"
    )


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        ValueError: No pants instances found.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)

    all_rows = read_manifest(manifest_path)

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
            in PANTS_CATEGORIES
        )
    ]

    if not rows:
        raise ValueError("No pants instances found.")

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

    print(f"pants cases    : {len(rows)}")

    print("features       : leg width + taper continuous geometry")

    output_rows = []

    taper_visuals = []
    hem_visuals = []
    fullness_visuals = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        crop, mask = load_crop_and_mask(row)

        features = extract_features(mask)

        record = {
            "source_image": (row["source_image"]),
            "garment_id": (row["garment_id"]),
            "garment_category": (row["garment_category"].strip().lower()),
            **features,
        }

        output_rows.append(record)

        print(
            f"[{index:03d}/{len(rows):03d}] "
            + f"{record['garment_id']:<30} "
            + f"taper={record['taper_ratio']:.3f} "
            + f"hem={record['hem_width_ratio']:.3f} "
            + f"fullness={record['lower_leg_fullness']:.3f} "
            + f"q={record['geometry_quality']}"
        )

        visual_path = per_instance_dir / (
            f"{index:03d}_" + f"{record['garment_id']}.jpg"
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
                    record["taper_ratio"],
                    visual_path,
                )
            )

            hem_visuals.append(
                (
                    record["hem_width_ratio"],
                    visual_path,
                )
            )

            fullness_visuals.append(
                (
                    record["lower_leg_fullness"],
                    visual_path,
                )
            )

    fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "upper_leg_width_ratio",
        "knee_width_ratio",
        "hem_width_ratio",
        "taper_ratio",
        "lower_width_slope",
        "lower_leg_fullness",
        "split_leg_fraction",
        "valid_row_fraction",
        "width_profile_cv",
        "geometry_quality",
    ]

    write_csv(
        REPORT_DIR / "pants_silhouette_features.csv",
        output_rows,
        fields,
    )

    quality_counts = {
        "good": 0,
        "usable": 0,
        "weak": 0,
    }

    for row in output_rows:
        quality_counts[row["geometry_quality"]] += 1

    summary_lines = [
        "PRD 3.1.3 pants silhouette continuous geometry v1",
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
            "upper_leg_width_ratio",
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
            "taper_ratio",
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
        "- taper_ratio compares lower hem width with upper-leg width.",
        "- lower_leg_fullness summarizes how much of the bbox width is occupied by garment material in the lower half.",
        "- total foreground width sums separate leg segments and does not count the empty gap between legs.",
        "- These are continuous silhouette descriptors, not skinny/straight/wide-leg labels.",
        "- No business thresholds are set.",
        "- DeepFashion2 has no direct fit/taper GT for this pilot; validate qualitatively.",
        "- Material/fabric and craftsmanship remain excluded.",
    ]

    (REPORT_DIR / "pants_silhouette_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 pants silhouette continuous geometry v1\n"
        + f"manifest={args.manifest}\n"
        + f"instances={len(output_rows)}\n"
        + "input=garment_instance_mask\n"
        + "output_features=upper_leg_width_ratio,knee_width_ratio,hem_width_ratio,taper_ratio,lower_leg_fullness\n"
        + "hard_classification=false\n"
        + "gt_available=false\n"
        + "material_and_craftsmanship=excluded\n"
    )

    (REPORT_DIR / "run_info.txt").write_text(
        run_info,
        encoding="utf-8",
    )

    make_contact_sheet(
        taper_visuals,
        OUTPUT_DIR / "contact_sheet_sorted_by_taper_ratio.jpg",
    )

    make_contact_sheet(
        hem_visuals,
        OUTPUT_DIR / "contact_sheet_sorted_by_hem_width.jpg",
    )

    make_contact_sheet(
        fullness_visuals,
        OUTPUT_DIR / "contact_sheet_sorted_by_lower_fullness.jpg",
    )

    print("\n=== FINISHED ===")

    print(
        "FEATURES : "
        + "reports/prd_attribute_extraction/pants_silhouette_v1/"
        + "pants_silhouette_features.csv"
    )

    print(
        "SUMMARY  : "
        + "reports/prd_attribute_extraction/pants_silhouette_v1/"
        + "pants_silhouette_summary.txt"
    )

    print(
        "TAPER    : "
        + "outputs/prd_attribute_extraction/pants_silhouette_v1/"
        + "contact_sheet_sorted_by_taper_ratio.jpg"
    )

    print(
        "HEM      : "
        + "outputs/prd_attribute_extraction/pants_silhouette_v1/"
        + "contact_sheet_sorted_by_hem_width.jpg"
    )

    print(
        "FULLNESS : "
        + "outputs/prd_attribute_extraction/pants_silhouette_v1/"
        + "contact_sheet_sorted_by_lower_fullness.jpg"
    )

    print("================")


if __name__ == "__main__":
    main()
