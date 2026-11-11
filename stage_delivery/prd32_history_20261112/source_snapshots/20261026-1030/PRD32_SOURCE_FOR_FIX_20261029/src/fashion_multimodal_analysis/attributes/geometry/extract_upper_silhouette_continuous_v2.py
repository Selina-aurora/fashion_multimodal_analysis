"""3.1.3 可视属性提取：关注颜色、图案和服装设计/几何属性；一致性不等于人工真值准确率。

PRD 3.1.3 upper-garment silhouette continuous features v2.

Why v2
------
v1 used the FULL foreground span of each horizontal row. For upper garments,
that can let sleeves/arms inflate the "chest width", which then makes
waist/chest and hem/chest ratios artificially small.

v2 therefore measures the CENTRAL TORSO SEGMENT instead of the full row span.

Core idea
---------
For each mask row:
1. find all contiguous foreground segments;
2. estimate a robust torso center from lower-middle garment rows;
3. choose the segment closest to that torso center;
4. measure chest / waist / hem from those torso-centered segments.

This is still a geometry pilot. No hard style labels or business thresholds.

Main outputs
------------
garment_length_proxy
    Image-space vertical elongation proxy. Not physical garment length.

torso_chest_width_ratio
torso_waist_width_ratio
torso_hem_width_ratio

waist_to_chest_ratio
    < 1 -> waist narrower than chest
    ~ 1 -> straighter torso silhouette
    > 1 -> waist at least as wide as chest

hem_to_chest_ratio
    < 1 -> hem narrows relative to chest
    ~ 1 -> straight lower silhouette
    > 1 -> hem expands relative to chest

lower_taper_slope
    Slope of torso-centered width from waist to hem.

arm_contamination_ratio
    Diagnostic: how often the upper band is much wider than the estimated
    torso reference. High values suggest sleeves/pose still contaminate shape.

Example
-------
python scripts/attributes/geometry/extract_upper_silhouette_continuous_v2.py
--manifest configs/garment_instances_gt_pilot_100.csv
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
    PROJECT_ROOT / "reports" / "prd_attribute_extraction" / "upper_silhouette_v2"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "prd_attribute_extraction" / "upper_silhouette_v2"
)

UPPER_CATEGORIES = {
    "top",
    "outerwear",
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


def tight_crop(
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """按目标前景的紧边界裁剪图像。

    Args:
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    x1, y1, x2, y2 = tight_bbox(mask)

    return (
        mask[y1:y2, x1:x2],
        (x1, y1, x2, y2),
    )


def contiguous_segments(
    row: np.ndarray,
) -> list[tuple[int, int]]:
    """Return foreground runs as [start, end) segments.

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


def choose_center_segment(
    row: np.ndarray,
    center_x: float,
) -> tuple[int, int] | None:
    """选择 center segment。

    Args:
        row: 一条实例、预测或审核记录。
        center_x: center x。

    Returns:
        按顺序返回 x1, x2 等结果。
    """
    segments = contiguous_segments(row)

    if not segments:
        return None

    # Prefer a segment containing the current torso center.
    for x1, x2 in segments:
        if x1 <= center_x < x2:
            return (x1, x2)

    # Otherwise choose the segment whose center is nearest.
    return min(
        segments,
        key=lambda seg: abs((seg[0] + seg[1]) / 2.0 - center_x),
    )


def estimate_torso_center(
    mask: np.ndarray,
) -> float:
    """Estimate torso center from lower-middle rows using iterative refinement.

    Args:
        mask: 掩码。

    Returns:
        返回 center，由函数体中同名变量的计算/收集过程得到。
    """
    h, w = mask.shape
    center = (w - 1) / 2.0

    y1 = int(round(h * 0.50))

    y2 = min(
        h,
        max(
            y1 + 1,
            int(round(h * 0.88)),
        ),
    )

    for _ in range(3):
        centers = []

        for y in range(y1, y2):
            seg = choose_center_segment(
                mask[y],
                center,
            )

            if seg is None:
                continue

            x1, x2 = seg

            centers.append((x1 + x2 - 1) / 2.0)

        if not centers:
            break

        center = float(
            np.median(
                np.asarray(
                    centers,
                    dtype=np.float64,
                )
            )
        )

    return center


def torso_width_at_row(
    mask: np.ndarray,
    y: int,
    center_x: float,
) -> float | None:
    """torso 宽度 at 记录。

    Args:
        mask: 掩码。
        y: 本步骤使用的原始值，转换/筛选规则见函数体。
        center_x: center x。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    seg = choose_center_segment(
        mask[y],
        center_x,
    )

    if seg is None:
        return None

    x1, x2 = seg

    return float(x2 - x1)


def torso_band_widths(
    mask: np.ndarray,
    center_x: float,
    start_frac: float,
    end_frac: float,
) -> list[float]:
    """torso band widths。

    Args:
        mask: 掩码。
        center_x: center x。
        start_frac: start frac。
        end_frac: end frac。

    Returns:
        返回 values，由函数体中同名变量的计算/收集过程得到。
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

    values = []

    for y in range(y1, y2):
        width = torso_width_at_row(
            mask,
            y,
            center_x,
        )

        if width is not None:
            values.append(
                width
                / max(
                    w,
                    1,
                )
            )

    return values


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


def torso_profile(
    mask: np.ndarray,
    center_x: float,
) -> tuple[np.ndarray, np.ndarray]:
    """torso profile。

    Args:
        mask: 掩码。
        center_x: center x。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    h, w = mask.shape

    ys = []
    widths = []

    for y in range(h):
        width = torso_width_at_row(
            mask,
            y,
            center_x,
        )

        if width is None:
            continue

        ys.append(
            y
            / max(
                h - 1,
                1,
            )
        )

        widths.append(
            width
            / max(
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


def arm_contamination_ratio(
    mask: np.ndarray,
    center_x: float,
    torso_reference: float,
) -> float:
    """Fraction of upper rows whose full span is much wider than torso core.

    Args:
        mask: 掩码。
        center_x: center x。
        torso_reference: torso reference。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    h, w = mask.shape

    y1 = int(round(h * 0.15))

    y2 = min(
        h,
        max(
            y1 + 1,
            int(round(h * 0.48)),
        ),
    )

    checked = 0
    contaminated = 0

    for y in range(y1, y2):
        xs = np.flatnonzero(mask[y])

        if len(xs) == 0:
            continue

        full_width = (int(xs.max()) - int(xs.min()) + 1) / max(
            w,
            1,
        )

        torso_width = torso_width_at_row(
            mask,
            y,
            center_x,
        )

        if torso_width is None:
            continue

        torso_width = torso_width / max(
            w,
            1,
        )

        checked += 1

        reference = max(
            torso_reference,
            torso_width,
            1e-6,
        )

        if full_width > (1.35 * reference):
            contaminated += 1

    if checked == 0:
        return 0.0

    return contaminated / checked


def extract_features(
    mask: np.ndarray,
) -> dict[str, float | str]:
    """按本实验的定义提取服饰区域特征。

    Args:
        mask: 掩码。

    Returns:
        结果字典，主要字段为 garment_length_proxy, torso_center_ratio, torso_chest_width_ratio,
        torso_waist_width_ratio, torso_hem_width_ratio, waist_to_chest_ratio,
        hem_to_chest_ratio, lower_taper_slope, width_profile_cv, valid_row_fraction。
    """
    tight, _ = tight_crop(mask)

    h, w = tight.shape

    center_x = estimate_torso_center(tight)

    garment_length_proxy = h / max(
        h + w,
        1,
    )

    # Lower chest band is intentional: it reduces sleeve/arm intrusion.
    chest_width_ratio = robust_median(
        torso_band_widths(
            tight,
            center_x,
            0.38,
            0.50,
        )
    )

    waist_width_ratio = robust_median(
        torso_band_widths(
            tight,
            center_x,
            0.55,
            0.69,
        )
    )

    hem_width_ratio = robust_median(
        torso_band_widths(
            tight,
            center_x,
            0.82,
            0.95,
        )
    )

    waist_to_chest_ratio = waist_width_ratio / max(
        chest_width_ratio,
        1e-6,
    )

    hem_to_chest_ratio = hem_width_ratio / max(
        chest_width_ratio,
        1e-6,
    )

    ys, widths = torso_profile(
        tight,
        center_x,
    )

    lower_mask = ys >= 0.48

    if lower_mask.sum() >= 3:
        lower_taper_slope = float(
            np.polyfit(
                ys[lower_mask],
                widths[lower_mask],
                deg=1,
            )[0]
        )
    else:
        lower_taper_slope = 0.0

    mean_width = float(widths.mean()) if len(widths) else 0.0

    width_profile_cv = (
        float(
            widths.std()
            / max(
                mean_width,
                1e-6,
            )
        )
        if len(widths)
        else 0.0
    )

    valid_row_fraction = len(widths) / max(
        h,
        1,
    )

    torso_reference = robust_median(
        torso_band_widths(
            tight,
            center_x,
            0.45,
            0.80,
        )
    )

    arm_contam = arm_contamination_ratio(
        tight,
        center_x,
        torso_reference,
    )

    if (
        valid_row_fraction >= 0.85
        and chest_width_ratio > 0.08
        and arm_contam <= 0.30
        and width_profile_cv <= 0.45
    ):
        geometry_quality = "good"
    elif valid_row_fraction >= 0.65 and chest_width_ratio > 0.05 and arm_contam <= 0.60:
        geometry_quality = "usable"
    else:
        geometry_quality = "weak"

    return {
        "garment_length_proxy": float(garment_length_proxy),
        "torso_center_ratio": float(
            center_x
            / max(
                w - 1,
                1,
            )
        ),
        "torso_chest_width_ratio": float(chest_width_ratio),
        "torso_waist_width_ratio": float(waist_width_ratio),
        "torso_hem_width_ratio": float(hem_width_ratio),
        "waist_to_chest_ratio": float(waist_to_chest_ratio),
        "hem_to_chest_ratio": float(hem_to_chest_ratio),
        "lower_taper_slope": float(lower_taper_slope),
        "width_profile_cv": float(width_profile_cv),
        "valid_row_fraction": float(valid_row_fraction),
        "arm_contamination_ratio": float(arm_contam),
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
        record: 记录字段，使用 torso_center_ratio, garment_id, garment_category,
        garment_length_proxy, waist_to_chest_ratio, hem_to_chest_ratio,
        lower_taper_slope。
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

    bbox_w = max(
        1,
        x2 - x1,
    )

    bbox_h = max(
        1,
        y2 - y1,
    )

    center_x = x1 + record["torso_center_ratio"] * max(
        bbox_w - 1,
        1,
    )

    draw_fg.line(
        (
            int(round(center_x)),
            y1,
            int(round(center_x)),
            y2,
        ),
        fill="black",
        width=1,
    )

    band_info = [
        (
            0.44,
            "chest",
        ),
        (
            0.62,
            "waist",
        ),
        (
            0.885,
            "hem",
        ),
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

    panel_h = 270

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
        ("length_proxy: " + f"{record['garment_length_proxy']:.3f}"),
        ("waist/chest: " + f"{record['waist_to_chest_ratio']:.3f}"),
        ("hem/chest: " + f"{record['hem_to_chest_ratio']:.3f}"),
        ("lower_slope: " + f"{record['lower_taper_slope']:.3f}"),
        ("arm_contam: " + f"{record['arm_contamination_ratio']:.3f}"),
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
        y += 34

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

            img.thumbnail((390, 550))

            tile = Image.new(
                "RGB",
                (400, 560),
                "white",
            )

            x = (400 - img.width) // 2

            y = (560 - img.height) // 2

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
            rows * 560,
        ),
        "white",
    )

    for idx, tile in enumerate(tiles):
        x = (idx % cols) * 400

        y = (idx // cols) * 560

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
        ValueError: No top/outerwear instances found.
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
            in UPPER_CATEGORIES
        )
    ]

    if not rows:
        raise ValueError("No top/outerwear instances found.")

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

    print(f"upper cases    : {len(rows)}")

    print("mode           : torso-centered silhouette geometry v2")

    output_rows = []

    length_visuals = []
    waist_visuals = []
    hem_visuals = []

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
            + f"{record['garment_id']:<35} "
            + f"waist/chest={record['waist_to_chest_ratio']:.3f} "
            + f"hem/chest={record['hem_to_chest_ratio']:.3f} "
            + f"arm={record['arm_contamination_ratio']:.2f} "
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
            length_visuals.append(
                (
                    record["garment_length_proxy"],
                    visual_path,
                )
            )

            waist_visuals.append(
                (
                    record["waist_to_chest_ratio"],
                    visual_path,
                )
            )

            hem_visuals.append(
                (
                    record["hem_to_chest_ratio"],
                    visual_path,
                )
            )

    fields = [
        "source_image",
        "garment_id",
        "garment_category",
        "garment_length_proxy",
        "torso_center_ratio",
        "torso_chest_width_ratio",
        "torso_waist_width_ratio",
        "torso_hem_width_ratio",
        "waist_to_chest_ratio",
        "hem_to_chest_ratio",
        "lower_taper_slope",
        "width_profile_cv",
        "valid_row_fraction",
        "arm_contamination_ratio",
        "geometry_quality",
    ]

    write_csv(
        REPORT_DIR / "upper_silhouette_features_v2.csv",
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
        "PRD 3.1.3 upper silhouette continuous features v2",
        "================================================",
        f"instances={len(output_rows)}",
        "method=torso-centered connected-segment geometry",
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
        summarize_feature(
            output_rows,
            "arm_contamination_ratio",
        ),
        "",
        "Interpretation",
        "--------------",
        "- v2 measures the torso-centered connected component per row rather than the full foreground span.",
        "- This specifically reduces sleeve/arm inflation of the chest-width denominator.",
        "- waist_to_chest_ratio and hem_to_chest_ratio are continuous silhouette descriptors, not style labels.",
        "- garment_length_proxy remains image-space only; it is not body-relative physical garment length.",
        "- No business thresholds are set.",
        "- DeepFashion2 has no direct GT for these fit/shape dimensions here, so validate qualitatively.",
        "- Material/fabric and craftsmanship remain excluded.",
    ]

    (REPORT_DIR / "upper_silhouette_summary_v2.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    run_info = (
        "PRD 3.1.3 upper silhouette continuous features v2\n"
        + f"manifest={args.manifest}\n"
        + f"instances={len(output_rows)}\n"
        + "garment_categories=top,outerwear\n"
        + "input=garment_instance_mask\n"
        + "method=torso_centered_connected_segment_widths\n"
        + "output_features=garment_length_proxy,waist_to_chest_ratio,hem_to_chest_ratio,lower_taper_slope\n"
        + "hard_classification=false\n"
        + "gt_available=false\n"
        + "material_and_craftsmanship=excluded\n"
    )

    (REPORT_DIR / "run_info.txt").write_text(
        run_info,
        encoding="utf-8",
    )

    make_contact_sheet(
        length_visuals,
        OUTPUT_DIR / "contact_sheet_sorted_by_length_proxy.jpg",
    )

    make_contact_sheet(
        waist_visuals,
        OUTPUT_DIR / "contact_sheet_sorted_by_waist_to_chest.jpg",
    )

    make_contact_sheet(
        hem_visuals,
        OUTPUT_DIR / "contact_sheet_sorted_by_hem_to_chest.jpg",
    )

    print("\n=== FINISHED ===")

    print(
        "FEATURES : "
        + "reports/prd_attribute_extraction/upper_silhouette_v2/"
        + "upper_silhouette_features_v2.csv"
    )

    print(
        "SUMMARY  : "
        + "reports/prd_attribute_extraction/upper_silhouette_v2/"
        + "upper_silhouette_summary_v2.txt"
    )

    print(
        "WAIST    : "
        + "outputs/prd_attribute_extraction/upper_silhouette_v2/"
        + "contact_sheet_sorted_by_waist_to_chest.jpg"
    )

    print(
        "HEM      : "
        + "outputs/prd_attribute_extraction/upper_silhouette_v2/"
        + "contact_sheet_sorted_by_hem_to_chest.jpg"
    )

    print("================")


if __name__ == "__main__":
    main()
