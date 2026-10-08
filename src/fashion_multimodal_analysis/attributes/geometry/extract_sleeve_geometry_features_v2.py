"""3.1.3 可视属性提取：关注颜色、图案和服装设计/几何属性；一致性不等于人工真值准确率。

PRD 3.1.3 sleeve continuous geometry feature pilot v2.

Why this exists
---------------
The v1 CLIP-anchor scalar preserved the broad order
    sleeveless < short < long
but compressed most samples into the middle.

For v2 we do NOT classify sleeves. We extract continuous geometric descriptors
directly from each garment mask, then inspect which descriptors actually track
sleeve length.

This is intentionally a diagnostic feature-extraction stage. It does not yet
force one final sleeve_length_ratio.

Features
--------
mask_aspect_ratio
    garment bbox width / garment bbox height

upper_width_ratio
    median silhouette width in y=10~45% / garment bbox width

mid_width_ratio
    median silhouette width in y=45~70% / garment bbox width

lower_width_ratio
    median silhouette width in y=70~90% / garment bbox width

upper_to_lower_width
    upper_width_ratio / lower_width_ratio

side_extension_ratio
    fraction of upper/middle garment pixels extending outside an estimated
    torso core derived from the lower-middle silhouette

upper_side_reach
    normalized maximum lateral reach outside the torso core

upper_area_ratio
    fraction of garment-mask area lying in the top 60% of the garment bbox

DeepFashion2 fine category names are used ONLY for weak ordinal diagnostics:
vest/sling -> sleeveless
short_sleeve_* -> short
long_sleeve_* -> long

Example
-------
python scripts/attributes/geometry/extract_sleeve_geometry_features_v2.py     --manifest
configs/garment_instances_gt_pilot_20.csv

Outputs
-------
reports/prd_attribute_extraction/continuous_v2_geometry/
    sleeve_geometry_features.csv
    feature_group_summary.csv
    feature_order_diagnostics.csv
    run_info.txt

outputs/prd_attribute_extraction/continuous_v2_geometry/
    per_instance/
    contact_sheet.jpg
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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
    PROJECT_ROOT / "reports" / "prd_attribute_extraction" / "continuous_v2_geometry"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "prd_attribute_extraction" / "continuous_v2_geometry"
)

UPPER_BODY_CATEGORIES = {"top", "outerwear", "dress"}

FEATURE_NAMES = [
    "mask_aspect_ratio",
    "upper_width_ratio",
    "mid_width_ratio",
    "lower_width_ratio",
    "upper_to_lower_width",
    "side_extension_ratio",
    "upper_side_reach",
    "upper_area_ratio",
]


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


def deepfashion2_gt_group(garment_id: str) -> str:
    """DeepFashion2 gt group。

    Args:
        garment_id: garment ID。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    gid = garment_id.lower()

    if "vest" in gid or "sling" in gid:
        return "sleeveless"
    if "short_sleeve" in gid:
        return "short"
    if "long_sleeve" in gid:
        return "long"
    return "unknown"


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
        # Conservative fallback: full crop foreground.
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
        raise ValueError(f"Empty mask for garment: {row['garment_id']}")

    return crop, mask


def tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
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


def crop_mask_to_bbox(
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """裁剪 掩码 转换 边界框。

    Args:
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    x1, y1, x2, y2 = tight_bbox(mask)
    return mask[y1:y2, x1:x2], (x1, y1, x2, y2)


def row_span(mask_row: np.ndarray) -> tuple[int, int] | None:
    """记录 span。

    Args:
        mask_row: 掩码 记录。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    xs = np.flatnonzero(mask_row)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(xs.max()) + 1


def band_row_widths(
    mask: np.ndarray,
    y_start_frac: float,
    y_end_frac: float,
) -> list[float]:
    """band 记录 widths。

    Args:
        mask: 掩码。
        y_start_frac: y start frac。
        y_end_frac: y end frac。

    Returns:
        返回 widths，由函数体中同名变量的计算/收集过程得到。
    """
    h, w = mask.shape

    y1 = max(
        0,
        min(
            h - 1,
            int(round(h * y_start_frac)),
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            h,
            int(round(h * y_end_frac)),
        ),
    )

    widths = []

    for y in range(y1, y2):
        span = row_span(mask[y])
        if span is None:
            continue

        x1, x2 = span
        widths.append((x2 - x1) / max(w, 1))

    return widths


def robust_median(values: list[float]) -> float:
    """计算对异常点较稳健的中位数统计。

    Args:
        values: 本步骤处理的数值或附加字段。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def estimate_torso_core(
    mask: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate torso center and half-width from lower-middle silhouette.

    Returns:
        center_x_normalized,
        half_width_normalized,
        full_width_normalized

    Args:
        mask: 掩码。
    """
    h, w = mask.shape

    y1 = int(round(h * 0.55))
    y2 = max(
        y1 + 1,
        int(round(h * 0.85)),
    )
    y2 = min(y2, h)

    centers = []
    widths = []

    for y in range(y1, y2):
        span = row_span(mask[y])
        if span is None:
            continue

        x1, x2 = span
        centers.append(((x1 + x2) / 2.0) / max(w, 1))
        widths.append((x2 - x1) / max(w, 1))

    if not centers or not widths:
        return 0.5, 0.20, 0.40

    center = float(np.median(centers))
    full_width = float(np.median(widths))

    # Slightly shrink the estimated torso width so sleeve-side pixels are not
    # swallowed by the torso core.
    half_width = max(
        0.05,
        full_width * 0.42,
    )

    return center, half_width, full_width


def extract_features(mask: np.ndarray) -> dict[str, float]:
    """按本实验的定义提取服饰区域特征。

    Args:
        mask: 掩码。

    Returns:
        结果字典，主要字段为 mask_aspect_ratio, upper_width_ratio, mid_width_ratio,
        lower_width_ratio, upper_to_lower_width, side_extension_ratio, upper_side_reach,
        upper_area_ratio。
    """
    tight, _ = crop_mask_to_bbox(mask)
    h, w = tight.shape

    mask_aspect_ratio = w / max(h, 1)

    upper_width_ratio = robust_median(
        band_row_widths(
            tight,
            0.10,
            0.45,
        )
    )

    mid_width_ratio = robust_median(
        band_row_widths(
            tight,
            0.45,
            0.70,
        )
    )

    lower_width_ratio = robust_median(
        band_row_widths(
            tight,
            0.70,
            0.90,
        )
    )

    upper_to_lower_width = upper_width_ratio / max(lower_width_ratio, 1e-6)

    center, torso_half_width, _ = estimate_torso_core(tight)

    x_coords = np.arange(w, dtype=np.float64) / max(w, 1)

    left_bound = center - torso_half_width
    right_bound = center + torso_half_width

    outside_torso = (x_coords < left_bound) | (x_coords > right_bound)

    upper_middle_limit = max(
        1,
        int(round(h * 0.72)),
    )

    region = tight[:upper_middle_limit, :]

    outside_pixels = region & outside_torso[None, :]

    side_extension_ratio = float(outside_pixels.sum()) / max(float(region.sum()), 1.0)

    # Maximum lateral reach beyond torso core, normalized by image width.
    ys, xs = np.nonzero(region)

    if len(xs) == 0:
        upper_side_reach = 0.0
    else:
        xs_norm = xs / max(w, 1)

        left_reach = max(
            0.0,
            left_bound - float(xs_norm.min()),
        )

        right_reach = max(
            0.0,
            float(xs_norm.max()) - right_bound,
        )

        upper_side_reach = left_reach + right_reach

    upper_area_limit = max(
        1,
        int(round(h * 0.60)),
    )

    upper_area_ratio = float(tight[:upper_area_limit].sum()) / max(
        float(tight.sum()), 1.0
    )

    return {
        "mask_aspect_ratio": float(mask_aspect_ratio),
        "upper_width_ratio": float(upper_width_ratio),
        "mid_width_ratio": float(mid_width_ratio),
        "lower_width_ratio": float(lower_width_ratio),
        "upper_to_lower_width": float(upper_to_lower_width),
        "side_extension_ratio": float(side_extension_ratio),
        "upper_side_reach": float(upper_side_reach),
        "upper_area_ratio": float(upper_area_ratio),
    }


def mean(values: list[float]) -> float:
    """计算本组数值的平均值，并按本模块策略处理空组。

    Args:
        values: 本步骤处理的数值或附加字段。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return sum(values) / len(values) if values else float("nan")


def median(values: list[float]) -> float:
    """计算本组数值的中位数，并处理空组。

    Args:
        values: 本步骤处理的数值或附加字段。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if not values:
        return float("nan")
    return float(
        np.median(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def pairwise_order_accuracy(
    lower_values: list[float],
    higher_values: list[float],
    higher_should_be_larger: bool,
) -> tuple[int, int, float]:
    """pairwise order 准确率。

    Args:
        lower_values: lower values。
        higher_values: higher values。
        higher_should_be_larger: higher should be larger。

    Returns:
        按顺序返回 correct, total 等结果。
    """
    total = 0
    correct = 0

    for low in lower_values:
        for high in higher_values:
            total += 1

            if higher_should_be_larger:
                ok = high > low
            else:
                ok = high < low

            if ok:
                correct += 1

    return (
        correct,
        total,
        correct / total if total else float("nan"),
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
        record: 记录字段，使用 garment_id, gt_group, upper_to_lower_width,
        side_extension_ratio, upper_side_reach, mask_aspect_ratio, upper_area_ratio。
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

    panel_h = 225

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
        (f"{record['garment_id']} | " + f"GT={record['gt_group']}"),
        ("upper/lower width: " + f"{record['upper_to_lower_width']:.3f}"),
        ("side extension: " + f"{record['side_extension_ratio']:.3f}"),
        ("upper side reach: " + f"{record['upper_side_reach']:.3f}"),
        ("aspect ratio: " + f"{record['mask_aspect_ratio']:.3f}"),
        ("upper area ratio: " + f"{record['upper_area_ratio']:.3f}"),
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
    paths: list[Path],
    output_path: Path,
) -> None:
    """把样本图与说明拼成审核联系表，便于逐例比较。

    Args:
        paths: 路径。
        output_path: 对应文件的相对路径或当前解析后的路径。
    """
    if not paths:
        return

    tiles = []

    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((380, 520))

            tile = Image.new(
                "RGB",
                (390, 530),
                "white",
            )

            x = (390 - img.width) // 2

            y = (530 - img.height) // 2

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
            cols * 390,
            rows * 530,
        ),
        "white",
    )

    for idx, tile in enumerate(tiles):
        x = (idx % cols) * 390

        y = (idx // cols) * 530

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


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        ValueError: No upper-body garment instances found.
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
            in UPPER_BODY_CATEGORIES
        )
    ]

    if not rows:
        raise ValueError("No upper-body garment instances found.")

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
    print(f"upper garments : {len(rows)}")
    print("mode           : continuous geometry features from garment mask")

    prediction_rows = []
    visual_paths = []

    grouped: dict[
        str,
        dict[str, list[float]],
    ] = defaultdict(lambda: defaultdict(list))

    for index, row in enumerate(
        rows,
        start=1,
    ):
        garment_id = row["garment_id"].strip()

        gt_group = deepfashion2_gt_group(garment_id)

        crop, mask = load_crop_and_mask(row)

        features = extract_features(mask)

        record = {
            "source_image": (row["source_image"]),
            "garment_id": garment_id,
            "garment_category": (row["garment_category"]),
            "gt_group": gt_group,
            **features,
        }

        prediction_rows.append(record)

        if gt_group != "unknown":
            for feature_name in FEATURE_NAMES:
                grouped[gt_group][feature_name].append(features[feature_name])

        print(
            f"[{index:02d}/{len(rows):02d}] "
            + f"{garment_id:<35} "
            + f"GT={gt_group:<10} "
            + f"side_ext={features['side_extension_ratio']:.3f} "
            + f"reach={features['upper_side_reach']:.3f}"
        )

        visual_path = per_instance_dir / f"{index:02d}_{garment_id}.jpg"

        save_visual(
            crop,
            mask,
            record,
            visual_path,
        )

        visual_paths.append(visual_path)

    write_csv(
        REPORT_DIR / "sleeve_geometry_features.csv",
        prediction_rows,
        [
            "source_image",
            "garment_id",
            "garment_category",
            "gt_group",
            *FEATURE_NAMES,
        ],
    )

    group_summary_rows = []

    for feature_name in FEATURE_NAMES:
        for group_name in [
            "sleeveless",
            "short",
            "long",
        ]:
            values = grouped[group_name].get(
                feature_name,
                [],
            )

            if not values:
                continue

            group_summary_rows.append(
                {
                    "feature": (feature_name),
                    "gt_group": (group_name),
                    "n": len(values),
                    "mean": mean(values),
                    "median": median(values),
                    "min": min(values),
                    "max": max(values),
                }
            )

    write_csv(
        REPORT_DIR / "feature_group_summary.csv",
        group_summary_rows,
        [
            "feature",
            "gt_group",
            "n",
            "mean",
            "median",
            "min",
            "max",
        ],
    )

    diagnostic_rows = []

    for feature_name in FEATURE_NAMES:
        means = {}

        for group_name in [
            "sleeveless",
            "short",
            "long",
        ]:
            values = grouped[group_name].get(
                feature_name,
                [],
            )

            if values:
                means[group_name] = mean(values)

        if not all(
            key in means
            for key in [
                "sleeveless",
                "short",
                "long",
            ]
        ):
            continue

        increasing = means["sleeveless"] < means["short"] < means["long"]

        decreasing = means["sleeveless"] > means["short"] > means["long"]

        if increasing:
            direction = "increasing"
            higher_should_be_larger = True
        elif decreasing:
            direction = "decreasing"
            higher_should_be_larger = False
        else:
            direction = "non_monotonic"

            # For diagnostics only, choose the endpoint direction.
            higher_should_be_larger = means["long"] > means["sleeveless"]

        sl = grouped["sleeveless"][feature_name]

        sh = grouped["short"][feature_name]

        lo = grouped["long"][feature_name]

        _, n1, acc1 = pairwise_order_accuracy(
            sl,
            sh,
            higher_should_be_larger,
        )

        _, n2, acc2 = pairwise_order_accuracy(
            sh,
            lo,
            higher_should_be_larger,
        )

        _, n3, acc3 = pairwise_order_accuracy(
            sl,
            lo,
            higher_should_be_larger,
        )

        diagnostic_rows.append(
            {
                "feature": (feature_name),
                "mean_sleeveless": (means["sleeveless"]),
                "mean_short": (means["short"]),
                "mean_long": (means["long"]),
                "mean_order": (direction),
                "pair_acc_sleeveless_short": (acc1),
                "pairs_sleeveless_short": (n1),
                "pair_acc_short_long": (acc2),
                "pairs_short_long": (n2),
                "pair_acc_sleeveless_long": (acc3),
                "pairs_sleeveless_long": (n3),
            }
        )

    # Sort by endpoint separation quality for convenience.
    diagnostic_rows.sort(
        key=lambda row: (
            row["pair_acc_sleeveless_long"],
            row["pair_acc_short_long"],
        ),
        reverse=True,
    )

    write_csv(
        REPORT_DIR / "feature_order_diagnostics.csv",
        diagnostic_rows,
        [
            "feature",
            "mean_sleeveless",
            "mean_short",
            "mean_long",
            "mean_order",
            "pair_acc_sleeveless_short",
            "pairs_sleeveless_short",
            "pair_acc_short_long",
            "pairs_short_long",
            "pair_acc_sleeveless_long",
            "pairs_sleeveless_long",
        ],
    )

    run_info = (
        "PRD 3.1.3 sleeve continuous geometry pilot v2\n"
        + f"manifest={args.manifest}\n"
        + f"upper_body_instances={len(rows)}\n"
        + "input=garment_instance_mask\n"
        + "output=continuous_geometry_feature_vector\n"
        + "hard_classification=false\n"
        + "gt_source=DeepFashion2_fine_category_name_for_diagnostic_only\n"
        + "material_and_craftsmanship=excluded\n"
    )

    (REPORT_DIR / "run_info.txt").write_text(
        run_info,
        encoding="utf-8",
    )

    make_contact_sheet(
        visual_paths,
        OUTPUT_DIR / "contact_sheet.jpg",
    )

    print("\n=== FINISHED ===")
    print(
        "FEATURES   : "
        + "reports/prd_attribute_extraction/continuous_v2_geometry/"
        + "sleeve_geometry_features.csv"
    )
    print(
        "SUMMARY    : "
        + "reports/prd_attribute_extraction/continuous_v2_geometry/"
        + "feature_group_summary.csv"
    )
    print(
        "DIAGNOSTIC : "
        + "reports/prd_attribute_extraction/continuous_v2_geometry/"
        + "feature_order_diagnostics.csv"
    )
    print(
        "CONTACT    : "
        + "outputs/prd_attribute_extraction/continuous_v2_geometry/"
        + "contact_sheet.jpg"
    )
    print("================")


if __name__ == "__main__":
    main()
