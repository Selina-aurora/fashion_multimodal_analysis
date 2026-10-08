"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。"""

from __future__ import annotations

import csv
import math
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

REBUILT_PARENT_DIR = OUTPUT_DIR / "rebuilt_parent_crops"
MANIFEST_OUT = REPORT_DIR / "refinement_manifest.csv"
CONTACT_OUT = REPORT_DIR / "refinement_windows_contact_sheet.jpg"
RESOLUTION_LOG = REPORT_DIR / "parent_crop_resolution.csv"

TARGET_REGIONS = {"shoulder", "waist", "pattern"}


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_project_path(raw: Any) -> Path:
    """将项目相对文件引用解析到当前工作副本。

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(raw)


def rel(path: Path) -> str:
    """rel。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # 路径不属于项目时，进入下方数据目录/外部路径分支；这是预期匹配失败。
        pass

    try:
        r = path.relative_to(PROJECT_ROOT.parent)
        return f"../{r.as_posix()}"
    except ValueError:
        return path.as_posix()


def build_garment_index() -> dict[tuple[str, str], dict[str, str]]:
    """Index existing garment-instance manifests by:
        (source image basename, garment/item id)

    This lets us recover a garment crop even when old outputs/prd_region_coverage/*
    crops were not uploaded to the GPU server.

    Returns:
        返回 index，由函数体中同名变量的计算/收集过程得到。
    """
    index: dict[tuple[str, str], dict[str, str]] = {}

    patterns = [
        "garment_instances_gt_pilot*.csv",
        "garment_instances*.csv",
        "prd_8class_train*.csv",
        "prd_8class_val*.csv",
    ]

    files = []
    for pattern in patterns:
        files.extend((PROJECT_ROOT / "configs").glob(pattern))

    # de-duplicate
    seen = set()
    unique_files = []
    for p in files:
        rp = p.resolve()
        if rp not in seen and p.is_file():
            seen.add(rp)
            unique_files.append(p)

    for csv_path in unique_files:
        try:
            rows = read_csv(csv_path)
        except Exception:
            continue

        for row in rows:
            source_image = (
                row.get("source_image")
                or row.get("image_path")
                or row.get("image_name")
                or ""
            ).strip()

            image_name = Path(source_image.replace("\\", "/")).name

            garment_id = (
                row.get("garment_id")
                or row.get("item_id")
                or row.get("annotation_item")
                or ""
            ).strip()

            if not image_name or not garment_id:
                continue

            key = (image_name, garment_id)

            # Prefer rows that actually expose crop/mask/bbox information.
            score = sum(
                bool(str(row.get(k, "")).strip())
                for k in (
                    "crop_path",
                    "masked_preview_path",
                    "bbox_x1",
                    "bbox_y1",
                    "bbox_x2",
                    "bbox_y2",
                    "source_image",
                )
            )

            prev = index.get(key)
            if prev is None:
                row["_manifest"] = csv_path.as_posix()
                row["_score"] = str(score)
                index[key] = row
            else:
                prev_score = int(prev.get("_score", "0"))
                if score > prev_score:
                    row["_manifest"] = csv_path.as_posix()
                    row["_score"] = str(score)
                    index[key] = row

    return index


def candidate_item_variants(item_id: str) -> list[str]:
    """Region coverage cases usually use item1 / item2, while garment manifests
    can use item1_short_sleeve_top etc. Match exact first, then prefix.

    Args:
        item_id: item ID。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    item_id = str(item_id).strip()
    return [item_id]


def find_manifest_row(
    index: dict[tuple[str, str], dict[str, str]],
    image_name: str,
    item_id: str,
) -> dict[str, str] | None:
    """查找 清单 记录。

    Args:
        index: 待访问样本的整数下标。
        image_name: 图像 name。
        item_id: item ID。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    image_name = Path(image_name.replace("\\", "/")).name
    item_id = str(item_id).strip()

    exact = index.get((image_name, item_id))
    if exact is not None:
        return exact

    # Typical mapping:
    #   region case item_id = item1
    #   garment manifest garment_id = item1_short_sleeve_top
    prefix = item_id + "_"

    candidates = [
        row
        for (img, gid), row in index.items()
        if img == image_name and (gid == item_id or gid.startswith(prefix))
    ]

    if len(candidates) == 1:
        return candidates[0]

    if candidates:
        # Prefer the richest manifest row.
        candidates.sort(
            key=lambda r: int(r.get("_score", "0")),
            reverse=True,
        )
        return candidates[0]

    return None


def valid_image(path: Path) -> bool:
    """有效 图像。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        当前条件的校验结果；失败条件及返回形式见函数体。
    """
    try:
        return path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def rebuild_parent_crop(
    case_row: dict[str, str],
    garment_index: dict[tuple[str, str], dict[str, str]],
) -> tuple[Path, str, str]:
    """Resolution order:
      1. original old region candidate crop, if present
      2. garment manifest crop_path
      3. garment manifest masked_preview_path
      4. rebuild crop from source_image + bbox
      5. DeepFashion2 raw image + manifest bbox

    Returns:
        path, resolution_method, source_manifest

    Args:
        case_row: 记录字段，使用 candidate_id, region。
        garment_index: garment index。

    Raises:
        FileNotFoundError: 需要的文件不存在。
        ValueError: 输入或实验状态不符合检查条件。
    """
    old = resolve_project_path(case_row.get("candidate_crop_path", ""))
    if valid_image(old):
        return old, "existing_region_candidate_crop", ""

    image_name = case_row.get("image_name", "").strip()
    item_id = case_row.get("item_id", "").strip()

    manifest_row = find_manifest_row(
        garment_index,
        image_name=image_name,
        item_id=item_id,
    )

    if manifest_row is None:
        raise FileNotFoundError(
            "Could not recover parent garment crop.\n"
            + f"  image_name={image_name}\n"
            + f"  item_id={item_id}\n"
            + f"  old_missing={old}\n"
            + "No matching garment instance was found in configs/*.csv."
        )

    manifest_name = manifest_row.get("_manifest", "")

    for key, method in (
        ("crop_path", "garment_manifest_crop_path"),
        ("masked_preview_path", "garment_manifest_masked_preview"),
    ):
        raw = manifest_row.get(key, "").strip()
        if raw:
            p = resolve_project_path(raw)
            if valid_image(p):
                return p, method, manifest_name

    source_raw = manifest_row.get("source_image", "").strip()

    source_candidates = []

    if source_raw:
        source_candidates.append(resolve_project_path(source_raw))

    # Known DeepFashion2 location in this project.
    source_candidates.append(
        (data_root() / "raw" / "train" / "train" / "image" / image_name).resolve()
    )

    source_image = next(
        (p for p in source_candidates if valid_image(p)),
        None,
    )

    if source_image is None:
        raise FileNotFoundError(
            "Matching garment manifest row was found, but source image/crop files "
            + "are unavailable.\n"
            + f"  image_name={image_name}\n"
            + f"  item_id={item_id}\n"
            + f"  manifest={manifest_name}"
        )

    needed = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    if not all(str(manifest_row.get(k, "")).strip() for k in needed):
        raise ValueError(
            "Cannot rebuild garment crop because bbox columns are missing.\n"
            + f"  image_name={image_name}\n"
            + f"  item_id={item_id}\n"
            + f"  manifest={manifest_name}"
        )

    x1 = int(float(manifest_row["bbox_x1"]))
    y1 = int(float(manifest_row["bbox_y1"]))
    x2 = int(float(manifest_row["bbox_x2"]))
    y2 = int(float(manifest_row["bbox_y2"]))

    image = Image.open(source_image).convert("RGB")
    w, h = image.size

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    crop = image.crop((x1, y1, x2, y2))

    rebuilt = (
        REBUILT_PARENT_DIR
        / case_row["region"].strip().lower()
        / f"{case_row['candidate_id']}_{image_name}"
    )
    rebuilt.parent.mkdir(parents=True, exist_ok=True)
    crop.save(rebuilt, quality=95)

    return rebuilt, "rebuilt_from_source_bbox", manifest_name


def windows_for(region: str) -> list[Any]:
    """windows for。

    Args:
        region: 区域。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    if region == "shoulder":
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


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        RuntimeError: No shoulder/waist/pattern rows found.
    """
    rows = [
        r
        for r in read_csv(INPUT_CASES)
        if r.get("region", "").strip().lower() in TARGET_REGIONS
    ]

    if not rows:
        raise RuntimeError("No shoulder/waist/pattern rows found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    garment_index = build_garment_index()
    print(f"Garment manifest index entries: {len(garment_index)}")

    out_rows = []
    resolution_rows = []
    preview_tiles = []

    for row in rows:
        region = row["region"].strip().lower()

        parent_path, resolution_method, source_manifest = rebuild_parent_crop(
            row,
            garment_index,
        )

        resolution_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "region": region,
                "image_name": row.get("image_name", ""),
                "item_id": row.get("item_id", ""),
                "original_candidate_crop_path": row.get("candidate_crop_path", ""),
                "resolved_parent_crop_path": rel(parent_path),
                "resolution_method": resolution_method,
                "source_manifest": source_manifest,
            }
        )

        image = Image.open(parent_path).convert("RGB")
        w, h = image.size

        for window_name, x1r, y1r, x2r, y2r in windows_for(region):
            x1 = max(0, min(w - 1, int(round(x1r * w))))
            y1 = max(0, min(h - 1, int(round(y1r * h))))
            x2 = max(x1 + 1, min(w, int(round(x2r * w))))
            y2 = max(y1 + 1, min(h, int(round(y2r * h))))

            crop = image.crop((x1, y1, x2, y2))

            scale = max(1.0, 900.0 / max(crop.size))
            if scale > 1.0:
                crop = crop.resize(
                    (
                        int(round(crop.width * scale)),
                        int(round(crop.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )

            out_path = OUTPUT_DIR / region / f"{row['candidate_id']}_{window_name}.jpg"
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
                    "parent_crop_path": rel(parent_path),
                    "parent_resolution_method": resolution_method,
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

            preview = image.copy()
            d = ImageDraw.Draw(preview)
            d.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=4)
            d.text(
                (5, 5),
                f"{row['candidate_id']} | {region} | {window_name}",
                fill=(255, 0, 0),
                font=ImageFont.load_default(),
            )
            preview.thumbnail((380, 330), Image.Resampling.LANCZOS)
            preview_tiles.append(preview)

    write_csv(MANIFEST_OUT, out_rows)
    write_csv(RESOLUTION_LOG, resolution_rows)

    cols = 4
    tile_w, tile_h = 400, 360
    rows_n = math.ceil(len(preview_tiles) / cols)

    sheet = Image.new(
        "RGB",
        (cols * tile_w, rows_n * tile_h),
        "white",
    )

    for i, tile in enumerate(preview_tiles):
        x = (i % cols) * tile_w + (tile_w - tile.width) // 2
        y = (i // cols) * tile_h + 10
        sheet.paste(tile, (x, y))

    sheet.save(CONTACT_OUT, quality=92)

    methods = {}
    for r in resolution_rows:
        methods[r["resolution_method"]] = methods.get(r["resolution_method"], 0) + 1

    print("=== REFINEMENT INPUTS BUILT ===")
    print(f"Source cases: {len(rows)}")
    print(f"Refinement crops: {len(out_rows)}")
    print("Parent crop resolution:")
    for k, v in sorted(methods.items()):
        print(f"  {k}: {v}")
    print()
    print("Manifest:", MANIFEST_OUT.relative_to(PROJECT_ROOT))
    print("Contact :", CONTACT_OUT.relative_to(PROJECT_ROOT))
    print("Resolve :", RESOLUTION_LOG.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
