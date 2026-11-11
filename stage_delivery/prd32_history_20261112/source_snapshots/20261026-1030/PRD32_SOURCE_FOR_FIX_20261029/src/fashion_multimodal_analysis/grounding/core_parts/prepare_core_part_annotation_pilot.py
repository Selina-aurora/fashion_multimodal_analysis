"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。

Prepare a leakage-safe core part annotation pilot.

Purpose:
    Build a small part-level supervision set for collar / cuff / hem without
    contaminating the frozen evaluation cases already used in the PRD coverage
    experiments.

Design:
    - Keep the existing 5 collar + 5 cuff + 5 hem evaluation cases held out.
    - Select 15 NEW candidates per region from the existing candidate CSVs.
    - Copy their garment crops into a dedicated annotation folder.
    - Create a manifest for manual part-level bbox annotation.

This script does NOT create labels automatically.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Any

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()
REPORT_ROOT = PROJECT_ROOT / "reports" / "prd_region_coverage"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "core_part_annotation_pilot"

REGIONS = ("collar", "cuff", "hem")
DEFAULT_PER_REGION = 15
DEFAULT_SEED = 20260914

# Frozen evaluation cases. Never use these for training.
HELD_OUT = {
    "collar": {
        "collar_001",
        "collar_003",
        "collar_010",
        "collar_012",
        "collar_020",
    },
    "cuff": {
        "cuff_006",
        "cuff_008",
        "cuff_013",
        "cuff_015",
        "cuff_019",
    },
    "hem": {
        "hem_003",
        "hem_005",
        "hem_008",
        "hem_017",
        "hem_023",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        已通过参数合法性检查的命令行配置。

    Raises:
        ValueError: --per-region must be greater than zero.
    """
    parser = argparse.ArgumentParser(
        description="Prepare 45 new collar/cuff/hem annotation images."
    )
    parser.add_argument(
        "--per-region",
        type=int,
        default=DEFAULT_PER_REGION,
        help=f"New annotation images per region (default: {DEFAULT_PER_REGION}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    args = parser.parse_args()

    if args.per_region <= 0:
        raise ValueError("--per-region must be greater than zero.")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file.

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def resolve_project_path(raw_path: str | Path) -> Path:
    """将项目相对文件引用解析到当前工作副本。

    Args:
        raw_path: 对应文件的相对路径或当前解析后的路径。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(raw_path)


def choose_rows(
    rows: list[dict[str, str]],
    region: str,
    count: int,
    seed: int,
) -> list[dict[str, str]]:
    """Choose new, non-held-out candidates with unique source images.

    Args:
        rows: 待处理的逐行记录。
        region: 区域。
        count: 数量。
        seed: 控制随机过程的种子。

    Returns:
        返回 selected，由函数体中同名变量的计算/收集过程得到。

    Raises:
        RuntimeError: 当前运行条件不满足实验要求。
    """
    pool = [row for row in rows if row["candidate_id"] not in HELD_OUT[region]]

    rng = random.Random(seed)
    rng.shuffle(pool)

    selected = []
    used_images = set()

    for row in pool:
        image_name = row["image_name"]
        if image_name in used_images:
            continue
        selected.append(row)
        used_images.add(image_name)
        if len(selected) >= count:
            break

    if len(selected) < count:
        raise RuntimeError(
            f"Not enough new candidates for {region}: "
            + f"needed {count}, found {len(selected)}."
        )
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV.

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。

    Raises:
        ValueError: No rows to write.
    """
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Prepare the annotation pilot.

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    args = parse_args()

    annotation_dir = OUTPUT_ROOT / "images"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    for region_index, region in enumerate(REGIONS):
        candidate_file = REPORT_ROOT / f"{region}_candidates.csv"
        candidate_rows = read_csv(candidate_file)

        selected = choose_rows(
            rows=candidate_rows,
            region=region,
            count=args.per_region,
            seed=args.seed + region_index,
        )

        print(f"{region}: selected {len(selected)} NEW cases")

        for local_index, row in enumerate(selected, start=1):
            source_path = resolve_project_path(row["candidate_crop_path"])
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing candidate crop: {source_path}")

            target_name = (
                f"{region}_{local_index:02d}_"
                + f"{row['candidate_id']}_"
                + f"{Path(row['image_name']).stem}.jpg"
            )
            target_path = annotation_dir / target_name
            shutil.copy2(source_path, target_path)

            manifest_rows.append(
                {
                    "annotation_id": f"{region}_train_{local_index:02d}",
                    "region": region,
                    "source_candidate_id": row["candidate_id"],
                    "image_name": row["image_name"],
                    "item_id": row["item_id"],
                    "category_name": row["category_name"],
                    "annotation_image_path": str(target_path.relative_to(PROJECT_ROOT)),
                    "split": "train_candidate",
                    "label_status": "pending",
                    "boxes_json": "",
                    "review_note": "",
                }
            )

    manifest_path = REPORT_ROOT / "core_part_annotation_pilot_manifest.csv"
    write_csv(manifest_path, manifest_rows)

    held_out_path = REPORT_ROOT / "core_part_annotation_held_out_cases.csv"
    held_rows = []
    for region in REGIONS:
        for candidate_id in sorted(HELD_OUT[region]):
            held_rows.append(
                {
                    "region": region,
                    "candidate_id": candidate_id,
                    "usage": "held_out_evaluation_only",
                }
            )
    write_csv(held_out_path, held_rows)

    print("\nFinished.")
    print(f"Annotation images: {annotation_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Held-out list: {held_out_path}")
    print("\nImportant: do NOT annotate/train on the held-out evaluation cases.")


if __name__ == "__main__":
    main()
