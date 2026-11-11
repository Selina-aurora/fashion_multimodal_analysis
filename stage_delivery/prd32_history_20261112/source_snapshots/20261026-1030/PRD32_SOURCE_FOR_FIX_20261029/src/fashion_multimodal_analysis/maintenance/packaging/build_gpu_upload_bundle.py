"""运行与维护：记录文件身份、阶段状态及依赖，长任务中断后先检查状态再续跑。"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)
from fashion_multimodal_analysis.maintenance.packaging.runtime import copy_runtime

PROJECT_ROOT = get_project_root()

SOURCE_DATASET_ROOT = data_root() / "raw" / "train" / "train"

SOURCE_IMAGE_DIR = SOURCE_DATASET_ROOT / "image"
SOURCE_ANNO_DIR = SOURCE_DATASET_ROOT / "annos"

BENCHMARK_CSV = (
    PROJECT_ROOT
    / "reports"
    / "verified_positive_benchmark"
    / "verified_positive_benchmark.csv"
)

EVAL_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "grounding/evaluation"
    / "evaluate_verified_positive_benchmark_v2.py"
)

BUNDLE_ROOT = PROJECT_ROOT / "gpu_eval"

BUNDLE_PROJECT_ROOT = BUNDLE_ROOT / "fashion_multimodal_analysis"

BUNDLE_IMAGE_DIR = BUNDLE_ROOT / "fashion_data" / "raw" / "train" / "train" / "image"

BUNDLE_ANNO_DIR = BUNDLE_ROOT / "fashion_data" / "raw" / "train" / "train" / "annos"

BUNDLE_BENCHMARK_DIR = BUNDLE_PROJECT_ROOT / "reports" / "verified_positive_benchmark"

BUNDLE_SCRIPT_DIR = BUNDLE_PROJECT_ROOT / "scripts"


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if BUNDLE_ROOT.exists():
        shutil.rmtree(BUNDLE_ROOT)

    BUNDLE_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BUNDLE_ANNO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BUNDLE_BENCHMARK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BUNDLE_SCRIPT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with BENCHMARK_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    rows = [
        row
        for row in rows
        if row["verified_positive"].strip().lower() == "yes"
        and row["usable_for_evaluation"].strip().lower() == "yes"
    ]

    print(f"Benchmark cases: {len(rows)}")

    copied_images = set()
    copied_annos = set()

    for index, row in enumerate(
        rows,
        start=1,
    ):
        image_name = row["image_name"].strip()

        image_source = SOURCE_IMAGE_DIR / image_name

        anno_name = f"{Path(image_name).stem}.json"

        anno_source = SOURCE_ANNO_DIR / anno_name

        if not image_source.is_file():
            raise FileNotFoundError(f"Missing image: {image_source}")

        if not anno_source.is_file():
            raise FileNotFoundError(f"Missing annotation: {anno_source}")

        image_target = BUNDLE_IMAGE_DIR / image_name

        anno_target = BUNDLE_ANNO_DIR / anno_name

        if image_name not in copied_images:
            shutil.copy2(
                image_source,
                image_target,
            )
            copied_images.add(image_name)

        if anno_name not in copied_annos:
            shutil.copy2(
                anno_source,
                anno_target,
            )
            copied_annos.add(anno_name)

        print(
            f"[{index:02d}/{len(rows)}] " + f"{row['benchmark_id']} " + f"{image_name}"
        )

    shutil.copy2(
        BENCHMARK_CSV,
        BUNDLE_BENCHMARK_DIR / "verified_positive_benchmark.csv",
    )

    copy_runtime(PROJECT_ROOT, BUNDLE_PROJECT_ROOT)

    print()
    print("GPU bundle completed.")
    print(f"Unique images: " + f"{len(copied_images)}")
    print(f"Unique annotations: " + f"{len(copied_annos)}")
    print(f"Bundle: {BUNDLE_ROOT}")


if __name__ == "__main__":
    main()
