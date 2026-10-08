"""运行与维护：记录文件身份、阶段状态及依赖，长任务中断后先检查状态再续跑。

Check package structure and optional manifest asset availability without loading models.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fashion_multimodal_analysis.cli import registry
from fashion_multimodal_analysis.common.paths import project_root, resolve_path


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        SystemExit: 执行本函数的操作失败。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        action="store_true",
        help="Also verify B1 train/validation/Core source images and masks",
    )
    parser.add_argument(
        "--checkpoint", default="", help="Optional checkpoint path to check"
    )
    args = parser.parse_args()
    root = project_root()
    missing = []
    for command in registry().values():
        for key in ("entrypoint", "implementation"):
            if not (root / command[key]).is_file():
                missing.append(command[key])
    print(
        f"Commands: {len(registry())}; missing entrypoints/implementations: {len(missing)}"
    )
    if args.assets:
        manifests = [
            "configs/prd_8class_train_v3.csv",
            "configs/prd_8class_val_v1.csv",
            "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv",
        ]
        for name in manifests:
            checked, absent = set(), set()
            with (root / name).open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    for key in ("source_image", "mask_path", "gt_mask_path"):
                        if row.get(key):
                            path = resolve_path(row[key])
                            checked.add(str(path))
                            if not path.is_file():
                                absent.add(str(path))
            print(
                f"{name}: {len(checked)} unique required files; {len(absent)} missing"
            )
            missing.extend(sorted(absent))
    if args.checkpoint and not resolve_path(args.checkpoint).is_file():
        missing.append(str(resolve_path(args.checkpoint)))
    if missing:
        print("Missing files (first 12):")
        for value in missing[:12]:
            print("  " + value)
        print(
            "See docs/data_and_weights.md. External datasets and model weights are not embedded in this repository."
        )
        raise SystemExit(1)
    print("PASS: requested checks completed. No model inference was run.")


if __name__ == "__main__":
    main()
