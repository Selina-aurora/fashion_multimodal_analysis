"""只读检查训练恢复所需资产，区分结果记录、模型字节和外部数据。

历史 JSON 中的 checkpoint 路径与 SHA-256 是身份记录，不能替代模型文件。
此命令不启动 GPU、不下载数据，也不覆盖历史训练或评估目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from fashion_multimodal_analysis.common.io import write_json
from fashion_multimodal_analysis.common.paths import data_root, project_root

EXPECTED = {
    "v5_checkpoint": (
        "outputs/prd_instance_segmentation/maskrcnn_8class_v5_full_targets/checkpoint_last.pth",
        "ea5a4ddf0f002256f1ce5164fc1b4b5b4d2263698f9525c9fbe7d8cf98bc1a65",
    ),
    "v8_checkpoint": (
        "outputs/prd_instance_segmentation/maskrcnn_8class_v8_full_dataset/checkpoint_last.pth",
        "e08d3426768f78953a81cf51d1cc2b664ac22a78ad8de10ae4a992f0dd3ad3b7",
    ),
    "v8_dataset_index": (
        "outputs/prd_instance_segmentation/maskrcnn_8class_v8_full_dataset/dataset.sqlite",
        "2aa79074bda85b5c400891f894e4ca11b1320ae887a1f74cbb2ea9e840e0cce8",
    ),
    "v5_training_manifest": (
        "configs/prd_8class_train_v5_full_targets.csv",
        "36ff78831b20e492d787c66161f05747433f8e85f7a089fd5df981c2bdf476e6",
    ),
    "full_target_development_manifest": (
        "configs/prd_8class_val_v2_full_targets.csv",
        "39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93",
    ),
    "core400_manifest": (
        "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv",
        "c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89",
    ),
}


def file_sha256(path: Path) -> str:
    """分块计算文件摘要，避免将大模型一次性读入内存。

    Args:
        path: 已存在的普通文件。

    Returns:
        文件原始字节的 SHA-256 十六进制字符串。

    Raises:
        OSError: 文件无法读取。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_assets(
    root: Path, data: Path, hash_weights: bool = False
) -> dict[str, Any]:
    """检查历史身份记录对应的文件，并列出原始数据目录的存在状态。

    Args:
        root: 本次使用的项目工作副本。
        data: 外部 fashion_data 目录。
        hash_weights: 是否读取完整模型字节核对摘要，可能耗时较长。

    Returns:
        包含 assets、external_data 和 ready_for_exact_v8_restore 的检查结果。
        ready 只表示所列身份文件齐备，不代表数据像素已完整校验或 GPU 可用。

    Raises:
        OSError: 已存在文件无法读取。
    """
    assets = []
    for name, (relative, expected) in EXPECTED.items():
        path = root / relative
        actual = None
        if path.is_file() and (hash_weights or not name.endswith("checkpoint")):
            actual = file_sha256(path)
        status = "MISSING"
        if path.is_file():
            status = (
                "PRESENT_UNHASHED"
                if actual is None
                else "HASH_MATCH" if actual == expected else "HASH_MISMATCH"
            )
        assets.append(
            {
                "name": name,
                "path": relative,
                "status": status,
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
    external = {
        name: (data / name).is_dir()
        for name in (
            "raw/train/train/annos",
            "raw/train/train/image",
            "raw/fashionpedia",
        )
    }
    return {
        "mode": "READ_ONLY",
        "assets": assets,
        "external_data": external,
        "ready_for_exact_v8_restore": (
            all(item["status"] == "HASH_MATCH" for item in assets)
            and all(external.values())
        ),
        "note": "原始图片、标注及派生掩码的逐项检查仍由训练准备步骤完成。",
    }


def main() -> None:
    """解析只读检查参数，并输出 JSON 状态。

    Raises:
        SystemExit: 参数非法，或 require-ready 条件未满足。
        OSError: 文件无法读取或检查报告无法写入。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None, help="项目目录，默认自动定位")
    parser.add_argument(
        "--data-root", default=None, help="数据目录，默认同级 fashion_data"
    )
    parser.add_argument("--hash-weights", action="store_true", help="核对模型完整字节")
    parser.add_argument("--json-out", default=None, help="可选的新检查报告路径")
    parser.add_argument(
        "--require-ready", action="store_true", help="缺资产时返回退出码 2"
    )
    args = parser.parse_args()
    root = Path(args.project).resolve() if args.project else project_root()
    data = Path(args.data_root).resolve() if args.data_root else data_root()
    result = inspect_assets(root, data, args.hash_weights)
    if args.json_out:
        output = Path(args.json_out)
        write_json(output if output.is_absolute() else root / output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_ready and not result["ready_for_exact_v8_restore"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
