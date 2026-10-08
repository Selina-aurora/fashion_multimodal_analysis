"""基准集管理：保留训练、开发、固定回归/审核划分，冻结后的标签不能随结果调整。

Audit PRD 3.1 data usage before freezing a real benchmark_test split.

Why this exists
---------------
The existing prd_8class_val_v1 split has already been used for threshold
selection / diagnostics / propagation studies, so it should NOT be treated
as a blind product acceptance set.

This script inventories every CSV under configs/ that contains
source_image + garment_id, records where each sample has already appeared,
and creates a used-sample registry to prevent benchmark leakage.

Outputs
-------
benchmark/prd_3_1_v1/audit/
├── config_manifest_inventory.csv
├── used_sample_registry.csv
└── data_usage_audit.txt

Run
---
cd fashion_multimodal_analysis

python scripts/benchmarking/audit_prd31_data_for_benchmark_v1.py
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
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
CONFIG_DIR = PROJECT_ROOT / "configs"
OUT_DIR = PROJECT_ROOT / "benchmark" / "prd_3_1_v1" / "audit"

CATEGORY_FIELDS = [
    "garment_category",
    "category",
    "class_name",
]

SOURCE_FIELDS = [
    "source_dataset",
    "dataset",
]

VALID_CATEGORIES = [
    "top",
    "pants",
    "skirt",
    "outerwear",
    "dress",
    "shoe",
    "bag",
    "accessory",
]


def read_csv(path: Path) -> Any:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def first(row: dict[str, Any], names: list[str]) -> Any:
    """first。

    Args:
        row: 一条实例、预测或审核记录。
        names: names。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    for n in names:
        v = str(row.get(n, "")).strip()
        if v:
            return v
    return ""


def norm_path(v: Any) -> Any:
    """规范化路径分隔符，兼容 Windows 与 Linux 清单。

    Args:
        v: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return str(v).replace("\\", "/").strip()


def infer_role(name: str) -> str:
    """推理 role。

    Args:
        name: 当前标签、字段或产物名称。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    n = name.lower()

    if "benchmark" in n or "test_v1" in n:
        return "benchmark_candidate_or_output"
    if "train" in n:
        return "train_or_training_source"
    if any(x in n for x in ["val", "pilot", "matched", "ablation", "diagnos", "roi"]):
        return "development_or_diagnostic"
    return "unknown_or_auxiliary"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory = []
    usage = defaultdict(
        lambda: {
            "source_image": "",
            "garment_id": "",
            "categories": set(),
            "source_datasets": set(),
            "configs": set(),
            "roles": set(),
        }
    )

    csv_files = sorted(CONFIG_DIR.glob("*.csv"))

    for path in csv_files:
        try:
            rows = read_csv(path)
        except Exception as e:
            inventory.append(
                {
                    "config_file": str(path.relative_to(PROJECT_ROOT)),
                    "read_status": f"ERROR: {type(e).__name__}: {e}",
                    "rows": "",
                    "has_source_image": "",
                    "has_garment_id": "",
                    "unique_images": "",
                    "unique_garments": "",
                    "role": infer_role(path.name),
                    "category_distribution": "",
                    "source_distribution": "",
                }
            )
            continue

        fields = set(rows[0].keys()) if rows else set()
        has_source = "source_image" in fields
        has_gid = "garment_id" in fields

        cats = Counter()
        sources = Counter()
        images = set()
        garments = set()

        for row in rows:
            cat = first(row, CATEGORY_FIELDS).lower()
            src = first(row, SOURCE_FIELDS)

            if cat:
                cats[cat] += 1
            if src:
                sources[src] += 1

            source_image = norm_path(row.get("source_image", ""))
            garment_id = str(row.get("garment_id", "")).strip()

            if source_image:
                images.add(source_image)

            if source_image and garment_id:
                key = (source_image, garment_id)
                garments.add(key)

                rec = usage[key]
                rec["source_image"] = source_image
                rec["garment_id"] = garment_id
                if cat:
                    rec["categories"].add(cat)
                if src:
                    rec["source_datasets"].add(src)
                rec["configs"].add(str(path.relative_to(PROJECT_ROOT)))
                rec["roles"].add(infer_role(path.name))

        inventory.append(
            {
                "config_file": str(path.relative_to(PROJECT_ROOT)),
                "read_status": "OK",
                "rows": len(rows),
                "has_source_image": int(has_source),
                "has_garment_id": int(has_gid),
                "unique_images": len(images),
                "unique_garments": len(garments),
                "role": infer_role(path.name),
                "category_distribution": "; ".join(
                    f"{k}={v}" for k, v in sorted(cats.items())
                ),
                "source_distribution": "; ".join(
                    f"{k}={v}" for k, v in sorted(sources.items())
                ),
            }
        )

    registry = []
    role_counts = Counter()
    category_counts = Counter()

    for _, rec in sorted(usage.items()):
        roles = sorted(rec["roles"])
        cats = sorted(rec["categories"])
        srcs = sorted(rec["source_datasets"])
        cfgs = sorted(rec["configs"])

        for role in roles:
            role_counts[role] += 1
        for cat in cats:
            category_counts[cat] += 1

        registry.append(
            {
                "source_image": rec["source_image"],
                "garment_id": rec["garment_id"],
                "garment_category": "|".join(cats),
                "source_dataset": "|".join(srcs),
                "usage_roles": "|".join(roles),
                "appears_in_config_count": len(cfgs),
                "source_configs": "|".join(cfgs),
                "exclude_from_blind_benchmark_v1": 1,
            }
        )

    write_csv(
        OUT_DIR / "config_manifest_inventory.csv",
        inventory,
    )

    write_csv(
        OUT_DIR / "used_sample_registry.csv",
        registry,
    )

    lines = [
        "PRD 3.1 Benchmark Data-Usage Audit v1",
        "====================================",
        "",
        f"configs_scanned={len(csv_files)}",
        f"unique_used_garment_samples={len(registry)}",
        "",
        "Important",
        "---------",
        "- Existing validation / pilot / matched / ablation samples have already been seen during development.",
        "- They should remain useful as development/regression diagnostics, but should not be promoted to the final blind product acceptance benchmark.",
        "- Any new benchmark_test builder should exclude every source_image + garment_id listed in used_sample_registry.csv.",
        "",
        "Unique used samples by inferred role",
        "------------------------------------",
    ]

    for role, n in sorted(role_counts.items()):
        lines.append(f"{role}={n}")

    lines += [
        "",
        "Unique used samples by category",
        "-------------------------------",
    ]

    for cat in VALID_CATEGORIES:
        lines.append(f"{cat}={category_counts.get(cat, 0)}")

    unknown = sum(
        n for cat, n in category_counts.items() if cat not in VALID_CATEGORIES
    )
    lines.append(f"other_or_unknown={unknown}")

    lines += [
        "",
        "Next step",
        "---------",
        "- Build a NEW untouched segmentation benchmark candidate pool from raw data.",
        "- Exclude used_sample_registry.csv.",
        "- Balance all 8 categories and source datasets as far as possible.",
        "- Human-review GT/masks before freezing benchmark_test.",
    ]

    (OUT_DIR / "data_usage_audit.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("FINISHED")
    print((OUT_DIR / "config_manifest_inventory.csv").relative_to(PROJECT_ROOT))
    print((OUT_DIR / "used_sample_registry.csv").relative_to(PROJECT_ROOT))
    print((OUT_DIR / "data_usage_audit.txt").relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
