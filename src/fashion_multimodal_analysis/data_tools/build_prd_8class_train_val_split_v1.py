"""数据处理：保存原图来源和实例标识，避免同一图片的不同实例跨训练与评估划分。

Build an image-grouped train/validation split for the unified PRD 3.1.1
8-class instance-segmentation manifest.

Input
-----
configs/prd_8class_unified_manifest_v2.csv

Key rule
--------
Split by (source_dataset, source_image), NOT by individual instance row.
This prevents instances from the same original image leaking across train/val.

Default target
--------------
80% train / 20% validation, with approximate per-class balance.

Outputs
-------
configs/prd_8class_train_v1.csv
configs/prd_8class_val_v1.csv

reports/prd_instance_segmentation/prd_8class_split_v1/
├── summary.txt
├── class_counts_train.csv
├── class_counts_val.csv
├── split_assignments.csv
└── validation_issues.csv

Run from project root:
    python scripts/data_tools/build_prd_8class_train_val_split_v1.py

Optional:
    python scripts/data_tools/build_prd_8class_train_val_split_v1.py --val-ratio 0.2
    --seed 20260918
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()

INPUT_CSV = PROJECT_ROOT / "configs" / "prd_8class_unified_manifest_v2.csv"

TRAIN_CSV = PROJECT_ROOT / "configs" / "prd_8class_train_v1.csv"

VAL_CSV = PROJECT_ROOT / "configs" / "prd_8class_val_v1.csv"

REPORT_DIR = (
    PROJECT_ROOT / "reports" / "prd_instance_segmentation" / "prd_8class_split_v1"
)

SUMMARY_PATH = REPORT_DIR / "summary.txt"
TRAIN_COUNTS_PATH = REPORT_DIR / "class_counts_train.csv"
VAL_COUNTS_PATH = REPORT_DIR / "class_counts_val.csv"
ASSIGNMENTS_PATH = REPORT_DIR / "split_assignments.csv"
ISSUES_PATH = REPORT_DIR / "validation_issues.csv"

PRD_CLASSES = (
    "top",
    "pants",
    "skirt",
    "outerwear",
    "dress",
    "shoe",
    "bag",
    "accessory",
)


def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Target fraction of instances in validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260918,
    )
    parser.add_argument(
        "--search-trials",
        type=int,
        default=4000,
        help="Randomized grouped-split search trials.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        按顺序返回 rows, fields 等结果。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Input manifest not found:\n  {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    return rows, fields


def write_csv(
    path: Path,
    rows: list[dict],
    fields: list[str],
) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
        fields: 输出字段或分组字段的顺序。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def group_rows(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """按共同标识分组记录，避免同一对象被重复统计。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        dataset = str(row.get("source_dataset", "")).strip()
        image = str(row.get("source_image", "")).strip()
        groups[(dataset, image)].append(row)

    return dict(groups)


def class_counter(rows: list[dict[str, str]]) -> Counter:
    """类别 counter。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return Counter(str(row.get("garment_category", "")).strip().lower() for row in rows)


def group_class_counter(
    group: list[dict[str, str]],
) -> Counter:
    """group 类别 counter。

    Args:
        group: group。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return class_counter(group)


def score_split(
    val_keys: set[tuple[str, str]],
    groups: dict[tuple[str, str], list[dict[str, str]]],
    total_counts: Counter,
    target_val_counts: dict[str, float],
    target_val_rows: float,
) -> float:
    """置信度 split。

    Args:
        val_keys: val 标识。
        groups: groups。
        total_counts: total 计数。
        target_val_counts: 目标 val 计数。
        target_val_rows: 待处理的 target_val 记录。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    val_rows = []
    for key in val_keys:
        val_rows.extend(groups[key])

    val_counts = class_counter(val_rows)

    # Strong class-distribution term.
    class_error = 0.0
    for cls in PRD_CLASSES:
        total = max(1, total_counts.get(cls, 0))
        target = target_val_counts[cls]
        actual = val_counts.get(cls, 0)
        class_error += ((actual - target) / total) ** 2

    # Total size term.
    total_rows = max(1, sum(total_counts.values()))
    size_error = ((len(val_rows) - target_val_rows) / total_rows) ** 2

    # Hard-ish penalties: every class should appear in both splits.
    coverage_penalty = 0.0
    for cls in PRD_CLASSES:
        total = total_counts.get(cls, 0)
        val_n = val_counts.get(cls, 0)
        train_n = total - val_n

        if total > 0 and val_n == 0:
            coverage_penalty += 1.0
        if total > 0 and train_n == 0:
            coverage_penalty += 3.0

    return class_error + 2.0 * size_error + 10.0 * coverage_penalty


def initial_group_probability(
    group_rows_list: list[dict[str, str]],
    val_ratio: float,
) -> float:
    # Same base probability for every image-group. Keeping this simple avoids
    # biasing larger images too heavily; randomized search handles the rest.
    """initial group probability。

    Args:
        group_rows_list: group 逐行记录 list。
        val_ratio: val 比例。

    Returns:
        返回 val_ratio，由函数体中同名变量的计算/收集过程得到。
    """
    return val_ratio


def randomized_search(
    groups: dict[tuple[str, str], list[dict[str, str]]],
    total_counts: Counter,
    val_ratio: float,
    seed: int,
    trials: int,
) -> set[tuple[str, str]]:
    """randomized search。

    Args:
        groups: groups。
        total_counts: total 计数。
        val_ratio: val 比例。
        seed: 控制随机过程的种子。
        trials: trials。

    Returns:
        返回 best_keys，由函数体中同名变量的计算/收集过程得到。
    """
    rng = random.Random(seed)

    keys = list(groups.keys())

    target_val_counts = {
        cls: total_counts.get(cls, 0) * val_ratio for cls in PRD_CLASSES
    }

    total_rows = sum(total_counts.values())
    target_val_rows = total_rows * val_ratio

    best_keys: set[tuple[str, str]] | None = None
    best_score = float("inf")

    # Seed a deterministic-ish baseline by sorting.
    for trial in range(trials):
        val_keys: set[tuple[str, str]] = set()

        # Random grouped assignment.
        for key in keys:
            p = initial_group_probability(groups[key], val_ratio)
            if rng.random() < p:
                val_keys.add(key)

        # Ensure nonempty split.
        if not val_keys:
            val_keys.add(rng.choice(keys))
        if len(val_keys) == len(keys):
            val_keys.remove(rng.choice(list(val_keys)))

        score = score_split(
            val_keys,
            groups,
            total_counts,
            target_val_counts,
            target_val_rows,
        )

        if score < best_score:
            best_score = score
            best_keys = set(val_keys)

    assert best_keys is not None

    # Local one-flip improvement.
    improved = True
    while improved:
        improved = False

        current_score = score_split(
            best_keys,
            groups,
            total_counts,
            target_val_counts,
            target_val_rows,
        )

        candidate_best = best_keys
        candidate_score = current_score

        shuffled = keys[:]
        rng.shuffle(shuffled)

        for key in shuffled:
            proposal = set(best_keys)

            if key in proposal:
                if len(proposal) <= 1:
                    continue
                proposal.remove(key)
            else:
                if len(proposal) >= len(keys) - 1:
                    continue
                proposal.add(key)

            score = score_split(
                proposal,
                groups,
                total_counts,
                target_val_counts,
                target_val_rows,
            )

            if score + 1e-12 < candidate_score:
                candidate_score = score
                candidate_best = proposal

        if candidate_score + 1e-12 < current_score:
            best_keys = candidate_best
            improved = True

    return best_keys


def validate_split(
    all_rows: list[dict[str, str]],
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """校验 split。

    Args:
        all_rows: 待处理的 all 记录。
        train_rows: 待处理的 train 记录。
        val_rows: 待处理的 val 记录。

    Returns:
        当前条件的校验结果；失败条件及返回形式见函数体。
    """
    issues: list[dict[str, str]] = []

    train_groups = {
        (
            str(r.get("source_dataset", "")).strip(),
            str(r.get("source_image", "")).strip(),
        )
        for r in train_rows
    }

    val_groups = {
        (
            str(r.get("source_dataset", "")).strip(),
            str(r.get("source_image", "")).strip(),
        )
        for r in val_rows
    }

    overlap = train_groups & val_groups

    for dataset, image in sorted(overlap):
        issues.append(
            {
                "issue_type": "image_group_leakage",
                "source_dataset": dataset,
                "source_image": image,
                "detail": "same source image appears in train and val",
            }
        )

    if len(train_rows) + len(val_rows) != len(all_rows):
        issues.append(
            {
                "issue_type": "row_count_mismatch",
                "source_dataset": "",
                "source_image": "",
                "detail": (
                    f"all={len(all_rows)}, "
                    + f"train={len(train_rows)}, "
                    + f"val={len(val_rows)}"
                ),
            }
        )

    train_counts = class_counter(train_rows)
    val_counts = class_counter(val_rows)

    for cls in PRD_CLASSES:
        if train_counts.get(cls, 0) == 0:
            issues.append(
                {
                    "issue_type": "missing_train_class",
                    "source_dataset": "",
                    "source_image": "",
                    "detail": cls,
                }
            )

        if val_counts.get(cls, 0) == 0:
            issues.append(
                {
                    "issue_type": "missing_val_class",
                    "source_dataset": "",
                    "source_image": "",
                    "detail": cls,
                }
            )

    return issues


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    args = parse_args()

    if not (0.05 <= args.val_ratio <= 0.5):
        raise ValueError("--val-ratio should be between 0.05 and 0.5")

    rows, fields = read_csv(INPUT_CSV)

    if not rows:
        raise ValueError("Unified manifest is empty.")

    groups = group_rows(rows)
    total_counts = class_counter(rows)

    unknown_classes = sorted(cls for cls in total_counts if cls not in PRD_CLASSES)

    if unknown_classes:
        raise ValueError(
            "Unexpected classes in unified manifest: " + ", ".join(unknown_classes)
        )

    print("Input:")
    print(f"  instances: {len(rows)}")
    print(f"  image groups: {len(groups)}")
    print(f"  target val ratio: {args.val_ratio:.2f}")

    val_keys = randomized_search(
        groups=groups,
        total_counts=total_counts,
        val_ratio=args.val_ratio,
        seed=args.seed,
        trials=args.search_trials,
    )

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []

    split_assignments = []

    for key, group in groups.items():
        dataset, source_image = key
        split = "val" if key in val_keys else "train"

        if split == "val":
            val_rows.extend(group)
        else:
            train_rows.extend(group)

        counts = group_class_counter(group)

        split_assignments.append(
            {
                "source_dataset": dataset,
                "source_image": source_image,
                "split": split,
                "instance_count": len(group),
                "classes": ";".join(
                    f"{cls}:{counts[cls]}"
                    for cls in PRD_CLASSES
                    if counts.get(cls, 0) > 0
                ),
            }
        )

    train_counts = class_counter(train_rows)
    val_counts = class_counter(val_rows)

    issues = validate_split(
        rows,
        train_rows,
        val_rows,
    )

    # Preserve stable original record ordering within each split.
    def record_num(row: dict[str, str]) -> int:
        """记录 num。

        Args:
            row: 一条实例、预测或审核记录。

        Returns:
            本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
        """
        rid = str(row.get("record_id", ""))
        try:
            return int(rid.rsplit("_", 1)[-1])
        except ValueError:
            return 10**9

    train_rows.sort(key=record_num)
    val_rows.sort(key=record_num)
    split_assignments.sort(
        key=lambda r: (
            r["split"],
            r["source_dataset"],
            r["source_image"],
        )
    )

    write_csv(TRAIN_CSV, train_rows, fields)
    write_csv(VAL_CSV, val_rows, fields)

    count_fields = [
        "garment_category",
        "count",
        "fraction_of_class",
    ]

    train_count_rows = []
    val_count_rows = []

    for cls in PRD_CLASSES:
        total = total_counts.get(cls, 0)
        train_n = train_counts.get(cls, 0)
        val_n = val_counts.get(cls, 0)

        train_count_rows.append(
            {
                "garment_category": cls,
                "count": train_n,
                "fraction_of_class": (f"{train_n / total:.4f}" if total else ""),
            }
        )

        val_count_rows.append(
            {
                "garment_category": cls,
                "count": val_n,
                "fraction_of_class": (f"{val_n / total:.4f}" if total else ""),
            }
        )

    write_csv(
        TRAIN_COUNTS_PATH,
        train_count_rows,
        count_fields,
    )

    write_csv(
        VAL_COUNTS_PATH,
        val_count_rows,
        count_fields,
    )

    write_csv(
        ASSIGNMENTS_PATH,
        split_assignments,
        [
            "source_dataset",
            "source_image",
            "split",
            "instance_count",
            "classes",
        ],
    )

    write_csv(
        ISSUES_PATH,
        issues,
        [
            "issue_type",
            "source_dataset",
            "source_image",
            "detail",
        ],
    )

    total_rows = len(rows)
    train_ratio = len(train_rows) / total_rows
    val_ratio_actual = len(val_rows) / total_rows

    train_group_count = len(groups) - len(val_keys)
    val_group_count = len(val_keys)

    summary = [
        "PRD 3.1.1 grouped train/val split v1",
        "====================================",
        "",
        "Split rule",
        "----------",
        "- Split unit = source_dataset + source_image.",
        "- Instances from the same original image cannot appear in both splits.",
        "",
        "Global counts",
        "-------------",
        f"total_instances={len(rows)}",
        f"total_image_groups={len(groups)}",
        f"train_instances={len(train_rows)}",
        f"val_instances={len(val_rows)}",
        f"train_image_groups={train_group_count}",
        f"val_image_groups={val_group_count}",
        f"target_val_ratio={args.val_ratio:.4f}",
        f"actual_train_ratio={train_ratio:.4f}",
        f"actual_val_ratio={val_ratio_actual:.4f}",
        f"seed={args.seed}",
        "",
        "Per-class counts: train / val / total",
        "-------------------------------------",
    ]

    for cls in PRD_CLASSES:
        summary.append(
            f"{cls}="
            + f"{train_counts.get(cls, 0)}/"
            + f"{val_counts.get(cls, 0)}/"
            + f"{total_counts.get(cls, 0)}"
        )

    dataset_train = Counter(r["source_dataset"] for r in train_rows)
    dataset_val = Counter(r["source_dataset"] for r in val_rows)

    summary += [
        "",
        "Per-source counts: train / val",
        "-----------------------------",
        f"DeepFashion2="
        + f"{dataset_train.get('DeepFashion2', 0)}/"
        + f"{dataset_val.get('DeepFashion2', 0)}",
        f"Fashionpedia="
        + f"{dataset_train.get('Fashionpedia', 0)}/"
        + f"{dataset_val.get('Fashionpedia', 0)}",
        "",
        "Validation",
        "----------",
        (
            f"image_group_overlap=0"
            if not any(i["issue_type"] == "image_group_leakage" for i in issues)
            else "image_group_overlap=ERROR"
        ),
        f"validation_issues={len(issues)}",
        "",
        "Interpretation",
        "--------------",
        "- This is a grouped data split for baseline training/evaluation.",
        "- It does not itself measure segmentation quality.",
        "- Keep the validation split fixed for later model comparisons.",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print()
    print("Split result:")
    print(f"  train: {len(train_rows)} " + f"({train_ratio:.3f})")
    print(f"  val  : {len(val_rows)} " + f"({val_ratio_actual:.3f})")

    print()
    print("Per class (train / val / total):")
    for cls in PRD_CLASSES:
        print(
            f"  {cls:10s} "
            + f"{train_counts.get(cls, 0):3d} / "
            + f"{val_counts.get(cls, 0):3d} / "
            + f"{total_counts.get(cls, 0):3d}"
        )

    print()
    print(f"Validation issues: {len(issues)}")
    print()
    print("Wrote:")
    print(f"  {TRAIN_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  {VAL_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {TRAIN_COUNTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {VAL_COUNTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {ASSIGNMENTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {ISSUES_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
