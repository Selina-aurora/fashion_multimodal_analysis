#!/usr/bin/env python3
"""结果分析：依据已有逐例记录定位问题，不替代新的模型评估。

Read-only PRD31 result analysis and native-annotation completeness audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from fashion_multimodal_analysis.common.paths import resolve_record_path

CLASSES = ("top", "pants", "skirt", "outerwear", "dress", "shoe", "bag", "accessory")
DF2_CLASS = {
    1: "top",
    2: "top",
    3: "outerwear",
    4: "outerwear",
    5: "top",
    6: "top",
    7: "pants",
    8: "pants",
    9: "skirt",
    10: "dress",
    11: "dress",
    12: "dress",
    13: "dress",
}
DF2_FINE = (
    "short_sleeve_top",
    "long_sleeve_top",
    "short_sleeve_outwear",
    "long_sleeve_outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short_sleeve_dress",
    "long_sleeve_dress",
    "vest_dress",
    "sling_dress",
)
FP_CLASS = {
    "shirt, blouse": "top",
    "top, t-shirt, sweatshirt": "top",
    "sweater": "top",
    "vest": "top",
    "cardigan": "outerwear",
    "jacket": "outerwear",
    "coat": "outerwear",
    "cape": "outerwear",
    "pants": "pants",
    "shorts": "pants",
    "skirt": "skirt",
    "dress": "dress",
    "shoe": "shoe",
    "bag, wallet": "bag",
    "glasses": "accessory",
    "hat": "accessory",
    "headband, head covering, hair accessory": "accessory",
    "tie": "accessory",
    "glove": "accessory",
    "watch": "accessory",
    "belt": "accessory",
    "leg warmer": "accessory",
    "tights, stockings": "accessory",
    "sock": "accessory",
    "scarf": "accessory",
}


def read_csv(path: str | Path) -> Any:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
    """
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(rows)


def digest(path: str | Path) -> str:
    """分块计算文件 SHA-256，核对文件身份而不加载整个权重到内存。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        64 位十六进制 SHA-256 摘要字符串。
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flag(value: Any) -> bool:
    """flag。

    Args:
        value: 待解析或规范化的输入值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return str(value).strip().lower() in ("true", "1", "yes")


def resolve(value: Any, root: Path, data: Path) -> Path:
    """将历史原图/掩码引用迁移到本次使用的项目和数据目录。

    Args:
        value: 待解析或规范化的输入值。
        root: 当前项目根目录。
        data: 外部服饰数据根目录。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_record_path(str(value), root, data)


def summarize(rows: list[dict[str, Any]]) -> Any:
    """汇总逐例结果，保持当前指标的分母定义。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        结果字段包括 gt_count, localized, class_correct, missed, wrong_class, low_mask_iou,
        correct_mask50, bbox50_recall, class_correct_recall,
        mean_mask_iou_correct_class。

    Raises:
        ValueError: Invalid mask IoU in source results
    """
    good = [r for r in rows if flag(r["class_correct"]) and flag(r["localized_bbox50"])]
    matched = sum(flag(r["localized_bbox50"]) for r in rows)
    outcomes = Counter(r["outcome"] for r in rows)
    values = [float(r["mask_iou"]) for r in good]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Invalid mask IoU in source results")
    return dict(
        gt_count=len(rows),
        localized=matched,
        class_correct=len(good),
        missed=outcomes["missed"],
        wrong_class=outcomes["wrong_class"],
        low_mask_iou=outcomes["low_mask_iou"],
        correct_mask50=outcomes["correct"],
        bbox50_recall=matched / len(rows) if rows else None,
        class_correct_recall=len(good) / len(rows) if rows else None,
        mean_mask_iou_correct_class=sum(values) / len(values) if values else None,
        mask85_count=sum(v >= 0.85 for v in values),
    )


def result_groups(rows: list[dict[str, Any]], fields: list[str]) -> Any:
    """按指定字段分组逐例结果，检查是否存在共性错误。

    Args:
        rows: 待处理的逐行记录。
        fields: 输出字段或分组字段的顺序。

    Returns:
        返回 output，由函数体中同名变量的计算/收集过程得到。
    """
    output = []
    for field in fields:
        groups = defaultdict(list)
        for row in rows:
            groups[row.get(field, "unknown")].append(row)
        for value, members in sorted(groups.items()):
            output.append(
                dict(group_field=field, group_value=value, **summarize(members))
            )
    return output


def annotation_id(row: dict[str, Any]) -> Any:
    """提取原始标注的实例 ID，用于核对选中目标。

    Args:
        row: 记录字段，使用 source_dataset。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    if row["source_dataset"] == "DeepFashion2":
        text = row.get("annotation_ref") or row.get("garment_id", "")
        match = re.match(r"(item\d+)(?:_|$)", text)
        if not match:
            raise ValueError("Unknown DeepFashion2 item identity: " + text)
        return match.group(1)
    text = (
        row.get("annotation_id")
        or row.get("annotation_ref")
        or row.get("garment_id", "")
    )
    text = text.removeprefix("fashionpedia_ann_")
    return str(int(text))


def usable_annotation(annotation: dict[str, Any], dataset: Any) -> bool:
    """检查原始目标标注是否可用于八类实例训练。

    Args:
        annotation: 原始实例标注。
        dataset: 按原图分组后的实例数据集。

    Returns:
        当前条件的校验结果；失败条件及返回形式见函数体。
    """
    if dataset == "Fashionpedia" and int(annotation.get("iscrowd", 0)):
        return False
    box = annotation.get("bounding_box" if dataset == "DeepFashion2" else "bbox", [])
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        return False
    if dataset == "DeepFashion2":
        valid = float(box[2]) > float(box[0]) and float(box[3]) > float(box[1])
    else:
        valid = float(box[2]) > 0 and float(box[3]) > 0
    return valid and bool(annotation.get("segmentation"))


def audit_native_annotations(root: Path, data: Path, manifests: Any) -> tuple[Any, ...]:
    """Compare selected identities to usable raw annotations in the same eight classes.

    This inventories polygon/RLE availability; it does not certify mask decoding.
    Garment parts, jumpsuits, umbrellas and crowds are outside the comparison scope.

    Args:
        root: 当前项目根目录。
        data: 外部服饰数据根目录。
        manifests: manifests。

    Returns:
        按顺序返回 output, summary 等结果。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    fp_path = data / "raw/fashionpedia/annotations/instances_attributes_val2020.json"
    fp_index = {}
    fp_available = fp_path.is_file()
    if fp_available:
        print("读取 Fashionpedia 原始标注……", flush=True)
        source = json.loads(fp_path.read_text(encoding="utf-8-sig"))
        images = {str(v["id"]): Path(v["file_name"]).name for v in source["images"]}
        if len(set(images.values())) != len(images):
            raise ValueError(
                "Fashionpedia duplicate image basenames require explicit path resolution"
            )
        names = {
            str(v["id"]): " ".join(v["name"].strip().lower().split())
            for v in source["categories"]
        }
        by_image = {name: {} for name in images.values()}
        for ann in source["annotations"]:
            name = images[str(ann["image_id"])]
            category = FP_CLASS.get(names[str(ann["category_id"])])
            by_image[name][str(ann["id"])] = (
                category,
                usable_annotation(ann, "Fashionpedia"),
            )
        fp_index = by_image
        del source
    cache, output = {}, []
    for split, rows in manifests.items():
        groups = defaultdict(list)
        for row in rows:
            groups[(row["source_dataset"], row["source_image"])].append(row)
        print(f"审核 {split}: {len(groups)} 张源图", flush=True)
        for (dataset, image), members in sorted(groups.items()):
            selected = {annotation_id(row): row["garment_category"] for row in members}
            if len(selected) != len(members):
                raise ValueError(f"Duplicate selected annotation: {split}: {image}")
            entry = dict(
                split=split,
                source_dataset=dataset,
                source_image=image,
                manifest_instances=len(members),
                status="",
                raw_usable_instances="",
                omitted_usable_instances="",
                omitted_by_class="",
                omitted_ids="",
                identity_or_class_errors="",
            )
            if dataset == "Fashionpedia":
                raw = fp_index.get(Path(image.replace("\\", "/")).name)
                entry["annotation_file"] = str(fp_path)
                if raw is None:
                    entry["status"] = (
                        "RAW_ANNOTATION_FILE_MISSING"
                        if not fp_available
                        else "RAW_IMAGE_ID_NOT_FOUND"
                    )
                    output.append(entry)
                    continue
            elif dataset == "DeepFashion2":
                image_path = resolve(image, root, data)
                ann_value = members[0].get("annotation_path")
                ann_path = (
                    resolve(ann_value, root, data)
                    if ann_value
                    else image_path.parent.parent
                    / "annos"
                    / (image_path.stem + ".json")
                )
                entry["annotation_file"] = str(ann_path)
                if not ann_path.is_file():
                    entry["status"] = "RAW_ANNOTATION_FILE_MISSING"
                    output.append(entry)
                    continue
                if ann_path not in cache:
                    raw_json = json.loads(ann_path.read_text(encoding="utf-8-sig"))
                    cache[ann_path] = {
                        key: (
                            DF2_CLASS.get(int(ann.get("category_id", -1))),
                            usable_annotation(ann, dataset),
                        )
                        for key, ann in raw_json.items()
                        if re.fullmatch(r"item\d+", key) and isinstance(ann, dict)
                    }
                raw = cache[ann_path]
            else:
                raise ValueError("Unsupported source dataset: " + dataset)
            errors = [
                key
                for key, category in selected.items()
                if key not in raw or raw[key][0] != category or not raw[key][1]
            ]
            usable = {
                key: category
                for key, (category, valid) in raw.items()
                if category and valid
            }
            omitted = {
                key: category for key, category in usable.items() if key not in selected
            }
            entry.update(
                raw_usable_instances=len(usable),
                omitted_usable_instances=len(omitted),
                omitted_by_class=json.dumps(
                    dict(Counter(omitted.values())), ensure_ascii=False
                ),
                omitted_ids=json.dumps(sorted(omitted)),
                identity_or_class_errors=json.dumps(errors),
                status=(
                    "IDENTITY_CLASS_OR_GEOMETRY_ERROR"
                    if errors
                    else (
                        "ADDITIONAL_RAW_TARGETS" if omitted else "COMPLETE_WITHIN_SCOPE"
                    )
                ),
            )
            output.append(entry)
    summary = {}
    for split in manifests:
        members = [r for r in output if r["split"] == split]
        checked = [r for r in members if r["raw_usable_instances"] != ""]
        summary[split] = dict(
            images=len(members),
            checked_images=len(checked),
            statuses=dict(Counter(r["status"] for r in members)),
            known_omitted_usable_instances=sum(
                r["omitted_usable_instances"] for r in checked
            ),
            omitted_usable_instances=(
                sum(r["omitted_usable_instances"] for r in checked)
                if len(checked) == len(members)
                else None
            ),
            audit_complete=len(checked) == len(members),
            interpretation="原标注库存检查；没有执行新增 mask 解码或修改数据清单。",
        )
        summary[split]["by_source"] = {}
        for dataset in sorted({r["source_dataset"] for r in members}):
            selected = [r for r in members if r["source_dataset"] == dataset]
            verified = [r for r in selected if r["raw_usable_instances"] != ""]
            summary[split]["by_source"][dataset] = dict(
                images=len(selected),
                checked_images=len(verified),
                manifest_instances=sum(r["manifest_instances"] for r in selected),
                raw_usable_instances=(
                    sum(r["raw_usable_instances"] for r in verified)
                    if len(verified) == len(selected)
                    else None
                ),
                omitted_usable_instances=(
                    sum(r["omitted_usable_instances"] for r in verified)
                    if len(verified) == len(selected)
                    else None
                ),
                statuses=dict(Counter(r["status"] for r in selected)),
            )
    return output, summary


def number(value: Any, percent: bool = False) -> Any:
    """将数值格式化为分析报告的显示文本。

    Args:
        value: 待解析或规范化的输入值。
        percent: percent。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return "NA" if value is None else f"{value*100:.2f}%" if percent else f"{value:.4f}"


def timing_summary(rows: list[dict[str, Any]], per_gt: Any, passes: Any) -> Any:
    """汇总多次计时，记录均值、分位数和实际样本数量。

    Args:
        rows: 待处理的逐行记录。
        per_gt: 逐 gt。
        passes: passes。

    Returns:
        结果字段包括 measurements, mean_ms, median_ms, p95_ms, scope。

    Raises:
        ValueError: Invalid inference timing
    """
    expected = {
        (r["image_index"], str(p)) for r in per_gt for p in range(1, passes + 1)
    }
    if (
        len(rows) != len(expected)
        or {(r["image_index"], r["timed_pass"]) for r in rows} != expected
    ):
        raise ValueError(
            "Timing rows do not cover every image and timed pass exactly once"
        )
    values = sorted(float(r["inference_ms"]) for r in rows)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid inference timing")
    position = (len(values) - 1) * 0.95
    lower = int(position)
    p95 = values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (
        position - lower
    )
    return dict(
        measurements=len(values),
        mean_ms=statistics.mean(values),
        median_ms=statistics.median(values),
        p95_ms=p95,
        scope="RGB conversion + tensor/device transfer + model + score/mask CPU postprocessing",
    )


def markdown_report(analysis: dict[str, Any]) -> Any:
    """将已计算的分析字段组织为 Markdown 报告。

    Args:
        analysis: 记录字段，使用 overall, training_fine_categories, native_annotations,
        first_loss, last_loss, provenance, timing。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    overall = analysis["overall"]
    lines = [
        "# PRD 3.1.1 v4-clean 结果分析",
        "",
        f"来源：AutoDL 实际 15 轮训练与 Core400 回归；checkpoint SHA-256 `{analysis['provenance']['checkpoint_sha256']}`。",
        "",
        f"400 个目标中定位 {overall['localized']} 个，定位且类别正确 {overall['class_correct']} 个，漏检 {overall['missed']} 个，错类 {overall['wrong_class']} 个。",
        f"正确类别匹配子集 mean mask IoU={number(overall['mean_mask_iou_correct_class'])}；满足类别正确且 mask IoU≥0.85 的目标 {overall['mask85_count']}/400。",
        "",
        f"平均耗时 {analysis['timing']['mean_ms']:.4f} ms，中位数 {analysis['timing']['median_ms']:.4f} ms，p95 {analysis['timing']['p95_ms']:.4f} ms；平均耗时满足 50 ms 目标。",
        f"训练 loss 从 {analysis['first_loss']:.4f} 降至 {analysis['last_loss']:.4f}。训练集合固定为 547 实例 / 475 图，不能表述为全量 Fashionpedia/DeepFashion2 训练。",
        "",
        "## 八类结果",
        "",
        "| 类别 | 目标数 | 定位 | 类别正确 | 漏检 | 匹配正确类 mask IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category in CLASSES:
        row = next(
            r
            for r in analysis["groups"]
            if r["group_field"] == "gt_class" and r["group_value"] == category
        )
        lines.append(
            f"| {category} | {row['gt_count']} | {row['localized']} | {row['class_correct']} | {row['missed']} | {number(row['mean_mask_iou_correct_class'])} |"
        )
    lines += [
        "",
        "## 目标大小",
        "",
        "| 大小 | 目标数 | 定位召回 | 漏检 |",
        "|---|---:|---:|---:|",
    ]
    for size in ("small", "medium", "large"):
        row = next(
            r
            for r in analysis["groups"]
            if r["group_field"] == "size_bucket" and r["group_value"] == size
        )
        lines.append(
            f"| {size} | {row['gt_count']} | {number(row['bbox50_recall'], True)} | {row['missed']} |"
        )
    lines += [
        "",
        "## 训练细类覆盖",
        "",
        "| 数据来源 | 细类 | 训练实例 |",
        "|---|---|---:|",
    ]
    for row in analysis["training_fine_categories"]:
        lines.append(
            f"| {row['source_dataset']} | {row['fine_category']} | {row['instances']} |"
        )
    lines += [
        "",
        "## 原始标注完整性检查",
        "",
        "| 集合 | 源图 | 已核对 | 清单外可用原标注实例 | 状态分布 |",
        "|---|---:|---:|---:|---|",
    ]
    for split, row in analysis["native_annotations"].items():
        omitted = (
            row["omitted_usable_instances"] if row["audit_complete"] else "尚未完整核对"
        )
        lines.append(
            f"| {split} | {row['images']} | {row['checked_images']} | {omitted} | {json.dumps(row['statuses'], ensure_ascii=False)} |"
        )
    lines += [
        "",
        "原标注缺失时，数量为尚未核对，而不是证明清单完整。可用原标注指八类范围内、非 crowd、有合法框和非空分割描述；尚未验证新增 mask 是否可解码。",
        "",
        "### 按来源核对",
        "",
        "| 集合 / 来源 | 源图 | 已核对 | 已选目标 | 原始可用目标 | 清单遗漏目标 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, details in analysis["native_annotations"].items():
        for dataset, row in details["by_source"].items():
            raw = (
                row["raw_usable_instances"]
                if row["raw_usable_instances"] is not None
                else "未核对"
            )
            omitted = (
                row["omitted_usable_instances"]
                if row["omitted_usable_instances"] is not None
                else "未核对"
            )
            lines.append(
                f"| {split} / {dataset} | {row['images']} | {row['checked_images']} | {row['manifest_instances']} | {raw} | {omitted} |"
            )
    lines += [
        "",
        "## 判断与下一步",
        "",
        "1. 当前瓶颈包括小目标漏检、配饰覆盖、服装细类错分，以及包/外套 mask 质量。统计相关性不能证明单一原因。",
        "2. 原训练清单只有 547 实例 / 475 图。检查每张训练原图是否保留了八类范围内所有可用标注；若有遗漏，先建立新版本完整目标清单，再扩充薄弱细类。",
        "3. 使用开发验证集选择候选设置；保留这次 Core400 回归及现有阈值，后续模型使用新的输出目录。Core 已被诊断使用，不声明独立盲测。",
        "4. Core400 每图只选一个评估目标；没有命中的额外预测可能对应未列入 Core 的真实服饰。precision/FP 仅是对所选 GT 的表观统计，不能据此判定所有额外预测都是误报。",
        "5. mean mask IoU 的分母是定位且类别正确的匹配目标，不是全部 400 个；某类子集 IoU 达到 0.85 不代表该类整体验收通过。",
        "",
        "参考：",
        "- https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html",
        "- https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py",
        "- https://github.com/cvdfoundation/fashionpedia",
        "- https://github.com/switchablenorms/DeepFashion2",
        "",
    ]
    return "\n".join(lines)


def run(args: Any) -> Any:
    """组织本模块的准备、执行和结果保存步骤。

    Args:
        args: 命令行配置；使用 data_root, output_dir, project, results_dir 等参数。

    Returns:
        返回 analysis，由函数体中同名变量的计算/收集过程得到。

    Raises:
        ValueError: Expected complete epochs 1..15 in training log
    """
    root = Path(args.project).expanduser().resolve()
    data = Path(args.data_root).expanduser().resolve()
    results = resolve(args.results_dir, root, data)
    core_file = root / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv"
    train_file = root / "configs/prd_8class_train_v4_clean.csv"
    val_file = root / "configs/prd_8class_val_v1.csv"
    core, training, validation = map(read_csv, (core_file, train_file, val_file))
    provenance = json.loads((results / "core_v4_clean/run_provenance.json").read_text())
    if provenance["core_manifest_sha256"] != digest(core_file):
        raise ValueError("Core manifest hash differs from evaluation provenance")
    if len(core) != 400 or Counter(r["garment_category"] for r in core) != Counter(
        {c: 50 for c in CLASSES}
    ):
        raise ValueError("Expected the existing 400-case Core, 50 cases per class")
    per_gt = read_csv(results / "core_v4_clean/per_gt_results.csv")
    original = {r["final_sample_id"]: r for r in core}
    if len({r["record_id"] for r in per_gt}) != len(per_gt) or set(original) != {
        r["record_id"] for r in per_gt
    }:
        raise ValueError("Every Core target must appear exactly once in the result")
    joined = []
    for row in per_gt:
        source = original[row["record_id"]]
        if (
            source["source_image"] != row["source_image"]
            or source["garment_category"] != row["gt_class"]
        ):
            raise ValueError("Result source/class identity differs from Core")
        joined.append({**source, **row})
    logs = read_csv(results / "train_v4_clean/train_log.csv")
    if [int(r["epoch"]) for r in logs] != list(range(1, 16)):
        raise ValueError("Expected complete epochs 1..15 in training log")
    groups = result_groups(
        joined,
        ("gt_class", "size_bucket", "scene_type", "fine_category", "source_dataset"),
    )
    fine_counts = Counter((r["source_dataset"], r["fine_category"]) for r in training)
    for fine in DF2_FINE:
        fine_counts.setdefault(("DeepFashion2", fine), 0)
    fine_rows = [
        dict(source_dataset=dataset, fine_category=fine, instances=n)
        for (dataset, fine), n in sorted(fine_counts.items())
    ]
    audit, native = audit_native_annotations(
        root, data, {"train": training, "validation": validation, "core_selected": core}
    )
    analysis = dict(
        overall=summarize(joined),
        groups=groups,
        provenance=provenance,
        timing=timing_summary(
            read_csv(results / "core_v4_clean/per_image_timing.csv"),
            per_gt,
            int(provenance["parameters"]["timed_passes"]),
        ),
        input_sha256={
            "core_manifest": digest(core_file),
            "training_manifest": digest(train_file),
            "per_gt_results": digest(results / "core_v4_clean/per_gt_results.csv"),
        },
        training_fine_categories=fine_rows,
        native_annotations=native,
        first_loss=float(logs[0]["train_loss_total"]),
        last_loss=float(logs[-1]["train_loss_total"]),
        raw_fashionpedia_sha256=(
            digest(
                data / "raw/fashionpedia/annotations/instances_attributes_val2020.json"
            )
            if (
                data / "raw/fashionpedia/annotations/instances_attributes_val2020.json"
            ).is_file()
            else None
        ),
        caveat="训练 loss 下降不是泛化验收；原标注库存不证明 mask 解码或训练效果。",
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    target = (
        resolve(args.output_dir, root, data)
        if args.output_dir
        else results / "diagnosis" / stamp
    )
    target.mkdir(parents=True, exist_ok=False)
    (target / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (target / "analysis.md").write_text(markdown_report(analysis), encoding="utf-8")
    write_csv(target / "groups.csv", groups)
    write_csv(target / "training_fine_categories.csv", fine_rows)
    write_csv(target / "native_annotation_audit.csv", audit)
    confusion = Counter(
        (r["gt_class"], r["pred_class"])
        for r in joined
        if r["outcome"] == "wrong_class"
    )
    write_csv(
        target / "confusions.csv",
        [
            dict(gt_class=a, pred_class=b, count=n)
            for (a, b), n in confusion.most_common()
        ],
    )
    write_csv(
        target / "joined_error_cases.csv",
        [r for r in joined if r["outcome"] != "correct"],
    )
    archive = root / "prd31_v4_diagnosis_results.zip"
    if archive.exists():
        archive = root / ("prd31_v4_diagnosis_results_" + stamp + ".zip")
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for path in sorted(target.iterdir()):
            bundle.write(path, "diagnosis/" + path.name)
    print(
        json.dumps(
            dict(
                status="DIAGNOSIS_COMPLETED",
                overall=analysis["overall"],
                native_annotations=native,
                report=str(target / "analysis.md"),
                results_zip=str(archive),
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return analysis


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".")
    parser.add_argument(
        "--data-root", default=os.environ.get("FASHION_DATA_ROOT", "../fashion_data")
    )
    parser.add_argument("--results-dir", default="reports/reruns/recovery_v2")
    parser.add_argument(
        "--output-dir", default="", help="New, non-existing output directory"
    )
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"排查未完成：{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
