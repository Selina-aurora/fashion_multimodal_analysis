"""3.1.1 实例分割：数据与模型版本分别记录，指标按固定匹配和计时口径解释。"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from collections import defaultdict
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
EVAL_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "segmentation/evaluation"
    / "eval_prd_8class_maskrcnn_baseline_v1.py"
)
VAL_CSV = PROJECT_ROOT / "configs" / "prd_8class_val_v1.csv"

BALANCED_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "prd_instance_segmentation"
    / "maskrcnn_8class_baseline_v2"
    / "checkpoint_last.pth"
)
NO_BALANCE_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "prd_instance_segmentation"
    / "maskrcnn_8class_baseline_v2_nobalance"
    / "checkpoint_last.pth"
)

TEMP_EVAL_DIR = (
    PROJECT_ROOT / "reports" / "prd_instance_segmentation" / "maskrcnn_8class_eval_v1"
)
BALANCED_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_instance_segmentation"
    / "maskrcnn_8class_eval_v2_balanced_fixed040"
)
NO_BALANCE_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_instance_segmentation"
    / "maskrcnn_8class_eval_v2_nobalance_fixed040"
)
COMPARE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_instance_segmentation"
    / "maskrcnn_balanced_vs_nobalance_v1"
)

CLASSES = ["top", "pants", "skirt", "outerwear", "dress", "shoe", "bag", "accessory"]
OLD5 = {"top", "pants", "skirt", "outerwear", "dress"}
NEW3 = {"shoe", "bag", "accessory"}


def parse_args() -> Any:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    p.add_argument("--score-threshold", type=float, default=0.40)
    p.add_argument("--mask-threshold", type=float, default=0.50)
    p.add_argument("--match-bbox-iou", type=float, default=0.50)
    return p.parse_args()


def read_csv(path: str | Path) -> Any:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
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
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def parse_summary(path: str | Path) -> Any:
    """解析 汇总。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        返回 out，由函数体中同名变量的计算/收集过程得到。
    """
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def fnum(v: Any) -> Any:
    """fnum。

    Args:
        v: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    s = str(v or "").strip()
    if not s or s.upper() == "NA":
        return math.nan
    try:
        return float(s)
    except Exception:
        return math.nan


def fmt(v: Any) -> Any:
    """把统计值转换为报告显示文本。

    Args:
        v: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return "NA" if math.isnan(v) else f"{v:.4f}"


def as_bool(v: Any) -> bool:
    """as bool。

    Args:
        v: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def run_eval(name: str, checkpoint: Any, destination: Any, args: Any) -> None:
    """执行 eval。

    Args:
        name: 当前标签、字段或产物名称。
        checkpoint: 待检查或使用的训练权重/状态。
        destination: 目标文件或目录。
        args: 已经解析的命令行配置。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not EVAL_SCRIPT.is_file():
        raise FileNotFoundError(EVAL_SCRIPT)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not VAL_CSV.is_file():
        raise FileNotFoundError(VAL_CSV)

    if TEMP_EVAL_DIR.exists():
        shutil.rmtree(TEMP_EVAL_DIR)

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--val-csv",
        str(VAL_CSV),
        "--device",
        args.device,
        "--score-threshold",
        str(args.score_threshold),
        "--mask-threshold",
        str(args.mask_threshold),
        "--match-bbox-iou",
        str(args.match_bbox_iou),
        "--max-visualizations",
        "0",
    ]

    print("\n" + "=" * 70)
    print("Evaluating:", name)
    print("=" * 70)

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(TEMP_EVAL_DIR, destination)


def class_exact(report_dir: Path) -> Any:
    """类别 exact。

    Args:
        report_dir: 保存本次报告的目录。

    Returns:
        返回 out，由函数体中同名变量的计算/收集过程得到。
    """
    rows = read_csv(report_dir / "per_gt_results.csv")
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["gt_class"].strip()].append(r)

    out = {}
    for cls in CLASSES:
        rr = grouped.get(cls, [])
        correct = [
            r
            for r in rr
            if as_bool(r.get("localized_bbox50")) and as_bool(r.get("class_correct"))
        ]
        ious = [fnum(r.get("mask_iou")) for r in correct]
        ious = [x for x in ious if not math.isnan(x)]
        high = sum(as_bool(r.get("mask_iou_ge_0_85")) for r in rr)

        out[cls] = {
            "gt_count": len(rr),
            "correct_class_count": len(correct),
            "mean_mask_iou_correct_class": (
                sum(ious) / len(ious) if ious else math.nan
            ),
            "mask_iou_ge_085_count": high,
            "mask_iou_ge_085_rate": (high / len(rr) if rr else math.nan),
        }
    return out


def load_by_key(path: str | Path, key: Any) -> dict[str, Any]:
    """加载 by 标识。

    Args:
        path: 要读取或写入的文件路径。
        key: 标识。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。
    """
    return {r[key].strip(): r for r in read_csv(path)}


def pooled(exact: Any, class_set: Any) -> dict[str, Any]:
    """pooled。

    Args:
        exact: exact。
        class_set: 类别 set。

    Returns:
        结果字典，主要字段为 gt_count, correct_class_count, mean_mask_iou_correct_class,
        mask_iou_ge_085_count, mask_iou_ge_085_rate。
    """
    gt = 0
    correct = 0
    high = 0
    weighted_sum = 0.0

    for cls in class_set:
        x = exact[cls]
        gt += x["gt_count"]
        correct += x["correct_class_count"]
        high += x["mask_iou_ge_085_count"]

        if x["correct_class_count"] > 0 and not math.isnan(
            x["mean_mask_iou_correct_class"]
        ):
            weighted_sum += x["mean_mask_iou_correct_class"] * x["correct_class_count"]

    mean_iou = weighted_sum / correct if correct else math.nan

    return {
        "gt_count": gt,
        "correct_class_count": correct,
        "mean_mask_iou_correct_class": mean_iou,
        "mask_iou_ge_085_count": high,
        "mask_iou_ge_085_rate": high / gt if gt else math.nan,
    }


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    args = parse_args()
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    run_eval("v2-balanced", BALANCED_CKPT, BALANCED_REPORT_DIR, args)
    run_eval("v2-no-balance", NO_BALANCE_CKPT, NO_BALANCE_REPORT_DIR, args)

    b_summary = parse_summary(BALANCED_REPORT_DIR / "evaluation_summary.txt")
    n_summary = parse_summary(NO_BALANCE_REPORT_DIR / "evaluation_summary.txt")

    b_class = load_by_key(
        BALANCED_REPORT_DIR / "per_class_metrics.csv", "garment_category"
    )
    n_class = load_by_key(
        NO_BALANCE_REPORT_DIR / "per_class_metrics.csv", "garment_category"
    )

    b_exact = class_exact(BALANCED_REPORT_DIR)
    n_exact = class_exact(NO_BALANCE_REPORT_DIR)

    per_class = []

    for cls in CLASSES:
        br = b_class[cls]
        nr = n_class[cls]
        be = b_exact[cls]
        ne = n_exact[cls]

        bmi = be["mean_mask_iou_correct_class"]
        nmi = ne["mean_mask_iou_correct_class"]

        per_class.append(
            {
                "garment_category": cls,
                "group": "DeepFashion2_old5" if cls in OLD5 else "Fashionpedia_new3",
                "gt_count": be["gt_count"],
                "balanced_bbox50_recall": br.get("bbox50_recall", ""),
                "nobalance_bbox50_recall": nr.get("bbox50_recall", ""),
                "delta_bbox50_recall": fmt(
                    fnum(br.get("bbox50_recall")) - fnum(nr.get("bbox50_recall"))
                ),
                "balanced_category_acc_localized": br.get(
                    "category_accuracy_on_localized", ""
                ),
                "nobalance_category_acc_localized": nr.get(
                    "category_accuracy_on_localized", ""
                ),
                "balanced_correct_class_count": be["correct_class_count"],
                "nobalance_correct_class_count": ne["correct_class_count"],
                "balanced_mean_mask_iou_correct_class": fmt(bmi),
                "nobalance_mean_mask_iou_correct_class": fmt(nmi),
                "delta_mean_mask_iou": fmt(bmi - nmi),
                "balanced_mask_iou_ge_085_count": be["mask_iou_ge_085_count"],
                "nobalance_mask_iou_ge_085_count": ne["mask_iou_ge_085_count"],
                "balanced_mask_iou_ge_085_rate": fmt(be["mask_iou_ge_085_rate"]),
                "nobalance_mask_iou_ge_085_rate": fmt(ne["mask_iou_ge_085_rate"]),
                "delta_mask_iou_ge_085_rate": fmt(
                    be["mask_iou_ge_085_rate"] - ne["mask_iou_ge_085_rate"]
                ),
            }
        )

    write_csv(COMPARE_DIR / "per_class_comparison.csv", per_class)

    b_source = load_by_key(
        BALANCED_REPORT_DIR / "per_source_metrics.csv", "source_dataset"
    )
    n_source = load_by_key(
        NO_BALANCE_REPORT_DIR / "per_source_metrics.csv", "source_dataset"
    )

    per_source = []
    for src in sorted(set(b_source) | set(n_source)):
        br = b_source.get(src, {})
        nr = n_source.get(src, {})

        per_source.append(
            {
                "source_dataset": src,
                "gt_count": br.get("gt_count", nr.get("gt_count", "")),
                "balanced_bbox50_recall": br.get("bbox50_recall", ""),
                "nobalance_bbox50_recall": nr.get("bbox50_recall", ""),
                "balanced_category_acc_localized": br.get(
                    "category_accuracy_on_localized", ""
                ),
                "nobalance_category_acc_localized": nr.get(
                    "category_accuracy_on_localized", ""
                ),
                "balanced_mean_mask_iou_correct_class": br.get(
                    "mean_mask_iou_correct_class", ""
                ),
                "nobalance_mean_mask_iou_correct_class": nr.get(
                    "mean_mask_iou_correct_class", ""
                ),
                "delta_mean_mask_iou": fmt(
                    fnum(br.get("mean_mask_iou_correct_class"))
                    - fnum(nr.get("mean_mask_iou_correct_class"))
                ),
                "balanced_mask_iou_ge_085_rate": br.get("mask_iou_ge_0_85_rate", ""),
                "nobalance_mask_iou_ge_085_rate": nr.get("mask_iou_ge_0_85_rate", ""),
            }
        )

    write_csv(COMPARE_DIR / "per_source_comparison.csv", per_source)

    grouped = []
    for name, classes in [
        ("DeepFashion2_old5", OLD5),
        ("Fashionpedia_new3", NEW3),
    ]:
        bg = pooled(b_exact, classes)
        ng = pooled(n_exact, classes)

        grouped.append(
            {
                "group": name,
                "gt_count": bg["gt_count"],
                "balanced_correct_class_count": bg["correct_class_count"],
                "nobalance_correct_class_count": ng["correct_class_count"],
                "balanced_mean_mask_iou_correct_class": fmt(
                    bg["mean_mask_iou_correct_class"]
                ),
                "nobalance_mean_mask_iou_correct_class": fmt(
                    ng["mean_mask_iou_correct_class"]
                ),
                "delta_mean_mask_iou": fmt(
                    bg["mean_mask_iou_correct_class"]
                    - ng["mean_mask_iou_correct_class"]
                ),
                "balanced_mask_iou_ge_085_count": bg["mask_iou_ge_085_count"],
                "nobalance_mask_iou_ge_085_count": ng["mask_iou_ge_085_count"],
                "balanced_mask_iou_ge_085_rate": fmt(bg["mask_iou_ge_085_rate"]),
                "nobalance_mask_iou_ge_085_rate": fmt(ng["mask_iou_ge_085_rate"]),
                "delta_mask_iou_ge_085_rate": fmt(
                    bg["mask_iou_ge_085_rate"] - ng["mask_iou_ge_085_rate"]
                ),
            }
        )

    write_csv(COMPARE_DIR / "grouped_mask_comparison.csv", grouped)

    speed_rows = []
    for name, s in [("v2-balanced", b_summary), ("v2-no-balance", n_summary)]:
        mean_ms = fnum(s.get("mean_inference_ms_per_image"))
        med_ms = fnum(s.get("median_inference_ms_per_image"))
        p95_ms = fnum(s.get("p95_inference_ms_per_image"))

        speed_rows.append(
            {
                "version": name,
                "mean_ms": fmt(mean_ms),
                "median_ms": fmt(med_ms),
                "p95_ms": fmt(p95_ms),
                "mean_le_50ms": "yes" if mean_ms <= 50 else "no",
                "p95_le_50ms": "yes" if p95_ms <= 50 else "no",
            }
        )

    write_csv(COMPARE_DIR / "speed_comparison.csv", speed_rows)

    overall_rows = []
    for metric in [
        "bbox50_recall",
        "category_accuracy_on_localized",
        "end_to_end_bbox50_class_recall",
        "mean_mask_iou_correct_class",
        "mask_iou_ge_0_85_rate",
        "mean_inference_ms_per_image",
        "median_inference_ms_per_image",
        "p95_inference_ms_per_image",
    ]:
        bv = fnum(b_summary.get(metric))
        nv = fnum(n_summary.get(metric))

        overall_rows.append(
            {
                "metric": metric,
                "balanced": fmt(bv),
                "no_balance": fmt(nv),
                "balanced_minus_no_balance": fmt(bv - nv),
            }
        )

    write_csv(COMPARE_DIR / "overall_comparison.csv", overall_rows)

    bg_old = pooled(b_exact, OLD5)
    ng_old = pooled(n_exact, OLD5)
    bg_new = pooled(b_exact, NEW3)
    ng_new = pooled(n_exact, NEW3)

    lines = [
        "PRD 3.1.1 Balanced vs No-Balance Diagnostic",
        "===========================================",
        "",
        f"score_threshold={args.score_threshold}",
        f"mask_threshold={args.mask_threshold}",
        f"match_bbox_iou={args.match_bbox_iou}",
        "",
        "[DeepFashion2 old5]",
        f"balanced_mean_mask_iou={fmt(bg_old['mean_mask_iou_correct_class'])}",
        f"nobalance_mean_mask_iou={fmt(ng_old['mean_mask_iou_correct_class'])}",
        f"delta_mean_mask_iou={fmt(bg_old['mean_mask_iou_correct_class'] - ng_old['mean_mask_iou_correct_class'])}",
        f"balanced_high_iou_count={bg_old['mask_iou_ge_085_count']}",
        f"nobalance_high_iou_count={ng_old['mask_iou_ge_085_count']}",
        "",
        "[Fashionpedia new3]",
        f"balanced_mean_mask_iou={fmt(bg_new['mean_mask_iou_correct_class'])}",
        f"nobalance_mean_mask_iou={fmt(ng_new['mean_mask_iou_correct_class'])}",
        f"delta_mean_mask_iou={fmt(bg_new['mean_mask_iou_correct_class'] - ng_new['mean_mask_iou_correct_class'])}",
        f"balanced_high_iou_count={bg_new['mask_iou_ge_085_count']}",
        f"nobalance_high_iou_count={ng_new['mask_iou_ge_085_count']}",
        "",
        "[Interpretation rule]",
        "- old5 stable + new3 decline: aggregate drop is mainly from the added/smaller classes.",
        "- old5 also declines clearly: balanced sampling is affecting original garment classes; consider reducing sampler strength.",
        "- mean IoU nearly unchanged but >=0.85 count drops: describe it as a high-IoU-tail change, not a general mask-head collapse.",
        "- small fixed validation set: descriptive evidence only, not proof of causal mechanism.",
        "",
        "[Speed]",
        f"balanced_mean_ms={b_summary.get('mean_inference_ms_per_image', 'NA')}",
        f"balanced_median_ms={b_summary.get('median_inference_ms_per_image', 'NA')}",
        f"balanced_p95_ms={b_summary.get('p95_inference_ms_per_image', 'NA')}",
        f"nobalance_mean_ms={n_summary.get('mean_inference_ms_per_image', 'NA')}",
        f"nobalance_median_ms={n_summary.get('median_inference_ms_per_image', 'NA')}",
        f"nobalance_p95_ms={n_summary.get('p95_inference_ms_per_image', 'NA')}",
    ]

    (COMPARE_DIR / "diagnosis.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n=== COMPARISON FINISHED ===")
    print("Output:", COMPARE_DIR.relative_to(PROJECT_ROOT))
    print("Key files:")
    print("  per_class_comparison.csv")
    print("  grouped_mask_comparison.csv")
    print("  per_source_comparison.csv")
    print("  speed_comparison.csv")
    print("  diagnosis.txt")


if __name__ == "__main__":
    main()
