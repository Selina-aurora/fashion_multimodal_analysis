from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FINAL_DIR = (
    ROOT
    / "reports"
    / "prd_3_1_v1"
    / "final_3_1_1_v1"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPERIMENTS = {
    "V2-balanced": {
        "change": "Reference baseline: class-balanced sampler",
        "report": ROOT / "reports/prd_3_1_v1/core_regression_v2_balanced_v1_evalfix",
    },
    "V3-A1-highres": {
        "change": "Higher input resolution: 800/1280",
        "report": ROOT / "reports/prd_3_1_v1/core_regression_v3_a1_highres_v1",
    },
    "V3-A2-smallanchors": {
        "change": "All RPN anchor scales shifted smaller",
        "report": ROOT / "reports/prd_3_1_v1/core_regression_v3_a2_smallanchors_v1",
    },
    "V3-A2b-P2small": {
        "change": "Only P2 anchor changed from 32 to 16",
        "report": ROOT / "reports/prd_3_1_v1/core_regression_v3_a2b_p2small_v1",
    },
    "V3-A3-smallsampling": {
        "change": "Small-object-aware training sampler",
        "report": ROOT / "reports/prd_3_1_v1/core_regression_v3_a3_smallsampling_v1",
    },
    "V3-B1-dataexp": {
        "change": "Unique-data expansion for underrepresented classes",
        "report": ROOT / "reports/prd_3_1_v1/core_regression_v3_b1_dataexp_v1",
    },
}

FINAL_MODEL = "V3-B1-dataexp"

FINAL_CHECKPOINT = (
    ROOT
    / "outputs"
    / "prd_instance_segmentation"
    / "maskrcnn_8class_v3_b1_dataexp"
    / "checkpoint_last.pth"
)

FINAL_TRAIN_MANIFEST = (
    ROOT
    / "configs"
    / "prd_8class_train_v3.csv"
)

CORE_MANIFEST = (
    ROOT
    / "benchmark"
    / "prd_3_1_v1"
    / "manifests"
    / "segmentation_test_v1.csv"
)


def parse_summary(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}

    result = {}

    for raw in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if "=" not in raw:
            continue

        key, value = raw.split("=", 1)
        result[key.strip()] = value.strip()

    return result


metrics = [
    "predictions_above_threshold",
    "bbox50_recall",
    "bbox50_precision",
    "mean_bbox_iou_localized",
    "category_accuracy_on_localized",
    "end_to_end_bbox50_class_recall",
    "mean_mask_iou_localized",
    "mean_mask_iou_correct_class",
    "mask_iou_ge_0_50_rate",
    "mask_iou_ge_0_85_rate",
    "mean_inference_ms_per_image",
    "correct",
    "low_mask_iou",
    "wrong_class",
    "missed",
]

rows = []

for name, info in EXPERIMENTS.items():
    summary_path = (
        info["report"]
        / "evaluation_summary.txt"
    )

    parsed = parse_summary(
        summary_path
    )

    if not parsed:
        print(
            f"WARNING: missing summary for {name}: "
            f"{summary_path}"
        )

    row = {
        "experiment": name,
        "controlled_change": info["change"],
        "selected_final": (
            "YES"
            if name == FINAL_MODEL
            else "NO"
        ),
    }

    for metric in metrics:
        row[metric] = parsed.get(
            metric,
            "",
        )

    rows.append(row)


# ------------------------------------------------------------
# Write ablation table.
# ------------------------------------------------------------
csv_path = (
    FINAL_DIR
    / "ablation_comparison.csv"
)

with csv_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "experiment",
            "controlled_change",
            "selected_final",
            *metrics,
        ],
    )
    writer.writeheader()
    writer.writerows(rows)


# ------------------------------------------------------------
# Selected model artifact copies.
# ------------------------------------------------------------
selected_report = (
    EXPERIMENTS[FINAL_MODEL]["report"]
)

copy_map = {
    "evaluation_summary.txt":
        "selected_model_evaluation_summary.txt",
    "per_class_metrics.csv":
        "selected_model_per_class_metrics.csv",
    "error_cases.csv":
        "selected_model_error_cases.csv",
    "per_gt_results.csv":
        "selected_model_per_gt_results.csv",
    "per_source_metrics.csv":
        "selected_model_per_source_metrics.csv",
    "per_image_timing.csv":
        "selected_model_per_image_timing.csv",
}

for src_name, dst_name in copy_map.items():
    src = selected_report / src_name
    dst = FINAL_DIR / dst_name

    if src.is_file():
        shutil.copy2(
            src,
            dst,
        )
    else:
        print(
            f"WARNING: missing optional file: {src}"
        )


# ------------------------------------------------------------
# Model pointer.
# ------------------------------------------------------------
pointer = [
    "PRD 3.1.1 FINAL MODEL POINTER",
    "=============================",
    "",
    f"selected_model={FINAL_MODEL}",
    f"checkpoint={FINAL_CHECKPOINT.relative_to(ROOT)}",
    f"train_manifest={FINAL_TRAIN_MANIFEST.relative_to(ROOT)}",
    f"core_manifest={CORE_MANIFEST.relative_to(ROOT)}",
    "score_threshold=0.40",
    "mask_threshold=0.50",
    "bbox_match_iou=0.50",
    "mask_reference_iou=0.85",
    "",
    "Important:",
    "- Do not overwrite the V2 reference baseline.",
    "- Core-400 remains frozen.",
    "- V3-B1 is the selected final model for PRD 3.1.1.",
]

(
    FINAL_DIR
    / "FINAL_MODEL_POINTER.txt"
).write_text(
    "\n".join(pointer) + "\n",
    encoding="utf-8",
)


# ------------------------------------------------------------
# Build final Markdown report.
# ------------------------------------------------------------
by_name = {
    r["experiment"]: r
    for r in rows
}

v2 = by_name.get(
    "V2-balanced",
    {},
)

b1 = by_name.get(
    "V3-B1-dataexp",
    {},
)


def pct(v: str) -> str:
    try:
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return "NA"


def num(v: str, n=4) -> str:
    try:
        return f"{float(v):.{n}f}"
    except Exception:
        return "NA"


lines = [
    "# PRD 3.1.1 Final Instance Segmentation Report",
    "",
    "## 1. Final decision",
    "",
    "**Selected final model: V3-B1 Data Expansion.**",
    "",
    "V2-balanced is retained as the reference baseline. "
    "V3-B1 is selected because it preserves approximately the same "
    "localization recall while improving localized category accuracy, "
    "end-to-end correct-class recall, mask quality, and the total number "
    "of correct Core-400 instances.",
    "",
    "## 2. Frozen evaluation protocol",
    "",
    "- Benchmark: Core-400",
    "- Classes: top, pants, skirt, outerwear, dress, shoe, bag, accessory",
    "- 50 GT cases per class; 400 total",
    "- Score threshold: 0.40",
    "- Prediction mask threshold: 0.50",
    "- BBox match IoU: 0.50",
    "- Mask reference threshold: IoU >= 0.85",
    "- Benchmark remained unchanged during V3 ablations",
    "",
    "## 3. Ablation experiments",
    "",
    "| Experiment | Controlled change | BBox Recall | "
    "Category Acc. | E2E Recall | Mask IoU (correct) | "
    "Mask>=0.85 | Correct | Missed | Mean ms |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for r in rows:
    lines.append(
        "| "
        + " | ".join(
            [
                r["experiment"],
                r["controlled_change"],
                num(r.get("bbox50_recall", "")),
                num(
                    r.get(
                        "category_accuracy_on_localized",
                        "",
                    )
                ),
                num(
                    r.get(
                        "end_to_end_bbox50_class_recall",
                        "",
                    )
                ),
                num(
                    r.get(
                        "mean_mask_iou_correct_class",
                        "",
                    )
                ),
                num(
                    r.get(
                        "mask_iou_ge_0_85_rate",
                        "",
                    )
                ),
                r.get("correct", "") or "NA",
                r.get("missed", "") or "NA",
                num(
                    r.get(
                        "mean_inference_ms_per_image",
                        "",
                    ),
                    2,
                ),
            ]
        )
        + " |"
    )


lines += [
    "",
    "## 4. V2 versus final V3-B1",
    "",
    f"- BBox50 recall: "
    f"{num(v2.get('bbox50_recall', ''))} -> "
    f"{num(b1.get('bbox50_recall', ''))}",
    f"- Localized category accuracy: "
    f"{num(v2.get('category_accuracy_on_localized', ''))} -> "
    f"{num(b1.get('category_accuracy_on_localized', ''))}",
    f"- End-to-end bbox50 + class recall: "
    f"{num(v2.get('end_to_end_bbox50_class_recall', ''))} -> "
    f"{num(b1.get('end_to_end_bbox50_class_recall', ''))}",
    f"- Mean mask IoU on correct class: "
    f"{num(v2.get('mean_mask_iou_correct_class', ''))} -> "
    f"{num(b1.get('mean_mask_iou_correct_class', ''))}",
    f"- Mask IoU >= 0.85 rate: "
    f"{num(v2.get('mask_iou_ge_0_85_rate', ''))} -> "
    f"{num(b1.get('mask_iou_ge_0_85_rate', ''))}",
    f"- Correct cases: "
    f"{v2.get('correct', 'NA')} -> "
    f"{b1.get('correct', 'NA')}",
    f"- Missed cases: "
    f"{v2.get('missed', 'NA')} -> "
    f"{b1.get('missed', 'NA')}",
    f"- Mean inference: "
    f"{num(v2.get('mean_inference_ms_per_image', ''), 2)} ms -> "
    f"{num(b1.get('mean_inference_ms_per_image', ''), 2)} ms",
    "",
    "## 5. Interpretation",
    "",
    "The V3 ablations show that the main remaining limitation is "
    "missed detection rather than mask refinement alone. Increasing "
    "input resolution did not improve the primary detection metrics. "
    "Smaller RPN anchors produced only limited gains and introduced "
    "trade-offs across object-size groups. Small-object-aware "
    "oversampling degraded overall detection. In contrast, adding "
    "new, non-overlapping training examples for underrepresented "
    "classes improved classification and mask quality without a "
    "meaningful speed penalty.",
    "",
    "V3-B1 does not materially improve pure localization recall over "
    "V2-balanced; the difference corresponds to approximately one "
    "localized GT instance on the 400-case benchmark. Its main benefit "
    "is better downstream correctness after localization.",
    "",
    "## 6. Final limitation",
    "",
    "**Missed detection remains the primary bottleneck.** "
    "Further work should therefore prioritize more diverse detection "
    "training data and/or a detector specifically optimized for small "
    "and weak-boundary garments, rather than continuing threshold, "
    "resolution, anchor, or sampler tuning on the frozen benchmark.",
    "",
    "## 7. Final artifacts",
    "",
    f"- Final checkpoint: `{FINAL_CHECKPOINT.relative_to(ROOT)}`",
    f"- Final train manifest: `{FINAL_TRAIN_MANIFEST.relative_to(ROOT)}`",
    f"- Frozen Core benchmark: `{CORE_MANIFEST.relative_to(ROOT)}`",
    f"- Final report directory: `{FINAL_DIR.relative_to(ROOT)}`",
    "",
    "## 8. Status",
    "",
    "**PRD 3.1.1 instance segmentation: CLOSED for the current phase.**",
    "",
]

(
    FINAL_DIR
    / "FINAL_MODEL_SELECTION.md"
).write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print()
print("=== PRD 3.1.1 FINALIZATION COMPLETE ===")
print()
print("Selected model:", FINAL_MODEL)
print(
    "Checkpoint:",
    FINAL_CHECKPOINT.relative_to(ROOT),
)
print(
    "Comparison:",
    csv_path.relative_to(ROOT),
)
print(
    "Final report:",
    (
        FINAL_DIR
        / "FINAL_MODEL_SELECTION.md"
    ).relative_to(ROOT),
)
print(
    "Final directory:",
    FINAL_DIR.relative_to(ROOT),
)
