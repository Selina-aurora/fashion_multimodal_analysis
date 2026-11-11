"""3.1 接口联调：区分预测 ROI 传递和使用 GT 的受控诊断，保留对象 ID 与坐标对应关系。

PRD 3.1 full-validation propagation expansion — Stage 1

Purpose
-------
Expand the previous 20-image / 13-pair pilot to the ENTIRE frozen 3.1.1
validation split, without overwriting the pilot outputs.

This wrapper reuses the two scripts already in scripts/:
1) build_prd31_predicted_roi_pilot_v1.py
2) build_prd31_matched_roi_manifest_v1.py

It will:
A. run v2-balanced on ALL validation image groups;
B. save every prediction >= 0.40 as predicted ROI;
C. build one-to-one same-class bbox-IoU>=0.50 matched pairs.

Outputs
-------
configs/
├── prd_3_1_predicted_roi_fullval_v1.csv
└── prd_3_1_predicted_roi_fullval_matched_v1.csv

reports/prd_integration/
├── predicted_roi_fullval_v1/
│   ├── pilot_image_selection.csv
│   ├── predicted_instances.csv
│   ├── gt_match_diagnostic.csv
│   └── summary.txt
└── predicted_roi_fullval_matched_v1/
    ├── matched_pairs.csv
    ├── unmatched_predictions.csv
    ├── unmatched_gt.csv
    └── summary.txt

outputs/prd_integration/predicted_roi_fullval_v1/
└── contact_sheet.jpg

Run
---
cd fashion_multimodal_analysis

python scripts/integration/run_prd31_fullval_expansion_stage1.py   --device cuda
"""

from __future__ import annotations

import argparse
import sys
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

PRED_MANIFEST = PROJECT_ROOT / "configs" / "prd_3_1_predicted_roi_fullval_v1.csv"

PRED_REPORT_DIR = (
    PROJECT_ROOT / "reports" / "prd_integration" / "predicted_roi_fullval_v1"
)

PRED_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "prd_integration" / "predicted_roi_fullval_v1"
)

MATCHED_MANIFEST = (
    PROJECT_ROOT / "configs" / "prd_3_1_predicted_roi_fullval_matched_v1.csv"
)

MATCHED_REPORT_DIR = (
    PROJECT_ROOT / "reports" / "prd_integration" / "predicted_roi_fullval_matched_v1"
)


def parse_args() -> Any:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.40,
    )
    p.add_argument(
        "--mask-threshold",
        type=float,
        default=0.50,
    )
    p.add_argument(
        "--match-bbox-iou",
        type=float,
        default=0.50,
    )
    return p.parse_args()


def run_with_argv(main_func: Any, argv: list[str]) -> None:
    """执行 with argv。

    Args:
        main_func: main func。
        argv: argv。
    """
    old = sys.argv[:]
    try:
        sys.argv = argv
        main_func()
    finally:
        sys.argv = old


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    args = parse_args()

    import fashion_multimodal_analysis.integration.build_prd31_matched_roi_manifest_v1 as match
    import fashion_multimodal_analysis.integration.build_prd31_predicted_roi_pilot_v1 as pred

    # ---------------------------------------------------------
    # A. Full validation prediction
    # ---------------------------------------------------------
    pred.MANIFEST_OUT = PRED_MANIFEST
    pred.REPORT_DIR = PRED_REPORT_DIR
    pred.OUTPUT_DIR = PRED_OUTPUT_DIR

    print("\n" + "=" * 72)
    print("STAGE A — v2-balanced predictions on FULL validation split")
    print("=" * 72)

    run_with_argv(
        pred.main,
        [
            "build_prd31_predicted_roi_pilot_v1.py",
            "--device",
            args.device,
            # Large number => category_aware_select returns all available samples.
            "--num-images",
            "999",
            "--score-threshold",
            str(args.score_threshold),
            "--mask-threshold",
            str(args.mask_threshold),
            "--match-bbox-iou",
            str(args.match_bbox_iou),
        ],
    )

    # ---------------------------------------------------------
    # B. One-to-one same-class matching
    # ---------------------------------------------------------
    match.DEFAULT_PREDICTED = PRED_MANIFEST
    match.DEFAULT_OUTPUT = MATCHED_MANIFEST
    match.REPORT_DIR = MATCHED_REPORT_DIR

    print("\n" + "=" * 72)
    print("STAGE B — one-to-one same-class matched ROI pairs")
    print("=" * 72)

    run_with_argv(
        match.main,
        [
            "build_prd31_matched_roi_manifest_v1.py",
            "--predicted",
            str(PRED_MANIFEST),
            "--gt",
            str(PROJECT_ROOT / "configs" / "prd_8class_val_v1.csv"),
            "--output",
            str(MATCHED_MANIFEST),
            "--iou-threshold",
            str(args.match_bbox_iou),
        ],
    )

    print("\n" + "=" * 72)
    print("FULL-VALIDATION EXPANSION STAGE 1 FINISHED")
    print("=" * 72)
    print(
        "Prediction summary:",
        PRED_REPORT_DIR.relative_to(PROJECT_ROOT) / "summary.txt",
    )
    print(
        "Matched summary   :",
        MATCHED_REPORT_DIR.relative_to(PROJECT_ROOT) / "summary.txt",
    )
    print(
        "Contact sheet     :",
        PRED_OUTPUT_DIR.relative_to(PROJECT_ROOT) / "contact_sheet.jpg",
    )
    print(
        "Matched manifest  :",
        MATCHED_MANIFEST.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
