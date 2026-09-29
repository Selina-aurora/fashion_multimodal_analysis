"""
Compatibility-fixed wrapper for PRD 3.1 predicted-ROI attribute pilot.

Fixes the current transformers incompatibility:
CLIPTextTransformer.forward() got an unexpected keyword argument 'return_dict'

The frozen baseline scripts are NOT modified. This wrapper monkey-patches only
their CLIP text/image encoder helpers at runtime, then runs:
1) primary color v1
2) pattern v3
3) categorical design labels v1
4) GT-ROI vs predicted-ROI label agreement

Run:
cd /workspace/fashion_multimodal_analysis
python scripts/run_prd31_predicted_roi_attributes_v1_fixed.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "configs"
    / "prd_3_1_predicted_roi_matched_v1.csv"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "prd_integration"
    / "predicted_roi_attributes_v1"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "prd_integration"
    / "predicted_roi_attributes_v1"
)

GT_COLOR = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "color_v1"
    / "primary_color_predictions.csv"
)

GT_PATTERN = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "pattern_v3_hierarchical"
    / "pattern_v3_predictions.csv"
)

GT_DESIGN = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "design_labels_v1_fixed"
    / "design_attribute_predictions_v1_fixed.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
    )
    return p.parse_args()


def resolve_path(raw: str | Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def norm_source(v: str) -> str:
    return str(v).replace("\\", "/").strip()


def key_of(row: dict[str, str]) -> tuple[str, str]:
    return (
        norm_source(row.get("source_image", "")),
        str(row.get("garment_id", "")).strip(),
    )


def run_module_main(module, argv: list[str]) -> None:
    old = sys.argv[:]
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old


def first_existing(
    row: dict[str, str],
    names: list[str],
) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def agreement(a: str, b: str) -> str:
    if not a or not b:
        return ""
    return "1" if a == b else "0"


def summarize_binary(
    rows: list[dict[str, Any]],
    field: str,
) -> tuple[int, int, float]:
    vals = []

    for row in rows:
        raw = str(row.get(field, "")).strip()
        if raw in {"0", "1"}:
            vals.append(int(raw))

    n = len(vals)
    correct = sum(vals)

    return (
        n,
        correct,
        correct / n if n else math.nan,
    )


def pooled_output(outputs):
    """
    Handle both ModelOutput-style objects and tuple outputs.
    """
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output

    if isinstance(outputs, (tuple, list)):
        if len(outputs) >= 2:
            return outputs[1]

        if len(outputs) == 1:
            # Fallback: CLS token from last hidden state.
            x = outputs[0]
            return x[:, 0]

    raise TypeError(
        f"Cannot extract pooled output from {type(outputs).__name__}"
    )


def install_clip_compat(module) -> None:
    """
    Patch module.encode_texts / module.encode_image so they do not pass
    return_dict to CLIPTextTransformer / CLIPVisionTransformer.
    """

    def encode_texts_compat(
        prompts,
        model,
        processor,
        device,
    ):
        inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            try:
                outputs = model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )

                pooled = pooled_output(outputs)

                features = model.text_projection(
                    pooled
                )

            except Exception:
                # Public HF API fallback for version differences.
                kwargs = {
                    "input_ids": inputs["input_ids"],
                }

                if inputs.get("attention_mask") is not None:
                    kwargs["attention_mask"] = inputs["attention_mask"]

                features = model.get_text_features(
                    **kwargs
                )

        return module.l2_normalize(
            features
        )

    def encode_image_compat(
        image,
        model,
        processor,
        device,
    ):
        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(device)

        with torch.inference_mode():
            try:
                outputs = model.vision_model(
                    pixel_values=pixel_values,
                )

                pooled = pooled_output(outputs)

                features = model.visual_projection(
                    pooled
                )

            except Exception:
                features = model.get_image_features(
                    pixel_values=pixel_values
                )

        return module.l2_normalize(
            features
        )

    module.encode_texts = encode_texts_compat
    module.encode_image = encode_image_compat


def main() -> None:
    args = parse_args()
    manifest = resolve_path(args.manifest)

    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    import run_primary_color_baseline_v1 as color_mod
    import run_pattern_hierarchical_v3 as pattern_mod
    import run_design_attribute_labels_v1 as design_mod

    # Compatibility fix for current transformers package.
    install_clip_compat(color_mod)
    install_clip_compat(pattern_mod)

    print(
        "CLIP compatibility patch installed "
        "(no return_dict passed to text/vision transformers)."
    )

    # 1) COLOR
    color_mod.REPORT_DIR = (
        REPORT_ROOT / "color_v1"
    )
    color_mod.OUTPUT_DIR = (
        OUTPUT_ROOT / "color_v1"
    )

    run_module_main(
        color_mod,
        [
            "run_primary_color_baseline_v1.py",
            "--manifest",
            str(manifest),
            "--holdout-size",
            "0",
            "--threads",
            str(args.threads),
        ],
    )

    # 2) PATTERN
    pattern_mod.REPORT_DIR = (
        REPORT_ROOT
        / "pattern_v3_hierarchical"
    )
    pattern_mod.OUTPUT_DIR = (
        OUTPUT_ROOT
        / "pattern_v3_hierarchical"
    )

    run_module_main(
        pattern_mod,
        [
            "run_pattern_hierarchical_v3.py",
            "--manifest",
            str(manifest),
            "--holdout-size",
            "0",
            "--threads",
            str(args.threads),
        ],
    )

    # 3) DESIGN
    empty_style = (
        REPORT_ROOT
        / "empty_style_for_predicted_roi.json"
    )

    empty_style.write_text(
        json.dumps(
            {
                "schema_version": (
                    "predicted-roi-empty-style-v1"
                ),
                "records": [],
                "note": (
                    "Intentionally empty to avoid "
                    "GT continuous-style leakage."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    design_mod.REPORT_DIR = (
        REPORT_ROOT
        / "design_labels_v1"
    )

    run_module_main(
        design_mod,
        [
            "run_design_attribute_labels_v1.py",
            "--manifest",
            str(manifest),
            "--style-json",
            str(empty_style),
            "--audit-size",
            "0",
            "--device",
            args.device,
        ],
    )

    # 4) COMPARE WITH FROZEN GT-ROI OUTPUTS
    pred_color_path = (
        REPORT_ROOT
        / "color_v1"
        / "primary_color_predictions.csv"
    )

    pred_pattern_path = (
        REPORT_ROOT
        / "pattern_v3_hierarchical"
        / "pattern_v3_predictions.csv"
    )

    pred_design_path = (
        REPORT_ROOT
        / "design_labels_v1"
        / "design_attribute_predictions_v1.csv"
    )

    for required in [
        pred_color_path,
        pred_pattern_path,
        pred_design_path,
        GT_COLOR,
        GT_PATTERN,
        GT_DESIGN,
    ]:
        if not required.is_file():
            raise FileNotFoundError(required)

    manifest_rows = read_csv(
        manifest
    )

    pred_color = {
        key_of(r): r
        for r in read_csv(pred_color_path)
    }

    pred_pattern = {
        key_of(r): r
        for r in read_csv(pred_pattern_path)
    }

    pred_design = {
        key_of(r): r
        for r in read_csv(pred_design_path)
    }

    gt_color = {
        key_of(r): r
        for r in read_csv(GT_COLOR)
    }

    gt_pattern = {
        key_of(r): r
        for r in read_csv(GT_PATTERN)
    }

    gt_design = {
        key_of(r): r
        for r in read_csv(GT_DESIGN)
    }

    comparisons = []

    for m in manifest_rows:
        key = key_of(m)

        pc = pred_color.get(
            key,
            {},
        )
        gc = gt_color.get(
            key,
            {},
        )

        pp = pred_pattern.get(
            key,
            {},
        )
        gp = gt_pattern.get(
            key,
            {},
        )

        pd = pred_design.get(
            key,
            {},
        )
        gd = gt_design.get(
            key,
            {},
        )

        pred_color_label = first_existing(
            pc,
            [
                "primary_color",
                "predicted_color",
                "label",
            ],
        )

        gt_color_label = first_existing(
            gc,
            [
                "primary_color",
                "predicted_color",
                "label",
            ],
        )

        pred_pattern_label = first_existing(
            pp,
            [
                "final_pattern",
                "pattern",
                "label",
            ],
        )

        gt_pattern_label = first_existing(
            gp,
            [
                "final_pattern",
                "pattern",
                "label",
            ],
        )

        row = {
            "source_image": key[0],
            "garment_id": key[1],
            "garment_category": (
                m.get(
                    "garment_category",
                    "",
                )
            ),
            "matched_gt_bbox_iou": (
                m.get(
                    "matched_gt_bbox_iou",
                    "",
                )
            ),
            "prediction_score": (
                m.get(
                    "prediction_score",
                    "",
                )
            ),
            "gt_primary_color": gt_color_label,
            "predroi_primary_color": pred_color_label,
            "color_agree": agreement(
                gt_color_label,
                pred_color_label,
            ),
            "gt_pattern": gt_pattern_label,
            "predroi_pattern": pred_pattern_label,
            "pattern_agree": agreement(
                gt_pattern_label,
                pred_pattern_label,
            ),
        }

        for attr in [
            "sleeve_length",
            "neckline",
            "silhouette_fit",
            "fashion_style",
        ]:
            gt_label = first_existing(
                gd,
                [
                    f"{attr}_label",
                    attr,
                ],
            )

            pred_label = first_existing(
                pd,
                [
                    f"{attr}_label",
                    attr,
                ],
            )

            row[
                f"gt_{attr}"
            ] = gt_label

            row[
                f"predroi_{attr}"
            ] = pred_label

            row[
                f"{attr}_agree"
            ] = agreement(
                gt_label,
                pred_label,
            )

        comparisons.append(
            row
        )

    compare_csv = (
        REPORT_ROOT
        / "per_instance_attribute_comparison.csv"
    )

    write_csv(
        compare_csv,
        comparisons,
    )

    metric_fields = [
        (
            "primary_color",
            "color_agree",
        ),
        (
            "pattern",
            "pattern_agree",
        ),
        (
            "sleeve_length",
            "sleeve_length_agree",
        ),
        (
            "neckline",
            "neckline_agree",
        ),
        (
            "silhouette_fit",
            "silhouette_fit_agree",
        ),
        (
            "fashion_style",
            "fashion_style_agree",
        ),
    ]

    summary_rows = []

    for metric, field in metric_fields:
        n, correct, rate = summarize_binary(
            comparisons,
            field,
        )

        summary_rows.append(
            {
                "attribute": metric,
                "comparable_pairs": n,
                "same_label_pairs": correct,
                "gt_vs_predroi_label_agreement": (
                    ""
                    if math.isnan(rate)
                    else f"{rate:.4f}"
                ),
            }
        )

    write_csv(
        REPORT_ROOT
        / "attribute_agreement_summary.csv",
        summary_rows,
    )

    category_counts = Counter(
        str(
            r.get(
                "garment_category",
                "",
            )
        )
        .strip()
        .lower()
        for r in manifest_rows
    )

    lines = [
        (
            "PRD 3.1 Predicted-ROI -> 3.1.3 "
            "Attribute Propagation Pilot v1"
        ),
        (
            "================================================"
            "============="
        ),
        "",
        f"manifest={manifest.relative_to(PROJECT_ROOT)}",
        f"matched_roi_pairs={len(manifest_rows)}",
        "",
        "Matched category distribution",
        "-----------------------------",
    ]

    for cls in [
        "top",
        "pants",
        "skirt",
        "outerwear",
        "dress",
        "shoe",
        "bag",
        "accessory",
    ]:
        lines.append(
            f"{cls}={category_counts.get(cls, 0)}"
        )

    lines += [
        "",
        "GT-ROI vs predicted-ROI label agreement",
        "---------------------------------------",
    ]

    for row in summary_rows:
        lines.append(
            f"{row['attribute']}: "
            f"{row['same_label_pairs']}/"
            f"{row['comparable_pairs']} "
            f"agreement="
            f"{row['gt_vs_predroi_label_agreement'] or 'NA'}"
        )

    lines += [
        "",
        "Interpretation",
        "--------------",
        (
            "- This is propagation consistency, "
            "NOT attribute accuracy against human GT."
        ),
        (
            "- A disagreement means the 3.1.3 label changed "
            "when GT ROI was replaced by matched predicted ROI."
        ),
        (
            "- No GT continuous style geometry was injected "
            "into predicted-ROI categorical inference."
        ),
        (
            "- Small matched pilot with incomplete class coverage; "
            "not final 8-class acceptance."
        ),
    ]

    (
        REPORT_ROOT
        / "summary.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=== PREDICTED ROI ATTRIBUTE PILOT FINISHED ==="
    )

    print(
        "Summary:",
        (
            REPORT_ROOT
            / "summary.txt"
        ).relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Agreement:",
        (
            REPORT_ROOT
            / "attribute_agreement_summary.csv"
        ).relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Per instance:",
        compare_csv.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()
