"""3.1 接口联调：区分预测 ROI 传递和使用 GT 的受控诊断，保留对象 ID 与坐标对应关系。

PRD 3.1 bbox/mask isolation ablation v1

Goal
----
Separate bbox-crop error from mask-shape error for 3.1.3 design attributes.

For every one-to-one matched garment in the full-validation set, build:

A. GT bbox   + GT mask
B. Pred bbox + Pred mask
C. GT bbox   + Pred mask
D. Pred bbox + GT mask

Then run the SAME design-label model on all four conditions and compare
each condition against A.

Primary diagnostic attributes:
- sleeve_length
- neckline

Secondary:
- silhouette_fit
- fashion_style

Interpretation
--------------
If C (GT bbox + Pred mask) drops much more than D (Pred bbox + GT mask),
mask quality is the stronger source of propagation error.

Inputs
------
configs/prd_3_1_predicted_roi_fullval_matched_v1.csv
configs/prd_8class_val_v1.csv

Outputs
-------
configs/
├── prd_3_1_ablation_a_gtbbox_gtmask_v1.csv
├── prd_3_1_ablation_b_predbbox_predmask_v1.csv
├── prd_3_1_ablation_c_gtbbox_predmask_v1.csv
└── prd_3_1_ablation_d_predbbox_gtmask_v1.csv

outputs/prd_integration/roi_mask_isolation_v1/<condition>/...

reports/prd_integration/roi_mask_isolation_v1/
├── A_gtbbox_gtmask/design_labels_v1/...
├── B_predbbox_predmask/design_labels_v1/...
├── C_gtbbox_predmask/design_labels_v1/...
├── D_predbbox_gtmask/design_labels_v1/...
├── per_instance_design_ablation.csv
├── attribute_agreement_by_condition.csv
└── summary.txt

Run
---
cd fashion_multimodal_analysis

python scripts/integration/run_prd31_bbox_mask_isolation_ablation_v1.py   --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()

MATCHED_CSV = PROJECT_ROOT / "configs" / "prd_3_1_predicted_roi_fullval_matched_v1.csv"

GT_CSV = PROJECT_ROOT / "configs" / "prd_8class_val_v1.csv"

REPORT_ROOT = PROJECT_ROOT / "reports" / "prd_integration" / "roi_mask_isolation_v1"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "prd_integration" / "roi_mask_isolation_v1"

CONFIG_ROOT = PROJECT_ROOT / "configs"

CONDITIONS = {
    "A_gtbbox_gtmask": ("gt", "gt"),
    "B_predbbox_predmask": ("pred", "pred"),
    "C_gtbbox_predmask": ("gt", "pred"),
    "D_predbbox_gtmask": ("pred", "gt"),
}

ATTRS = (
    "sleeve_length",
    "neckline",
    "silhouette_fit",
    "fashion_style",
)


def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。

    Returns:
        已通过参数合法性检查的命令行配置。
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
    )
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
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
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def resolve_path(raw: Any) -> Path:
    """将清单或配置中的相对路径解析到当前项目/数据目录。

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(raw)


def project_relative(path: Path) -> str:
    """project relative。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace(
            "\\", "/"
        )
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def norm_source(v: str) -> str:
    """规范化来源标识，使不同表格中的同一原图可连接。

    Args:
        v: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return str(v).replace("\\", "/").strip()


def key_of(row: dict[str, str]) -> tuple[str, str]:
    """提取用于跨表连接的记录标识。

    Args:
        row: 一条实例、预测或审核记录。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return (
        norm_source(row.get("source_image", "")),
        str(row.get("garment_id", "")).strip(),
    )


def box_from_row(
    row: dict[str, str],
) -> tuple[int, int, int, int]:
    """边界框 from 记录。

    Args:
        row: 一条实例、预测或审核记录。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return tuple(
        int(round(float(row[k]))) for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    )


def load_binary_mask(path: Path) -> np.ndarray:
    """加载 二值 掩码。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。
    """
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 0


def reconstruct_full_mask(
    row: dict[str, str],
    image_size: tuple[int, int],
) -> np.ndarray:
    """Supports either:
    - full-image mask
    - bbox/crop mask

    Args:
        row: 记录字段，使用 mask_path。
        image_size: 图像的宽、高，用于处理坐标和掩码尺寸。

    Returns:
        返回 canvas，由函数体中同名变量的计算/收集过程得到。
    """
    w, h = image_size

    mask_path = resolve_path(row["mask_path"])
    m = load_binary_mask(mask_path)

    if m.shape == (h, w):
        return m

    x1, y1, x2, y2 = box_from_row(row)

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    tw = x2 - x1
    th = y2 - y1

    if m.shape != (th, tw):
        pil = Image.fromarray(
            (m.astype(np.uint8) * 255),
            mode="L",
        )
        pil = pil.resize(
            (tw, th),
            Image.Resampling.NEAREST,
        )
        m = np.asarray(pil) > 0

    canvas = np.zeros((h, w), dtype=bool)
    canvas[y1:y2, x1:x2] = m[:th, :tw]

    return canvas


def clip_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """CLIP 边界框。

    Args:
        box: xyxy 坐标的候选框。
        image_size: 图像的宽、高，用于处理坐标和掩码尺寸。

    Returns:
        按顺序返回 x1, y1, x2, y2 等结果。
    """
    w, h = image_size
    x1, y1, x2, y2 = box

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    return x1, y1, x2, y2


def save_condition_roi(
    image: Image.Image,
    full_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path, Path]:
    """保存 condition roi。

    Args:
        image: 本步骤处理的图像对象。
        full_mask: 整图 掩码。
        crop_box: 裁剪 边界框。
        output_dir: 当前版本的输出目录。
        stem: stem。

    Returns:
        按顺序返回 crop_path, mask_path, masked_path 等结果。
    """
    x1, y1, x2, y2 = clip_box(
        crop_box,
        image.size,
    )

    crop = image.crop((x1, y1, x2, y2))

    mask_crop_arr = full_mask[y1:y2, x1:x2].astype(np.uint8)

    mask_crop = Image.fromarray(
        mask_crop_arr * 255,
        mode="L",
    )

    arr = np.asarray(crop).copy()

    masked_arr = np.full_like(
        arr,
        255,
    )

    mask_bool = mask_crop_arr > 0

    masked_arr[mask_bool] = arr[mask_bool]

    masked = Image.fromarray(masked_arr)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop_path = output_dir / f"{stem}_crop.jpg"
    mask_path = output_dir / f"{stem}_mask.png"
    masked_path = output_dir / f"{stem}_masked.jpg"

    crop.save(crop_path, quality=95)
    mask_crop.save(mask_path)
    masked.save(masked_path, quality=95)

    return crop_path, mask_path, masked_path


def run_module_main(module: str, argv: list[str]) -> None:
    """调用目标实验模块，保留命令行入口的运行语义。

    Args:
        module: 要执行的 Python 模块名称。
        argv: argv。
    """
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
    """按优先顺序返回存在的候选路径。

    Args:
        row: 一条实例、预测或审核记录。
        names: names。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        KeyError: 所需字段不存在。
    """
    args = parse_args()

    matched_rows = read_csv(MATCHED_CSV)
    gt_rows = read_csv(GT_CSV)

    gt_idx = {key_of(r): r for r in gt_rows}

    manifests: dict[str, list[dict[str, Any]]] = {name: [] for name in CONDITIONS}

    print(f"matched pairs: {len(matched_rows)}")

    for idx, pred in enumerate(
        matched_rows,
        start=1,
    ):
        key = key_of(pred)
        gt = gt_idx.get(key)

        if gt is None:
            raise KeyError(f"GT row not found for {key}")

        image_path = resolve_path(gt["source_image"])

        image = Image.open(image_path).convert("RGB")

        gt_full_mask = reconstruct_full_mask(
            gt,
            image.size,
        )

        pred_full_mask = reconstruct_full_mask(
            pred,
            image.size,
        )

        gt_box = box_from_row(gt)
        pred_box = box_from_row(pred)

        for condition, (
            bbox_source,
            mask_source,
        ) in CONDITIONS.items():

            crop_box = gt_box if bbox_source == "gt" else pred_box

            full_mask = gt_full_mask if mask_source == "gt" else pred_full_mask

            out_dir = OUTPUT_ROOT / condition / Path(gt["source_image"]).stem

            safe_gid = str(gt["garment_id"]).replace("/", "_").replace("\\", "_")

            (
                crop_path,
                mask_path,
                masked_path,
            ) = save_condition_roi(
                image,
                full_mask,
                crop_box,
                out_dir,
                f"{idx:03d}_{safe_gid}",
            )

            x1, y1, x2, y2 = clip_box(
                crop_box,
                image.size,
            )

            manifests[condition].append(
                {
                    "source_dataset": gt.get(
                        "source_dataset",
                        "",
                    ),
                    "source_image": gt["source_image"],
                    "garment_id": gt["garment_id"],
                    "garment_category": gt["garment_category"],
                    "fine_category": gt.get(
                        "fine_category",
                        "",
                    ),
                    "crop_path": project_relative(crop_path),
                    "mask_path": project_relative(mask_path),
                    "masked_preview_path": (project_relative(masked_path)),
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "ablation_condition": condition,
                    "bbox_source": bbox_source,
                    "mask_source": mask_source,
                    "matched_gt_bbox_iou": pred.get(
                        "matched_gt_bbox_iou",
                        "",
                    ),
                    "prediction_score": pred.get(
                        "prediction_score",
                        "",
                    ),
                }
            )

    condition_manifest_paths = {}

    for condition, rows in manifests.items():
        tag = {
            "A_gtbbox_gtmask": "A_gtbbox_gtmask",
            "B_predbbox_predmask": "B_predbbox_predmask",
            "C_gtbbox_predmask": "C_gtbbox_predmask",
            "D_predbbox_gtmask": "D_predbbox_gtmask",
        }[condition]

        path = CONFIG_ROOT / f"prd_3_1_ablation_{tag}_v1.csv"

        write_csv(
            path,
            rows,
        )

        condition_manifest_paths[condition] = path

    # ---------------------------------------------------------
    # Run same design-label model on all four conditions.
    # ---------------------------------------------------------
    import fashion_multimodal_analysis.attributes.design.run_design_attribute_labels_v1 as design_mod

    condition_predictions: dict[
        str,
        dict[tuple[str, str], dict[str, str]],
    ] = {}

    for condition in CONDITIONS:
        print()
        print("=" * 72)
        print(f"RUNNING DESIGN ATTRIBUTES: {condition}")
        print("=" * 72)

        condition_report_dir = REPORT_ROOT / condition / "design_labels_v1"

        condition_report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        empty_style_path = condition_report_dir / "empty_style.json"

        empty_style_path.write_text(
            json.dumps(
                {
                    "schema_version": ("bbox-mask-isolation-empty-style-v1"),
                    "records": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        design_mod.REPORT_DIR = condition_report_dir

        run_module_main(
            design_mod,
            [
                "run_design_attribute_labels_v1.py",
                "--manifest",
                str(condition_manifest_paths[condition]),
                "--style-json",
                str(empty_style_path),
                "--audit-size",
                "0",
                "--device",
                args.device,
            ],
        )

        pred_csv = condition_report_dir / "design_attribute_predictions_v1.csv"

        condition_predictions[condition] = {key_of(r): r for r in read_csv(pred_csv)}

    # ---------------------------------------------------------
    # Compare B/C/D against A.
    # ---------------------------------------------------------
    baseline = condition_predictions["A_gtbbox_gtmask"]

    per_instance_rows = []

    for key, a in baseline.items():
        row = {
            "source_image": key[0],
            "garment_id": key[1],
            "garment_category": a.get(
                "garment_category",
                "",
            ),
        }

        for condition in CONDITIONS:
            r = condition_predictions[condition].get(
                key,
                {},
            )

            for attr in ATTRS:
                row[f"{condition}_{attr}"] = first_existing(
                    r,
                    [
                        f"{attr}_label",
                        attr,
                    ],
                )

        for condition in [
            "B_predbbox_predmask",
            "C_gtbbox_predmask",
            "D_predbbox_gtmask",
        ]:
            for attr in ATTRS:
                a_label = row.get(
                    f"A_gtbbox_gtmask_{attr}",
                    "",
                )

                c_label = row.get(
                    f"{condition}_{attr}",
                    "",
                )

                if a_label and c_label:
                    row[f"{condition}_{attr}_agree_with_A"] = int(a_label == c_label)
                else:
                    row[f"{condition}_{attr}_agree_with_A"] = ""

        per_instance_rows.append(row)

    write_csv(
        REPORT_ROOT / "per_instance_design_ablation.csv",
        per_instance_rows,
    )

    summary_rows = []

    for condition in [
        "B_predbbox_predmask",
        "C_gtbbox_predmask",
        "D_predbbox_gtmask",
    ]:
        for attr in ATTRS:
            field = f"{condition}_{attr}_agree_with_A"

            vals = [
                int(r[field])
                for r in per_instance_rows
                if str(r.get(field, "")).strip() in {"0", "1"}
            ]

            summary_rows.append(
                {
                    "condition": condition,
                    "attribute": attr,
                    "comparable_pairs": len(vals),
                    "same_as_A_pairs": sum(vals),
                    "agreement_with_A": (f"{sum(vals)/len(vals):.4f}" if vals else ""),
                }
            )

    write_csv(
        REPORT_ROOT / "attribute_agreement_by_condition.csv",
        summary_rows,
    )

    lookup = {
        (
            row["condition"],
            row["attribute"],
        ): row
        for row in summary_rows
    }

    lines = [
        "PRD 3.1 BBox / Mask Isolation Ablation v1",
        "========================================",
        "",
        f"matched_pairs={len(matched_rows)}",
        "",
        "Conditions",
        "----------",
        "A = GT bbox + GT mask (reference)",
        "B = Pred bbox + Pred mask (real pipeline)",
        "C = GT bbox + Pred mask (isolates mask error)",
        "D = Pred bbox + GT mask (isolates bbox error)",
        "",
        "Agreement with A",
        "----------------",
    ]

    for attr in ATTRS:
        lines.append(f"{attr}:")

        for condition in [
            "B_predbbox_predmask",
            "C_gtbbox_predmask",
            "D_predbbox_gtmask",
        ]:
            r = lookup[
                (
                    condition,
                    attr,
                )
            ]

            lines.append(
                f"  {condition}: "
                + f"{r['same_as_A_pairs']}/"
                + f"{r['comparable_pairs']} "
                + f"agreement="
                + f"{r['agreement_with_A'] or 'NA'}"
            )

    lines += [
        "",
        "Decision rule",
        "-------------",
        (
            "- If C is much worse than D, "
            + "predicted mask quality is the stronger error source."
        ),
        (
            "- If D is much worse than C, "
            + "bbox crop quality is the stronger error source."
        ),
        (
            "- If both C and D are degraded, "
            + "both components contribute or interact."
        ),
        (
            "- Compare sleeve_length and neckline first; "
            + "they are the main local-structure attributes."
        ),
        "",
        "Caution",
        "-------",
        (
            "- This is a controlled diagnostic on the fixed matched set, "
            + "not human-ground-truth attribute accuracy."
        ),
        (
            "- Some categories are unsupported by design-label v1 "
            + "and therefore are excluded from attribute comparisons."
        ),
    ]

    (REPORT_ROOT / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== BBOX/MASK ISOLATION ABLATION FINISHED ===")
    print(
        "Summary:",
        (REPORT_ROOT / "summary.txt").relative_to(PROJECT_ROOT),
    )
    print(
        "Agreement:",
        (REPORT_ROOT / "attribute_agreement_by_condition.csv").relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        "Per-instance:",
        (REPORT_ROOT / "per_instance_design_ablation.csv").relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
