"""基准集管理：保留训练、开发、固定回归/审核划分，冻结后的标签不能随结果调整。

Build a focused human-review packet for PRD 3.1.1 QC v2.

This packet contains ONLY cases that are not GEOMETRY_PASS in QC v2:
- REVIEW_REQUIRED
- REVIEW_PRIORITY

It does not auto-approve or auto-reject annotations.

For each case it creates:
1) original full image with GT bbox;
2) zoomed crop;
3) GT mask overlay crop;
4) a review CSV row.

Inputs
------
benchmark/prd_3_1_v1/candidates/segmentation_core_stress_v1/qc/
    human_review_priority_v2.csv

Outputs
-------
benchmark/prd_3_1_v1/candidates/segmentation_core_stress_v1/human_review_packet_v1/
├── review_cases_v1.csv
├── review_packet_summary.txt
├── core/
│   ├── REVIEW_REQUIRED/
│   └── REVIEW_PRIORITY/
└── stress/
    ├── REVIEW_REQUIRED/
    └── REVIEW_PRIORITY/

Run
---
cd fashion_multimodal_analysis

python scripts/benchmarking/build_prd31_human_review_packet_v1.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()

POOL_ROOT = (
    PROJECT_ROOT
    / "benchmark"
    / "prd_3_1_v1"
    / "candidates"
    / "segmentation_core_stress_v1"
)

QC_CSV = POOL_ROOT / "qc" / "human_review_priority_v2.csv"

OUT_DIR = POOL_ROOT / "human_review_packet_v1"


def read_csv(path: Path) -> Any:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。
    """
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
    """按确定的字段顺序保存实验 CSV。

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def resolve_project_path(raw: Any) -> Path:
    """将项目相对文件引用解析到当前工作副本。

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        当前工作副本可使用的路径。
    """
    return resolve_artifact_path(raw)


def draw_full_image(
    source_path: Path,
    bbox: Any,
    out_path: Path,
) -> None:
    """绘制 整图 图像。

    Args:
        source_path: 对应文件的相对路径或当前解析后的路径。
        bbox: xyxy 坐标的目标框。
        out_path: 结果文件路径。
    """
    image = Image.open(source_path).convert("RGB")

    draw = ImageDraw.Draw(image)

    x1, y1, x2, y2 = bbox

    draw.rectangle(
        (x1, y1, x2, y2),
        outline="white",
        width=max(
            2,
            int(
                min(
                    image.width,
                    image.height,
                )
                / 250
            ),
        ),
    )

    image.thumbnail((1200, 1200))

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        out_path,
        quality=92,
    )


def draw_zoom(
    source_path: Path,
    bbox: Any,
    out_path: Path,
) -> None:
    """绘制 zoom。

    Args:
        source_path: 对应文件的相对路径或当前解析后的路径。
        bbox: xyxy 坐标的目标框。
        out_path: 结果文件路径。
    """
    image = Image.open(source_path).convert("RGB")

    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    padx = max(
        15,
        int(
            0.30
            * max(
                1,
                w,
            )
        ),
    )

    pady = max(
        15,
        int(
            0.30
            * max(
                1,
                h,
            )
        ),
    )

    px1 = max(
        0,
        int(math.floor(x1 - padx)),
    )

    py1 = max(
        0,
        int(math.floor(y1 - pady)),
    )

    px2 = min(
        image.width,
        int(math.ceil(x2 + padx)),
    )

    py2 = min(
        image.height,
        int(math.ceil(y2 + pady)),
    )

    crop = image.crop(
        (
            px1,
            py1,
            px2,
            py2,
        )
    )

    draw = ImageDraw.Draw(crop)

    draw.rectangle(
        (
            x1 - px1,
            y1 - py1,
            x2 - px1,
            y2 - py1,
        ),
        outline="white",
        width=3,
    )

    crop.thumbnail((900, 900))

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop.save(
        out_path,
        quality=92,
    )


def build_panel(
    full_path: Path,
    zoom_path: Path,
    overlay_path: Path,
    row: dict[str, Any],
    out_path: Path,
) -> None:
    """构建 panel。

    Args:
        full_path: 对应文件的相对路径或当前解析后的路径。
        zoom_path: 对应文件的相对路径或当前解析后的路径。
        overlay_path: 对应文件的相对路径或当前解析后的路径。
        row: 记录字段，使用 review_sample_id, review_subset, garment_category,
        fine_or_source_category, qc_v2_status。
        out_path: 结果文件路径。
    """
    full = Image.open(full_path).convert("RGB")

    zoom = Image.open(zoom_path).convert("RGB")

    overlay = Image.open(overlay_path).convert("RGB")

    each_w = 520
    each_h = 420
    header_h = 130

    canvas = Image.new(
        "RGB",
        (
            each_w * 3,
            each_h + header_h,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    font = ImageFont.load_default()

    title = (
        f"{row['review_sample_id']} | "
        + f"{row['review_subset']} | "
        + f"{row['garment_category']} | "
        + f"{row['fine_or_source_category']}\n"
        + f"QC={row['qc_v2_status']} | "
        + f"reason={row.get('qc_v2_reasons','')} | "
        + f"advisory={row.get('qc_v2_advisory','')}\n"
        + f"mask_bbox_iou="
        + f"{row.get('mask_bbox_iou_with_gt_bbox','')} | "
        + f"inside_bbox="
        + f"{row.get('mask_inside_bbox_ratio','')} | "
        + f"fill="
        + f"{row.get('mask_fill_ratio_in_bbox','')}"
    )

    draw.multiline_text(
        (10, 10),
        title,
        fill="black",
        font=font,
        spacing=4,
    )

    labels = [
        ("FULL + GT BBOX", full),
        ("ZOOM + GT BBOX", zoom),
        ("GT MASK OVERLAY", overlay),
    ]

    for i, (
        label,
        image,
    ) in enumerate(labels):
        image.thumbnail(
            (
                each_w - 20,
                each_h - 45,
            )
        )

        x0 = i * each_w + (each_w - image.width) // 2

        y0 = header_h + 30 + (each_h - 45 - image.height) // 2

        canvas.paste(
            image,
            (
                x0,
                y0,
            ),
        )

        draw.text(
            (
                i * each_w + 10,
                header_h + 5,
            ),
            label,
            fill="black",
            font=font,
        )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        out_path,
        quality=93,
    )


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    rows = read_csv(QC_CSV)

    review_rows = [
        row
        for row in rows
        if str(
            row.get(
                "qc_v2_status",
                "",
            )
        )
        != "GEOMETRY_PASS"
    ]

    result_rows = []

    for idx, row in enumerate(
        review_rows,
        start=1,
    ):
        sid = row["review_sample_id"]

        subset = row["review_subset"]

        status = row["qc_v2_status"]

        source_path = resolve_project_path(row["source_image"])

        overlay_path = resolve_project_path(row["gt_overlay_path"])

        bbox = (
            float(row["gt_bbox_x1"]),
            float(row["gt_bbox_y1"]),
            float(row["gt_bbox_x2"]),
            float(row["gt_bbox_y2"]),
        )

        case_dir = OUT_DIR / subset / status / sid

        full_out = case_dir / "01_full_bbox.jpg"

        zoom_out = case_dir / "02_zoom_bbox.jpg"

        panel_out = case_dir / "03_review_panel.jpg"

        draw_full_image(
            source_path,
            bbox,
            full_out,
        )

        draw_zoom(
            source_path,
            bbox,
            zoom_out,
        )

        build_panel(
            full_out,
            zoom_out,
            overlay_path,
            row,
            panel_out,
        )

        result_rows.append(
            {
                "review_sample_id": sid,
                "review_subset": subset,
                "garment_category": row["garment_category"],
                "fine_or_source_category": row["fine_or_source_category"],
                "qc_v2_status": status,
                "qc_v2_reasons": row.get(
                    "qc_v2_reasons",
                    "",
                ),
                "qc_v2_advisory": row.get(
                    "qc_v2_advisory",
                    "",
                ),
                "mask_bbox_iou_with_gt_bbox": row.get(
                    "mask_bbox_iou_with_gt_bbox",
                    "",
                ),
                "mask_inside_bbox_ratio": row.get(
                    "mask_inside_bbox_ratio",
                    "",
                ),
                "mask_fill_ratio_in_bbox": row.get(
                    "mask_fill_ratio_in_bbox",
                    "",
                ),
                "review_panel_path": str(panel_out.relative_to(PROJECT_ROOT)).replace(
                    "\\",
                    "/",
                ),
                "category_valid": "",
                "bbox_valid": "",
                "mask_valid": "",
                "image_quality_valid": "",
                "ambiguous": "",
                "human_decision": "",
                "exclude_reason": "",
                "reviewer_note": "",
            }
        )

        print(f"[{idx}/{len(review_rows)}] " + f"{sid}")

    write_csv(
        OUT_DIR / "review_cases_v1.csv",
        result_rows,
    )

    from collections import Counter

    status_counts = Counter(row["qc_v2_status"] for row in result_rows)

    core = [x for x in result_rows if x["review_subset"] == "core"]

    stress = [x for x in result_rows if x["review_subset"] == "stress"]

    lines = [
        ("PRD 3.1.1 Focused Human Review " + "Packet v1"),
        ("======================================"),
        "",
        f"total_review_cases={len(result_rows)}",
        ("REVIEW_REQUIRED=" + f"{status_counts['REVIEW_REQUIRED']}"),
        ("REVIEW_PRIORITY=" + f"{status_counts['REVIEW_PRIORITY']}"),
        f"core_review_cases={len(core)}",
        f"stress_review_cases={len(stress)}",
        "",
        "Human decision rule",
        "-------------------",
        ("- PASS only when category, bbox, " + "mask and image quality are all valid."),
        ("- EXCLUDE when source annotation is " + "clearly wrong or unsuitable."),
        ("- AMBIGUOUS when a stable annotation " + "cannot be established visually."),
        ("- REVIEW_PRIORITY is not an automatic " + "failure."),
        ("- Never use model predictions when " + "deciding whether a GT case stays."),
    ]

    (OUT_DIR / "review_packet_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== HUMAN REVIEW PACKET READY ===")

    print((OUT_DIR / "review_cases_v1.csv").relative_to(PROJECT_ROOT))

    print((OUT_DIR / "review_packet_summary.txt").relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
