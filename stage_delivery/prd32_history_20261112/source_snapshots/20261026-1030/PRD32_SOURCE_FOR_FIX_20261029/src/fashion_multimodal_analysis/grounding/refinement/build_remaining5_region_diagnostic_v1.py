"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
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
REPORT_ROOT = PROJECT_ROOT / "reports" / "prd_region_coverage"

FORMAL_RESULTS = REPORT_ROOT / "formal_case_results.csv"
MANUAL_AUDIT = REPORT_ROOT / "prd_40case_formal_manual_audit_final.csv"

OUTPUT_DIR = REPORT_ROOT / "remaining5_diagnostic_v1"
CASES_OUT = OUTPUT_DIR / "cases.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
SUMMARY_TXT = OUTPUT_DIR / "summary.txt"

TARGET_REGIONS = ["pocket", "shoulder", "waist", "pattern", "decoration"]


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
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
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


def is_yes(v: str) -> bool:
    """判断 yes。

    Args:
        v: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        当前条件的校验结果；失败条件及返回形式见函数体。
    """
    return str(v).strip().lower() in {"yes", "y", "true", "1"}


def as_float(v: str, default: float = 0.0) -> float:
    """as float。

    Args:
        v: 本步骤使用的原始值，转换/筛选规则见函数体。
        default: default。

    Returns:
        返回 default，由函数体中同名变量的计算/收集过程得到。
    """
    try:
        return float(v)
    except Exception:
        return default


def parse_detections(raw: str) -> list[dict]:
    """解析 detections。

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def merge_rows(
    results: list[dict[str, str]],
    audits: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """merge 逐行记录。

    Args:
        results: 结果。
        audits: audits。

    Returns:
        返回 merged，由函数体中同名变量的计算/收集过程得到。
    """
    audit_by_id = {
        r["candidate_id"].strip(): r for r in audits if r.get("candidate_id")
    }

    merged = []

    for r in results:
        region = r.get("region", "").strip().lower()
        if region not in TARGET_REGIONS:
            continue

        cid = r.get("candidate_id", "").strip()
        a = audit_by_id.get(cid, {})

        quality = (
            (
                a.get("localization_quality", "")
                or a.get("manual_quality", "")
                or a.get("quality", "")
            )
            .strip()
            .lower()
        )

        if not quality:
            quality = "missed" if not is_yes(r.get("detected", "")) else "unreviewed"

        merged.append(
            {
                "candidate_id": cid,
                "region": region,
                "image_name": r.get("image_name", ""),
                "item_id": r.get("item_id", ""),
                "category_name": r.get("category_name", ""),
                "candidate_crop_path": r.get("candidate_crop_path", ""),
                "prompt": r.get("prompt", ""),
                "threshold": r.get("threshold", ""),
                "detected": r.get("detected", ""),
                "num_detections": r.get("num_detections", ""),
                "top_score": r.get("top_score", ""),
                "top_box_area_ratio": r.get("top_box_area_ratio", ""),
                "inference_seconds": r.get("inference_seconds", ""),
                "target_present": a.get("target_present", r.get("target_present", "")),
                "visibility": a.get("visibility", r.get("visibility", "")),
                "localization_quality": quality,
                "review_note": a.get("review_note", ""),
                "detections_json": r.get("detections_json", ""),
            }
        )

    return merged


def make_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """生成 汇总。

    Args:
        rows: 待处理的逐行记录。

    Returns:
        返回 out，由函数体中同名变量的计算/收集过程得到。
    """
    out = []

    for region in TARGET_REGIONS:
        rr = [r for r in rows if r["region"] == region]
        q = Counter(str(r["localization_quality"]).strip().lower() for r in rr)

        n = len(rr)
        correct = q.get("correct", 0)
        coarse = q.get("coarse", 0)
        wrong = q.get("wrong", 0)
        missed = q.get("missed", 0)
        usable = correct + coarse
        non_empty = sum(is_yes(str(r["detected"])) for r in rr)

        areas = [
            as_float(str(r["top_box_area_ratio"]))
            for r in rr
            if str(r["top_box_area_ratio"]).strip()
        ]

        out.append(
            {
                "region": region,
                "N": n,
                "correct": correct,
                "coarse": coarse,
                "wrong": wrong,
                "missed": missed,
                "strict_accuracy_pct": round(100 * correct / n, 2) if n else 0.0,
                "usable_rate_pct": round(100 * usable / n, 2) if n else 0.0,
                "missed_rate_pct": round(100 * missed / n, 2) if n else 0.0,
                "non_empty_prediction_rate_pct": (
                    round(100 * non_empty / n, 2) if n else 0.0
                ),
                "mean_top_box_area_ratio": (
                    round(sum(areas) / len(areas), 4) if areas else ""
                ),
            }
        )

    return out


def draw_case(row: dict[str, Any], size: Any = (420, 420)) -> Image.Image:
    """绘制 案例。

    Args:
        row: 记录字段，使用 candidate_crop_path, candidate_id, region, localization_quality,
        detected, top_score, top_box_area_ratio。
        size: size。

    Returns:
        返回 canvas，由函数体中同名变量的计算/收集过程得到。
    """
    w, h = size
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    p = resolve_project_path(row["candidate_crop_path"])

    if p.is_file():
        image = Image.open(p).convert("RGB")
    else:
        image = Image.new("RGB", (320, 280), (235, 235, 235))

    orig_w, orig_h = image.size
    d = ImageDraw.Draw(image)

    for det in parse_detections(str(row.get("detections_json", ""))):
        box = det.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue

        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except Exception:
            continue

        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1 *= orig_w
            x2 *= orig_w
            y1 *= orig_h
            y2 *= orig_h

        d.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)

    image.thumbnail((w - 20, 300), Image.Resampling.LANCZOS)
    canvas.paste(image, ((w - image.width) // 2, 5))

    y = 312
    lines = [
        f"{row['candidate_id']} | {row['region']} | {row['localization_quality']}",
        f"detected={row['detected']} score={row['top_score']} area={row['top_box_area_ratio']}",
        f"image={row['image_name']} item={row['item_id']}",
    ]

    note = str(row.get("review_note", "")).strip()
    if note:
        lines.append(note[:75])

    for line in lines:
        draw.text((8, y), line, fill="black", font=font)
        y += 18

    draw.rectangle([1, 1, w - 2, h - 2], outline="gray", width=1)
    return canvas


def make_contact_sheet(rows: list[dict[str, Any]], region: str) -> None:
    """把样本图与说明拼成审核联系表，便于逐例比较。

    Args:
        rows: 待处理的逐行记录。
        region: 区域。
    """
    rr = [r for r in rows if r["region"] == region]
    if not rr:
        return

    tile_w, tile_h = 420, 420
    cols = 3
    rows_n = math.ceil(len(rr) / cols)
    title_h = 42

    sheet = Image.new(
        "RGB",
        (cols * tile_w, title_h + rows_n * tile_h),
        "white",
    )

    d = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    d.text(
        (10, 14), f"PRD 3.1.2 remaining5 diagnostic - {region}", fill="black", font=font
    )

    for i, row in enumerate(rr):
        tile = draw_case(row, (tile_w, tile_h))
        x = (i % cols) * tile_w
        y = title_h + (i // cols) * tile_h
        sheet.paste(tile, (x, y))

    sheet.save(OUTPUT_DIR / f"{region}.jpg", quality=92)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        RuntimeError: No remaining-five rows found.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = read_csv(FORMAL_RESULTS)
    audits = read_csv(MANUAL_AUDIT)

    merged = merge_rows(results, audits)
    if not merged:
        raise RuntimeError("No remaining-five rows found.")

    write_csv(CASES_OUT, merged)

    summary = make_summary(merged)
    write_csv(SUMMARY_CSV, summary)

    for region in TARGET_REGIONS:
        make_contact_sheet(merged, region)

    lines = [
        "PRD 3.1.2 Remaining-Five Region Diagnostic v1",
        "=============================================",
        "",
        "Source: existing 40-case formal Grounding-DINO pilot.",
        "No model rerun and no new samples are introduced.",
        "",
    ]

    for row in summary:
        lines.append(
            f"{row['region']}: N={row['N']}, correct={row['correct']}, "
            + f"coarse={row['coarse']}, wrong={row['wrong']}, missed={row['missed']}, "
            + f"usable={row['usable_rate_pct']}%, missed_rate={row['missed_rate_pct']}%"
        )

    lines += [
        "",
        "Interpretation:",
        "- correct = tight/semantically correct localization.",
        "- coarse = target is included but the box is too broad.",
        "- wrong = response is present but points to the wrong region.",
        "- missed = no usable localization response.",
        "- usable = correct + coarse.",
        "- pattern here means LOCAL PATTERN REGION localization, not 3.1.3 pattern semantic classification.",
    ]

    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== REMAINING-5 DIAGNOSTIC BUILT ===")
    print(f"Cases: {len(merged)}")
    print()

    for row in summary:
        print(
            f"{row['region']:10s} "
            + f"N={row['N']} "
            + f"correct={row['correct']} "
            + f"coarse={row['coarse']} "
            + f"wrong={row['wrong']} "
            + f"missed={row['missed']} "
            + f"usable={row['usable_rate_pct']:.1f}%"
        )

    print()
    print("Output:", OUTPUT_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
