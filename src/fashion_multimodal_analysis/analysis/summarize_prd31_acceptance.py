"""结果分析：依据已有逐例记录定位问题，不替代新的模型评估。

Finalize strict semantic reviews, PRD gates and complete-chain failure attribution.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from fashion_multimodal_analysis.benchmarking.build_prd31_review_packet import image_url
from fashion_multimodal_analysis.benchmarking.freeze_prd31_reviewed_tests import (
    validate_identity,
    verify_frozen,
)
from fashion_multimodal_analysis.benchmarking.prepare_prd31_recovery import (
    ATTRIBUTES,
    REGIONS,
    applies,
    digest,
)
from fashion_multimodal_analysis.common.io import read_csv, write_csv, write_json
from fashion_multimodal_analysis.common.paths import project_root, resolve_path
from fashion_multimodal_analysis.common.schema import CLASS_TO_ID
from fashion_multimodal_analysis.evaluation.protocol import (
    flag,
    metric_status,
    per_attribute_metrics,
)

GRADES = ("correct", "coarse", "wrong", "missed")


def strict_grounding(
    rows: list[dict[str, Any]], audits: Any, threshold: float = 0.5
) -> tuple[Any, ...]:
    """strict grounding。

    Args:
        rows: 待处理的逐行记录。
        audits: audits。
        threshold: 当前判定规则使用的阈值。

    Returns:
        按顺序返回 result, details 等结果。
    """
    if audits is not None:
        audits = validate_identity(
            rows,
            audits,
            ("track", "query_id"),
            {"manual_grade", "reviewer", "review_status", "annotation_note"},
        )
    else:
        audits = rows
    details = []
    for row in audits:
        prediction_detected = row["prediction_status"] == "detected"
        reviewed = (
            row.get("review_status") == "reviewed"
            and bool(row.get("reviewer", "").strip())
            and row.get("manual_grade") in GRADES
        )
        known = not prediction_detected or reviewed
        correct = bool(
            prediction_detected
            and reviewed
            and row["manual_grade"] == "correct"
            and float(row["bbox_iou"]) >= threshold
            and flag(row["garment_class_correct"])
        )
        details.append(
            {
                **row,
                "semantic_known": int(known),
                "strict_correct": int(correct),
                "failure_stage": (
                    "SEG_WRONG_CLASS"
                    if not flag(row["garment_class_correct"])
                    else (
                        "GROUNDING_MISS"
                        if not prediction_detected
                        else (
                            "REQUIRES_SEMANTIC_REVIEW"
                            if not reviewed
                            else "" if correct else "GROUNDING_WRONG_OR_COARSE"
                        )
                    )
                ),
            }
        )
    result = {}
    for track in ("A", "B"):
        values = [r for r in details if r["track"] == track]
        known = sum(r["semantic_known"] for r in values)
        n = len(values)
        correct = sum(r["strict_correct"] for r in values)
        result[track] = {
            "n": n,
            "known": known,
            "pending_semantic_reviews": n - known,
            "correct": correct,
            "strict_accuracy": correct / n if n and n == known else None,
            "accuracy_lower_bound": correct / n if n else None,
            "accuracy_upper_bound": (correct + n - known) / n if n else None,
            "manual_grade_counts": dict(
                Counter(
                    r.get("manual_grade")
                    or ("missed" if r["prediction_status"] != "detected" else "pending")
                    for r in values
                )
            ),
            "per_region": {
                region: {
                    "n": len(v := [r for r in values if r["target_region"] == region]),
                    "known": sum(r["semantic_known"] for r in v),
                    "correct": sum(r["strict_correct"] for r in v),
                    "accuracy": (
                        sum(r["strict_correct"] for r in v) / len(v)
                        if v and all(r["semantic_known"] for r in v)
                        else None
                    ),
                }
                for region in REGIONS
            },
        }
    return result, details


def summarize(
    root: Path | None = None, output: Any = None, semantic_audit: Any = None
) -> Any:
    """汇总逐例结果，保持当前指标的分母定义。

    Args:
        root: 当前项目根目录。
        output: 模型预测或输出文件位置，具体由本函数的读写操作决定。
        semantic_audit: semantic 检查。

    Returns:
        结果字典包含 benchmark_version, prd_module_status, gates, segmentation, grounding,
        attributes, latency, end_to_end, pending_actions, boundary。

    Raises:
        ValueError: Invalid measured module time
    """
    root = root or project_root()
    output = output or root / "reports/reruns/recovery_v2/acceptance"
    meta = json.loads((output / "run_metadata.json").read_text())
    frozen = root / "benchmark/prd_3_1_v2/frozen"
    verify_frozen(root)
    if digest(frozen / "freeze_meta.json") != meta["benchmark_freeze_sha256"]:
        raise ValueError("Run belongs to a different benchmark freeze")
    for name, expected in meta["artifact_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError("Evaluation artifact changed: " + name)
    config = meta["config"]
    seg = json.loads((output / "segmentation_metrics.json").read_text())
    truth = read_csv(frozen / "attribute_test.csv")
    predictions = read_csv(output / "attribute_predictions.csv")
    attr_metrics = {}
    attr_details = []
    for track in ("A", "B"):
        index = {
            (r["sample_id"], r["attribute_name"]): r
            for r in predictions
            if r["track"] == track
        }
        attr_metrics[track], detail = per_attribute_metrics(truth, index, track)
        attr_details.extend(detail)
    rows = read_csv(output / "grounding_predictions.csv")
    audit_path = semantic_audit or output / "grounding_semantic_audit_reviewed.csv"
    grounding, ground_details = strict_grounding(
        rows,
        read_csv(audit_path) if audit_path.is_file() else None,
        config["bbox_match_iou"],
    )
    timings = read_csv(output / "timing.csv")
    timing_metrics = {}
    for track in ("A", "B"):
        timing_metrics[track] = {}
        for module in ("segmentation", "grounding", "attributes"):
            values = [
                float(r["ms"])
                for r in timings
                if r["module"] == module and r["track"] == track
            ]
            if any(not np.isfinite(v) or v < 0 for v in values):
                raise ValueError("Invalid measured module time")
            timing_metrics[track][module] = {
                "n": len(values),
                "mean_ms": float(np.mean(values)) if values else None,
                "median_ms": float(np.median(values)) if values else None,
                "p95_ms": float(np.percentile(values, 95)) if values else None,
            }
    attr_coverage = {
        (r["attribute_name"], r["garment_category"])
        for r in attr_details
        if r["track"] == "A"
    }
    required = {(a, c) for a in ATTRIBUTES for c in CLASS_TO_ID if applies(a, c)}
    complete_attrs = required <= attr_coverage
    complete_seg = (
        all(m["gt_count"] == 50 for m in seg["per_class"].values())
        and len(seg["per_class"]) == 8
    )
    mask_coverage = complete_seg and all(
        m["mean_mask_iou_correct_class"] is not None for m in seg["per_class"].values()
    )
    complete_ground = all(
        grounding["A"]["per_region"][r]["n"]
        >= config["min_reviewed_positive_per_region"]
        for r in REGIONS
    )
    expected_timing = {
        "segmentation": len(
            {r["source_image"] for r in read_csv(frozen / "segmentation_test.csv")}
        )
        * config["timed_passes"],
        "grounding": len(read_csv(frozen / "grounding_test.csv"))
        * config["timed_passes"],
        "attributes": len(
            {
                r["sample_id"]
                for r in truth
                if r["applicability"] == "eligible" and not flag(r.get("ambiguous"))
            }
        )
        * config["timed_passes"],
    }
    gates = []
    for name, value, target, coverage in [
        (
            "3.1.1_mean_mask_iou",
            seg["micro"]["mean_mask_iou_correct_class"],
            config["accuracy_targets"]["mask_iou"],
            mask_coverage,
        ),
        (
            "3.1.2_strict_accuracy",
            grounding["A"]["strict_accuracy"],
            config["accuracy_targets"]["grounding"],
            complete_ground,
        ),
        (
            "3.1.3_human_gt_micro_accuracy",
            attr_metrics["A"]["micro_accuracy"],
            config["accuracy_targets"]["attributes"],
            complete_attrs,
        ),
    ]:
        gates.append(
            {
                "metric": name,
                "track": "A",
                "value": value,
                "target": target,
                "status": metric_status(value, target, coverage=coverage),
            }
        )
    is_gpu = meta["device"].startswith("cuda") and meta.get("gpu") != "none"
    for module, target in config["latency_targets_ms"].items():
        m = timing_metrics["A"][module]
        gates.append(
            {
                "metric": module + "_mean_latency_ms",
                "track": "A",
                "value": m["mean_ms"],
                "target": target,
                "status": metric_status(
                    m["mean_ms"],
                    target,
                    higher=False,
                    coverage=m["n"] == expected_timing[module],
                    evaluated=is_gpu,
                ),
            }
        )
    ground_b = {
        (r["sample_id"], r["query_id"]): r for r in ground_details if r["track"] == "B"
    }
    attr_b = {
        (r["sample_id"], r["attribute_name"]): r
        for r in attr_details
        if r["track"] == "B"
    }
    chain = read_csv(output / "end_to_end_cases.csv")
    final_chain = []
    for row in chain:
        sid = row["sample_id"]
        stage = (
            "SEG_MISS"
            if not flag(row["localized_bbox50"])
            else "SEG_WRONG_CLASS" if not flag(row["class_correct"]) else ""
        )
        if not stage:
            for query_id in json.loads(row["required_queries"]):
                q = ground_b[(sid, query_id)]
                if q["failure_stage"]:
                    stage = q["failure_stage"]
                    break
        if not stage:
            for name in json.loads(row["required_attributes"]):
                a = attr_b[(sid, name)]
                if not a["correct"]:
                    stage = (
                        "GROUNDING_MISS"
                        if a["prediction_status"] == "localization_missing"
                        else (
                            "ATTRIBUTE_MISSING"
                            if a["prediction_status"] != "predicted"
                            else "ATTRIBUTE_WRONG_LABEL"
                        )
                    )
                    break
        final_chain.append(
            {
                **row,
                "failure_stage": stage or "COMPLETE",
                "chain_complete": int(not stage),
            }
        )
    failure_counts = dict(Counter(r["failure_stage"] for r in final_chain))
    status = (
        "PASS"
        if all(g["status"] == "PASS" for g in gates)
        else "FAIL" if any(g["status"] == "FAIL" for g in gates) else "NOT_EVALUATED"
    )
    result = {
        "benchmark_version": meta["benchmark_version"],
        "prd_module_status": status,
        "gates": gates,
        "segmentation": seg,
        "grounding": grounding,
        "attributes": attr_metrics,
        "latency": timing_metrics,
        "end_to_end": {
            "n": len(final_chain),
            "correct_class_coverage": (
                sum(flag(r["class_correct"]) for r in chain) / len(chain)
                if chain
                else None
            ),
            "complete_cases": sum(r["chain_complete"] for r in final_chain),
            "failure_stages": failure_counts,
            "numerical_prd_gate": "NONE_SPECIFIED",
            "track_b_attribute_denominator": attr_metrics["B"]["n"],
            "coordinate_consistency": "Original-image boxes and full-image binary masks; prediction ROI lineage recorded",
        },
        "pending_actions": (
            [
                "Review detected localizations in grounding_semantic_review.html and export reviewed CSV"
            ]
            if grounding["A"]["pending_semantic_reviews"]
            or grounding["B"]["pending_semantic_reviews"]
            else []
        ),
        "boundary": config["test_isolation_note"],
    }
    write_json(output / "acceptance_summary.json", result)
    write_csv(output / "acceptance_gates.csv", gates)
    write_csv(output / "grounding_reviewed_results.csv", ground_details)
    write_csv(output / "attribute_human_gt_results.csv", attr_details)
    write_csv(output / "end_to_end_failure_stages.csv", final_chain)
    lines = [
        "# PRD 3.1 v2 实测验收",
        "",
        f"模块总体状态：**{status}**。数值来自本次真实推理与人工 GT。",
        "",
        "| 指标 | 实测 | 目标 | 状态 |",
        "|---|---:|---:|---|",
    ]
    for g in gates:
        lines.append(
            f"| {g['metric']} | {g['value'] if g['value'] is not None else '待评价'} | {g['target']} | {g['status']} |"
        )
    lines.extend(
        [
            "",
            f"Track B 属性 micro：{attr_metrics['B']['micro_accuracy']}，分母：{attr_metrics['B']['n']}。",
            f"端到端 {len(final_chain)} 例，完整成功 {sum(r['chain_complete'] for r in final_chain)} 例；失败分布：{json.dumps(failure_counts,ensure_ascii=False)}。",
            "",
            "3.1.2 的 coarse 不算严格成功。未审核预测保持 NOT_EVALUATED。CPU 延迟不用于 GPU PRD 验收。",
            "",
            config["test_isolation_note"],
        ]
    )
    (output / "ACCEPTANCE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def build_semantic_review_page(root: Path, output: Any) -> None:
    """构建 semantic 审核 page。

    Args:
        root: 当前项目根目录。
        output: 模型预测或输出文件位置，具体由本函数的读写操作决定。
    """
    rows = read_csv(output / "grounding_predictions.csv")
    payload = {
        "rows": rows,
        "fields": list(rows[0]),
        "images": {
            r["track"] + "_" + r["query_id"]: image_url(root / r["visualization_path"])
            for r in rows
        },
        "fingerprint": digest(output / "grounding_predictions.csv"),
    }
    html = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>局部定位严格审核</title><style>body{font:16px/1.6 system-ui;background:#f4f6f9;color:#18253a;margin:24px auto;max-width:1100px;padding:16px}img{max-width:100%;max-height:650px}button,input,select{font:inherit;padding:9px;margin:6px;border-radius:6px;border:1px solid #bac8d9}article{background:white;padding:20px;border-radius:10px}h1{font-size:24px}</style>
<h1>局部定位严格审核</h1><p>黄色：服装 GT；绿色：目标区域 GT；红色：模型输出。只判断语义目标是否正确，框 IoU 由程序核算。框太粗选 coarse，找错位置选 wrong，漏检选 missed。coarse 不算 PRD 成功。</p>
<label>审核人 <input id="reviewer"></label><button id="prev">上一例</button><button id="next">下一例</button><span id="progress"></span>
<article><h2 id="title"></h2><img id="preview" alt="服装、目标区域与预测框"><p id="metric"></p>
<select id="grade"><option value="">待审核</option><option value="correct">correct：语义位置正确</option><option value="coarse">coarse：整件或过粗</option><option value="wrong">wrong：位置/对象错误</option><option value="missed">missed：漏检</option></select><input id="note" placeholder="说明"><button id="save">保存本例</button></article>
<button id="export">导出 grounding_semantic_audit_reviewed.csv</button><p>把导出 CSV 放回本次 acceptance 报告目录，运行 --phase report。不需要重复模型推理。</p>
<script id="data" type="application/json">__DATA__</script><script>'use strict';const d=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),key='prd31-semantic-'+d.fingerprint;let rows=d.rows,i=0;
try{let c=JSON.parse(localStorage.getItem(key)||'null');if(c){rows=c.rows;i=c.i;$('reviewer').value=c.reviewer;}}catch(e){}
function show(){let r=rows[i];$('title').textContent=r.track+' · '+r.query_id+' · '+r.sample_id;$('preview').src=d.images[r.track+'_'+r.query_id];$('metric').textContent='状态 '+r.prediction_status+'；bbox IoU '+Number(r.bbox_iou).toFixed(3)+'；模型短语 '+r.model_phrase;$('grade').value=r.manual_grade;$('note').value=r.annotation_note;$('progress').textContent=(i+1)+' / '+rows.length+'；已审核 '+rows.filter(r=>r.review_status==='reviewed').length;}
function save(){let r=rows[i];r.manual_grade=$('grade').value;r.annotation_note=$('note').value;r.reviewer=$('reviewer').value.trim();r.review_status=r.manual_grade&&r.reviewer?'reviewed':'pending';try{localStorage.setItem(key,JSON.stringify({rows,i,reviewer:$('reviewer').value}));}catch(e){}show();}
$('save').onclick=save;$('prev').onclick=()=>{save();i=Math.max(0,i-1);show();};$('next').onclick=()=>{save();i=Math.min(rows.length-1,i+1);show();};$('export').onclick=()=>{save();let esc=v=>'"'+String(v??'').replaceAll('"','""')+'"',csv='\ufeff'+[d.fields.map(esc).join(','),...rows.map(r=>d.fields.map(k=>esc(r[k])).join(','))].join('\r\n'),a=document.createElement('a'),url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));a.href=url;a.download='grounding_semantic_audit_reviewed.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};show();</script></html>"""
    (output / "grounding_semantic_review.html").write_text(
        html.replace(
            "__DATA__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
        ),
        encoding="utf-8",
    )


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir", default="reports/reruns/recovery_v2/acceptance")
    p.add_argument("--grounding-audit")
    args = p.parse_args()
    result = summarize(
        output=resolve_path(args.report_dir),
        semantic_audit=(
            resolve_path(args.grounding_audit) if args.grounding_audit else None
        ),
    )
    print(
        json.dumps(
            {
                "prd_module_status": result["prd_module_status"],
                "gates": result["gates"],
                "pending_actions": result["pending_actions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
