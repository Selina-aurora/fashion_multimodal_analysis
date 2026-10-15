"""处理已保存的固定八图预测，仅检查一致性，不再次运行模型。"""

import argparse
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from fashion_multimodal_analysis.integration.semantic_consistency import (
    apply_consistency_review,
)


def main() -> None:
    """运行指定配置的一致性复核并汇总状态；复核提示不作为人工真值。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir")
    p.add_argument("--output-dir")
    args = p.parse_args()
    root = Path.cwd()
    if args.input_dir:
        source = Path(args.input_dir).resolve()
    else:
        candidates = sorted(
            (root / "reports/reruns/recovery_20261009").glob(
                "prd31_neckline_context_*"
            ),
            reverse=True,
        )
        candidates = [
            f for f in candidates if f.is_dir() and (f / "summary.json").is_file()
        ]
        if not candidates:
            raise SystemExit("找不到本次八图缓存结果，请指定 --input-dir。")
        source = candidates[0]
    summary = json.loads((source / "summary.json").read_text())
    if summary.get("images") != 8 or summary.get("pipeline_status_counts") != {
        "completed": 8
    }:
        raise SystemExit("输入不是已完成的八图回归。")
    if (
        summary.get("input_sha256", {}).get("segmentation_checkpoint")
        != "c5ba795225349b1c521aac0517ac7c493d9818dcc17d558ac4a3f6f8669452f0"
    ):
        raise SystemExit("缓存权重与原 V5 不符。")
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root
        / "reports/reruns/recovery_20261009"
        / ("semantic_consistency_review_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    if output == source or source in output.parents:
        raise SystemExit("输出必须是独立的新目录。")
    shutil.copytree(source, output)
    cases = []
    all_flags = []
    for index in range(1, 9):
        case = f"case_{index:02d}"
        folder = output / case
        payload = json.loads((folder / "predictions.json").read_text())
        with Image.open(output / "source_images" / (case + ".jpg")) as raw:
            canvas = raw.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for instance in payload["instances"]:
            # 写入的是交付状态，raw 模型记录单独保留，不重写原输入目录。
            instance["raw_regions_before_consistency_review"] = json.loads(
                json.dumps(instance["regions"])
            )
            instance["raw_attributes_before_consistency_review"] = json.loads(
                json.dumps(instance["attributes"])
            )
            flags = apply_consistency_review(
                instance["regions"], instance["attributes"]
            )
            instance.setdefault("review_flags", []).extend(flags)
            all_flags.extend(flags)
            for flag in flags:
                cases.append(
                    {
                        "case_id": case,
                        "instance_id": instance["instance_id"],
                        "category_candidate": instance["category"],
                        "evidence": flag,
                        "semantic_review_status": "pending",
                    }
                )
            for name, region in instance["regions"].items():
                if region.get("status") == "requires_review" and region.get("bbox"):
                    draw.rectangle(region["bbox"], outline="orange", width=3)
                    draw.text(
                        (region["bbox"][0], region["bbox"][1]),
                        name + " REVIEW",
                        fill="orange",
                    )
        payload["review_processing_status"] = "completed"
        payload["consistency_policy"] = "semantic_consistency_review_20261009_v1"
        payload["model_inference_run_in_this_step"] = False
        (folder / "predictions.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        canvas.thumbnail((1400, 1400))
        canvas.save(folder / "consistency_review.jpg", quality=93)
    result = {
        "status": "cached_review_completed",
        "images": 8,
        "predicted_instances": summary["predicted_instances"],
        "conflict_evidence_counts": dict(Counter(f["reason"] for f in all_flags)),
        "region_review_counts": dict(
            Counter(
                name
                for i in range(1, 9)
                for instance in json.loads(
                    (output / f"case_{i:02d}" / "predictions.json").read_text()
                )["instances"]
                for name, r in instance["regions"].items()
                if r.get("status") == "requires_review"
            )
        ),
        "attribute_review_counts": dict(
            Counter(
                name
                for i in range(1, 9)
                for instance in json.loads(
                    (output / f"case_{i:02d}" / "predictions.json").read_text()
                )["instances"]
                for name, r in instance["attributes"].items()
                if r.get("status") == "requires_review"
            )
        ),
        "source_report": str(source),
        "model_inference_run": False,
        "classes_boxes_masks_changed": False,
        "accuracy_evaluated": False,
        "prd_acceptance_evaluated": False,
        "note": "Review statuses are abstentions, not semantic corrections or accuracy gains. Raw model candidates are preserved. Prior summary.json describes the original inference.",
    }
    (output / "consistency_review_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "consistency_review_evidence.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(output.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(output))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("UPLOAD_THIS_ZIP:", archive, flush=True)


if __name__ == "__main__":
    main()
