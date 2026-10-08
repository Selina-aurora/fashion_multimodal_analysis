"""结果分析：依据已有逐例记录定位问题，不替代新的模型评估。"""

from __future__ import annotations

from pathlib import Path

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    import csv
    from pathlib import Path

    BASELINE = [
        [
            "collar",
            5,
            0,
            5,
            0,
            0,
            0.0,
            100.0,
            0.0,
            "Baseline: all 5 responses are coarse garment-level boxes.",
        ],
        ["cuff", 5, 0, 4, 0, 1, 0.0, 80.0, 20.0, "Baseline: mostly coarse; 1 miss."],
        [
            "hem",
            5,
            0,
            3,
            0,
            2,
            0.0,
            60.0,
            40.0,
            "Baseline: partial feasibility, 2 misses.",
        ],
        ["pocket", 5, 0, 0, 0, 5, 0.0, 0.0, 100.0, "Baseline: complete miss."],
        ["shoulder", 5, 0, 2, 0, 3, 0.0, 40.0, 60.0, "Baseline: 2 coarse, 3 misses."],
        ["waist", 5, 0, 3, 0, 2, 0.0, 60.0, 40.0, "Baseline: 3 coarse, 2 misses."],
        [
            "pattern",
            5,
            1,
            2,
            0,
            2,
            20.0,
            60.0,
            40.0,
            "Baseline: 1 strict-correct, 2 coarse, 2 misses.",
        ],
        ["decoration", 5, 0, 0, 0, 5, 0.0, 0.0, 100.0, "Baseline: complete miss."],
    ]

    FOLLOWUP = [
        [
            "collar",
            "Spatial-HR",
            5,
            0,
            4,
            0,
            1,
            0.0,
            80.0,
            "Spatial crop did not tighten localization; SAM pilot also remained coarse.",
        ],
        [
            "cuff",
            "Spatial-HR",
            5,
            0,
            5,
            0,
            0,
            0.0,
            100.0,
            "Response availability improved, but localization remained coarse; supervised/top-k diagnostics still showed unstable candidate quality/ranking.",
        ],
        [
            "hem",
            "Spatial-HR",
            5,
            0,
            0,
            0,
            5,
            0.0,
            0.0,
            "Fixed spatial prior was unstable for hem; later supervised candidate diagnostics showed some feasibility but not robust acceptance-level localization.",
        ],
        [
            "shoulder",
            "Spatial refinement visual review",
            5,
            0,
            5,
            0,
            0,
            0.0,
            100.0,
            "Misses were recovered, but boxes remained broad upper-body/shoulder-inclusive regions.",
        ],
        [
            "waist",
            "Spatial refinement visual review",
            5,
            0,
            5,
            0,
            0,
            0.0,
            100.0,
            "Misses were recovered, but boxes remained broad waist-inclusive regions.",
        ],
        [
            "pattern",
            "Spatial refinement visual review",
            5,
            1,
            4,
            0,
            0,
            20.0,
            100.0,
            "Misses were recovered; only the case where the pattern occupies nearly the whole garment is strict-correct.",
        ],
        [
            "pocket",
            "Prompt-scale diagnostic visual review",
            5,
            0,
            5,
            0,
            0,
            0.0,
            100.0,
            "Candidate availability recovered, but boxes generally covered the whole trousers/outerwear rather than a tight pocket.",
        ],
        [
            "decoration",
            "Prompt-scale diagnostic visual review",
            5,
            1,
            4,
            0,
            0,
            20.0,
            100.0,
            "Candidate availability recovered; one large graphic was localized reasonably, while most boxes remained coarse.",
        ],
    ]

    OUT = get_project_root() / "reports/prd_region_coverage/final_summary_v1"
    OUT.mkdir(parents=True, exist_ok=True)

    baseline_csv = OUT / "prd_3_1_2_final_baseline_40case.csv"
    followup_csv = OUT / "prd_3_1_2_final_targeted_followup.csv"
    summary_md = OUT / "prd_3_1_2_final_summary_v1.md"

    with baseline_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "region",
                "N",
                "correct",
                "coarse",
                "wrong",
                "missed",
                "strict_accuracy_pct",
                "usable_rate_pct",
                "missed_rate_pct",
                "note",
            ]
        )
        w.writerows(BASELINE)
        w.writerow(
            [
                "OVERALL",
                40,
                1,
                19,
                0,
                20,
                2.5,
                50.0,
                50.0,
                "Comparable frozen 40-case baseline across all 8 PRD regions.",
            ]
        )

    with followup_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "region",
                "followup_condition",
                "N",
                "correct",
                "coarse",
                "wrong",
                "missed",
                "strict_accuracy_pct",
                "usable_rate_pct",
                "interpretation",
            ]
        )
        w.writerows(FOLLOWUP)

    lines = []
    lines.append("# PRD 3.1.2 Final Summary v1")
    lines.append("")
    lines.append("## Evaluation rubric")
    lines.append("")
    lines.append(
        "- correct: requested region is localized tightly and semantically correctly"
    )
    lines.append(
        "- coarse: requested region is included, but the box is substantially broader than the target"
    )
    lines.append("- wrong: prediction falls on another region/object")
    lines.append("- missed: no usable prediction")
    lines.append("- usable rate = (correct + coarse) / N")
    lines.append("")
    lines.append("## A. Frozen 40-case baseline (comparable across all 8 regions)")
    lines.append("")
    lines.append(
        "|Region|N|Correct|Coarse|Wrong|Missed|Strict Acc.|Usable|Missed Rate|"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in BASELINE:
        lines.append(
            f"|{r[0]}|{r[1]}|{r[2]}|{r[3]}|{r[4]}|{r[5]}|{r[6]:.1f}%|{r[7]:.1f}%|{r[8]:.1f}%|"
        )
    lines.append("|OVERALL|40|1|19|0|20|2.5%|50.0%|50.0%|")
    lines.append("")
    lines.append("## B. Targeted follow-up diagnostics")
    lines.append("")
    lines.append(
        "> These follow-up conditions are region-specific diagnostics and should not be aggregated as a single acceptance metric."
    )
    lines.append("")
    lines.append(
        "|Region|Condition|N|Correct|Coarse|Wrong|Missed|Strict Acc.|Usable|Interpretation|"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in FOLLOWUP:
        lines.append(
            f"|{r[0]}|{r[1]}|{r[2]}|{r[3]}|{r[4]}|{r[5]}|{r[6]}|{r[7]:.1f}%|{r[8]:.1f}%|{r[9]}|"
        )
    lines.append("")
    lines.append("## Final conclusion")
    lines.append("")
    lines.append(
        "PRD 3.1.2 has completed an 8-region Grounding-DINO baseline, manual error audit, and region-specific diagnostics. The main failure mode has shifted from candidate absence to localization granularity. Spatial cropping can recover missed shoulder/waist/pattern cases, and prompt/scale changes can recover pocket/decoration responses, but most outputs remain garment-level or coarse. Collar/cuff/hem diagnostics also show that local high-resolution crops, SAM refinement, and a small supervised detector do not consistently produce tight local-region boxes. The current results are therefore retained as a feasibility/diagnostic baseline rather than an acceptance-level localization result; further minor prompt/threshold/crop tuning is not recommended at this stage."
    )

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Wrote:")
    print(baseline_csv)
    print(followup_csv)
    print(summary_md)


if __name__ == "__main__":
    main()
