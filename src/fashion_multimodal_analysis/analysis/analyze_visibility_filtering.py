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
    from pathlib import Path

    import pandas as pd

    src = Path(
        "reports/grounding_group_evaluation/manual_localization_audit_reviewed.csv"
    )

    out_dir = Path("reports/repository_audit/visibility_filtering")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)

    print("Total samples:", len(df))

    # baseline
    baseline = df["error_type"].value_counts().reset_index()

    baseline.columns = ["error_type", "count"]

    baseline.to_csv(out_dir / "baseline_error_distribution.csv", index=False)

    # simulate visibility filtering
    filtered = df[df["error_type"] != "target_absent"]

    print("After visibility filtering:", len(filtered))

    filtered_error = filtered["error_type"].value_counts().reset_index()

    filtered_error.columns = ["error_type", "count"]

    filtered_error.to_csv(out_dir / "filtered_error_distribution.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "total_cases",
                "target_absent_cases",
                "remaining_cases_after_filter",
            ],
            "value": [
                len(df),
                (df["error_type"] == "target_absent").sum(),
                len(filtered),
            ],
        }
    )

    summary.to_csv(out_dir / "visibility_filtering_summary.csv", index=False)

    print(summary)
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
