"""3.1.1 实例分割：数据与模型版本分别记录，指标按固定匹配和计时口径解释。

Train the path-isolated v4 candidate using the existing B1 architecture/settings.
"""

from __future__ import annotations

import sys

from fashion_multimodal_analysis.common.paths import project_root


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        FileNotFoundError: Run prepare_prd31_recovery first
    """
    root = project_root()
    if not (root / "configs/prd_8class_train_v4_clean.csv").is_file():
        raise FileNotFoundError("Run prepare_prd31_recovery first")
    from fashion_multimodal_analysis.segmentation.training import (
        train_prd_8class_maskrcnn_v3_b1_dataexp as trainer,
    )

    arguments = sys.argv[1:]
    defaults = [
        "--train-csv",
        "configs/prd_8class_train_v4_clean.csv",
        "--val-csv",
        "configs/prd_8class_val_v1.csv",
        "--output-dir",
        "outputs/prd_instance_segmentation/maskrcnn_8class_v4_clean",
        "--report-dir",
        "reports/reruns/recovery_v2/train_v4_clean",
        "--experiment-name",
        "v4-clean",
        "--verify-isolation",
    ]
    previous = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *defaults, *arguments]
        trainer.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
