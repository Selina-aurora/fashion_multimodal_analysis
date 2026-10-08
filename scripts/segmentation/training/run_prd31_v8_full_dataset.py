"""八类原始训练数据：准备索引、从 V5 微调、断点续跑、固定口径评估。

入口只负责把命令转交给 src 中的实现；模型加载和实验逻辑均在对应模块内。
从项目根目录执行本文件。--help 只显示参数，不启动训练或加载模型。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "src/fashion_multimodal_analysis").is_dir()
)
sys.path.insert(0, str(ROOT / "src"))


if __name__ == "__main__":
    from fashion_multimodal_analysis.cli import run_entry

    run_entry("run_prd31_v8_full_dataset")
