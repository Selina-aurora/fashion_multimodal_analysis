"""分类命令入口，实际算法在 src 中实现；--help 仅显示参数。

基准集管理：保留训练、开发、固定回归/审核划分，冻结后的标签不能随结果调整。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "src/fashion_multimodal_analysis").is_dir()
)
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from fashion_multimodal_analysis.cli import run_entry

    run_entry("prepare_prd31_recovery")
