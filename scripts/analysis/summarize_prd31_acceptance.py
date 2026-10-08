"""分类命令入口，实际算法在 src 中实现；--help 仅显示参数。

结果分析：依据已有逐例记录定位问题，不替代新的模型评估。
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

    run_entry("summarize_prd31_acceptance")
