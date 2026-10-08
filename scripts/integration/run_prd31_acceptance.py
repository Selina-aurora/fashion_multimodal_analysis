"""分类命令入口，实际算法在 src 中实现；--help 仅显示参数。

3.1 接口联调：区分预测 ROI 传递和使用 GT 的受控诊断，保留对象 ID 与坐标对应关系。
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

    run_entry("run_prd31_acceptance")
