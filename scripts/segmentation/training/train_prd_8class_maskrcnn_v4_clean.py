"""分类命令入口，实际算法在 src 中实现；--help 仅显示参数。

3.1.1 实例分割：数据与模型版本分别记录，指标按固定匹配和计时口径解释。
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

    run_entry("train_prd_8class_maskrcnn_v4_clean")
