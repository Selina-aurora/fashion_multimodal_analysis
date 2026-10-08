"""从全部已保存评估建立独立汇总表，不重新运行 GPU。

命令参数和实现位于 src，入口不重复实验逻辑。
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

    run_entry("summarize_saved_results")
