"""只读检查 V5/V8 权重、冻结清单、SQLite 索引及外部数据。

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

    run_entry("check_assets")
