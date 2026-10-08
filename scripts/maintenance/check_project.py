"""分类命令入口，实际算法在 src 中实现；--help 仅显示参数。

运行与维护：记录文件身份、阶段状态及依赖，长任务中断后先检查状态再续跑。
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

    run_entry("check_project")
