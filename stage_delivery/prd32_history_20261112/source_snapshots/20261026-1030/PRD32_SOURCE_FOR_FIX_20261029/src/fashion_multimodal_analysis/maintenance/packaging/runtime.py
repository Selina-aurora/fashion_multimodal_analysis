"""运行与维护：记录文件身份、阶段状态及依赖，长任务中断后先检查状态再续跑。

Copy the runtime required by categorized entrypoints into a portable bundle.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def copy_runtime(source: Path, destination: Path) -> None:
    """复制分类入口所需的源码、注册表和名称别名。

    Args:
        source: 已整理的项目根目录。
        destination: 新的便携运行工作目录。
    """
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "configs").mkdir(exist_ok=True)
    # 同时保留 JSON 注册表和路径别名，冻结 CSV 才能找到重命名后的掩码。
    for directory in ("src", "scripts"):
        shutil.copytree(
            source / directory,
            destination / directory,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
    shutil.copy2(source / "pyproject.toml", destination / "pyproject.toml")
