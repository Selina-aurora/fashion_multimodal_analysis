"""项目公共接口：参数约定与返回结构供其他模块复用。

统一定位项目、外部数据和整理前的文件名。

代码与配置保存相对路径，项目搬到另一台 GPU 后只需保留同级的
fashion_data 目录，或者设置 FASHION_DATA_ROOT。历史评估清单保持原字节，
文件名通过 path_aliases.json 映射，避免整理名称时改变冻结集的哈希。

主要接口为 project_root、data_root、resolve_path 和 resolve_record_path。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


def project_root() -> Path:
    """定位包含配置与源码的项目根目录。

    Returns:
        当前工作副本的目录。环境变量可用于显式选择另一份副本。

    Raises:
        FileNotFoundError: 环境变量指向无效项目，或者无法找到源码工作副本。
    """
    value = os.environ.get("FASHION_PROJECT_ROOT")
    if value:
        root = Path(value).expanduser().resolve()
        if not (root / "configs").is_dir():
            raise FileNotFoundError(f"项目缺少 configs 目录：{root}")
        return root
    for root in Path(__file__).resolve().parents:
        if (root / "pyproject.toml").is_file() and (root / "configs").is_dir():
            return root
    raise FileNotFoundError("请从源码工作副本运行，或设置 FASHION_PROJECT_ROOT。")


def data_root() -> Path:
    """定位与项目同级的外部数据目录。

    Returns:
        FASHION_DATA_ROOT 指定的目录；未指定时为项目同级 fashion_data。
    """
    value = os.environ.get("FASHION_DATA_ROOT")
    return (
        Path(value).expanduser().resolve()
        if value
        else project_root().parent / "fashion_data"
    )


@lru_cache(maxsize=8)
def _aliases(root: Path) -> dict[str, str]:
    """缓存名称映射，避免逐实例读取同一份 JSON。

    Args:
        root: 当前项目根目录。

    Returns:
        整理前相对路径到当前相对路径的映射；未找到映射文件时为空字典。
    """
    path = root / "src/fashion_multimodal_analysis/common/path_aliases.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def resolve_record_path(value: str | Path, root: Path, data: Path) -> Path:
    """将配置或历史清单中的路径映射到当前工作副本。

    Args:
        value: 项目相对路径，或以 ../fashion_data 开头的数据路径。
        root: 当前项目根目录。
        data: 当前外部数据根目录。

    Returns:
        可直接交给文件读取函数的路径。原来使用服务器目录的历史记录
        也能依据项目名/数据目录名迁移；外部数据文件名保持原样。

    Notes:
        只映射项目内的已整理文件。不会修改原清单、标签、掩码像素或结果表。
    """
    text = str(value).strip().replace("\\", "/")
    path = Path(text).expanduser()
    if text.startswith("../fashion_data/"):
        return (data / text[len("../fashion_data/") :]).resolve()
    parts = path.parts
    if path.is_absolute():
        if "fashion_data" in parts:
            index = parts.index("fashion_data")
            return data.joinpath(*parts[index + 1 :]).resolve()
        if "fashion_multimodal_analysis" in parts:
            index = parts.index("fashion_multimodal_analysis")
            path = root.joinpath(*parts[index + 1 :])
        elif root.name in parts:
            index = len(parts) - 1 - tuple(reversed(parts)).index(root.name)
            path = root.joinpath(*parts[index + 1 :])
    else:
        path = root / path
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        return path.resolve()
    return (root / _aliases(root).get(relative, relative)).resolve()


def resolve_path(value: str | Path) -> Path:
    """使用当前项目与外部数据目录解析一个文件引用。

    Args:
        value: 配置/CSV 中的路径字符串或 pathlib.Path。

    Returns:
        按当前工作副本和文件名映射解析后的路径。
    """
    return resolve_record_path(value, project_root(), data_root())
