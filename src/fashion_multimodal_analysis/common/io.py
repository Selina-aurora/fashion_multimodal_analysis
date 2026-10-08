"""项目公共接口：参数约定与返回结构供其他模块复用。

读取清单并保存 CSV/JSON，统一 UTF-8 编码。

CSV 写入 BOM，方便用 Excel 打开中文备注；JSON 禁止写入 NaN。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """读取含表头的 CSV，保留原始字符串值，不自动转换标签或数值。

    Args:
        path: 清单或结果表路径。

    Returns:
        每行对应一个字段名到字符串值的字典。

    Raises:
        OSError: 文件不存在或无法读取。
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None
) -> None:
    """按指定字段顺序写入表格，自动创建上级目录。

    Args:
        path: 输出路径。
        rows: 要保存的行；不在指定字段中的值不会写入。
        fields: 表头顺序，省略时按字段首次出现的顺序推导。

    Raises:
        OSError: 无法创建目录或写入文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, data: Any) -> None:
    """写入标准 JSON，遇到 NaN 或不可序列化对象时抛出异常。

    Args:
        path: 输出路径，自动创建上级目录。
        data: 可序列化的指标或元数据。

    Raises:
        ValueError: 数据含 NaN 或 Infinity。
        TypeError: 数据含不可序列化的对象。
        OSError: 无法写入目标文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
