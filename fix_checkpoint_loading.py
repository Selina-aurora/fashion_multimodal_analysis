"""项目公共接口：参数约定与返回结构供其他模块复用。

Repair full locally trained checkpoint loading; retain source backups and model files.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def repair(project: Any) -> dict[str, Any]:
    """repair。

    Args:
        project: project。

    Returns:
        结果字典，主要字段为 changed_files, backup, next_action。

    Raises:
        FileNotFoundError: Run this from the organized project root or set --project
    """
    root = Path(project).expanduser().resolve()
    package = root / "src/fashion_multimodal_analysis"
    if not package.is_dir():
        raise FileNotFoundError(
            "Run this from the organized project root or set --project"
        )
    backup = (
        root
        / "archive"
        / (
            "checkpoint_loading_fix_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    changes = []
    for path in sorted(package.rglob("*.py")):
        source = path.read_bytes()
        text = source.decode("utf-8")
        tree = ast.parse(text)
        lines = source.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        replacements = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "load"
            ):
                continue
            if any(keyword.arg == "weights_only" for keyword in node.keywords):
                continue
            node.keywords.append(
                ast.keyword(arg="weights_only", value=ast.Constant(value=False))
            )
            start = offsets[node.lineno - 1] + node.col_offset
            end = offsets[node.end_lineno - 1] + node.end_col_offset
            replacements.append((start, end, ast.unparse(node).encode("utf-8")))
        if not replacements:
            continue
        modified = source
        for start, end, value in sorted(replacements, reverse=True):
            modified = modified[:start] + value + modified[end:]
        ast.parse(modified.decode("utf-8"))
        old = backup / path.relative_to(root)
        old.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, old)
        temporary = path.with_suffix(".py.tmp")
        temporary.write_bytes(modified)
        temporary.replace(path)
        changes.append(str(path.relative_to(root)))
    if changes:
        (backup / "FIX_LOG.json").write_text(
            json.dumps(
                {
                    "changed_files": changes,
                    "scope": "Full checkpoints produced by this project; model weights, manifests and reports unchanged",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    return {
        "changed_files": changes,
        "backup": str(backup) if changes else None,
        "next_action": "Rerun --phase all to reuse the completed checkpoint and continue Core evaluation",
    }


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".")
    args = parser.parse_args()
    print(json.dumps(repair(args.project), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
