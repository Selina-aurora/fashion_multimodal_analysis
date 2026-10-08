"""项目公共接口：参数约定与返回结构供其他模块复用。

Lazy command dispatch: listing and help never load model dependencies.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

from fashion_multimodal_analysis.common.paths import project_root


def registry() -> dict[str, dict[str, Any]]:
    """读取命令注册表，连接分类脚本与 src 中的实现。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return json.loads(
        Path(__file__).with_name("command_registry.json").read_text(encoding="utf-8")
    )


def command_help(command: str, spec: dict[str, Any]) -> None:
    """显示静态参数说明，不加载模型或启动实验。

    Args:
        command: 命令注册表中的实验名称。
        spec: 记录字段，使用 options, description, entrypoint, implementation。
    """
    print(spec["description"])
    print("Entry: " + spec["entrypoint"])
    print("Implementation: " + spec["implementation"])
    print(
        "Static help (no model loaded); run arguments are parsed by the implementation."
    )
    if not spec["options"]:
        print(
            "This historical experiment has no argparse options; see its module constants and docs/commands.md."
        )
    for option in spec["options"]:
        details = "; ".join(
            key + "=" + option[key]
            for key in ("default", "choices", "required", "action", "help")
            if key in option
        )
        print("  " + ", ".join(option["flags"]) + ("  " + details if details else ""))


def run_entry(command: str, arguments: list[str] | None = None) -> None:
    """按命令注册表分派实验，运行结束后恢复 argv 和工作目录。

    Args:
        command: 命令注册表中的实验名称。
        arguments: 转交给实验入口的命令行参数。

    Raises:
        SystemExit: 执行本函数的操作失败。
    """
    records = registry()
    if command not in records:
        command = next(
            (key for key, item in records.items() if item["name"] == command), command
        )
    if command not in records:
        raise SystemExit("Unknown command: " + command + "; use fashion-analysis list")
    spec = records[command]
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if "--help" in arguments or "-h" in arguments:
        command_help(command, spec)
        return
    old_argv, old_cwd = sys.argv[:], Path.cwd()
    try:
        os.chdir(project_root())
        sys.argv = [spec["entrypoint"], *arguments]
        runpy.run_module(spec["module"], run_name="__main__")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. See docs/setup.md and install the dependencies for this command."
        ) from exc
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        SystemExit: Usage: fashion-analysis list [category] | run COMMAND [arguments]
    """
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("Usage: fashion-analysis list [category] | run COMMAND [arguments]")
        return
    if args[0] == "list":
        category = args[1] if len(args) > 1 else ""
        for key, item in registry().items():
            if category in item["category"]:
                print(item["category"] + " / " + item["name"])
        return
    if args[0] == "run" and len(args) >= 2:
        run_entry(args[1], args[2:])
        return
    raise SystemExit(
        "Usage: fashion-analysis list [category] | run COMMAND [arguments]"
    )


if __name__ == "__main__":
    main()
