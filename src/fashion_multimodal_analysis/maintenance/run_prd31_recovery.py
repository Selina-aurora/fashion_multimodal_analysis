"""运行与维护：记录文件身份、阶段状态及依赖，长任务中断后先检查状态再续跑。

One workflow for clean retraining, reviewed tests and reproducible PRD reports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from fashion_multimodal_analysis.benchmarking.build_prd31_review_packet import (
    build as build_review,
)
from fashion_multimodal_analysis.benchmarking.prepare_prd31_recovery import (
    digest,
    prepare,
)
from fashion_multimodal_analysis.common.io import read_csv, write_json
from fashion_multimodal_analysis.common.paths import project_root, resolve_path
from fashion_multimodal_analysis.common.schema import CLASS_TO_ID
from fashion_multimodal_analysis.segmentation.training.checkpoint_state import (
    validate_checkpoint,
)


def record_progress(root: Path, stage: str, detail: str = "", **extra: Any) -> Any:
    """Persist the current stage before lengthy work, independently of the final result.

    Args:
        root: 当前项目根目录。
        stage: 当前流程阶段。
        detail: 用于进度记录的说明文本。

    Returns:
        结果字典包含 stage, detail, updated_at, workflow_pid。
    """
    path = root / "reports/reruns/recovery_v2/workflow_progress.json"
    value = {
        "stage": stage,
        "detail": detail,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "workflow_pid": os.getpid(),
        **extra,
    }
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)
    print(f"[{value['updated_at']}] {stage}: {detail}", flush=True)
    return value


@contextmanager
def workflow_lock(root: Path) -> Iterator[Any]:
    """On AutoDL, prevent a second workflow from writing to the same experiment.

    Args:
        root: 当前项目根目录。

    Yields:
        按需产生的记录/任务；不一次性加载全部条目。

    Raises:
        FileExistsError: A recovery workflow is already running in this checkout; use
        --status
    """
    path = root / "reports/reruns/recovery_v2/workflow.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        if os.name == "posix":
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise FileExistsError(
                    "A recovery workflow is already running in this checkout; use --status"
                )
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def show_status(root: Path) -> Any:
    """读取最近保存的流程状态；状态记录不等于进程仍在运行。

    Args:
        root: 当前项目根目录。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    report = root / "reports/reruns/recovery_v2"
    result = {}
    for name in ("workflow_progress.json", "workflow_status.json"):
        path = report / name
        result[name[:-5]] = (
            json.loads(path.read_text())
            if path.is_file()
            else {"status": "NOT_STARTED"}
        )
    log = report / "workflow.log"
    if log.is_file():
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 12000))
            result["recent_log"] = (
                handle.read().decode("utf-8", errors="replace").splitlines()[-12:]
            )
    train_log = report / "train_v4_clean/train_log.csv"
    rows = read_csv(train_log) if train_log.is_file() else []
    result["last_logged_epoch"] = rows[-1]["epoch"] if rows else None
    result["note"] = "阶段是最后保存的进度；是否仍在运行请结合进程和日志更新时间判断。"
    return compact_state(result)


def compact_state(value: dict[str, Any]) -> Any:
    """压缩状态展示，完整问题列表保留在原报告文件中。

    Args:
        value: 记录字段，使用 missing_assets。

    Returns:
        返回 value，由函数体中同名变量的计算/收集过程得到。
    """
    if isinstance(value, dict):
        result = {
            k: compact_state(v) for k, v in value.items() if k != "missing_assets"
        }
        if "missing_assets" in value:
            result["missing_assets_preview"] = value["missing_assets"][:8]
            result["full_preflight_report"] = (
                "reports/reruns/recovery_v2/gpu_preflight.json"
            )
        return result
    if isinstance(value, list):
        return [compact_state(v) for v in value]
    return value


def preflight(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """在训练前检查文件、依赖、CUDA 和 torchvision 运算可用性。

    Args:
        root: 当前项目根目录。
        config: 记录字段，使用 train_manifest, val_manifest。

    Returns:
        结果字典，主要字段为 status, missing_asset_count, missing_assets, dependencies,
        cuda_available, runtime_error, next_action。
    """
    manifests = [
        root / config["train_manifest"],
        root / config["val_manifest"],
        root / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv",
    ]
    missing = set()
    for path in manifests:
        for row in read_csv(path):
            for key in (
                "source_image",
                "mask_path" if "mask_path" in row else "gt_mask_path",
            ):
                if row.get(key) and not resolve_path(row[key]).is_file():
                    missing.add(str(resolve_path(row[key])))
    required = ("torch", "torchvision", "transformers", "numpy", "PIL", "pandas")
    dependencies = {name: bool(importlib.util.find_spec(name)) for name in required}
    cuda = False
    runtime_error = ""
    if dependencies["torch"]:
        try:
            import torch

            cuda = torch.cuda.is_available()
            if dependencies["torchvision"]:
                import torchvision

                # Real NMS execution detects mismatched torch/torchvision builds before training.
                nms_device = "cuda" if cuda else "cpu"
                torchvision.ops.nms(
                    torch.tensor([[0.0, 0.0, 2.0, 2.0]], device=nms_device),
                    torch.tensor([1.0], device=nms_device),
                    0.5,
                )
                if cuda:
                    torch.cuda.synchronize()
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
    return {
        "status": (
            "READY"
            if not missing and all(dependencies.values()) and cuda and not runtime_error
            else "BLOCKED"
        ),
        "missing_asset_count": len(missing),
        "missing_assets": sorted(missing),
        "dependencies": dependencies,
        "cuda_available": cuda,
        "runtime_error": runtime_error,
        "next_action": "Run this workflow in the original AutoDL checkout with the external fashion_data dataset and CUDA.",
    }


def command(root: Path, module: str, arguments: list[str]) -> None:
    """启动当前项目的子命令，并保留日志与失败状态。

    Args:
        root: 当前项目根目录。
        module: 要执行的 Python 模块名称。
        arguments: 转交给实验入口的命令行参数。

    Raises:
        subprocess.CalledProcessError: 执行本函数的操作失败。
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    print("Running: " + module + " " + " ".join(arguments), flush=True)
    record_progress(root, "RUNNING_COMMAND", module)
    path = root / "reports/reruns/recovery_v2/workflow.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-u", "-m", module, *arguments]
    with path.open("ab", buffering=0) as log, path.open("rb") as reader:
        log.write(("\nRunning: " + module + " " + " ".join(arguments) + "\n").encode())
        reader.seek(0, os.SEEK_END)
        process = subprocess.Popen(
            argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        started = last_notice = time.monotonic()

        def display_new_output() -> bool:
            """display new 输出。

            Returns:
                本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
            """
            chunk = reader.read()
            if chunk:
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            return bool(chunk)

        try:
            while True:
                try:
                    code = process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    code = None
                displayed = display_new_output()
                now = time.monotonic()
                if displayed:
                    last_notice = now
                if code is not None:
                    break
                if now - last_notice >= 30:
                    record_progress(
                        root,
                        "RUNNING_COMMAND",
                        module + "；子进程仍在运行，等待下一条输出",
                        elapsed_seconds=round(now - started),
                        child_pid=process.pid,
                    )
                    last_notice = now
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            display_new_output()
            raise
    if code:
        raise subprocess.CalledProcessError(code, argv)


def gpu_phase(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """执行 GPU 训练/评估阶段，校验已有模型后再决定续训或复用。

    Args:
        root: 当前项目根目录。
        config: 记录字段，使用 segmentation_checkpoint, train_manifest, val_manifest,
        score_threshold, mask_threshold, bbox_match_iou, warmup_runs。

    Returns:
        结果字典，主要字段为 status, training, human_review_packet。

    Raises:
        ValueError: Core report belongs to another checkpoint or protocol; select a new
        report directory
        FileExistsError: Unverified Core report exists; preserve it before selecting a
        new report directory
    """
    report = root / "reports/reruns/recovery_v2"
    record_progress(root, "PREFLIGHT", "检查原图、掩码、依赖与 CUDA")
    readiness = preflight(root, config)
    write_json(report / "gpu_preflight.json", readiness)
    if readiness["status"] != "READY":
        return {"status": "BLOCKED_GPU_OR_ASSETS", "preflight": readiness}
    checkpoint = resolve_path(config["segmentation_checkpoint"])
    train_args = [
        "--epochs",
        "15",
        "--device",
        "cuda",
        "--train-csv",
        config["train_manifest"],
        "--val-csv",
        config["val_manifest"],
        "--output-dir",
        str(checkpoint.parent),
        "--report-dir",
        str(report / "train_v4_clean"),
    ]
    if checkpoint.exists():
        import torch

        cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
        resume = type(cp.get("epoch")) is int and cp["epoch"] < 15
        validate_checkpoint(
            cp,
            digest(root / config["train_manifest"]),
            digest(root / config["val_manifest"]),
            for_resume=resume,
        )
        completed = cp["epoch"]
        del cp
        if resume:
            record_progress(
                root, "RESUME_TRAINING", f"从已完成的第 {completed} 轮继续，目标 15 轮"
            )
            command(
                root,
                "fashion_multimodal_analysis.segmentation.training.train_prd_8class_maskrcnn_v4_clean",
                [*train_args, "--resume", str(checkpoint)],
            )
        else:
            record_progress(
                root, "REUSE_TRAINING", "15 轮 checkpoint 与清单、参数校验相符"
            )
    else:
        record_progress(root, "TRAINING", "先检查训练数据，再开始 15 轮干净重训")
        command(
            root,
            "fashion_multimodal_analysis.segmentation.training.train_prd_8class_maskrcnn_v4_clean",
            [*train_args, "--dry-run"],
        )
        command(
            root,
            "fashion_multimodal_analysis.segmentation.training.train_prd_8class_maskrcnn_v4_clean",
            train_args,
        )
    provenance = {
        "checkpoint": config["segmentation_checkpoint"],
        "checkpoint_sha256": digest(checkpoint),
        "train_manifest": config["train_manifest"],
        "train_manifest_sha256": digest(root / config["train_manifest"]),
        "val_manifest_sha256": digest(root / config["val_manifest"]),
        "epoch": 15,
        "status": "ACTUAL_GPU_TRAINING_COMPLETED",
    }
    write_json(report / "training_provenance.json", provenance)
    core_report = report / "core_v4_clean"
    meta_file = core_report / "run_provenance.json"
    parameters = {
        "score_threshold": config["score_threshold"],
        "mask_threshold": config["mask_threshold"],
        "match_bbox_iou": config["bbox_match_iou"],
        "warmup_runs": config["warmup_runs"],
        "timed_passes": config["timed_passes"],
    }
    evaluation_provenance = {
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "core_manifest_sha256": digest(
            root / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv"
        ),
        "parameters": parameters,
    }
    if meta_file.exists():
        if json.loads(meta_file.read_text()) != evaluation_provenance:
            raise ValueError(
                "Core report belongs to another checkpoint or protocol; select a new report directory"
            )
    else:
        if (core_report / "per_gt_results.csv").exists():
            raise FileExistsError(
                "Unverified Core report exists; preserve it before selecting a new report directory"
            )
        command(
            root,
            "fashion_multimodal_analysis.segmentation.evaluation.eval_prd_8class_maskrcnn_v3_b1_dataexp",
            [
                "--checkpoint",
                config["segmentation_checkpoint"],
                "--val-csv",
                "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv",
                "--report-dir",
                str(core_report),
                "--device",
                "cuda",
                "--score-threshold",
                str(config["score_threshold"]),
                "--mask-threshold",
                str(config["mask_threshold"]),
                "--match-bbox-iou",
                str(config["bbox_match_iou"]),
                "--warmup-runs",
                str(config["warmup_runs"]),
                "--timed-passes",
                str(config["timed_passes"]),
            ],
        )
        write_json(meta_file, evaluation_provenance)
    record_progress(root, "BUILD_REVIEW", "重建包含原图的人工审核页面")
    packet = build_review(root)
    return {
        "status": "GPU_RETRAIN_AND_CORE_EVALUATION_COMPLETED",
        "training": provenance,
        "human_review_packet": packet,
    }


def finish_phase(
    root: Path, config: dict[str, Any], report_dir: Path
) -> dict[str, Any]:
    """处理人工审核后的评估阶段。

    Args:
        root: 当前项目根目录。
        config: 记录字段，使用 segmentation_checkpoint。
        report_dir: 保存本次报告的目录。

    Returns:
        结果字典，主要字段为 status, gates, pending_actions, report。

    Raises:
        ValueError: Acceptance report belongs to a different checkpoint; retain it and
        choose a new --rep
    """
    from fashion_multimodal_analysis.analysis.summarize_prd31_acceptance import (
        summarize,
    )
    from fashion_multimodal_analysis.benchmarking.freeze_prd31_reviewed_tests import (
        freeze,
        verify_frozen,
    )
    from fashion_multimodal_analysis.integration.run_prd31_acceptance import evaluate

    base = root / "benchmark/prd_3_1_v2"
    paths = [
        base / "review/grounding_reviewed.csv",
        base / "review/attribute_reviewed.csv",
    ]
    if not (base / "frozen").is_dir():
        if not all(p.is_file() for p in paths):
            return {
                "status": "HUMAN_GT_REVIEW_REQUIRED",
                "review_page": "benchmark/prd_3_1_v2/review/prd31_review.html",
                "required_review_files": [str(p.relative_to(root)) for p in paths],
                "next_command": "python scripts/maintenance/run_prd31_recovery.py --phase finish",
            }
        freeze(root)
    verify_frozen(root)
    if (report_dir / "run_metadata.json").is_file():
        recorded = json.loads((report_dir / "run_metadata.json").read_text())
        checkpoint = resolve_path(config["segmentation_checkpoint"])
        if not checkpoint.is_file() or recorded.get("checkpoint_sha256") != digest(
            checkpoint
        ):
            raise ValueError(
                "Acceptance report belongs to a different checkpoint; retain it and choose a new --report-dir"
            )
        record_progress(root, "SUMMARIZE", "复用本模型已有推理，更新人工审核统计")
        result = summarize(root, report_dir)
    else:
        readiness = preflight(root, config)
        if readiness["status"] != "READY":
            return {"status": "BLOCKED_GPU_OR_ASSETS", "preflight": readiness}
        record_progress(
            root, "ACCEPTANCE", "加载模型并运行冻结验收，模型下载及加载可能需要等待"
        )
        result = evaluate(root, report_dir)
    return {
        "status": result["prd_module_status"],
        "gates": result["gates"],
        "pending_actions": result["pending_actions"],
        "report": str((report_dir / "ACCEPTANCE_REPORT.md").relative_to(root)),
    }


def run(
    phase: str = "all", root: Path | None = None, report_dir: Path | None = None
) -> Any:
    """组织本模块的准备、执行和结果保存步骤。

    Args:
        phase: phase。
        root: 当前项目根目录。
        report_dir: 保存本次报告的目录。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    root = root or project_root()
    report_dir = report_dir or root / "reports/reruns/recovery_v2/acceptance"
    state = {"phase": phase}
    record_progress(root, "START", f"运行阶段 {phase}")
    if phase == "check":
        config = json.loads((root / "configs/prd_3_1_recovery_v2.json").read_text())
        readiness = preflight(root, config)
        write_json(root / "reports/reruns/recovery_v2/gpu_preflight.json", readiness)
        return save_state(
            root,
            {
                **state,
                "status": (
                    "READY_FOR_GPU_AND_ASSETS"
                    if readiness["status"] == "READY"
                    else "BLOCKED_GPU_OR_ASSETS"
                ),
                "preflight": readiness,
            },
        )
    if phase == "report":
        from fashion_multimodal_analysis.analysis.summarize_prd31_acceptance import (
            summarize,
        )

        result = summarize(root, report_dir)
        state = {
            "status": result["prd_module_status"],
            "gates": result["gates"],
            "pending_actions": result["pending_actions"],
        }
    else:
        record_progress(root, "PREPARE", "检查干净训练清单和人工审核候选")
        state["preparation"] = prepare(root)
        config = json.loads((root / "configs/prd_3_1_recovery_v2.json").read_text())
        if phase == "prepare":
            state.update(
                status="PREPARED_REVIEW_AND_GPU_RUNS_REQUIRED",
                review_packet=build_review(root),
            )
        if phase in ("all", "gpu"):
            gpu = gpu_phase(root, config)
            state["gpu"] = gpu
            state["status"] = gpu["status"]
            if gpu["status"] == "BLOCKED_GPU_OR_ASSETS":
                return save_state(root, state)
        if phase in ("all", "finish"):
            state["acceptance"] = finish_phase(root, config, report_dir)
            state["status"] = state["acceptance"]["status"]
    return save_state(root, state)


def save_state(root: Path, state: dict[str, Any]) -> Any:
    """保存 状态。

    Args:
        root: 当前项目根目录。
        state: 记录字段，使用 status。

    Returns:
        结果字典包含 updated_at。
    """
    state = {**state, "updated_at": datetime.now(timezone.utc).isoformat()}
    write_json(root / "reports/reruns/recovery_v2/workflow_status.json", state)
    record_progress(
        root, state["status"], "本次运行已结束；详细结果见 workflow_status.json"
    )
    return state


def main() -> None:
    """解析运行参数并执行本模块的实验入口。

    Raises:
        SystemExit: 执行本函数的操作失败。
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=["all", "prepare", "check", "gpu", "finish", "report"],
        default="all",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Read saved stages and recent log without starting any work",
    )
    p.add_argument("--report-dir", default="reports/reruns/recovery_v2/acceptance")
    args = p.parse_args()
    if args.status:
        print(json.dumps(show_status(project_root()), ensure_ascii=False, indent=2))
        return
    try:
        with workflow_lock(project_root()):
            result = run(args.phase, report_dir=resolve_path(args.report_dir))
    except FileExistsError as exc:
        if "already running in this checkout" in str(exc):
            print(
                json.dumps(
                    {"status": "BLOCKED_ALREADY_RUNNING", "error": str(exc)},
                    ensure_ascii=False,
                )
            )
            raise SystemExit(2)
        result = save_state(
            project_root(), {"status": "BLOCKED_INPUT_OR_RUN", "error": str(exc)}
        )
    except KeyboardInterrupt:
        save_state(
            project_root(),
            {
                "status": "INTERRUPTED",
                "next_action": "再次运行原命令；新版未完成 checkpoint 会按已完成轮次恢复",
            },
        )
        raise SystemExit(130)
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        subprocess.CalledProcessError,
    ) as exc:
        result = save_state(
            project_root(),
            {
                "status": "BLOCKED_INPUT_OR_RUN",
                "error": f"{type(exc).__name__}: {exc}",
                "next_action": "Read docs/recovery_guide.md and retain existing frozen GT/checkpoints/reports.",
            },
        )
    print(json.dumps(compact_state(result), ensure_ascii=False, indent=2))
    if result["status"].startswith("BLOCKED"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
