"""3.1.1 实例分割：数据与模型版本分别记录，指标按固定匹配和计时口径解释。

Validate controlled experiments and retain complete epoch-boundary training state.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from fashion_multimodal_analysis.common.schema import CLASS_TO_ID

CONTROLLED_SETTINGS = {
    "batch_size": 1,
    "workers": 0,
    "min_size": 640,
    "max_size": 1024,
    "seed": 20260921,
    "lr": 0.0025,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "step_size": 5,
    "gamma": 0.1,
    "sampler_alpha": 0.5,
    "no_hflip": False,
    "no_balanced_sampler": False,
}


def validate_checkpoint(
    checkpoint: dict[str, Any],
    train_sha256: Any,
    val_sha256: Any,
    settings: Any = None,
    epochs: int = 15,
    for_resume: bool = False,
) -> Any:
    """Reject another experiment, an unfinished final model, or incomplete resume state.

    Args:
        checkpoint: 记录字段，使用 rng_state。
        train_sha256: 训练 sha256。
        val_sha256: val sha256。
        settings: 配置。
        epochs: epochs。
        for_resume: for resume。

    Returns:
        当前条件的校验结果；失败条件及返回形式见函数体。

    Raises:
        ValueError: Checkpoint is missing complete, ordered epoch logs
    """
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain the recorded training state")
    for key, expected in {
        "class_to_id": CLASS_TO_ID,
        "train_manifest_sha256": train_sha256,
        "val_manifest_sha256": val_sha256,
    }.items():
        if checkpoint.get(key) != expected:
            raise ValueError("Checkpoint provenance mismatch: " + key)
    expected_settings = CONTROLLED_SETTINGS if settings is None else settings
    saved = checkpoint.get("args", {})
    for key, expected in expected_settings.items():
        if saved.get(key) != expected:
            raise ValueError("Checkpoint training setting differs: " + key)
    if saved.get("epochs") != epochs:
        raise ValueError("Checkpoint total training epochs differ")
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int or not 1 <= epoch <= epochs:
        raise ValueError("Checkpoint epoch is invalid")
    if not for_resume:
        if epoch != epochs:
            raise ValueError(
                f"Checkpoint is unfinished ({epoch}/{epochs} epochs); resume training first"
            )
        return epoch
    if epoch == epochs:
        raise ValueError(
            "Training is already complete; reuse this checkpoint for evaluation"
        )
    if checkpoint.get("checkpoint_format") != 2:
        raise ValueError(
            "This older partial checkpoint has no complete resume state; preserve it and use a new output path"
        )
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "sampler"}
    if not required <= checkpoint.get("rng_state", {}).keys():
        raise ValueError("Checkpoint is missing random-generator state")
    if (
        not expected_settings.get("no_balanced_sampler", False)
        and checkpoint["rng_state"]["sampler"] is None
    ):
        raise ValueError("Checkpoint is missing the weighted-sampler generator state")
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict"):
        if key not in checkpoint:
            raise ValueError("Checkpoint is missing " + key)
    logs = checkpoint.get("epoch_log", [])
    if [row.get("epoch") for row in logs] != list(range(1, epoch + 1)):
        raise ValueError("Checkpoint is missing complete, ordered epoch logs")
    return epoch


def capture_rng_state(
    torch_module: Any, sampler_generator: Any = None
) -> dict[str, Any]:
    """capture rng 状态。

    Args:
        torch_module: 调用方提供的 PyTorch 模块。
        sampler_generator: 采样器 generator。

    Returns:
        结果字典，主要字段为 python, numpy, torch_cpu, torch_cuda, sampler。
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch_module.get_rng_state(),
        "torch_cuda": (
            torch_module.cuda.get_rng_state_all()
            if torch_module.cuda.is_available()
            else []
        ),
        "sampler": (
            sampler_generator.get_state() if sampler_generator is not None else None
        ),
    }


def restore_rng_state(
    state: dict[str, Any], torch_module: Any, sampler_generator: Any = None
) -> None:
    """restore rng 状态。

    Args:
        state: 记录字段，使用 python, numpy, torch_cpu, torch_cuda, sampler。
        torch_module: 调用方提供的 PyTorch 模块。
        sampler_generator: 采样器 generator。

    Raises:
        ValueError: Resume sampler configuration differs
    """
    cuda_count = (
        torch_module.cuda.device_count() if torch_module.cuda.is_available() else 0
    )
    if len(state["torch_cuda"]) != cuda_count:
        raise ValueError("Resume requires the same visible CUDA device count")
    if (sampler_generator is None) != (state["sampler"] is None):
        raise ValueError("Resume sampler configuration differs")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_module.set_rng_state(state["torch_cpu"])
    if cuda_count:
        torch_module.cuda.set_rng_state_all(state["torch_cuda"])
    if sampler_generator is not None:
        sampler_generator.set_state(state["sampler"])


def atomic_save(payload: Any, path: str | Path, torch_module: Any) -> None:
    """A failed save must leave the previous completed-epoch checkpoint intact.

    Args:
        payload: 要保存的权重、优化器和进度状态。
        path: 要读取或写入的文件路径。
        torch_module: 调用方提供的 PyTorch 模块。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".checkpoint-", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch_module.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
