"""CPU 检查：验证几何、数据身份和入口行为，不生成新的 GPU 精度或速度成绩。

Resume guards, failed-write recovery and visible workflow progress.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fashion_multimodal_analysis.common.schema import CLASS_TO_ID
from fashion_multimodal_analysis.maintenance.run_prd31_recovery import (
    command,
    record_progress,
    show_status,
    workflow_lock,
)
from fashion_multimodal_analysis.segmentation.training.checkpoint_state import (
    CONTROLLED_SETTINGS,
    atomic_save,
    capture_rng_state,
    restore_rng_state,
    validate_checkpoint,
)


def recorded_checkpoint(epoch: int = 15) -> dict[str, Any]:
    """recorded checkpoint。

    Args:
        epoch: 当前训练轮次。

    Returns:
        结果字典，主要字段为 epoch, class_to_id, train_manifest_sha256, val_manifest_sha256, args,
        model_state_dict, optimizer_state_dict, scheduler_state_dict, checkpoint_format,
        rng_state。
    """
    return {
        "epoch": epoch,
        "class_to_id": CLASS_TO_ID,
        "train_manifest_sha256": "training",
        "val_manifest_sha256": "validation",
        "args": {**CONTROLLED_SETTINGS, "epochs": 15},
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "checkpoint_format": 2,
        "rng_state": {
            "python": (),
            "numpy": (),
            "torch_cpu": b"",
            "torch_cuda": [],
            "sampler": b"fixture",
        },
        "epoch_log": [{"epoch": i, "train_loss_total": 1} for i in range(1, epoch + 1)],
    }


class CheckpointGuards(unittest.TestCase):
    """保存 CheckpointGuards 的数据/运行职责。"""

    def test_completed_prior_format_is_still_usable(self) -> None:
        """验证较早格式的完整 checkpoint 仍可使用。"""
        cp = recorded_checkpoint()
        for key in ("checkpoint_format", "rng_state", "epoch_log"):
            cp.pop(key)
        self.assertEqual(validate_checkpoint(cp, "training", "validation"), 15)

    def test_final_acceptance_rejects_partial_model(self) -> None:
        """验证未完成训练的模型不能当作最终验收模型。"""
        with self.assertRaisesRegex(ValueError, "unfinished"):
            validate_checkpoint(recorded_checkpoint(7), "training", "validation")

    def test_other_data_or_training_setup_cannot_be_resumed(self) -> None:
        """验证不同数据或训练设置不能混入同一断点续跑。"""
        for kind in ("training", "validation", "batch_size", "epochs"):
            cp = recorded_checkpoint(7)
            if kind in ("training", "validation"):
                cp[
                    {
                        "training": "train_manifest_sha256",
                        "validation": "val_manifest_sha256",
                    }[kind]
                ] = "other"
            else:
                cp["args"][kind] += 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_checkpoint(cp, "training", "validation", for_resume=True)

    def test_partial_resume_requires_rng_and_complete_epoch_history(self) -> None:
        """验证中途续跑要求随机状态和完整 epoch 历史。"""
        self.assertEqual(
            validate_checkpoint(
                recorded_checkpoint(7), "training", "validation", for_resume=True
            ),
            7,
        )
        for kind in (
            "old_format",
            "missing_sampler",
            "missing_optimizer",
            "missing_log",
        ):
            cp = recorded_checkpoint(7)
            if kind == "old_format":
                cp.pop("checkpoint_format")
            elif kind == "missing_sampler":
                cp["rng_state"]["sampler"] = None
            elif kind == "missing_optimizer":
                cp.pop("optimizer_state_dict")
            else:
                cp["epoch_log"].pop(2)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_checkpoint(cp, "training", "validation", for_resume=True)

    def test_failed_checkpoint_write_keeps_previous_completed_epoch(self) -> None:
        """验证 checkpoint 写入失败仍保留前一个完成版本。"""

        def failed_save(payload: Any, destination: Any) -> None:
            """failed 保存。

            Args:
                payload: 要保存的权重、优化器和进度状态。
                destination: 目标文件或目录。

            Raises:
                OSError: fixture disk failure
            """
            Path(destination).write_bytes(b"half-written")
            raise OSError("fixture disk failure")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_last.pth"
            path.write_bytes(b"previous-completed-epoch")
            with self.assertRaises(OSError):
                atomic_save({}, path, SimpleNamespace(save=failed_save))
            self.assertEqual(path.read_bytes(), b"previous-completed-epoch")
            self.assertEqual(list(Path(directory).iterdir()), [path])
            writer = SimpleNamespace(
                save=lambda payload, destination: Path(destination).write_text(
                    json.dumps(payload)
                )
            )
            atomic_save({"epoch": 8}, path, writer)
            self.assertEqual(json.loads(path.read_text())["epoch"], 8)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
        "PyTorch/torchvision are not installed; check the actual Core loader in AutoDL",
    )
    def test_core_entrypoint_loads_numpy_rng_checkpoint_before_model_initialization(
        self,
    ) -> None:
        """验证 Core 入口在模型初始化前正确加载含 NumPy 随机状态的断点。"""
        import importlib
        from unittest.mock import patch

        import torch

        module = importlib.import_module(
            "fashion_multimodal_analysis.segmentation.evaluation.eval_prd_8class_maskrcnn_v3_b1_dataexp"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "recorded_training.pth"
            torch.save(
                {
                    "epoch": 15,
                    "args": {"min_size": 640, "max_size": 1024},
                    "model_state_dict": {},
                    "rng_state": {"numpy": np.random.get_state()},
                },
                checkpoint,
            )
            args = SimpleNamespace(
                report_dir=str(root / "report"),
                device="cpu",
                checkpoint=str(checkpoint),
                val_csv=str(root / "unused_manifest.csv"),
                warmup_runs=0,
                timed_passes=1,
            )
            with (
                patch.object(module, "args_parse", return_value=args),
                patch.object(module, "ValDataset", return_value=[]),
                patch.object(
                    module,
                    "build_model",
                    side_effect=RuntimeError("fixture checkpoint loaded"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "fixture checkpoint loaded"):
                    module.main()

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is not installed; actual training-state restore must be checked in AutoDL",
    )
    def test_real_torch_epoch_resume_matches_uninterrupted_toy_training(self) -> None:
        """验证真实 PyTorch 小模型的续跑与不中断训练一致。"""
        import torch

        settings = {**CONTROLLED_SETTINGS, "lr": 0.1, "step_size": 2, "gamma": 0.7}

        def session(seed: int) -> tuple[Any, ...]:
            """session。

            Args:
                seed: 控制随机过程的种子。

            Returns:
                按顺序返回 model, optimizer, scheduler, generator 等结果。
            """
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=2, gamma=0.7
            )
            generator = torch.Generator().manual_seed(seed)
            return model, optimizer, scheduler, generator

        def epoch(state: Any) -> None:
            """epoch。

            Args:
                state: 状态。
            """
            model, optimizer, scheduler, generator = state
            indices = torch.multinomial(
                torch.ones(8), 8, replacement=True, generator=generator
            )
            for index in indices:
                x = (
                    torch.randn(1, 2)
                    + random.random()
                    + np.random.random()
                    + float(index) / 10
                )
                loss = model(x).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        first = session(23)
        epoch(first)
        epoch(first)
        model, optimizer, scheduler, generator = first
        payload = {
            **recorded_checkpoint(2),
            "args": {**settings, "epochs": 5},
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": capture_rng_state(torch, generator),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pth"
            atomic_save(payload, path, torch)
            for _ in range(3):
                epoch(first)
            expected = {k: v.clone() for k, v in first[0].state_dict().items()}
            expected_draws = (
                random.random(),
                np.random.random(),
                torch.rand(3),
                torch.rand(3, generator=first[3]),
            )
            resumed = session(99)
            cp = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(
                validate_checkpoint(
                    cp,
                    "training",
                    "validation",
                    settings=settings,
                    epochs=5,
                    for_resume=True,
                ),
                2,
            )
            resumed[0].load_state_dict(cp["model_state_dict"])
            resumed[2].load_state_dict(cp["scheduler_state_dict"])
            resumed[1].load_state_dict(cp["optimizer_state_dict"])
            restore_rng_state(cp["rng_state"], torch, resumed[3])
            for _ in range(3):
                epoch(resumed)
            for key, value in expected.items():
                self.assertTrue(torch.equal(value, resumed[0].state_dict()[key]))
            self.assertEqual(
                first[1].param_groups[0]["lr"], resumed[1].param_groups[0]["lr"]
            )
            self.assertEqual(expected_draws[:2], (random.random(), np.random.random()))
            self.assertTrue(torch.equal(expected_draws[2], torch.rand(3)))
            self.assertTrue(
                torch.equal(expected_draws[3], torch.rand(3, generator=resumed[3]))
            )


class WorkflowProgress(unittest.TestCase):
    """保存 WorkflowProgress 的数据/运行职责。"""

    def test_finish_cannot_reuse_report_for_another_checkpoint(self) -> None:
        """验证结束阶段不能复用属于另一个模型的报告。"""
        from unittest.mock import patch

        from test_recovery import WorkflowIntegrationTests

        from fashion_multimodal_analysis.benchmarking.freeze_prd31_reviewed_tests import (
            freeze,
        )
        from fashion_multimodal_analysis.maintenance.run_prd31_recovery import (
            finish_phase,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"FASHION_PROJECT_ROOT": str(root)}):
                WorkflowIntegrationTests().fixture(root)
                freeze(root)
                config = json.loads(
                    (root / "configs/prd_3_1_recovery_v2.json").read_text()
                )
                checkpoint = root / config["segmentation_checkpoint"]
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"new fixture model")
                output = root / "reports/fixture"
                output.mkdir(parents=True)
                (output / "run_metadata.json").write_text(
                    json.dumps({"checkpoint_sha256": "other fixture model"})
                )
                with self.assertRaisesRegex(ValueError, "different checkpoint"):
                    finish_phase(root, config, output)

    def test_status_reads_saved_progress_without_preparing_or_training(self) -> None:
        """验证状态查询只读已保存进度，不准备数据或启动训练。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                record_progress(root, "TRAINING", "fixture epoch 3")
            before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            result = show_status(root)
            self.assertEqual(result["workflow_progress"]["stage"], "TRAINING")
            self.assertEqual(result["workflow_status"]["status"], "NOT_STARTED")
            self.assertEqual(
                before, sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            )

    @unittest.skipUnless(os.name == "posix", "AutoDL uses POSIX advisory locks")
    def test_second_workflow_cannot_write_until_first_releases_lock(self) -> None:
        """验证同目录第二个工作流在锁释放前不能写入。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with workflow_lock(root):
                with self.assertRaisesRegex(FileExistsError, "already running"):
                    with workflow_lock(root):
                        self.fail("Second workflow entered")
            with workflow_lock(root):
                pass

    def test_child_error_and_output_are_retained_in_log(self) -> None:
        """验证子进程错误和输出均保留在日志中。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src/fixture_child.py").write_text(
                "print('fixture epoch output', flush=True)\nraise SystemExit(4)\n"
            )
            import subprocess

            with (
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(subprocess.CalledProcessError) as error,
            ):
                command(root, "fixture_child", [])
            self.assertEqual(error.exception.returncode, 4)
            self.assertIn(
                "fixture epoch output",
                (root / "reports/reruns/recovery_v2/workflow.log").read_text(),
            )
            self.assertEqual(
                show_status(root)["workflow_progress"]["detail"], "fixture_child"
            )


if __name__ == "__main__":
    unittest.main()
