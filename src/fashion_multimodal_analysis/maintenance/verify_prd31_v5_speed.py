"""运行与维护：记录文件身份、阶段状态及依赖，长任务中断后先检查状态再续跑。

Verify filtered mask transfers on CUDA, then evaluate the existing V5 checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path
from typing import Any

BASE_RUNTIME_SHA = "24616489db986f3b9f21a7ef337ac79bcf0ad52bd5b21f4d73667401be23ea23"
CHECKPOINT_SHA = "ea5a4ddf0f002256f1ce5164fc1b4b5b4d2263698f9525c9fbe7d8cf98bc1a65"
CORE_SHA = "c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89"
DEV_SHA = "39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93"
# Reuse the documented candidate instead of reinstalling an unannotated copy.
REFERENCE_RUNTIME_SHA = (
    "ff85ca020695498ab287d73e8afd915b44345ecd2168b38eb24410fb8ddddfb8"
)
FAST_RUNTIME = (
    Path(__file__).resolve().parents[1] / "evaluation/runtime.py"
).read_text(encoding="utf-8")


def digest(path: str | Path) -> str:
    """分块计算文件 SHA-256，核对文件身份而不加载整个权重到内存。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        64 位十六进制 SHA-256 摘要字符串。
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: str | Path) -> Any:
    """读取 CSV 逐行记录，供既有结果回算。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def atomic_write(path: str | Path, data: Any) -> None:
    """先写临时文件再替换目标，避免中断产生不完整文件。

    Args:
        path: 要读取或写入的文件路径。
        data: 外部数据目录或本函数处理的数据结构；具体用途由操作对象决定。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def original_decode(
    output: dict[str, Any], score_threshold: float, mask_threshold: float
) -> tuple[Any, ...]:
    """按加速前的操作顺序解码预测，用于逐项验证加速结果相同。

    Args:
        output: 记录字段，使用 scores, boxes, labels, masks。
        score_threshold: 保留候选的置信度阈值。
        mask_threshold: 将预测掩码转为前景的概率阈值。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    import numpy as np

    scores = output["scores"].detach().cpu().numpy()
    keep = np.isfinite(scores) & (scores >= score_threshold)
    return (
        output["boxes"].detach().cpu().numpy()[keep],
        output["labels"].detach().cpu().numpy()[keep],
        scores[keep],
        (output["masks"].detach().cpu().numpy()[keep, 0] >= mask_threshold).astype(
            np.uint8
        ),
    )


def verify_equal(output: Any, fast_decode: Any) -> Any:
    """逐项比较框、标签、置信度和掩码，拒绝改变预测结果的加速方案。

    Args:
        output: 模型预测或输出文件位置，具体由本函数的读写操作决定。
        fast_decode: fast 解码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    import numpy as np

    old = original_decode(output, 0.4, 0.5)
    new = fast_decode(output, 0.4, 0.5)
    for name, a, b in zip(("boxes", "labels", "scores", "masks"), old, new):
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError("Postprocessing changed " + name)
    return len(new[2])


def verify_cuda(
    model: Any, device: Any, samples: Any, fast_decode: Any
) -> dict[str, Any]:
    """在 CUDA 上检查后处理一致性，并覆盖空结果及阈值边界。

    Args:
        model: 已构建的模型对象，由调用方负责选择权重。
        device: 当前计算设备，与输入张量和模型设备保持一致。
        samples: samples。
        fast_decode: fast 解码。

    Returns:
        结果字典，主要字段为 boundary_cases_passed, images, scope。
    """
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F

    from fashion_multimodal_analysis.common.paths import resolve_path

    # Exercise empty results, NaN/inf scores and score/mask threshold boundaries.
    scores = np.array(
        [
            np.nan,
            np.inf,
            -np.inf,
            0.0,
            np.nextafter(np.float32(0.4), np.float32(0)),
            0.4,
            0.9,
        ],
        dtype=np.float32,
    )
    masks = np.tile(
        np.array([0.0, 0.49999997, 0.5, 1.0], dtype=np.float32), (7, 1)
    ).reshape(7, 1, 2, 2)
    synthetic = {
        "scores": torch.tensor(scores, device=device),
        "boxes": torch.arange(28, device=device, dtype=torch.float32).reshape(7, 4),
        "labels": torch.arange(7, device=device),
        "masks": torch.tensor(masks, device=device),
    }
    verify_equal(synthetic, fast_decode)
    verify_equal({k: v[:0] for k, v in synthetic.items()}, fast_decode)
    verify_equal(
        {**synthetic, "scores": torch.zeros_like(synthetic["scores"])}, fast_decode
    )
    evidence = []
    with torch.inference_mode():
        for index, source in enumerate(samples, 1):
            with Image.open(resolve_path(source)) as raw:
                tensor = F.to_tensor(raw.convert("RGB")).to(device)
                output = model([tensor])[0]
            retained = verify_equal(output, fast_decode)
            evidence.append(
                {
                    "source_image": source,
                    "model_outputs": len(output["scores"]),
                    "retained_outputs": retained,
                    "all_arrays_exactly_equal": True,
                }
            )
            print(
                f"VERIFY {index:02d}/{len(samples)} PASS: retained={retained}",
                flush=True,
            )
    return {
        "boundary_cases_passed": True,
        "images": evidence,
        "scope": "Exact boxes/labels/scores/binary masks for two decoders on the SAME model output. No model or threshold changes.",
    }


def check_accuracy(before: Any, after: Any) -> dict[str, Any]:
    """连接两次逐例表，核对加速前后指标与判定是否一致。

    Args:
        before: before。
        after: after。

    Returns:
        结果字典，主要字段为 accuracy_equal, gt_rows, differences。

    Raises:
        ValueError: Expected 400 Core GT rows in both reports
    """
    old, new = rows(before / "per_gt_results.csv"), rows(after / "per_gt_results.csv")
    if len(old) != 400 or len(new) != 400:
        raise ValueError("Expected 400 Core GT rows in both reports")
    exact = (
        "source_dataset",
        "source_image",
        "record_id",
        "garment_id",
        "gt_class",
        "fine_category",
        "pred_class",
        "localized_bbox50",
        "class_correct",
        "mask_iou_ge_0_50",
        "mask_iou_ge_0_85",
        "outcome",
    )
    differences = []
    for a, b in zip(old, new):
        changed = [key for key in exact if a[key] != b[key]]
        changed += [
            key
            for key, tol in [
                ("mask_iou", 0.0),
                ("bbox_iou", 1e-6),
                ("pred_score", 1e-6),
            ]
            if abs(float(a[key]) - float(b[key])) > tol
        ]
        if changed:
            differences.append({"record_id": a["record_id"], "changed_fields": changed})
    old_metrics = json.loads((before / "protocol_metrics.json").read_text())
    new_metrics = json.loads((after / "protocol_metrics.json").read_text())
    for key in (
        "tp",
        "fp",
        "fn",
        "gt_count",
        "prediction_count",
        "localization_matches",
        "mask_iou_ge_0_85_count",
    ):
        if old_metrics["micro"][key] != new_metrics["micro"][key]:
            differences.append({"metric": key})
    return {
        "accuracy_equal": not differences,
        "gt_rows": 400,
        "differences": differences[:20],
    }


def summary(path: str | Path) -> Any:
    """读取或整理当前实验的汇总字段。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return dict(
        line.split("=", 1)
        for line in Path(path).read_text().splitlines()
        if "=" in line and not line.startswith("-")
    )


def package(root: Path, run: Any, original_runtime: Any) -> None:
    """package。

    Args:
        root: 当前项目根目录。
        run: 执行。
        original_runtime: original 运行环境。
    """
    destination = root / "prd31_v5_speed_results.zip"
    if destination.exists():
        destination = root / ("prd31_v5_speed_results_" + run.name + ".zip")
    folders = [
        root / "reports/reruns/recovery_v2/core_v5_full_targets",
        root / "reports/reruns/recovery_v2/v5_full_targets",
        run,
    ]
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in folders:
            if folder.is_dir():
                for p in sorted(folder.rglob("*")):
                    if p.is_file():
                        z.write(p, p.relative_to(root))
        z.writestr("patch/runtime_before.py", original_runtime)
        z.writestr("patch/runtime_candidate.py", FAST_RUNTIME)
        z.write(Path(__file__), "patch/FIX_PRD31_V5_SPEED.py")
    print("RESULTS_ZIP=" + str(destination), flush=True)


def run(root: Path, data: Any) -> None:
    """组织本模块的准备、执行和结果保存步骤。

    Args:
        root: 当前项目根目录。
        data: 外部数据目录或本函数处理的数据结构；具体用途由操作对象决定。

    Raises:
        ValueError: Core accuracy changed across runs; restored original runtime
        RuntimeError: CUDA is required; no CPU timing fallback
    """
    os.environ["FASHION_PROJECT_ROOT"] = str(root)
    os.environ["FASHION_DATA_ROOT"] = str(data)
    sys.path.insert(0, str(root / "src"))
    runtime = root / "src/fashion_multimodal_analysis/evaluation/runtime.py"
    checkpoint = (
        root
        / "outputs/prd_instance_segmentation/maskrcnn_8class_v5_full_targets/checkpoint_last.pth"
    )
    core = root / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv"
    dev = root / "configs/prd_8class_val_v2_full_targets.csv"
    for path, expected in [
        (checkpoint, CHECKPOINT_SHA),
        (core, CORE_SHA),
        (dev, DEV_SHA),
    ]:
        if digest(path) != expected:
            raise ValueError("Frozen file SHA256 differs: " + str(path))
    candidate = FAST_RUNTIME.encode()
    if hashlib.sha256(candidate).hexdigest() != REFERENCE_RUNTIME_SHA:
        raise ValueError("Documented runtime candidate differs from its release hash")
    if digest(runtime) not in (BASE_RUNTIME_SHA, hashlib.sha256(candidate).hexdigest()):
        raise ValueError("runtime.py has other edits; no overwrite performed")
    backup = runtime.with_name("runtime.py.before_prd31_speed_fix.bak")
    if backup.exists() and digest(backup) != BASE_RUNTIME_SHA:
        raise ValueError("Existing runtime backup differs")
    original = backup.read_bytes() if backup.exists() else runtime.read_bytes()
    if hashlib.sha256(original).hexdigest() != BASE_RUNTIME_SHA:
        raise ValueError("Original runtime backup is missing")
    baseline = root / "reports/reruns/recovery_v2/core_v5_full_targets"
    previous = summary(baseline / "evaluation_summary.txt")
    for key, value in [
        ("validation_images", "400"),
        ("gt_instances", "400"),
        ("checkpoint_epoch", "15"),
        ("score_threshold", "0.4"),
        ("match_bbox_iou", "0.5"),
        ("mask_threshold", "0.5"),
        ("warmup_runs", "10"),
        ("timed_passes", "5"),
        ("device", "cuda"),
    ]:
        if previous.get(key) != value:
            raise ValueError("Original Core protocol differs: " + key)
    for name in ("per_gt_results.csv", "protocol_metrics.json"):
        if not (baseline / name).is_file():
            raise ValueError("Original Core report incomplete: " + name)
    import torch

    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU timing fallback")
    cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(cp.get("epoch", -1)) != 15:
        raise ValueError("Expected completed epoch 15")
    saved = cp.get("args", {})
    device = torch.device("cuda")
    model = build_evaluation_model(
        int(saved.get("min_size", 640)), int(saved.get("max_size", 1024))
    )
    model.load_state_dict(cp["model_state_dict"])
    model.to(device).eval()
    samples = sorted({r["source_image"] for r in rows(dev)})
    if len(samples) != 32:
        raise ValueError("Expected 32 development images")
    fast = types.ModuleType("candidate_prd31_runtime")
    exec(compile(FAST_RUNTIME, "<candidate_runtime>", "exec"), fast.__dict__)
    verified = verify_cuda(model, device, samples, fast.postprocess_filtered)
    del model, cp
    torch.cuda.empty_cache()
    run_dir = (
        root
        / "reports/reruns/recovery_v2/v5_speed_fix"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    new_report = run_dir / "core_v5_filtered_transfer"
    status = {
        "checkpoint_sha256": CHECKPOINT_SHA,
        "core_manifest_sha256": CORE_SHA,
        "development_manifest_sha256": DEV_SHA,
        "verification": verified,
        "score_threshold": 0.4,
        "mask_threshold": 0.5,
        "match_bbox_iou": 0.5,
        "warmup_runs": 10,
        "timed_passes": 5,
        "baseline_report": str(baseline),
        "candidate_runtime_sha256": hashlib.sha256(candidate).hexdigest(),
    }
    applied = False
    try:
        if not backup.exists():
            atomic_write(backup, original)
        atomic_write(runtime, candidate)
        applied = True
        print("PATCH_APPLIED: checking the existing V5 on Core400", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(
                    root
                    / "scripts/segmentation/evaluation/eval_prd_8class_maskrcnn_v3_b1_dataexp.py"
                ),
                "--checkpoint",
                str(checkpoint),
                "--val-csv",
                str(core),
                "--report-dir",
                str(new_report),
                "--device",
                "cuda",
                "--score-threshold",
                "0.4",
                "--mask-threshold",
                "0.5",
                "--match-bbox-iou",
                "0.5",
                "--warmup-runs",
                "10",
                "--timed-passes",
                "5",
            ],
            cwd=root,
            check=True,
        )
        status["core_accuracy_check"] = check_accuracy(baseline, new_report)
        if not status["core_accuracy_check"]["accuracy_equal"]:
            raise ValueError(
                "Core accuracy changed across runs; restored original runtime"
            )
        current = summary(new_report / "evaluation_summary.txt")
        old_ms, new_ms = float(previous["mean_inference_ms_per_image"]), float(
            current["mean_inference_ms_per_image"]
        )
        status.update(
            original_report_mean_ms=old_ms,
            candidate_mean_ms=new_ms,
            candidate_time_le_50ms=new_ms <= 50,
            speed_improved_vs_recorded_run=new_ms < old_ms,
        )
        if new_ms >= old_ms:
            atomic_write(runtime, original)
            status["status"] = "ACCURACY_EQUAL_NO_SPEED_GAIN_ORIGINAL_RUNTIME_RESTORED"
        else:
            status["status"] = "ACCURACY_EQUAL_FASTER_RUNTIME_INSTALLED"
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        if applied:
            atomic_write(runtime, original)
        status.update(
            status="FAILED_ORIGINAL_RUNTIME_RESTORED",
            error=type(error).__name__ + ": " + str(error),
        )
        raise
    finally:
        atomic_write(
            run_dir / "speed_fix_status.json",
            (json.dumps(status, ensure_ascii=False, indent=2) + "\n").encode(),
        )
        package(root, run_dir, original)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".")
    parser.add_argument(
        "--data-root", default=os.environ.get("FASHION_DATA_ROOT", "../fashion_data")
    )
    args = parser.parse_args()
    root, data = (
        Path(args.project).expanduser().resolve(),
        Path(args.data_root).expanduser().resolve(),
    )
    lock = root / "reports/reruns/recovery_v2/workflow.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl

        with lock.open("a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(root, data)
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(2, f"速度修复未完成：{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
