"""3.1.1 实例分割：数据与模型版本分别记录，指标按固定匹配和计时口径解释。

Add complete DF2 training images for all 13 native labels; compare V5/V6 on development.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

V5_TRAIN_SHA = "36ff78831b20e492d787c66161f05747433f8e85f7a089fd5df981c2bdf476e6"
DEV_SHA = "39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93"
CORE_SHA = "c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89"
V5_CP_SHA = "ea5a4ddf0f002256f1ce5164fc1b4b5b4d2263698f9525c9fbe7d8cf98bc1a65"
FAST_RUNTIME_SHA = "ff85ca020695498ab287d73e8afd915b44345ecd2168b38eb24410fb8ddddfb8"
REPORT_BASE = "reports/reruns/recovery_v2/v6_coverage"
TRAIN_MANIFEST = "configs/prd_8class_train_v6_coverage.csv"
DEV_MANIFEST = "configs/prd_8class_val_v2_full_targets.csv"


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


def read_csv(path: str | Path) -> Any:
    """读取带表头的 CSV，保留每列原始字符串。

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def same_or_write(path: str | Path, content: Any) -> None:
    """新建文件或确认已有内容相同，拒绝覆盖不同的历史产物。

    Args:
        path: 要读取或写入的文件路径。
        content: 待发布的完整文件内容。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Existing V6 artifact differs; preserved: " + str(path))
        return
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(content)
    temp.replace(path)


def write_json(path: str | Path, value: Any) -> None:
    """保存结构化实验记录。

    Args:
        path: 要读取或写入的文件路径。
        value: 待解析或规范化的输入值。
    """
    same_or_write(
        path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    )


def load_helper(root: Path) -> Any:
    """取得已迁移到包内的 V5 完整目标处理实现。

    Args:
        root: 当前项目根目录。

    Returns:
        当前读取操作得到的记录、图像或模型对象；保持原始格式约定。
    """
    from fashion_multimodal_analysis.segmentation.training import (
        run_prd31_v5_full_targets,
    )

    return run_prd31_v5_full_targets


def coverage(rows: list[dict[str, Any]], fine_names: Any) -> dict[str, Any]:
    """按细类别统计现有训练覆盖，供补充样本选择。

    Args:
        rows: 待处理的逐行记录。
        fine_names: fine names。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    images = {name: set() for name in fine_names}
    for row in rows:
        if row["source_dataset"] == "DeepFashion2" and row["fine_category"] in images:
            images[row["fine_category"]].add(row["source_image"])
    return {name: len(members) for name, members in images.items()}


def protected_images(root: Path, data: Any, helper: Any) -> tuple[Any, ...]:
    """收集本轮训练必须排除的固定评估/审核原图。

    Args:
        root: 当前项目根目录。
        data: 外部数据目录或本函数处理的数据结构；具体用途由操作对象决定。
        helper: helper。

    Returns:
        按顺序返回 strict, extra_paths, evidence 等结果。
    """
    mandatory = [
        root / DEV_MANIFEST,
        root / "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv",
    ]
    extra = sorted(
        {
            *root.glob("benchmark/**/manifests/*.csv"),
            *root.glob("benchmark/**/review/*.csv"),
            *root.glob("configs/*val*.csv"),
            *root.glob("configs/*test*.csv"),
        }
    )
    all_files = sorted(set(mandatory + extra))
    strict, extra_paths, evidence = set(), set(), []
    for path in all_files:
        targets = set()
        for row in read_csv(path):
            for key in ("source_image", "image_path", "query_image", "image"):
                value = row.get(key, "").strip()
                if value and (
                    value.startswith("../fashion_data/")
                    or value.startswith("/workspace/fashion_data/")
                ):
                    targets.add(str(helper.diag.resolve(value, root, data).resolve()))
        extra_paths.update(targets)
        if path in mandatory:
            strict.update(targets)
        evidence.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": digest(path),
                "image_paths": len(targets),
            }
        )
    return strict, extra_paths, evidence


def expand_df2_image(
    annotation_file: str | Path,
    image_file: str | Path,
    root: Path,
    data: Path,
    helper: Any,
    start: Any,
) -> tuple[Any, ...]:
    """补充一张 DeepFashion2 图片的完整实例，并保持原图坐标。

    Args:
        annotation_file: 对应文件的相对路径或当前解析后的路径。
        image_file: 对应文件的相对路径或当前解析后的路径。
        root: 当前项目根目录。
        data: 外部服饰数据根目录。
        helper: helper。
        start: start。

    Returns:
        按顺序返回 result, masks 等结果。
    """
    from PIL import Image

    source = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
    items = [
        (key, ann)
        for key, ann in source.items()
        if re.fullmatch(r"item\d+", key)
        and isinstance(ann, dict)
        and int(ann.get("category_id", -1)) in helper.diag.DF2_CLASS
        and helper.diag.usable_annotation(ann, "DeepFashion2")
    ]
    if not items:
        return [], []
    with Image.open(image_file) as image:
        width, height = image.size
    result, masks = [], []
    source_image = "../fashion_data/" + image_file.relative_to(data).as_posix()
    for key, annotation in sorted(items):
        cat_id = int(annotation["category_id"])
        fine = helper.diag.DF2_FINE[cat_id - 1]
        mask = helper.decode_mask(annotation, width, height)
        x1, y1, x2, y2 = mask.getbbox()
        relative = Path("processed/prd8_full_targets_v6/train") / (
            "df2_" + image_file.stem + "_" + key + ".png"
        )
        buf = io.BytesIO()
        mask.save(buf, format="PNG")
        masks.append((data / relative, buf.getvalue()))
        result.append(
            dict(
                record_id=f"v6_new_{start + len(result) + 1:05d}",
                source_dataset="DeepFashion2",
                source_image=source_image,
                garment_id=key + "_" + fine,
                garment_category=helper.diag.DF2_CLASS[cat_id],
                fine_category=fine,
                source_category_id=cat_id,
                annotation_ref=key,
                mask_path="../fashion_data/" + relative.as_posix(),
                bbox_x1=x1,
                bbox_y1=y1,
                bbox_x2=x2,
                bbox_y2=y2,
                bbox_width=x2 - x1,
                bbox_height=y2 - y1,
                image_width=width,
                image_height=height,
                source_manifest="raw_complete_targets_v6_coverage",
            )
        )
    return result, masks


def prepare(
    root: Path, data: Path, helper: Any, target: Any, seed: int, max_scan: Any
) -> Any:
    """准备当前实验所需数据，并执行来源/划分/有效性检查。

    Args:
        root: 当前项目根目录。
        data: 外部服饰数据根目录。
        helper: helper。
        target: 目标。
        seed: 控制随机过程的种子。
        max_scan: max 扫描。

    Returns:
        本版本的数据准备凭据，包含清单/索引身份、有效样本数和排除记录。

    Raises:
        ValueError: Eight training classes must remain covered
        FileNotFoundError: 需要的文件不存在。
    """
    for relative, expected in [
        ("configs/prd_8class_train_v5_full_targets.csv", V5_TRAIN_SHA),
        (DEV_MANIFEST, DEV_SHA),
        ("benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv", CORE_SHA),
    ]:
        if digest(root / relative) != expected:
            raise ValueError("Input manifest SHA differs: " + relative)
    base = read_csv(root / "configs/prd_8class_train_v5_full_targets.csv")
    strict, protected, protected_files = protected_images(root, data, helper)
    existing_paths = {
        str(helper.diag.resolve(r["source_image"], root, data).resolve()) for r in base
    }
    if existing_paths & strict:
        raise ValueError("V5 base training overlaps Core/development")
    before = coverage(base, helper.diag.DF2_FINE)
    counts = dict(before)
    annotations = data / "raw/train/train/annos"
    images = annotations.parent / "image"
    candidates = sorted(annotations.glob("*.json"))
    if not candidates:
        raise FileNotFoundError("Native DF2 annotations not found: " + str(annotations))
    random.Random(seed).shuffle(candidates)
    added, selected, skipped = [], [], Counter()
    scanned = 0
    for path in candidates:
        if all(n >= target for n in counts.values()):
            break
        if max_scan and scanned >= max_scan:
            break
        scanned += 1
        if scanned % 1000 == 0:
            print("SCAN", scanned, "coverage", counts, flush=True)
        image_file = images / (path.stem + ".jpg")
        resolved = str(image_file.resolve())
        if resolved in protected or resolved in existing_paths:
            skipped["protected_or_existing_image"] += 1
            continue
        try:
            source = json.loads(path.read_text(encoding="utf-8-sig"))
            fine = {
                helper.diag.DF2_FINE[int(a["category_id"]) - 1]
                for key, a in source.items()
                if re.fullmatch(r"item\d+", key)
                and isinstance(a, dict)
                and int(a.get("category_id", -1)) in helper.diag.DF2_CLASS
                and helper.diag.usable_annotation(a, "DeepFashion2")
            }
            if not any(counts[name] < target for name in fine):
                continue
            new_rows, masks = expand_df2_image(
                path, image_file, root, data, helper, len(added)
            )
        except (OSError, ValueError, KeyError, TypeError):
            skipped["invalid_or_missing_raw_image_annotation_or_mask"] += 1
            continue
        if not new_rows:
            skipped["no_usable_targets"] += 1
            continue
        for destination, content in masks:
            same_or_write(destination, content)
        added.extend(new_rows)
        existing_paths.add(resolved)
        for name in {r["fine_category"] for r in new_rows}:
            counts[name] += 1
        selected.append(
            {
                "source_image": new_rows[0]["source_image"],
                "annotation_sha256": digest(path),
                "instances": len(new_rows),
            }
        )
        if len(selected) % 50 == 0:
            print(
                "SELECTED",
                len(selected),
                "new complete images; coverage",
                counts,
                flush=True,
            )
    missing = {name: target - n for name, n in counts.items() if n < target}
    if missing:
        raise ValueError(
            f"Coverage incomplete after {scanned} annotations: {missing}. No V6 manifest published. Increase --max-scan or use --max-scan 0."
        )
    combined = base + added
    new_paths = {
        str(helper.diag.resolve(r["source_image"], root, data).resolve()) for r in added
    }
    if new_paths & protected or len({r["record_id"] for r in combined}) != len(
        combined
    ):
        raise ValueError("New image isolation or record uniqueness failed")
    counts_by_class = Counter(r["garment_category"] for r in combined)
    if set(counts_by_class) != set(helper.diag.CLASSES):
        raise ValueError("Eight training classes must remain covered")
    report = dict(
        status="V6_COMPLETE_TARGET_COVERAGE_PREPARED",
        target_images_per_df2_fine_category=target,
        selection_seed=seed,
        max_scan=max_scan,
        scanned_annotations=scanned,
        skipped=dict(skipped),
        original_instances=len(base),
        new_instances=len(added),
        total_instances=len(combined),
        original_images=len({r["source_image"] for r in base}),
        new_images=len(selected),
        total_images=len({r["source_image"] for r in combined}),
        fine_image_coverage_before=before,
        fine_image_coverage_after=counts,
        class_counts=dict(counts_by_class),
        selected_new_images=selected,
        protected_manifests=protected_files,
        new_protected_image_overlap=0,
        inherited_base_overlap_with_other_benchmark_images=len(
            {
                str(helper.diag.resolve(r["source_image"], root, data).resolve())
                for r in base
            }
            & (protected - strict)
        ),
        input_sha256={
            "v5_training": V5_TRAIN_SHA,
            "development": DEV_SHA,
            "core": CORE_SHA,
        },
        scope="V5 training images retained; new images exclude Core, development and listed benchmark/review images. Path isolation only. Existing broader benchmark overlap is disclosed; no blind acceptance claimed.",
    )
    manifest = root / TRAIN_MANIFEST
    same_or_write(manifest, helper.csv_bytes(combined))
    report["v6_training_manifest_sha256"] = digest(manifest)
    write_json(root / REPORT_BASE / "data_preparation.json", report)
    print(
        "PREPARED",
        report["total_images"],
        "images;",
        report["total_instances"],
        "instances",
        flush=True,
    )
    print("DF2 fine coverage:", counts, flush=True)
    return report


def command(root: Path, module: str, arguments: list[str]) -> None:
    """启动当前项目的子命令，并保留日志与失败状态。

    Args:
        root: 当前项目根目录。
        module: 要执行的 Python 模块名称。
        arguments: 转交给实验入口的命令行参数。
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "-u", "-m", module, *arguments], cwd=root, env=env, check=True
    )


def evaluate(root: Path, checkpoint: Any, label: Any) -> Any:
    """按固定评估设置计算结果，保留逐例记录和运行凭据。

    Args:
        root: 当前项目根目录。
        checkpoint: 待检查或使用的训练权重/状态。
        label: 标签。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    folder = root / REPORT_BASE / label
    required = [
        "evaluation_summary.txt",
        "protocol_metrics.json",
        "per_gt_results.csv",
        "per_class_metrics.csv",
        "per_source_metrics.csv",
        "per_image_timing.csv",
        "error_cases.csv",
    ]
    inputs = dict(
        checkpoint_sha256=digest(checkpoint),
        val_manifest_sha256=DEV_SHA,
        runtime_sha256=FAST_RUNTIME_SHA,
        score_threshold=0.4,
        mask_threshold=0.5,
        match_bbox_iou=0.5,
        warmup_runs=10,
        timed_passes=5,
        split="DEVELOPMENT_FULL_TARGETS_NOT_CORE_ACCEPTANCE",
    )
    provenance = folder / "comparison_provenance.json"
    if provenance.exists():
        saved = json.loads(provenance.read_text())
        if saved["inputs"] != inputs or any(
            not (folder / name).is_file()
            or digest(folder / name) != saved["report_sha256"].get(name)
            for name in required
        ):
            raise ValueError(
                "Existing development report provenance or content differs: " + label
            )
        print("REUSE_EVALUATION:", label, flush=True)
        return json.loads((folder / "protocol_metrics.json").read_text())["micro"]
    if folder.exists() and any(folder.iterdir()):
        preserved = folder.with_name(
            folder.name + "_interrupted_" + time.strftime("%Y%m%d_%H%M%S")
        )
        folder.rename(preserved)
    command(
        root,
        "fashion_multimodal_analysis.segmentation.evaluation.eval_prd_8class_maskrcnn_v3_b1_dataexp",
        [
            "--checkpoint",
            str(checkpoint),
            "--val-csv",
            DEV_MANIFEST,
            "--report-dir",
            str(folder),
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
    )
    write_json(
        provenance,
        {
            "inputs": inputs,
            "report_sha256": {name: digest(folder / name) for name in required},
        },
    )
    return json.loads((folder / "protocol_metrics.json").read_text())["micro"]


def run(root: Path, data: Any, args: Any) -> None:
    """组织本模块的准备、执行和结果保存步骤。

    Args:
        root: 当前项目根目录。
        data: 外部数据目录或本函数处理的数据结构；具体用途由操作对象决定。
        args: 命令行配置；使用 images_per_fine, max_scan, phase, selection_seed 等参数。

    Raises:
        ValueError: V5 reference checkpoint changed
        RuntimeError: CUDA is required
    """
    os.environ["FASHION_PROJECT_ROOT"], os.environ["FASHION_DATA_ROOT"] = str(
        root
    ), str(data)
    sys.path.insert(0, str(root / "src"))
    helper = load_helper(root)
    if (
        digest(root / "src/fashion_multimodal_analysis/evaluation/runtime.py")
        != FAST_RUNTIME_SHA
    ):
        raise ValueError(
            "Expected verified filtered-transfer runtime from FIX_PRD31_V5_SPEED.py"
        )
    if args.phase in ("all", "prepare"):
        prepare(
            root, data, helper, args.images_per_fine, args.selection_seed, args.max_scan
        )
    if args.phase == "prepare":
        return
    prepared = json.loads((root / REPORT_BASE / "data_preparation.json").read_text())
    if (
        digest(root / TRAIN_MANIFEST) != prepared["v6_training_manifest_sha256"]
        or digest(root / DEV_MANIFEST) != DEV_SHA
    ):
        raise ValueError("Prepared V6 manifests changed")
    import torch

    from fashion_multimodal_analysis.segmentation.training.checkpoint_state import (
        validate_checkpoint,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    v5 = (
        root
        / "outputs/prd_instance_segmentation/maskrcnn_8class_v5_full_targets/checkpoint_last.pth"
    )
    if digest(v5) != V5_CP_SHA:
        raise ValueError("V5 reference checkpoint changed")
    v6 = (
        root
        / "outputs/prd_instance_segmentation/maskrcnn_8class_v6_coverage/checkpoint_last.pth"
    )
    if args.phase == "all":
        arguments = [
            "--epochs",
            "15",
            "--device",
            "cuda",
            "--train-csv",
            TRAIN_MANIFEST,
            "--val-csv",
            DEV_MANIFEST,
            "--output-dir",
            str(v6.parent),
            "--report-dir",
            REPORT_BASE + "/training",
            "--experiment-name",
            "v6-native-fine-coverage",
            "--verify-isolation",
        ]
        completed = False
        if v6.exists():
            cp = torch.load(v6, map_location="cpu", weights_only=False)
            resume = cp.get("epoch", 0) < 15
            validate_checkpoint(
                cp, digest(root / TRAIN_MANIFEST), DEV_SHA, for_resume=resume
            )
            del cp
            if resume:
                arguments += ["--resume", str(v6)]
            else:
                completed = True
                print("REUSE_V6_TRAINING: matching epoch 15 checkpoint", flush=True)
        if not completed:
            command(
                root,
                "fashion_multimodal_analysis.segmentation.training.train_prd_8class_maskrcnn_v3_b1_dataexp",
                arguments,
            )
    cp = torch.load(v6, map_location="cpu", weights_only=False)
    validate_checkpoint(cp, digest(root / TRAIN_MANIFEST), DEV_SHA)
    del cp
    result = {
        "v5": evaluate(root, v5, "v5_development_fast"),
        "v6": evaluate(root, v6, "v6_development_fast"),
        "status": "V6_DEVELOPMENT_COMPARISON_COMPLETED",
        "formal_acceptance": "NOT_CLAIMED",
    }
    write_json(root / REPORT_BASE / "development_comparison.json", result)
    archive = root / "prd31_v6_results.zip"
    if archive.exists():
        archive = root / ("prd31_v6_results_" + time.strftime("%Y%m%d_%H%M%S") + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted((root / REPORT_BASE).rglob("*")):
            if path.is_file() and path.suffix in (".csv", ".txt", ".json"):
                z.write(path, path.relative_to(root))
        z.write(root / TRAIN_MANIFEST, TRAIN_MANIFEST)
        z.write(Path(__file__), "patch/RUN_PRD31_V6_COVERAGE.py")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("RESULTS_ZIP=" + str(archive), flush=True)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default=".")
    p.add_argument(
        "--data-root", default=os.environ.get("FASHION_DATA_ROOT", "../fashion_data")
    )
    p.add_argument("--phase", choices=("all", "prepare", "compare"), default="all")
    p.add_argument("--images-per-fine", type=int, default=40)
    p.add_argument("--selection-seed", type=int, default=20260930)
    p.add_argument(
        "--max-scan",
        type=int,
        default=50000,
        help="0 scans the full native training annotation directory",
    )
    args = p.parse_args()
    if args.images_per_fine < 1 or args.max_scan < 0:
        p.error("images-per-fine must be positive and max-scan nonnegative")
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
            run(root, data, args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        ImportError,
        subprocess.CalledProcessError,
    ) as error:
        p.exit(2, f"V6 未完成：{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
