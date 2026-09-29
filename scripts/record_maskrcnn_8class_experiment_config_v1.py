from __future__ import annotations

import csv
import platform
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = PROJECT_ROOT / "configs" / "prd_8class_train_v1.csv"
VAL_CSV = PROJECT_ROOT / "configs" / "prd_8class_val_v1.csv"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "environment"
    / "maskrcnn_8class_experiment_config_v1.txt"
)

CLASS_TO_ID = {
    "top": 1,
    "pants": 2,
    "skirt": 3,
    "outerwear": 4,
    "dress": 5,
    "shoe": 6,
    "bag": 7,
    "accessory": 8,
}

CONFIG = {
    "task": "PRD 3.1.1 8-class instance segmentation",
    "model": "Mask R-CNN ResNet50-FPN",
    "pretrained_weights": "MaskRCNN_ResNet50_FPN_Weights.DEFAULT",
    "num_foreground_classes": 8,
    "num_classes_with_background": 9,
    "epochs": 15,
    "batch_size": 1,
    "optimizer": "SGD",
    "learning_rate": 0.0025,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_scheduler": "StepLR",
    "lr_step_size": 5,
    "lr_gamma": 0.1,
    "input_min_size": 640,
    "input_max_size": 1024,
    "random_horizontal_flip": True,
    "random_horizontal_flip_probability": 0.5,
    "seed": 20260918,
    "num_workers": 0,
    "device": "cuda",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize_split(rows: list[dict[str, str]]) -> dict:
    image_groups = {
        (row["source_dataset"].strip(), row["source_image"].strip())
        for row in rows
    }
    class_counts = Counter(
        row["garment_category"].strip().lower()
        for row in rows
    )
    source_counts = Counter(
        row["source_dataset"].strip()
        for row in rows
    )
    return {
        "instances": len(rows),
        "image_groups": len(image_groups),
        "class_counts": class_counts,
        "source_counts": source_counts,
    }


def main() -> None:
    train_rows = read_rows(TRAIN_CSV)
    val_rows = read_rows(VAL_CSV)

    train = summarize_split(train_rows)
    val = summarize_split(val_rows)

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "NO CUDA"
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if cuda_available else 0.0
    )

    lines = [
        "PRD 3.1.1 Mask R-CNN 8-Class Baseline Experiment Configuration",
        "================================================================",
        "",
        f"recorded_at={datetime.now().isoformat(timespec='seconds')}",
        "",
        "[Task / Model]",
        f"task={CONFIG['task']}",
        f"model={CONFIG['model']}",
        f"pretrained_weights={CONFIG['pretrained_weights']}",
        f"foreground_classes={CONFIG['num_foreground_classes']}",
        f"classes_with_background={CONFIG['num_classes_with_background']}",
        "",
        "[Class Mapping]",
    ]

    for name, idx in CLASS_TO_ID.items():
        lines.append(f"{idx}={name}")

    lines += [
        "",
        "[Dataset Split]",
        f"train_instances={train['instances']}",
        f"train_image_groups={train['image_groups']}",
        f"val_instances={val['instances']}",
        f"val_image_groups={val['image_groups']}",
        "",
        "[Train Class Counts]",
    ]

    for cls in CLASS_TO_ID:
        lines.append(f"{cls}={train['class_counts'].get(cls, 0)}")

    lines += ["", "[Validation Class Counts]"]
    for cls in CLASS_TO_ID:
        lines.append(f"{cls}={val['class_counts'].get(cls, 0)}")

    lines += [
        "",
        "[Dataset Sources]",
        f"train_DeepFashion2={train['source_counts'].get('DeepFashion2', 0)}",
        f"train_Fashionpedia={train['source_counts'].get('Fashionpedia', 0)}",
        f"val_DeepFashion2={val['source_counts'].get('DeepFashion2', 0)}",
        f"val_Fashionpedia={val['source_counts'].get('Fashionpedia', 0)}",
        "",
        "[Training Hyperparameters]",
        f"epochs={CONFIG['epochs']}",
        f"batch_size={CONFIG['batch_size']}",
        f"optimizer={CONFIG['optimizer']}",
        f"learning_rate={CONFIG['learning_rate']}",
        f"momentum={CONFIG['momentum']}",
        f"weight_decay={CONFIG['weight_decay']}",
        f"lr_scheduler={CONFIG['lr_scheduler']}",
        f"lr_step_size={CONFIG['lr_step_size']}",
        f"lr_gamma={CONFIG['lr_gamma']}",
        f"input_min_size={CONFIG['input_min_size']}",
        f"input_max_size={CONFIG['input_max_size']}",
        f"random_horizontal_flip={CONFIG['random_horizontal_flip']}",
        f"random_horizontal_flip_probability={CONFIG['random_horizontal_flip_probability']}",
        f"seed={CONFIG['seed']}",
        f"num_workers={CONFIG['num_workers']}",
        f"device_requested={CONFIG['device']}",
        "",
        "[Runtime Environment]",
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"pytorch={torch.__version__}",
        f"torchvision={torchvision.__version__}",
        f"cuda_available={cuda_available}",
        f"pytorch_cuda={torch.version.cuda}",
        f"gpu={gpu_name}",
        f"vram_gb={vram_gb:.2f}",
        f"cudnn={torch.backends.cudnn.version()}",
        "",
        "[Training Command]",
        "python scripts/train_prd_8class_maskrcnn_baseline_v1.py --epochs 15 --device cuda",
        "",
        "[Notes]",
        "- Train/validation split is grouped by source image to avoid leakage.",
        "- DeepFashion2 provides top/pants/skirt/outerwear/dress.",
        "- Fashionpedia provides shoe/bag/accessory.",
        "- This file records the baseline configuration; evaluation metrics are reported separately.",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Experiment configuration saved:")
    print(OUTPUT_PATH)
    print()
    print(f"Train instances: {train['instances']}")
    print(f"Val instances  : {val['instances']}")
    print(f"GPU            : {gpu_name}")
    print(f"VRAM           : {vram_gb:.2f} GB")


if __name__ == "__main__":
    main()
