"""
Prepare a compact, portable GPU package for the PRD 3.1.1 Mask R-CNN baseline.

Why
---
Instead of uploading the entire fashion_data directory, this script copies only
the images and masks referenced by:
    configs/prd_8class_train_v1.csv
    configs/prd_8class_val_v1.csv

It builds a portable package with the SAME sibling-folder convention expected by
the current training script:

prd8_gpu_package_v1/
├── fashion_multimodal_analysis/
│   ├── scripts/
│   │   └── train_prd_8class_maskrcnn_baseline_v1.py
│   └── configs/
│       ├── prd_8class_train_v1.csv
│       └── prd_8class_val_v1.csv
└── fashion_data/
    └── gpu_prd8_v1/
        ├── images/
        └── masks/

The two copied manifests are rewritten so that source_image and mask_path point
to the compact package copies.

A ZIP is also created automatically:
    prd8_gpu_package_v1.zip

Run from project root:
    python scripts/prepare_prd8_gpu_package_v1.py
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = PROJECT_ROOT.parent

TRAIN_MANIFEST = PROJECT_ROOT / "configs" / "prd_8class_train_v1.csv"
VAL_MANIFEST = PROJECT_ROOT / "configs" / "prd_8class_val_v1.csv"

TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "train_prd_8class_maskrcnn_baseline_v1.py"
)

PACKAGE_ROOT = PROJECT_ROOT / "gpu_packages" / "prd8_gpu_package_v1"

PACKAGE_PROJECT = PACKAGE_ROOT / "fashion_multimodal_analysis"
PACKAGE_CONFIGS = PACKAGE_PROJECT / "configs"
PACKAGE_SCRIPTS = PACKAGE_PROJECT / "scripts"

PACKAGE_DATA = PACKAGE_ROOT / "fashion_data" / "gpu_prd8_v1"
PACKAGE_IMAGES = PACKAGE_DATA / "images"
PACKAGE_MASKS = PACKAGE_DATA / "masks"

ZIP_PATH = PROJECT_ROOT / "gpu_packages" / "prd8_gpu_package_v1.zip"


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV:\n  {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    return rows, fields


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_from_project(value: str) -> Path:
    p = Path(str(value).strip())

    if p.is_absolute():
        return p.resolve()

    return (PROJECT_ROOT / p).resolve()


def short_hash(text: str) -> str:
    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:10]


def unique_target_name(
    src: Path,
    logical_key: str,
) -> str:
    suffix = src.suffix.lower() or ".bin"
    return f"{short_hash(logical_key)}_{src.stem}{suffix}"


def copy_once(
    src: Path,
    dst_dir: Path,
    cache: dict[str, Path],
) -> Path:
    src = src.resolve()
    key = str(src)

    if key in cache:
        return cache[key]

    if not src.is_file():
        raise FileNotFoundError(f"Missing referenced file:\n  {src}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / unique_target_name(src, key)
    shutil.copy2(src, dst)

    cache[key] = dst
    return dst


def package_relative_to_project(path: Path) -> str:
    """
    Return path relative to package's fashion_multimodal_analysis folder.
    Expected result:
        ../fashion_data/gpu_prd8_v1/...
    """
    return path.resolve().relative_to(PACKAGE_PROJECT.parent).as_posix()


def rewrite_manifest(
    src_manifest: Path,
    dst_manifest: Path,
    image_cache: dict[str, Path],
    mask_cache: dict[str, Path],
):
    rows, fields = read_csv(src_manifest)
    rewritten = []

    for row in rows:
        src_image = resolve_from_project(row["source_image"])
        src_mask = resolve_from_project(row["mask_path"])

        dst_image = copy_once(
            src_image,
            PACKAGE_IMAGES,
            image_cache,
        )

        dst_mask = copy_once(
            src_mask,
            PACKAGE_MASKS,
            mask_cache,
        )

        new_row = dict(row)

        # Keep same sibling convention expected by training script.
        image_rel = (
            Path("..")
            / "fashion_data"
            / "gpu_prd8_v1"
            / "images"
            / dst_image.name
        )
        mask_rel = (
            Path("..")
            / "fashion_data"
            / "gpu_prd8_v1"
            / "masks"
            / dst_mask.name
        )

        new_row["source_image"] = image_rel.as_posix()
        new_row["mask_path"] = mask_rel.as_posix()

        # These are not needed for training. Blank them to avoid stale local paths.
        for optional_path_field in (
            "crop_path",
            "masked_preview_path",
            "visualization_path",
        ):
            if optional_path_field in new_row:
                new_row[optional_path_field] = ""

        rewritten.append(new_row)

    write_csv(
        dst_manifest,
        rewritten,
        fields,
    )

    return len(rows)


def make_zip():
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        for file_path in PACKAGE_ROOT.rglob("*"):
            if not file_path.is_file():
                continue

            arcname = file_path.relative_to(PACKAGE_ROOT.parent)
            zf.write(
                file_path,
                arcname.as_posix(),
            )


def main():
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)

    PACKAGE_CONFIGS.mkdir(parents=True, exist_ok=True)
    PACKAGE_SCRIPTS.mkdir(parents=True, exist_ok=True)
    PACKAGE_IMAGES.mkdir(parents=True, exist_ok=True)
    PACKAGE_MASKS.mkdir(parents=True, exist_ok=True)

    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Training script not found:\n  {TRAIN_SCRIPT}"
        )

    shutil.copy2(
        TRAIN_SCRIPT,
        PACKAGE_SCRIPTS / TRAIN_SCRIPT.name,
    )

    image_cache: dict[str, Path] = {}
    mask_cache: dict[str, Path] = {}

    train_rows = rewrite_manifest(
        TRAIN_MANIFEST,
        PACKAGE_CONFIGS / TRAIN_MANIFEST.name,
        image_cache,
        mask_cache,
    )

    val_rows = rewrite_manifest(
        VAL_MANIFEST,
        PACKAGE_CONFIGS / VAL_MANIFEST.name,
        image_cache,
        mask_cache,
    )

    readme = f"""PRD 3.1.1 Mask R-CNN compact GPU package

Contents
--------
train instances: {train_rows}
val instances: {val_rows}
unique source images: {len(image_cache)}
unique masks: {len(mask_cache)}

GPU usage
---------
1. Unzip this package.
2. cd prd8_gpu_package_v1/fashion_multimodal_analysis
3. Install dependencies:
       pip install torch torchvision pillow numpy
4. Check CUDA:
       python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
5. Dry-run:
       python scripts/train_prd_8class_maskrcnn_baseline_v1.py --dry-run
6. Train:
       python scripts/train_prd_8class_maskrcnn_baseline_v1.py --epochs 15 --device cuda
"""
    (PACKAGE_ROOT / "README_GPU.txt").write_text(
        readme,
        encoding="utf-8",
    )

    print("Package folder created:")
    print(f"  {PACKAGE_ROOT}")
    print()
    print(f"Train instances: {train_rows}")
    print(f"Val instances  : {val_rows}")
    print(f"Unique images  : {len(image_cache)}")
    print(f"Unique masks   : {len(mask_cache)}")
    print()
    print("Creating ZIP...")
    make_zip()
    print()
    print("ZIP created:")
    print(f"  {ZIP_PATH}")
    print(
        f"ZIP size: {ZIP_PATH.stat().st_size / (1024**2):.1f} MB"
    )


if __name__ == "__main__":
    main()
