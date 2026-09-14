from pathlib import Path
import csv
import shutil


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT.parent
    / "fashion_data"
    / "raw"
    / "train"
    / "train"
)

IMAGE_DIR = DATASET_ROOT / "image"
ANNO_DIR = DATASET_ROOT / "annos"

BENCHMARK_CSV = (
    PROJECT_ROOT
    / "reports"
    / "verified_positive_benchmark"
    / "verified_positive_benchmark.csv"
)

EVAL_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "evaluate_verified_positive_benchmark_v3.py"
)

BUNDLE_ROOT = (
    PROJECT_ROOT
    / "gpu_eval_v3"
)

BUNDLE_PROJECT_ROOT = (
    BUNDLE_ROOT
    / "fashion_multimodal_analysis"
)

BUNDLE_DATASET_ROOT = (
    BUNDLE_ROOT
    / "fashion_data"
    / "raw"
    / "train"
    / "train"
)

BUNDLE_IMAGE_DIR = (
    BUNDLE_DATASET_ROOT
    / "image"
)

BUNDLE_ANNO_DIR = (
    BUNDLE_DATASET_ROOT
    / "annos"
)

BUNDLE_SCRIPT_DIR = (
    BUNDLE_PROJECT_ROOT
    / "scripts"
)

BUNDLE_BENCHMARK_DIR = (
    BUNDLE_PROJECT_ROOT
    / "reports"
    / "verified_positive_benchmark"
)

ZIP_PATH = (
    PROJECT_ROOT
    / "gpu_eval_v3.zip"
)


# ============================================================
# Helpers
# ============================================================


def ensure_file(path: Path, description: str):
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def load_benchmark_rows():
    ensure_file(
        BENCHMARK_CSV,
        "Benchmark CSV",
    )

    with BENCHMARK_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    required_fields = {
        "benchmark_id",
        "target",
        "image_name",
        "item_id",
        "category_name",
        "visibility",
        "verified_positive",
        "usable_for_evaluation",
    }

    if not rows:
        raise ValueError(
            "Benchmark CSV is empty."
        )

    missing = (
        required_fields
        - set(rows[0].keys())
    )

    if missing:
        raise ValueError(
            "Benchmark CSV missing fields: "
            f"{sorted(missing)}"
        )

    usable_rows = []

    for row in rows:
        if (
            row["verified_positive"]
            .strip()
            .lower()
            != "yes"
        ):
            continue

        if (
            row["usable_for_evaluation"]
            .strip()
            .lower()
            != "yes"
        ):
            continue

        usable_rows.append(
            row
        )

    return usable_rows


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 60)
    print("Building GPU evaluation bundle v3")
    print("=" * 60)

    ensure_file(
        EVAL_SCRIPT,
        "v3 evaluation script",
    )

    rows = load_benchmark_rows()

    print()
    print(
        f"Usable benchmark cases: "
        f"{len(rows)}"
    )

    if len(rows) != 88:
        print(
            "WARNING: Expected 88 cases, "
            f"but found {len(rows)}."
        )
        print(
            "The bundle will still be built "
            "using all usable rows."
        )

    # --------------------------------------------------------
    # Clean old bundle
    # --------------------------------------------------------

    if BUNDLE_ROOT.exists():
        print()
        print(
            f"Removing old bundle:\n"
            f"{BUNDLE_ROOT}"
        )
        shutil.rmtree(
            BUNDLE_ROOT
        )

    if ZIP_PATH.exists():
        print(
            f"Removing old ZIP:\n"
            f"{ZIP_PATH}"
        )
        ZIP_PATH.unlink()

    # --------------------------------------------------------
    # Create folders
    # --------------------------------------------------------

    BUNDLE_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BUNDLE_ANNO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BUNDLE_SCRIPT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BUNDLE_BENCHMARK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Copy images and annotations
    # --------------------------------------------------------

    copied_images = set()
    copied_annos = set()

    print()
    print(
        "Copying benchmark data..."
    )

    for index, row in enumerate(
        rows,
        start=1,
    ):

        benchmark_id = (
            row["benchmark_id"]
            .strip()
        )

        target = (
            row["target"]
            .strip()
        )

        image_name = (
            row["image_name"]
            .strip()
        )

        image_source = (
            IMAGE_DIR
            / image_name
        )

        anno_name = (
            f"{Path(image_name).stem}.json"
        )

        anno_source = (
            ANNO_DIR
            / anno_name
        )

        ensure_file(
            image_source,
            f"Image for {benchmark_id}",
        )

        ensure_file(
            anno_source,
            f"Annotation for {benchmark_id}",
        )

        # Same image may be reused for multiple targets.
        if image_name not in copied_images:
            shutil.copy2(
                image_source,
                BUNDLE_IMAGE_DIR
                / image_name,
            )

            copied_images.add(
                image_name
            )

        if anno_name not in copied_annos:
            shutil.copy2(
                anno_source,
                BUNDLE_ANNO_DIR
                / anno_name,
            )

            copied_annos.add(
                anno_name
            )

        print(
            f"[{index:02d}/{len(rows)}] "
            f"{benchmark_id:<15} "
            f"{target:<7} "
            f"{image_name}"
        )

    # --------------------------------------------------------
    # Copy evaluation script
    # --------------------------------------------------------

    shutil.copy2(
        EVAL_SCRIPT,
        BUNDLE_SCRIPT_DIR
        / EVAL_SCRIPT.name,
    )

    # --------------------------------------------------------
    # Copy benchmark CSV
    # --------------------------------------------------------

    shutil.copy2(
        BENCHMARK_CSV,
        BUNDLE_BENCHMARK_DIR
        / BENCHMARK_CSV.name,
    )

    # --------------------------------------------------------
    # requirements_gpu.txt
    # Do NOT reinstall torch because the cloud image already
    # provides CUDA-enabled PyTorch.
    # --------------------------------------------------------

    requirements_path = (
        BUNDLE_PROJECT_ROOT
        / "requirements_gpu.txt"
    )

    requirements_path.write_text(
        "\n".join(
            [
                "transformers",
                "pillow",
                "tqdm",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------

    readme_path = (
        BUNDLE_ROOT
        / "README_GPU.txt"
    )

    readme_path.write_text(
        """
GPU EVALUATION BUNDLE V3
========================

Directory structure:

gpu_eval_v3/
├── fashion_multimodal_analysis/
│   ├── scripts/
│   │   └── evaluate_verified_positive_benchmark_v3.py
│   ├── reports/
│   │   └── verified_positive_benchmark/
│   │       └── verified_positive_benchmark.csv
│   └── requirements_gpu.txt
│
└── fashion_data/
    └── raw/
        └── train/
            └── train/
                ├── image/
                └── annos/


After uploading and extracting on the GPU server:

cd gpu_eval_v3/fashion_multimodal_analysis

Check CUDA:

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

Install missing packages if needed:

pip install -r requirements_gpu.txt

Smoke test:

python scripts/evaluate_verified_positive_benchmark_v3.py --per-target-limit 1

Full 88-case evaluation:

python scripts/evaluate_verified_positive_benchmark_v3.py
""".strip()
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # ZIP
    # --------------------------------------------------------

    print()
    print(
        "Creating ZIP..."
    )

    shutil.make_archive(
        base_name=str(
            ZIP_PATH.with_suffix("")
        ),
        format="zip",
        root_dir=PROJECT_ROOT,
        base_dir=BUNDLE_ROOT.name,
    )

    zip_size_mb = (
        ZIP_PATH.stat().st_size
        / 1024
        / 1024
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "GPU bundle completed successfully."
    )
    print("=" * 60)

    print(
        f"Benchmark cases: "
        f"{len(rows)}"
    )

    print(
        f"Unique images: "
        f"{len(copied_images)}"
    )

    print(
        f"Unique annotations: "
        f"{len(copied_annos)}"
    )

    print(
        f"ZIP size: "
        f"{zip_size_mb:.2f} MB"
    )

    print()
    print(
        "Upload this file:"
    )

    print(
        ZIP_PATH
    )


if __name__ == "__main__":
    main()