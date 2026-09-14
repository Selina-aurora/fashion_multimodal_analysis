from pathlib import Path
import csv
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DATASET_ROOT = (
    PROJECT_ROOT.parent
    / "fashion_data"
    / "raw"
    / "train"
    / "train"
)

SOURCE_IMAGE_DIR = SOURCE_DATASET_ROOT / "image"
SOURCE_ANNO_DIR = SOURCE_DATASET_ROOT / "annos"

BENCHMARK_CSV = (
    PROJECT_ROOT
    / "reports"
    / "verified_positive_benchmark"
    / "verified_positive_benchmark.csv"
)

EVAL_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "evaluate_verified_positive_benchmark_v2.py"
)

BUNDLE_ROOT = (
    PROJECT_ROOT
    / "gpu_eval"
)

BUNDLE_PROJECT_ROOT = (
    BUNDLE_ROOT
    / "fashion_multimodal_analysis"
)

BUNDLE_IMAGE_DIR = (
    BUNDLE_ROOT
    / "fashion_data"
    / "raw"
    / "train"
    / "train"
    / "image"
)

BUNDLE_ANNO_DIR = (
    BUNDLE_ROOT
    / "fashion_data"
    / "raw"
    / "train"
    / "train"
    / "annos"
)

BUNDLE_BENCHMARK_DIR = (
    BUNDLE_PROJECT_ROOT
    / "reports"
    / "verified_positive_benchmark"
)

BUNDLE_SCRIPT_DIR = (
    BUNDLE_PROJECT_ROOT
    / "scripts"
)


def main():
    if BUNDLE_ROOT.exists():
        shutil.rmtree(BUNDLE_ROOT)

    BUNDLE_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BUNDLE_ANNO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BUNDLE_BENCHMARK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BUNDLE_SCRIPT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with BENCHMARK_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    rows = [
        row
        for row in rows
        if row["verified_positive"].strip().lower() == "yes"
        and row["usable_for_evaluation"].strip().lower() == "yes"
    ]

    print(f"Benchmark cases: {len(rows)}")

    copied_images = set()
    copied_annos = set()

    for index, row in enumerate(
        rows,
        start=1,
    ):
        image_name = row["image_name"].strip()

        image_source = (
            SOURCE_IMAGE_DIR
            / image_name
        )

        anno_name = (
            f"{Path(image_name).stem}.json"
        )

        anno_source = (
            SOURCE_ANNO_DIR
            / anno_name
        )

        if not image_source.is_file():
            raise FileNotFoundError(
                f"Missing image: {image_source}"
            )

        if not anno_source.is_file():
            raise FileNotFoundError(
                f"Missing annotation: {anno_source}"
            )

        image_target = (
            BUNDLE_IMAGE_DIR
            / image_name
        )

        anno_target = (
            BUNDLE_ANNO_DIR
            / anno_name
        )

        if image_name not in copied_images:
            shutil.copy2(
                image_source,
                image_target,
            )
            copied_images.add(
                image_name
            )

        if anno_name not in copied_annos:
            shutil.copy2(
                anno_source,
                anno_target,
            )
            copied_annos.add(
                anno_name
            )

        print(
            f"[{index:02d}/{len(rows)}] "
            f"{row['benchmark_id']} "
            f"{image_name}"
        )

    shutil.copy2(
        BENCHMARK_CSV,
        BUNDLE_BENCHMARK_DIR
        / "verified_positive_benchmark.csv",
    )

    shutil.copy2(
        EVAL_SCRIPT,
        BUNDLE_SCRIPT_DIR
        / "evaluate_verified_positive_benchmark_v2.py",
    )

    print()
    print("GPU bundle completed.")
    print(
        f"Unique images: "
        f"{len(copied_images)}"
    )
    print(
        f"Unique annotations: "
        f"{len(copied_annos)}"
    )
    print(
        f"Bundle: {BUNDLE_ROOT}"
    )


if __name__ == "__main__":
    main()