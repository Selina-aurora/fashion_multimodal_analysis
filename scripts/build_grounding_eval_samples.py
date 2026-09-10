"""Build reproducible evaluation samples from DeepFashion2."""

import csv
import random
from pathlib import Path
from typing import List, Sequence


POOL_SIZE = 300
GROUP_SIZE = 100
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "fashion_data"

IMAGE_DIR = (
    DATASET_ROOT
    / "raw"
    / "train"
    / "train"
    / "image"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "evaluation_sampling"
)


def collect_image_files(image_dir: Path) -> List[Path]:
    """Collect supported image files from the dataset.

    Args:
        image_dir: Directory containing DeepFashion2 training images.

    Returns:
        Sorted list of image paths.

    Raises:
        FileNotFoundError: If the image directory does not exist.
    """
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_dir}"
        )

    image_files = [
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    return sorted(image_files)


def build_pool(
    image_files: Sequence[Path],
    pool_size: int,
    random_seed: int,
) -> List[Path]:
    """Randomly build a fixed evaluation pool.

    Args:
        image_files: Available dataset images.
        pool_size: Number of images in the evaluation pool.
        random_seed: Seed used for reproducible sampling.

    Returns:
        Randomly sampled image paths.

    Raises:
        ValueError: If pool_size exceeds available images.
    """
    if pool_size > len(image_files):
        raise ValueError(
            "Pool size exceeds the number of available images."
        )

    generator = random.Random(random_seed)

    return generator.sample(
        list(image_files),
        pool_size,
    )


def split_pool(
    pool: Sequence[Path],
    group_size: int,
) -> List[List[Path]]:
    """Split the evaluation pool into non-overlapping groups.

    Args:
        pool: Evaluation image pool.
        group_size: Number of images per evaluation group.

    Returns:
        List of non-overlapping image groups.

    Raises:
        ValueError: If pool size is not divisible by group size.
    """
    if len(pool) % group_size != 0:
        raise ValueError(
            "Pool size must be divisible by group size."
        )

    groups = []

    for start_index in range(
        0,
        len(pool),
        group_size,
    ):
        groups.append(
            list(
                pool[
                    start_index:
                    start_index + group_size
                ]
            )
        )

    return groups


def save_manifest(
    image_paths: Sequence[Path],
    output_file: Path,
    dataset_root: Path,
) -> None:
    """Save image paths to a CSV manifest.

    Args:
        image_paths: Images included in the manifest.
        output_file: Destination CSV file.
        dataset_root: Dataset root used for relative paths.
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "image_name",
                "relative_path",
            ],
        )

        writer.writeheader()

        for sample_id, image_path in enumerate(
            image_paths,
            start=1,
        ):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "image_name": image_path.name,
                    "relative_path": image_path.relative_to(
                        dataset_root
                    ).as_posix(),
                }
            )


def main() -> None:
    """Build the evaluation pool and three evaluation groups."""
    image_files = collect_image_files(
        IMAGE_DIR
    )

    print(
        f"Available images: {len(image_files)}"
    )

    pool = build_pool(
        image_files=image_files,
        pool_size=POOL_SIZE,
        random_seed=RANDOM_SEED,
    )

    groups = split_pool(
        pool=pool,
        group_size=GROUP_SIZE,
    )

    pool_file = (
        OUTPUT_DIR
        / "eval_pool_300.csv"
    )

    save_manifest(
        image_paths=pool,
        output_file=pool_file,
        dataset_root=DATASET_ROOT,
    )

    for group_index, group in enumerate(
        groups,
        start=1,
    ):
        group_file = (
            OUTPUT_DIR
            / f"eval_group_{group_index}_100.csv"
        )

        save_manifest(
            image_paths=group,
            output_file=group_file,
            dataset_root=DATASET_ROOT,
        )

        print(
            f"Group {group_index}: "
            f"{len(group)} images -> "
            f"{group_file}"
        )

    print()
    print(
        f"Evaluation pool: {len(pool)} images"
    )
    print(
        f"Random seed: {RANDOM_SEED}"
    )
    print(
        f"Saved pool: {pool_file}"
    )


if __name__ == "__main__":
    main()