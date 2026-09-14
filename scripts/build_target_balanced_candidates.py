"""Build target-balanced candidate pools for manual verification.

DeepFashion2 does not directly annotate local-part presence for targets such
as collar, button, or zipper. Therefore, this script builds candidate pools
only. Human review is required before constructing a verified-positive
localization benchmark.

Candidate sampling is target-specific:
- sleeve uses sleeve-bearing garment categories;
- collar prioritizes tops and outerwear;
- button prioritizes garments likely to contain visible buttons;
- zipper prioritizes outerwear and lower-body garments.

Contact sheets are automatically split into multiple pages.
"""

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


LOGGER = logging.getLogger(__name__)

RANDOM_SEED = 42
DEFAULT_MIN_BBOX_AREA_RATIO = 0.08

DEFAULT_TARGET_COUNTS = {
    "sleeve": 50,
    "collar": 80,
    "button": 120,
    "zipper": 200,
}

CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_ROWS = 5
CONTACT_SHEET_PAGE_SIZE = (
    CONTACT_SHEET_COLUMNS
    * CONTACT_SHEET_ROWS
)

CONTACT_SHEET_CELL_WIDTH = 320
CONTACT_SHEET_CELL_HEIGHT = 400
CONTACT_SHEET_PADDING = 12
LABEL_HEIGHT = 50

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT.parent
    / "fashion_data"
    / "raw"
    / "train"
    / "train"
)

IMAGE_DIR = (
    DATASET_ROOT
    / "image"
)

ANNOTATION_DIR = (
    DATASET_ROOT
    / "annos"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "target_balanced_candidates"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "target_balanced_candidates"
)


TARGET_CATEGORY_PRIORITY = {
    "sleeve": [
        {
            "short sleeve top",
            "long sleeve top",
            "short sleeve outwear",
            "long sleeve outwear",
            "short sleeve dress",
            "long sleeve dress",
        },
    ],
    "collar": [
        {
            "short sleeve top",
            "long sleeve top",
            "short sleeve outwear",
            "long sleeve outwear",
        },
        {
            "short sleeve dress",
            "long sleeve dress",
            "vest dress",
        },
        {
            "vest",
            "sling dress",
            "sling",
        },
    ],
    "button": [
        {
            "short sleeve outwear",
            "long sleeve outwear",
            "short sleeve top",
            "long sleeve top",
        },
        {
            "short sleeve dress",
            "long sleeve dress",
            "vest dress",
        },
        {
            "trousers",
            "shorts",
            "skirt",
            "vest",
        },
    ],
    "zipper": [
        {
            "short sleeve outwear",
            "long sleeve outwear",
            "trousers",
            "shorts",
            "skirt",
        },
        {
            "short sleeve dress",
            "long sleeve dress",
            "vest dress",
        },
        {
            "short sleeve top",
            "long sleeve top",
        },
    ],
}


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build target-specific candidate pools "
            "for manual localization verification."
        )
    )

    parser.add_argument(
        "--sleeve-count",
        type=int,
        default=DEFAULT_TARGET_COUNTS["sleeve"],
    )

    parser.add_argument(
        "--collar-count",
        type=int,
        default=DEFAULT_TARGET_COUNTS["collar"],
    )

    parser.add_argument(
        "--button-count",
        type=int,
        default=DEFAULT_TARGET_COUNTS["button"],
    )

    parser.add_argument(
        "--zipper-count",
        type=int,
        default=DEFAULT_TARGET_COUNTS["zipper"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    parser.add_argument(
        "--min-bbox-area-ratio",
        type=float,
        default=DEFAULT_MIN_BBOX_AREA_RATIO,
    )

    parser.add_argument(
        "--limit-annotations",
        type=int,
        default=None,
        help=(
            "Optional annotation scan limit. "
            "Useful for smoke tests."
        ),
    )

    args = parser.parse_args()

    target_counts = [
        args.sleeve_count,
        args.collar_count,
        args.button_count,
        args.zipper_count,
    ]

    if any(
        count <= 0
        for count in target_counts
    ):
        raise ValueError(
            "All target counts must be greater than zero."
        )

    if not (
        0.0
        <= args.min_bbox_area_ratio
        <= 1.0
    ):
        raise ValueError(
            "--min-bbox-area-ratio must be "
            "between 0 and 1."
        )

    return args


def load_annotation(
    annotation_path: Path,
) -> dict[str, Any] | None:
    """Load one DeepFashion2 annotation."""
    try:
        with annotation_path.open(
            "r",
            encoding="utf-8",
        ) as json_file:
            return json.load(
                json_file
            )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        LOGGER.warning(
            "Failed annotation %s: %s",
            annotation_path.name,
            exc,
        )

        return None


def extract_garments(
    annotation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract valid garment instances."""
    garments = []

    for (
        item_id,
        value,
    ) in annotation.items():
        if not item_id.startswith(
            "item"
        ):
            continue

        if not isinstance(
            value,
            dict,
        ):
            continue

        category_name = (
            value.get(
                "category_name"
            )
        )

        bbox = value.get(
            "bounding_box"
        )

        if (
            category_name is None
            or bbox is None
            or len(bbox) != 4
        ):
            continue

        try:
            bbox_values = [
                int(
                    round(
                        float(coord)
                    )
                )
                for coord in bbox
            ]

        except (
            TypeError,
            ValueError,
        ):
            continue

        garments.append(
            {
                "item_id": item_id,
                "category_name": str(
                    category_name
                ),
                "bbox": bbox_values,
            }
        )

    return garments


def clip_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
) -> list[int]:
    """Clip bbox to image boundaries."""
    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = bbox

    return [
        max(
            0,
            min(
                image_width - 1,
                x_min,
            ),
        ),
        max(
            0,
            min(
                image_height - 1,
                y_min,
            ),
        ),
        max(
            0,
            min(
                image_width - 1,
                x_max,
            ),
        ),
        max(
            0,
            min(
                image_height - 1,
                y_max,
            ),
        ),
    ]


def calculate_bbox_area_ratio(
    bbox: list[int],
    image_width: int,
    image_height: int,
) -> float:
    """Calculate bbox area relative to image area."""
    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = bbox

    width = max(
        0,
        x_max - x_min + 1,
    )

    height = max(
        0,
        y_max - y_min + 1,
    )

    image_area = (
        image_width
        * image_height
    )

    if image_area <= 0:
        return 0.0

    return (
        width
        * height
        / image_area
    )


def get_target_priority(
    target: str,
    category_name: str,
) -> int | None:
    """Return category priority tier for one target.

    Lower numbers mean higher priority.
    """
    priority_groups = (
        TARGET_CATEGORY_PRIORITY[
            target
        ]
    )

    for (
        priority,
        categories,
    ) in enumerate(
        priority_groups
    ):
        if (
            category_name
            in categories
        ):
            return priority

    return None


def collect_candidates(
    min_bbox_area_ratio: float,
    annotation_limit: int | None,
) -> dict[
    str,
    list[dict[str, Any]],
]:
    """Collect target-specific candidate garments."""
    candidates: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    annotation_paths = sorted(
        ANNOTATION_DIR.glob(
            "*.json"
        )
    )

    if (
        annotation_limit
        is not None
    ):
        annotation_paths = (
            annotation_paths[
                :annotation_limit
            ]
        )

    LOGGER.info(
        "Scanning %d annotations",
        len(annotation_paths),
    )

    for (
        index,
        annotation_path,
    ) in enumerate(
        annotation_paths,
        start=1,
    ):
        if (
            index % 10000
            == 0
        ):
            LOGGER.info(
                "Scanned %d annotations",
                index,
            )

        image_name = (
            f"{annotation_path.stem}.jpg"
        )

        image_path = (
            IMAGE_DIR
            / image_name
        )

        if not image_path.is_file():
            continue

        annotation = load_annotation(
            annotation_path
        )

        if annotation is None:
            continue

        try:
            with Image.open(
                image_path
            ) as image_file:
                image_width = (
                    image_file.width
                )

                image_height = (
                    image_file.height
                )

        except OSError:
            continue

        garments = (
            extract_garments(
                annotation
            )
        )

        for garment in garments:
            bbox = clip_bbox(
                garment["bbox"],
                image_width,
                image_height,
            )

            area_ratio = (
                calculate_bbox_area_ratio(
                    bbox,
                    image_width,
                    image_height,
                )
            )

            if (
                area_ratio
                < min_bbox_area_ratio
            ):
                continue

            category_name = (
                garment[
                    "category_name"
                ]
            )

            for target in (
                TARGET_CATEGORY_PRIORITY
            ):
                priority = (
                    get_target_priority(
                        target,
                        category_name,
                    )
                )

                if priority is None:
                    continue

                candidates[
                    target
                ].append(
                    {
                        "target": target,
                        "priority": (
                            priority
                        ),
                        "image_name": (
                            image_name
                        ),
                        "item_id": (
                            garment[
                                "item_id"
                            ]
                        ),
                        "category_name": (
                            category_name
                        ),
                        "bbox": bbox,
                        "bbox_area_ratio": (
                            area_ratio
                        ),
                        "image_width": (
                            image_width
                        ),
                        "image_height": (
                            image_height
                        ),
                    }
                )

    return candidates


def sample_by_priority(
    candidates: list[
        dict[str, Any]
    ],
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample candidates while respecting category priority."""
    grouped: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in candidates:
        grouped[
            record["priority"]
        ].append(
            record
        )

    selected = []

    seen_keys = set()

    for priority in sorted(
        grouped.keys()
    ):
        current_group = (
            grouped[
                priority
            ]
        )

        rng.shuffle(
            current_group
        )

        current_group.sort(
            key=lambda row: row[
                "bbox_area_ratio"
            ],
            reverse=True,
        )

        for record in current_group:
            unique_key = (
                record[
                    "image_name"
                ],
                record[
                    "item_id"
                ],
            )

            if unique_key in seen_keys:
                continue

            selected.append(
                record
            )

            seen_keys.add(
                unique_key
            )

            if (
                len(selected)
                >= target_count
            ):
                return selected

    return selected


def sample_candidates(
    candidate_pool: dict[
        str,
        list[dict[str, Any]],
    ],
    target_counts: dict[
        str,
        int,
    ],
    seed: int,
) -> dict[
    str,
    list[dict[str, Any]],
]:
    """Sample candidates independently for each target."""
    sampled = {}

    for target in (
        TARGET_CATEGORY_PRIORITY
    ):
        target_seed = (
            seed
            + sum(
                ord(char)
                for char in target
            )
        )

        rng = random.Random(
            target_seed
        )

        candidates = list(
            candidate_pool.get(
                target,
                [],
            )
        )

        LOGGER.info(
            "%s candidate pool: %d",
            target,
            len(candidates),
        )

        selected = (
            sample_by_priority(
                candidates,
                target_counts[
                    target
                ],
                rng,
            )
        )

        sampled[
            target
        ] = selected

        LOGGER.info(
            "%s selected: %d / %d",
            target,
            len(selected),
            target_counts[
                target
            ],
        )

        priority_counter: dict[
            int,
            int,
        ] = defaultdict(int)

        for record in selected:
            priority_counter[
                record["priority"]
            ] += 1

        LOGGER.info(
            "%s priority distribution: %s",
            target,
            dict(
                priority_counter
            ),
        )

    return sampled


def save_crop(
    record: dict[str, Any],
    candidate_id: int,
    target: str,
) -> Path:
    """Save one garment ROI crop."""
    image_path = (
        IMAGE_DIR
        / record["image_name"]
    )

    crop_dir = (
        OUTPUT_ROOT
        / target
        / "crops"
    )

    crop_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop_path = (
        crop_dir
        / (
            f"{candidate_id:03d}_"
            f"{Path(record['image_name']).stem}_"
            f"{record['item_id']}.jpg"
        )
    )

    with Image.open(
        image_path
    ) as image_file:
        image = (
            image_file
            .convert("RGB")
        )

        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = record[
            "bbox"
        ]

        crop = image.crop(
            (
                x_min,
                y_min,
                x_max + 1,
                y_max + 1,
            )
        )

        crop.save(
            crop_path,
            quality=95,
        )

    return crop_path


def resize_to_fit(
    image: Image.Image,
    max_width: int,
    max_height: int,
) -> Image.Image:
    """Resize an image while preserving aspect ratio."""
    resized = image.copy()

    resized.thumbnail(
        (
            max_width,
            max_height,
        ),
        Image.Resampling.LANCZOS,
    )

    return resized


def build_contact_sheet_page(
    target: str,
    page_records: list[
        dict[str, Any]
    ],
    page_number: int,
) -> Path:
    """Build one contact-sheet page."""
    row_count = (
        len(page_records)
        + CONTACT_SHEET_COLUMNS
        - 1
    ) // CONTACT_SHEET_COLUMNS

    sheet_width = (
        CONTACT_SHEET_COLUMNS
        * CONTACT_SHEET_CELL_WIDTH
    )

    sheet_height = (
        row_count
        * CONTACT_SHEET_CELL_HEIGHT
    )

    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        sheet
    )

    font = (
        ImageFont.load_default()
    )

    for index, record in enumerate(
        page_records
    ):
        crop_path = Path(
            record[
                "crop_path_absolute"
            ]
        )

        with Image.open(
            crop_path
        ) as image_file:
            crop = (
                image_file
                .convert("RGB")
            )

        max_width = (
            CONTACT_SHEET_CELL_WIDTH
            - (
                2
                * CONTACT_SHEET_PADDING
            )
        )

        max_height = (
            CONTACT_SHEET_CELL_HEIGHT
            - LABEL_HEIGHT
            - (
                2
                * CONTACT_SHEET_PADDING
            )
        )

        resized = (
            resize_to_fit(
                crop,
                max_width,
                max_height,
            )
        )

        column = (
            index
            % CONTACT_SHEET_COLUMNS
        )

        row = (
            index
            // CONTACT_SHEET_COLUMNS
        )

        cell_x = (
            column
            * CONTACT_SHEET_CELL_WIDTH
        )

        cell_y = (
            row
            * CONTACT_SHEET_CELL_HEIGHT
        )

        image_x = (
            cell_x
            + (
                CONTACT_SHEET_CELL_WIDTH
                - resized.width
            )
            // 2
        )

        image_y = (
            cell_y
            + CONTACT_SHEET_PADDING
        )

        sheet.paste(
            resized,
            (
                image_x,
                image_y,
            ),
        )

        label = (
            f"{record['candidate_id']:03d} | "
            f"{record['image_name']} | "
            f"{record['category_name']} | "
            f"P{record['priority'] + 1}"
        )

        draw.text(
            (
                cell_x
                + CONTACT_SHEET_PADDING,
                cell_y
                + CONTACT_SHEET_CELL_HEIGHT
                - LABEL_HEIGHT
                + 10,
            ),
            label,
            fill="black",
            font=font,
        )

        draw.rectangle(
            (
                cell_x,
                cell_y,
                cell_x
                + CONTACT_SHEET_CELL_WIDTH
                - 1,
                cell_y
                + CONTACT_SHEET_CELL_HEIGHT
                - 1,
            ),
            outline="gray",
            width=1,
        )

    output_dir = (
        OUTPUT_ROOT
        / target
        / "contact_sheets"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            f"{target}_"
            f"contact_sheet_"
            f"{page_number:02d}.jpg"
        )
    )

    sheet.save(
        output_path,
        quality=92,
    )

    return output_path


def build_contact_sheets(
    target: str,
    records: list[
        dict[str, Any]
    ],
) -> list[Path]:
    """Build paginated contact sheets."""
    output_paths = []

    for start_index in range(
        0,
        len(records),
        CONTACT_SHEET_PAGE_SIZE,
    ):
        end_index = (
            start_index
            + CONTACT_SHEET_PAGE_SIZE
        )

        page_records = records[
            start_index:end_index
        ]

        page_number = (
            start_index
            // CONTACT_SHEET_PAGE_SIZE
            + 1
        )

        output_path = (
            build_contact_sheet_page(
                target,
                page_records,
                page_number,
            )
        )

        output_paths.append(
            output_path
        )

    return output_paths


def save_review_csv(
    target: str,
    records: list[
        dict[str, Any]
    ],
) -> Path:
    """Save manual verification sheet."""
    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_ROOT
        / (
            f"{target}_"
            "candidate_review.csv"
        )
    )

    fieldnames = [
        "candidate_id",
        "target",
        "sampling_priority",
        "image_name",
        "item_id",
        "category_name",
        "bbox_area_ratio",
        "crop_path",
        "target_present",
        "visibility",
        "usable_for_evaluation",
        "notes",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "candidate_id": (
                        record[
                            "candidate_id"
                        ]
                    ),
                    "target": target,
                    "sampling_priority": (
                        record[
                            "priority"
                        ]
                        + 1
                    ),
                    "image_name": (
                        record[
                            "image_name"
                        ]
                    ),
                    "item_id": (
                        record[
                            "item_id"
                        ]
                    ),
                    "category_name": (
                        record[
                            "category_name"
                        ]
                    ),
                    "bbox_area_ratio": round(
                        record[
                            "bbox_area_ratio"
                        ],
                        6,
                    ),
                    "crop_path": (
                        record[
                            "crop_path_relative"
                        ]
                    ),
                    "target_present": "",
                    "visibility": "",
                    "usable_for_evaluation": "",
                    "notes": "",
                }
            )

    return output_path


def process_target(
    target: str,
    records: list[
        dict[str, Any]
    ],
) -> None:
    """Create crops, CSV, and contact sheets."""
    processed_records = []

    for (
        candidate_id,
        record,
    ) in enumerate(
        records,
        start=1,
    ):
        crop_path = save_crop(
            record=record,
            candidate_id=(
                candidate_id
            ),
            target=target,
        )

        enriched_record = {
            **record,
            "candidate_id": (
                candidate_id
            ),
            "crop_path_absolute": str(
                crop_path
            ),
            "crop_path_relative": str(
                crop_path.relative_to(
                    PROJECT_ROOT
                )
            ),
        }

        processed_records.append(
            enriched_record
        )

    review_path = save_review_csv(
        target,
        processed_records,
    )

    contact_paths = (
        build_contact_sheets(
            target,
            processed_records,
        )
    )

    LOGGER.info(
        "%s review CSV: %s",
        target,
        review_path,
    )

    LOGGER.info(
        "%s contact sheets: %d",
        target,
        len(contact_paths),
    )


def save_manifest(
    sampled: dict[
        str,
        list[dict[str, Any]],
    ],
    target_counts: dict[
        str,
        int,
    ],
    args: argparse.Namespace,
) -> None:
    """Save candidate-generation metadata."""
    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_ROOT
        / "candidate_sampling_manifest.json"
    )

    payload = {
        "seed": args.seed,
        "min_bbox_area_ratio": (
            args.min_bbox_area_ratio
        ),
        "requested_counts": (
            target_counts
        ),
        "generated_counts": {
            target: len(records)
            for (
                target,
                records,
            ) in sampled.items()
        },
        "contact_sheet_page_size": (
            CONTACT_SHEET_PAGE_SIZE
        ),
        "candidate_policy": (
            "Target-specific garment-category "
            "priority sampling."
        ),
        "important_note": (
            "Candidate samples are not verified "
            "target-positive examples. Manual review "
            "is mandatory."
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            payload,
            json_file,
            indent=4,
            ensure_ascii=False,
        )

    LOGGER.info(
        "Saved manifest: %s",
        output_path,
    )


def main() -> None:
    """Build target-balanced candidate pools."""
    configure_logging()

    args = parse_args()

    target_counts = {
        "sleeve": (
            args.sleeve_count
        ),
        "collar": (
            args.collar_count
        ),
        "button": (
            args.button_count
        ),
        "zipper": (
            args.zipper_count
        ),
    }

    LOGGER.info(
        "Seed: %d",
        args.seed,
    )

    LOGGER.info(
        "Target counts: %s",
        target_counts,
    )

    LOGGER.info(
        "Minimum bbox area ratio: %.3f",
        args.min_bbox_area_ratio,
    )

    candidate_pool = (
        collect_candidates(
            min_bbox_area_ratio=(
                args.min_bbox_area_ratio
            ),
            annotation_limit=(
                args.limit_annotations
            ),
        )
    )

    sampled = (
        sample_candidates(
            candidate_pool=(
                candidate_pool
            ),
            target_counts=(
                target_counts
            ),
            seed=args.seed,
        )
    )

    for (
        target,
        records,
    ) in sampled.items():
        process_target(
            target,
            records,
        )

    save_manifest(
        sampled=sampled,
        target_counts=(
            target_counts
        ),
        args=args,
    )

    print()
    print(
        "Candidate generation completed."
    )

    print(
        f"Outputs: {OUTPUT_ROOT}"
    )

    print(
        f"Reports: {REPORT_ROOT}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These are candidates only, "
        "not verified positive examples."
    )

    print(
        "Manual review fields:"
    )

    print(
        "target_present = yes / no / uncertain"
    )

    print(
        "visibility = clear / partial / tiny / occluded"
    )

    print(
        "usable_for_evaluation = yes / no"
    )


if __name__ == "__main__":
    main()