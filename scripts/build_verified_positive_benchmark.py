"""Build the final verified-positive localization benchmark.

The benchmark contains 88 manually verified garment-part examples:

- sleeve: 22
- collar: 22
- button: 22
- zipper: 22

Important:
The exact crop filenames are used instead of candidate IDs alone because
older smoke-test runs may have produced duplicate candidate IDs.
"""

import csv
import json
import logging
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "target_balanced_candidates"
)

SOURCE_REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "target_balanced_candidates"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "verified_positive_benchmark"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "verified_positive_benchmark"
)

DATASET_IMAGE_ROOT = (
    PROJECT_ROOT.parent
    / "fashion_data"
    / "raw"
    / "train"
    / "train"
    / "image"
)


SELECTED_FILES = {
    "sleeve": [
        "003_017524_item1.jpg",
        "005_018629_item1.jpg",
        "008_028725_item1.jpg",
        "011_030741_item1.jpg",
        "012_182793_item1.jpg",
        "014_151051_item1.jpg",
        "016_067370_item1.jpg",
        "017_116985_item1.jpg",
        "018_061410_item1.jpg",
        "019_114181_item1.jpg",
        "020_119516_item1.jpg",
        "022_177713_item1.jpg",
        "026_102651_item1.jpg",
        "027_027786_item1.jpg",
        "031_027831_item1.jpg",
        "035_129567_item1.jpg",
        "037_153420_item1.jpg",
        "038_017518_item1.jpg",
        "040_133110_item1.jpg",
        "044_154526_item1.jpg",
        "045_123488_item1.jpg",
        "050_045056_item1.jpg",
    ],

    "collar": [
        "008_032686_item1.jpg",
        "010_145442_item1.jpg",
        "011_082889_item1.jpg",
        "017_087323_item1.jpg",
        "021_141532_item1.jpg",
        "024_109191_item1.jpg",
        "030_170066_item1.jpg",
        "031_034543_item1.jpg",
        "032_150951_item1.jpg",
        "033_043744_item1.jpg",
        "036_127679_item1.jpg",
        "038_118608_item1.jpg",
        "040_053073_item1.jpg",
        "041_172132_item1.jpg",
        "042_026314_item1.jpg",
        "045_005237_item1.jpg",
        "050_058348_item1.jpg",
        "054_152404_item1.jpg",
        "066_140980_item1.jpg",
        "074_189367_item1.jpg",
        "076_105891_item1.jpg",
        "080_147261_item1.jpg",
    ],

    "button": [
        "002_100023_item1.jpg",
        "010_091865_item1.jpg",
        "012_105965_item1.jpg",
        "016_178720_item1.jpg",
        "017_054074_item1.jpg",
        "025_181706_item1.jpg",
        "038_098376_item1.jpg",
        "039_174082_item1.jpg",
        "049_025070_item1.jpg",
        "056_042346_item1.jpg",
        "061_152404_item1.jpg",
        "064_119326_item1.jpg",
        "066_021683_item1.jpg",
        "072_130559_item1.jpg",
        "076_032653_item1.jpg",
        "083_105891_item1.jpg",
        "086_102501_item1.jpg",
        "101_151346_item1.jpg",
        "104_180957_item1.jpg",
        "116_009673_item1.jpg",
        "117_032686_item1.jpg",
        "118_156639_item1.jpg",
    ],

    "zipper": [
        "036_191459_item1.jpg",
        "041_151528_item1.jpg",
        "042_176369_item1.jpg",
        "043_183837_item1.jpg",
        "048_116992_item1.jpg",
        "051_051339_item1.jpg",
        "058_173202_item1.jpg",
        "074_115518_item1.jpg",
        "086_164368_item1.jpg",
        "087_000503_item1.jpg",
        "114_152093_item1.jpg",
        "115_082100_item1.jpg",
        "126_082097_item1.jpg",
        "131_085215_item1.jpg",
        "142_030741_item1.jpg",
        "165_135191_item1.jpg",
        "168_168672_item1.jpg",
        "171_140428_item1.jpg",
        "181_128187_item1.jpg",
        "183_107554_item1.jpg",
        "194_106749_item1.jpg",
        "199_133082_item1.jpg",
    ],
}


PARTIAL_VISIBILITY = {
    "zipper": {
        "058_173202_item1.jpg",
        "074_115518_item1.jpg",
    },
}


CONTACT_COLUMNS = 4
CELL_WIDTH = 320
CELL_HEIGHT = 400
LABEL_HEIGHT = 45
PADDING = 10


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def load_review_csv(
    target: str,
) -> dict[str, dict[str, str]]:
    """Load candidate metadata indexed by crop filename."""
    csv_path = (
        SOURCE_REPORT_ROOT
        / f"{target}_candidate_review.csv"
    )

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing review CSV: {csv_path}"
        )

    records = {}

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            crop_path = row.get(
                "crop_path",
                "",
            )

            crop_name = Path(
                crop_path
            ).name

            if crop_name:
                records[crop_name] = row

    return records


def build_record(
    target: str,
    filename: str,
    metadata: dict[str, str],
    benchmark_index: int,
) -> dict[str, str]:
    """Build one final benchmark record."""
    source_crop = (
        SOURCE_OUTPUT_ROOT
        / target
        / "crops"
        / filename
    )

    if not source_crop.is_file():
        raise FileNotFoundError(
            f"Selected crop not found: {source_crop}"
        )

    if filename not in metadata:
        raise ValueError(
            "Selected crop is not present in the "
            f"formal review CSV: {filename}"
        )

    row = metadata[filename]

    candidate_id = row[
        "candidate_id"
    ]

    image_name = row[
        "image_name"
    ]

    item_id = row[
        "item_id"
    ]

    category_name = row[
        "category_name"
    ]

    bbox_area_ratio = row.get(
        "bbox_area_ratio",
        "",
    )

    original_image_path = (
        DATASET_IMAGE_ROOT
        / image_name
    )

    if not original_image_path.is_file():
        LOGGER.warning(
            "Original image not found: %s",
            original_image_path,
        )

    destination_dir = (
        OUTPUT_ROOT
        / target
        / "crops"
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_crop = (
        destination_dir
        / filename
    )

    shutil.copy2(
        source_crop,
        destination_crop,
    )

    visibility = "clear"

    if filename in PARTIAL_VISIBILITY.get(
        target,
        set(),
    ):
        visibility = "partial"

    benchmark_id = (
        f"{target}_{benchmark_index:03d}"
    )

    return {
        "benchmark_id": benchmark_id,
        "target": target,
        "candidate_id": candidate_id,
        "image_name": image_name,
        "item_id": item_id,
        "category_name": category_name,
        "bbox_area_ratio": bbox_area_ratio,
        "target_present": "yes",
        "visibility": visibility,
        "usable_for_evaluation": "yes",
        "verified_positive": "yes",
        "source_crop_filename": filename,
        "benchmark_crop_path": str(
            destination_crop.relative_to(
                PROJECT_ROOT
            )
        ),
        "original_image_path": str(
            original_image_path
        ),
    }


def save_benchmark_csv(
    records: list[dict[str, str]],
) -> Path:
    """Save the final benchmark CSV."""
    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_ROOT
        / "verified_positive_benchmark.csv"
    )

    fieldnames = [
        "benchmark_id",
        "target",
        "candidate_id",
        "image_name",
        "item_id",
        "category_name",
        "bbox_area_ratio",
        "target_present",
        "visibility",
        "usable_for_evaluation",
        "verified_positive",
        "source_crop_filename",
        "benchmark_crop_path",
        "original_image_path",
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
        writer.writerows(records)

    return output_path


def resize_to_fit(
    image: Image.Image,
    width: int,
    height: int,
) -> Image.Image:
    resized = image.copy()

    resized.thumbnail(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    return resized


def build_contact_sheet(
    target: str,
    records: list[dict[str, str]],
) -> Path:
    """Build one final contact sheet for a target."""
    rows = (
        len(records)
        + CONTACT_COLUMNS
        - 1
    ) // CONTACT_COLUMNS

    sheet_width = (
        CONTACT_COLUMNS
        * CELL_WIDTH
    )

    sheet_height = (
        rows
        * CELL_HEIGHT
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

    font = ImageFont.load_default()

    for index, record in enumerate(
        records
    ):
        crop_path = (
            PROJECT_ROOT
            / record[
                "benchmark_crop_path"
            ]
        )

        with Image.open(
            crop_path
        ) as image_file:
            image = image_file.convert(
                "RGB"
            )

        max_width = (
            CELL_WIDTH
            - 2 * PADDING
        )

        max_height = (
            CELL_HEIGHT
            - LABEL_HEIGHT
            - 2 * PADDING
        )

        resized = resize_to_fit(
            image,
            max_width,
            max_height,
        )

        column = (
            index
            % CONTACT_COLUMNS
        )

        row = (
            index
            // CONTACT_COLUMNS
        )

        cell_x = (
            column
            * CELL_WIDTH
        )

        cell_y = (
            row
            * CELL_HEIGHT
        )

        image_x = (
            cell_x
            + (
                CELL_WIDTH
                - resized.width
            )
            // 2
        )

        image_y = (
            cell_y
            + PADDING
        )

        sheet.paste(
            resized,
            (
                image_x,
                image_y,
            ),
        )

        label = (
            f"{record['benchmark_id']} | "
            f"{record['image_name']} | "
            f"{record['visibility']}"
        )

        draw.text(
            (
                cell_x + PADDING,
                cell_y
                + CELL_HEIGHT
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
                + CELL_WIDTH
                - 1,
                cell_y
                + CELL_HEIGHT
                - 1,
            ),
            outline="gray",
            width=1,
        )

    contact_dir = (
        OUTPUT_ROOT
        / "contact_sheets"
    )

    contact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        contact_dir
        / f"{target}_verified_positive.jpg"
    )

    sheet.save(
        output_path,
        quality=92,
    )

    return output_path


def save_manifest(
    records: list[dict[str, str]],
) -> Path:
    """Save benchmark metadata."""
    counts = {}

    for target in SELECTED_FILES:
        counts[target] = sum(
            record["target"] == target
            for record in records
        )

    manifest = {
        "benchmark_name": (
            "DeepFashion2 Verified Positive "
            "Fine-Grained Localization Benchmark"
        ),
        "total_cases": len(records),
        "target_counts": counts,
        "targets": list(
            SELECTED_FILES.keys()
        ),
        "verification_policy": (
            "Target must be directly visible in the image. "
            "Examples requiring inference from garment type "
            "are excluded."
        ),
        "balance_policy": (
            "22 verified-positive cases per target."
        ),
        "targets_grouping": {
            "larger_part": [
                "sleeve",
                "collar",
            ],
            "small_object": [
                "button",
                "zipper",
            ],
        },
    }

    output_path = (
        REPORT_ROOT
        / "verified_positive_manifest.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            manifest,
            json_file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


def validate_balance(
    records: list[dict[str, str]],
) -> None:
    """Validate final benchmark balance."""
    for target in SELECTED_FILES:
        count = sum(
            record["target"] == target
            for record in records
        )

        if count != 22:
            raise ValueError(
                f"{target} has {count} cases, "
                "expected 22."
            )

    if len(records) != 88:
        raise ValueError(
            f"Benchmark contains {len(records)} cases, "
            "expected 88."
        )


def main() -> None:
    configure_logging()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_records = []

    for target, filenames in (
        SELECTED_FILES.items()
    ):
        LOGGER.info(
            "Processing %s",
            target,
        )

        metadata = load_review_csv(
            target
        )

        target_records = []

        for index, filename in enumerate(
            filenames,
            start=1,
        ):
            record = build_record(
                target=target,
                filename=filename,
                metadata=metadata,
                benchmark_index=index,
            )

            target_records.append(
                record
            )

            all_records.append(
                record
            )

        contact_path = (
            build_contact_sheet(
                target,
                target_records,
            )
        )

        LOGGER.info(
            "%s: %d verified cases",
            target,
            len(target_records),
        )

        LOGGER.info(
            "Contact sheet: %s",
            contact_path,
        )

    validate_balance(
        all_records
    )

    csv_path = save_benchmark_csv(
        all_records
    )

    manifest_path = save_manifest(
        all_records
    )

    print()
    print(
        "Verified-positive benchmark completed."
    )

    print(
        f"Total cases: {len(all_records)}"
    )

    print(
        "sleeve: 22"
    )

    print(
        "collar: 22"
    )

    print(
        "button: 22"
    )

    print(
        "zipper: 22"
    )

    print()
    print(
        f"CSV: {csv_path}"
    )

    print(
        f"Manifest: {manifest_path}"
    )

    print(
        f"Outputs: {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()