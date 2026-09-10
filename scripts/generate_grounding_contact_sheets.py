"""Generate contact sheets for manual grounding annotations."""

import csv
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageOps


GROUP_ID = 1
ROWS = 5
COLUMNS = 5

CELL_WIDTH = 260
CELL_HEIGHT = 340
LABEL_HEIGHT = 42
PADDING = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "fashion_data"

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "reports"
    / "evaluation_sampling"
    / "grounding_presence_annotations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation_contact_sheets"
    / f"group_{GROUP_ID}"
)


def read_group_rows(
    annotation_file: Path,
    group_id: int,
) -> List[Dict[str, str]]:
    """Read annotation rows belonging to one evaluation group.

    Args:
        annotation_file: Grounding annotation CSV file.
        group_id: Evaluation group identifier.

    Returns:
        Rows belonging to the selected evaluation group.

    Raises:
        FileNotFoundError: If the annotation file does not exist.
    """
    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file does not exist: {annotation_file}"
        )

    with annotation_file.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        return [
            row
            for row in reader
            if int(row["group_id"]) == group_id
        ]


def load_thumbnail(
    image_path: Path,
) -> Image.Image:
    """Load an image and resize it for the contact sheet.

    Args:
        image_path: Source image path.

    Returns:
        Resized RGB image.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        max_size = (
            CELL_WIDTH - (2 * PADDING),
            CELL_HEIGHT - LABEL_HEIGHT - (2 * PADDING),
        )

        return ImageOps.contain(
            image,
            max_size,
        )


def paste_sample(
    sheet: Image.Image,
    row: Dict[str, str],
    position: int,
) -> None:
    """Paste one sample image and label onto a contact sheet.

    Args:
        sheet: Contact sheet canvas.
        row: Annotation row containing image information.
        position: Zero-based position on the current sheet.
    """
    grid_row = position // COLUMNS
    grid_column = position % COLUMNS

    cell_x = grid_column * CELL_WIDTH
    cell_y = grid_row * CELL_HEIGHT

    image_path = (
        DATASET_ROOT
        / Path(row["relative_path"])
    )

    draw = ImageDraw.Draw(sheet)

    if not image_path.exists():
        draw.text(
            (
                cell_x + PADDING,
                cell_y + PADDING,
            ),
            "IMAGE NOT FOUND",
            fill="black",
        )
        return

    thumbnail = load_thumbnail(image_path)

    image_x = (
        cell_x
        + (CELL_WIDTH - thumbnail.width) // 2
    )

    image_area_height = CELL_HEIGHT - LABEL_HEIGHT

    image_y = (
        cell_y
        + (image_area_height - thumbnail.height) // 2
    )

    sheet.paste(
        thumbnail,
        (image_x, image_y),
    )

    label = (
        f'ID {row["sample_id"]} | '
        f'{row["image_name"]}'
    )

    draw.text(
        (
            cell_x + PADDING,
            cell_y + CELL_HEIGHT - LABEL_HEIGHT + 10,
        ),
        label,
        fill="black",
    )


def save_contact_sheet(
    rows: List[Dict[str, str]],
    page_number: int,
) -> Path:
    """Create and save one contact sheet page.

    Args:
        rows: Samples displayed on this page.
        page_number: One-based contact sheet page number.

    Returns:
        Saved contact sheet path.
    """
    sheet_width = COLUMNS * CELL_WIDTH
    sheet_height = ROWS * CELL_HEIGHT

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white",
    )

    for position, row in enumerate(rows):
        paste_sample(
            sheet=sheet,
            row=row,
            position=position,
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / f"contact_sheet_{page_number:02d}.jpg"
    )

    sheet.save(
        output_file,
        quality=92,
    )

    return output_file


def main() -> None:
    """Generate contact sheets for the selected evaluation group."""
    group_rows = read_group_rows(
        annotation_file=ANNOTATION_FILE,
        group_id=GROUP_ID,
    )

    samples_per_page = ROWS * COLUMNS

    print(
        f"Group {GROUP_ID}: {len(group_rows)} images"
    )

    for start_index in range(
        0,
        len(group_rows),
        samples_per_page,
    ):
        page_rows = group_rows[
            start_index:
            start_index + samples_per_page
        ]

        page_number = (
            start_index // samples_per_page
        ) + 1

        output_file = save_contact_sheet(
            rows=page_rows,
            page_number=page_number,
        )

        print(
            f"Page {page_number}: {output_file}"
        )


if __name__ == "__main__":
    main()