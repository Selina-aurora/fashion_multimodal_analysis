"""OpenCV bbox annotator with previous/next navigation for the core part pilot.

Controls:
    Left mouse drag  draw a bbox
    Z                undo last bbox
    C                clear all bboxes
    S / Enter        save current image and move next
    N                mark unusable and move next
    P                go to previous image
    Q / Esc          save progress and quit

Previously saved images can be revisited with P and edited.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "prd_region_coverage"
MANIFEST_FILE = REPORT_ROOT / "core_part_annotation_pilot_manifest.csv"
OUTPUT_FILE = REPORT_ROOT / "core_part_annotation_labels.csv"

WINDOW_NAME = "Core Part BBox Annotator"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write annotation rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "annotation_id",
        "region",
        "source_candidate_id",
        "image_name",
        "annotation_image_path",
        "label_status",
        "boxes_json",
        "review_note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing() -> dict[str, dict[str, str]]:
    """Load previous annotation progress when available."""
    if not OUTPUT_FILE.is_file():
        return {}
    return {
        row["annotation_id"]: row
        for row in read_csv(OUTPUT_FILE)
    }


def draw_ui(
    image,
    boxes: list[list[int]],
    region: str,
    annotation_id: str,
    index: int,
    total: int,
    status: str,
):
    """Render current boxes and help text."""
    canvas = image.copy()

    for box_index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"{region} {box_index}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    lines = [
        f"[{index + 1}/{total}] {annotation_id} | target={region} | status={status}",
        "drag bbox | Z undo | C clear | S/Enter save+next | N skip | P previous | Q/Esc quit",
    ]
    for line_index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (10, 24 + line_index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return canvas


def normalize_box(
    start: tuple[int, int],
    end: tuple[int, int],
    width: int,
    height: int,
) -> list[int] | None:
    """Normalize a drag box and reject tiny accidental boxes."""
    x1 = max(0, min(start[0], end[0]))
    y1 = max(0, min(start[1], end[1]))
    x2 = min(width - 1, max(start[0], end[0]))
    y2 = min(height - 1, max(start[1], end[1]))

    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return [x1, y1, x2, y2]


def parse_boxes(raw: str) -> list[list[int]]:
    """Parse saved bbox JSON safely."""
    if not raw:
        return []
    try:
        boxes = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(boxes, list):
        return []
    return boxes


def main() -> None:
    """Run interactive annotation with backward navigation."""
    manifest = read_csv(MANIFEST_FILE)
    existing = load_existing()

    output_rows = []
    for row in manifest:
        old = existing.get(row["annotation_id"])
        if old:
            output_rows.append(old)
        else:
            output_rows.append(
                {
                    "annotation_id": row["annotation_id"],
                    "region": row["region"],
                    "source_candidate_id": row["source_candidate_id"],
                    "image_name": row["image_name"],
                    "annotation_image_path": row["annotation_image_path"],
                    "label_status": "pending",
                    "boxes_json": "",
                    "review_note": "",
                }
            )

    first_pending = next(
        (
            index
            for index, row in enumerate(output_rows)
            if row["label_status"] == "pending"
        ),
        0,
    )

    index = first_pending
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while 0 <= index < len(output_rows):
            row = output_rows[index]

            image_path = PROJECT_ROOT / Path(
                row["annotation_image_path"].replace("\\", "/")
            )
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")

            height, width = image.shape[:2]
            boxes = parse_boxes(row.get("boxes_json", ""))

            dragging = False
            drag_start = (0, 0)
            drag_end = (0, 0)

            def mouse_callback(event, x, y, flags, param):
                nonlocal dragging, drag_start, drag_end

                if event == cv2.EVENT_LBUTTONDOWN:
                    dragging = True
                    drag_start = (x, y)
                    drag_end = (x, y)

                elif event == cv2.EVENT_MOUSEMOVE and dragging:
                    drag_end = (x, y)

                elif event == cv2.EVENT_LBUTTONUP and dragging:
                    dragging = False
                    drag_end = (x, y)
                    box = normalize_box(
                        drag_start,
                        drag_end,
                        width,
                        height,
                    )
                    if box is not None:
                        boxes.append(box)

            cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

            move = 0

            while True:
                canvas = draw_ui(
                    image=image,
                    boxes=boxes,
                    region=row["region"],
                    annotation_id=row["annotation_id"],
                    index=index,
                    total=len(output_rows),
                    status=row["label_status"],
                )

                if dragging:
                    temp_box = normalize_box(
                        drag_start,
                        drag_end,
                        width,
                        height,
                    )
                    if temp_box is not None:
                        x1, y1, x2, y2 = temp_box
                        cv2.rectangle(
                            canvas,
                            (x1, y1),
                            (x2, y2),
                            (255, 0, 0),
                            1,
                        )

                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(20) & 0xFF

                if key in (ord("z"), ord("Z")):
                    if boxes:
                        boxes.pop()

                elif key in (ord("c"), ord("C")):
                    boxes.clear()

                elif key in (ord("s"), ord("S"), 13):
                    if not boxes:
                        print("Draw at least one bbox, or press N to skip.")
                        continue
                    row["label_status"] = "labeled"
                    row["boxes_json"] = json.dumps(boxes)
                    row["review_note"] = ""
                    write_csv(OUTPUT_FILE, output_rows)
                    print(
                        f"[{index + 1}/{len(output_rows)}] "
                        f"saved {row['annotation_id']} "
                        f"with {len(boxes)} box(es)"
                    )
                    move = 1
                    break

                elif key in (ord("n"), ord("N")):
                    row["label_status"] = "unusable"
                    row["boxes_json"] = "[]"
                    row["review_note"] = "Skipped during manual annotation."
                    write_csv(OUTPUT_FILE, output_rows)
                    print(
                        f"[{index + 1}/{len(output_rows)}] "
                        f"skipped {row['annotation_id']}"
                    )
                    move = 1
                    break

                elif key in (ord("p"), ord("P")):
                    # Keep the current row unchanged unless it was explicitly saved.
                    write_csv(OUTPUT_FILE, output_rows)
                    move = -1
                    break

                elif key in (ord("q"), ord("Q"), 27):
                    write_csv(OUTPUT_FILE, output_rows)
                    print(f"Progress saved: {OUTPUT_FILE}")
                    return

            if move == -1:
                index = max(0, index - 1)
            elif move == 1:
                index += 1

        write_csv(OUTPUT_FILE, output_rows)
        print("\nReached the end of the annotation list.")
        print(f"Labels: {OUTPUT_FILE}")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
