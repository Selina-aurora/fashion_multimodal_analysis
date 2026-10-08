"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。

Review only the semantic-QC flagged core-part annotations.

Inputs:
    reports/prd_region_coverage/core_part_annotation_labels.csv
    reports/prd_region_coverage/core_part_annotation_semantic_qc_flags.csv

Controls:
    Left mouse drag  draw bbox
    Z                undo last bbox
    C                clear all bboxes
    S / Enter        save edited boxes and go next flagged case
    N                mark unusable and go next
    K                keep current annotation unchanged and go next
    P                previous flagged case
    Q / Esc          save progress and quit

The script updates core_part_annotation_labels.csv in place and creates a
backup once before the first edit.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import cv2

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()
REPORT_ROOT = PROJECT_ROOT / "reports" / "prd_region_coverage"

LABEL_FILE = REPORT_ROOT / "core_part_annotation_labels.csv"
FLAGS_FILE = REPORT_ROOT / "core_part_annotation_semantic_qc_flags.csv"
BACKUP_FILE = REPORT_ROOT / "core_part_annotation_labels_before_semantic_qc.csv"

WINDOW_NAME = "Flagged Core Part Review"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows.

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        每行一个字段字典；值保留原始字符串，不自动改变标签。

    Raises:
        FileNotFoundError: 需要的文件不存在。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows.

    Args:
        path: 要读取或写入的文件路径。
        rows: 待处理的逐行记录。

    Raises:
        ValueError: No rows to write.
    """
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_boxes(raw: str) -> list[list[int]]:
    """Parse bbox JSON.

    Args:
        raw: 本步骤使用的原始值，转换/筛选规则见函数体。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def normalize_box(
    start: tuple[int, int],
    end: tuple[int, int],
    width: int,
    height: int,
) -> list[int] | None:
    """Normalize a mouse-drag box.

    Args:
        start: start。
        end: end。
        width: 图像或目标表示的宽度。
        height: 图像或目标表示的高度。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    x1 = max(0, min(start[0], end[0]))
    y1 = max(0, min(start[1], end[1]))
    x2 = min(width - 1, max(start[0], end[0]))
    y2 = min(height - 1, max(start[1], end[1]))

    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return [x1, y1, x2, y2]


def draw_ui(
    image: Any,
    row: dict[str, str],
    flag: dict[str, str],
    boxes: list[list[int]],
    index: int,
    total: int,
) -> Any:
    """Draw current labels and semantic-QC instructions.

    Args:
        image: 本步骤处理的图像对象。
        row: 记录字段，使用 annotation_id, region。
        flag: 记录字段，使用 reason, priority, recommended_action。
        boxes: 边界框。
        index: 待访问样本的整数下标。
        total: total。

    Returns:
        返回 canvas，由函数体中同名变量的计算/收集过程得到。
    """
    canvas = image.copy()

    for box_index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"{row['region']} {box_index}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    lines = [
        f"[{index + 1}/{total}] {row['annotation_id']} | "
        + f"{flag['priority']} | {flag['recommended_action']}",
        flag["reason"][:115],
        "drag bbox | Z undo | C clear | S save | N unusable | "
        + "K keep | P previous | Q quit",
    ]

    for line_index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (8, 22 + 22 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 0),
            1,
        )

    return canvas


def main() -> None:
    """Review only flagged annotations.

    Raises:
        KeyError: 所需字段不存在。
        FileNotFoundError: 需要的文件不存在。
    """
    rows = read_csv(LABEL_FILE)
    flags = read_csv(FLAGS_FILE)

    if not BACKUP_FILE.exists():
        shutil.copy2(LABEL_FILE, BACKUP_FILE)
        print(f"Backup created: {BACKUP_FILE}")

    row_map = {row["annotation_id"]: row for row in rows}

    for flag in flags:
        if flag["annotation_id"] not in row_map:
            raise KeyError(f"Missing annotation row: {flag['annotation_id']}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    index = 0

    try:
        while 0 <= index < len(flags):
            flag = flags[index]
            row = row_map[flag["annotation_id"]]

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

            def mouse_callback(
                event: str, x: Any, y: Any, flags_: Any, param: Any
            ) -> None:
                """mouse callback。

                Args:
                    event: 写入进度日志的事件名称。
                    x: 本步骤使用的原始值，转换/筛选规则见函数体。
                    y: 本步骤使用的原始值，转换/筛选规则见函数体。
                    flags_: flags 。
                    param: param。
                """
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
                    row=row,
                    flag=flag,
                    boxes=boxes,
                    index=index,
                    total=len(flags),
                )

                if dragging:
                    temp = normalize_box(
                        drag_start,
                        drag_end,
                        width,
                        height,
                    )
                    if temp is not None:
                        x1, y1, x2, y2 = temp
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
                        print("Draw a bbox or press N for unusable.")
                        continue

                    row["label_status"] = "labeled"
                    row["boxes_json"] = json.dumps(boxes)
                    row["review_note"] = "Reviewed during semantic QC."
                    write_csv(LABEL_FILE, rows)
                    move = 1
                    break

                elif key in (ord("n"), ord("N")):
                    row["label_status"] = "unusable"
                    row["boxes_json"] = "[]"
                    row["review_note"] = "Marked unusable during semantic QC."
                    write_csv(LABEL_FILE, rows)
                    move = 1
                    break

                elif key in (ord("k"), ord("K")):
                    row["review_note"] = (
                        "Reviewed and kept unchanged during semantic QC."
                    )
                    write_csv(LABEL_FILE, rows)
                    move = 1
                    break

                elif key in (ord("p"), ord("P")):
                    move = -1
                    break

                elif key in (ord("q"), ord("Q"), 27):
                    write_csv(LABEL_FILE, rows)
                    print(f"Progress saved: {LABEL_FILE}")
                    return

            index = max(0, index + move)

        write_csv(LABEL_FILE, rows)
        print("\nFinished flagged semantic-QC review.")
        print(f"Updated labels: {LABEL_FILE}")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
