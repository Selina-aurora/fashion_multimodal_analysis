"""数据处理：保存原图来源和实例标识，避免同一图片的不同实例跨训练与评估划分。

Prepare per-garment crops/masks from DeepFashion2 annotations.

Purpose
-------
Create clean 3.1.3 inputs where one row = one garment instance.

This is an annotation-assisted pilot:
- garment bbox/mask comes from DeepFashion2 ground truth
- used to validate 3.1.3 independently of 3.1.1 prediction errors
- later replace these GT regions with automatic 3.1.1 outputs

Expected dataset layout
-----------------------
../fashion_data/raw/train/train/
    image/
    annos/

Example: single image
---------------------
python scripts/data_tools/prepare_garment_instances_from_deepfashion2.py     --image
000001.jpg

Example: deterministic 20-image pilot
-------------------------------------
python scripts/data_tools/prepare_garment_instances_from_deepfashion2.py
--sample-size 20     --seed 20260915

Outputs
-------
configs/garment_instances_gt_pilot.csv

outputs/garment_instances/gt_pilot/
    <image_stem>/
        <garment_id>_crop.jpg
        <garment_id>_mask.png
        <garment_id>_masked.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ImageDraw

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)

PROJECT_ROOT = get_project_root()

DEFAULT_IMAGE_DIR = data_root() / "raw" / "train" / "train" / "image"

DEFAULT_ANNO_DIR = data_root() / "raw" / "train" / "train" / "annos"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "garment_instances" / "gt_pilot"

DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "garment_instances_gt_pilot.csv"

CATEGORY_MAP = {
    1: ("short_sleeve_top", "top"),
    2: ("long_sleeve_top", "top"),
    3: ("short_sleeve_outwear", "outerwear"),
    4: ("long_sleeve_outwear", "outerwear"),
    5: ("vest", "top"),
    6: ("sling", "top"),
    7: ("shorts", "pants"),
    8: ("trousers", "pants"),
    9: ("skirt", "skirt"),
    10: ("short_sleeve_dress", "dress"),
    11: ("long_sleeve_dress", "dress"),
    12: ("vest_dress", "dress"),
    13: ("sling_dress", "dress"),
}


def to_project_relative(path: Path) -> str:
    """Prefer a path relative to PROJECT_ROOT; fall back to normalized string.

    Args:
        path: 要读取或写入的文件路径。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        try:
            return "../" + path.relative_to(PROJECT_ROOT.parent).as_posix()
        except ValueError:
            return path.as_posix()


def anno_path_for(image_path: Path, anno_dir: Path) -> Path:
    """anno 路径 for。

    Args:
        image_path: 目标原图的文件引用。
        anno_dir: 本步骤使用的目录。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    return anno_dir / f"{image_path.stem}.json"


def extract_bbox(
    item: dict[str, Any], width: int, height: int
) -> tuple[int, int, int, int]:
    """Return clipped xyxy bbox. Supports DeepFashion2 bounding_box and common bbox
    variants.

    Args:
        item: item。
        width: 图像或目标表示的宽度。
        height: 图像或目标表示的高度。

    Returns:
        按顺序返回 x1, y1, x2, y2 等结果。

    Raises:
        ValueError: 输入或实验状态不符合检查条件。
    """
    bbox = item.get("bounding_box")

    if bbox is None:
        bbox = item.get("bbox")

    if bbox is None:
        raise ValueError("Garment item has no bounding_box/bbox.")

    if len(bbox) != 4:
        raise ValueError(f"Unexpected bbox: {bbox}")

    x1, y1, a, b = [float(v) for v in bbox]

    # DeepFashion2 bounding_box is normally [x1, y1, x2, y2].
    # If values look like width/height instead, convert conservatively.
    if a <= x1 or b <= y1:
        x2 = x1 + a
        y2 = y1 + b
    else:
        x2 = a
        y2 = b

    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))

    return x1, y1, x2, y2


def iter_polygons(segmentation: Any) -> Iterator[Any]:
    """Yield flat polygon coordinate lists from DeepFashion2-style segmentation.

    Args:
        segmentation: 原始分割标注，可能为多边形或 COCO RLE。

    Yields:
        按需产生的记录/任务；不一次性加载全部条目。
    """
    if not segmentation:
        return

    if isinstance(segmentation, list):
        # Flat polygon: [x1,y1,x2,y2,...]
        if segmentation and all(isinstance(v, (int, float)) for v in segmentation):
            if len(segmentation) >= 6:
                yield segmentation
            return

        # List of polygons.
        for poly in segmentation:
            if (
                isinstance(poly, list)
                and len(poly) >= 6
                and all(isinstance(v, (int, float)) for v in poly)
            ):
                yield poly


def build_full_mask(
    item: dict[str, Any],
    image_size: tuple[int, int],
) -> Image.Image | None:
    """构建 整图 掩码。

    Args:
        item: item。
        image_size: 图像的宽、高，用于处理坐标和掩码尺寸。

    Returns:
        返回 mask，由函数体中同名变量的计算/收集过程得到。
    """
    width, height = image_size
    segmentation = item.get("segmentation")

    polygons = list(iter_polygons(segmentation))
    if not polygons:
        return None

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for poly in polygons:
        pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)

    return mask


def mask_crop_on_white(crop: Image.Image, mask: Image.Image) -> Image.Image:
    """掩码 裁剪 on white。

    Args:
        crop: 裁剪。
        mask: 掩码。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    white = Image.new("RGB", crop.size, "white")
    return Image.composite(crop, white, mask)


def garment_items(annotation: dict[str, Any]) -> list[Any]:
    """Return item1/item2/... entries only.

    Args:
        annotation: 原始实例标注。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。
    """
    pairs = []

    for key, value in annotation.items():
        if not key.startswith("item"):
            continue
        if not isinstance(value, dict):
            continue

        suffix = key[4:]
        try:
            order = int(suffix)
        except ValueError:
            order = 999999

        pairs.append((order, key, value))

    pairs.sort(key=lambda x: x[0])
    return [(key, value) for _, key, value in pairs]


def select_images(
    image_dir: Path,
    single_image: str | None,
    sample_size: int | None,
    seed: int,
) -> list[Path]:
    """选择 图像。

    Args:
        image_dir: 本步骤使用的目录。
        single_image: single 图像。
        sample_size: sample size。
        seed: 控制随机过程的种子。

    Returns:
        本步骤计算或解析得到的结果；函数体保留了具体结构和空值处理规则。

    Raises:
        FileNotFoundError: 需要的文件不存在。
        RuntimeError: 当前运行条件不满足实验要求。
    """
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    if single_image:
        p = Path(single_image)
        if not p.is_absolute():
            p = image_dir / p
        p = p.resolve()

        if not p.is_file():
            raise FileNotFoundError(f"Image not found: {p}")
        return [p]

    all_images = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )

    if not all_images:
        raise RuntimeError(f"No images found under {image_dir}")

    if sample_size is None:
        return all_images

    sample_size = min(sample_size, len(all_images))
    rng = random.Random(seed)
    return sorted(rng.sample(all_images, sample_size))


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default=None,
        help="One image filename, e.g. 000001.jpg",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Deterministically sample N source images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260915,
    )
    parser.add_argument(
        "--image-dir",
        default=str(DEFAULT_IMAGE_DIR),
    )
    parser.add_argument(
        "--anno-dir",
        default=str(DEFAULT_ANNO_DIR),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir).expanduser().resolve()
    anno_dir = Path(args.anno_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser()

    if not manifest_path.is_absolute():
        manifest_path = (PROJECT_ROOT / manifest_path).resolve()

    selected = select_images(
        image_dir=image_dir,
        single_image=args.image,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped_images = 0
    skipped_items = 0

    print(f"image dir     : {to_project_relative(image_dir)}")
    print(f"annotation dir: {to_project_relative(anno_dir)}")
    print(f"source images : {len(selected)}")

    for image_index, image_path in enumerate(selected, start=1):
        anno_path = anno_path_for(image_path, anno_dir)

        if not anno_path.is_file():
            print(f"[skip] no annotation: {image_path.name}")
            skipped_images += 1
            continue

        annotation = json.loads(anno_path.read_text(encoding="utf-8"))

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        items = garment_items(annotation)

        if not items:
            print(f"[skip] no item*: {image_path.name}")
            skipped_images += 1
            continue

        image_output_dir = OUTPUT_ROOT / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        saved_for_image = 0

        for item_index, (item_key, item) in enumerate(items, start=1):
            try:
                category_id = int(item.get("category_id", -1))
            except (TypeError, ValueError):
                category_id = -1

            fine_category, coarse_category = CATEGORY_MAP.get(
                category_id,
                ("unknown", "unknown"),
            )

            try:
                x1, y1, x2, y2 = extract_bbox(
                    item,
                    width=width,
                    height=height,
                )
            except Exception as exc:
                print(f"[skip item] {image_path.name} {item_key}: {exc}")
                skipped_items += 1
                continue

            garment_id = f"{item_key}_{fine_category}"

            crop = image.crop((x1, y1, x2, y2))

            crop_path = image_output_dir / f"{garment_id}_crop.jpg"
            crop.save(crop_path, quality=95)

            full_mask = build_full_mask(
                item,
                image.size,
            )

            mask_path_str = ""
            masked_path_str = ""

            if full_mask is not None:
                mask_crop = full_mask.crop((x1, y1, x2, y2))

                mask_path = image_output_dir / f"{garment_id}_mask.png"
                mask_crop.save(mask_path)

                masked = mask_crop_on_white(
                    crop,
                    mask_crop,
                )

                masked_path = image_output_dir / f"{garment_id}_masked.jpg"
                masked.save(masked_path, quality=95)

                mask_path_str = to_project_relative(mask_path)
                masked_path_str = to_project_relative(masked_path)

            rows.append(
                {
                    "source_image": to_project_relative(image_path),
                    "garment_id": garment_id,
                    "garment_category": coarse_category,
                    "fine_category": fine_category,
                    "category_id": category_id,
                    "annotation_item": item_key,
                    "crop_path": to_project_relative(crop_path),
                    "mask_path": mask_path_str,
                    "masked_preview_path": masked_path_str,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                }
            )

            saved_for_image += 1

        print(
            f"[{image_index:03d}/{len(selected):03d}] "
            + f"{image_path.name}: {saved_for_image} garment instance(s)"
        )

    fieldnames = [
        "source_image",
        "garment_id",
        "garment_category",
        "fine_category",
        "category_id",
        "annotation_item",
        "crop_path",
        "mask_path",
        "masked_preview_path",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]

    with manifest_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== FINISHED ===")
    print(f"garment instances : {len(rows)}")
    print(f"skipped images    : {skipped_images}")
    print(f"skipped items     : {skipped_items}")
    print("manifest          : " + f"{to_project_relative(manifest_path)}")
    print("crops/masks       : " + "outputs/garment_instances/gt_pilot/")
    print("================")


if __name__ == "__main__":
    main()
