
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "fashion_data"

INPUT_CASES = (
    PROJECT_ROOT
    / "reports"
    / "prd_region_coverage"
    / "remaining5_diagnostic_v1"
    / "cases.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "prd_region_coverage"
    / "shoulder_waist_pattern_refinement_v1"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_region_coverage"
    / "shoulder_waist_pattern_refinement_v1"
)

REBUILT_PARENT_DIR = OUTPUT_DIR / "rebuilt_parent_crops"
MANIFEST_OUT = REPORT_DIR / "refinement_manifest.csv"
CONTACT_OUT = REPORT_DIR / "refinement_windows_contact_sheet.jpg"
RESOLUTION_LOG = REPORT_DIR / "parent_crop_resolution.csv"

TARGET_REGIONS = {"shoulder", "waist", "pattern"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def resolve_project_path(raw: str) -> Path:
    raw = str(raw or "").strip().replace("\\", "/")
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def rel(path: Path) -> str:
    path = path.resolve()

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        pass

    try:
        p = path.relative_to(PROJECT_ROOT.parent)
        return f"../{p.as_posix()}"
    except ValueError:
        return path.as_posix()


def valid_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def build_manifest_index() -> dict[tuple[str, str], dict[str, str]]:
    index = {}

    for csv_path in (PROJECT_ROOT / "configs").glob("*.csv"):
        try:
            rows = read_csv(csv_path)
        except Exception:
            continue

        for row in rows:
            source_image = str(
                row.get("source_image")
                or row.get("image_path")
                or row.get("image_name")
                or ""
            ).strip()

            image_name = Path(source_image.replace("\\", "/")).name

            garment_id = str(
                row.get("garment_id")
                or row.get("item_id")
                or row.get("annotation_item")
                or ""
            ).strip()

            if not image_name or not garment_id:
                continue

            key = (image_name, garment_id)

            score = sum(
                bool(str(row.get(k, "")).strip())
                for k in (
                    "crop_path",
                    "masked_preview_path",
                    "bbox_x1",
                    "bbox_y1",
                    "bbox_x2",
                    "bbox_y2",
                    "source_image",
                )
            )

            row = dict(row)
            row["_manifest"] = csv_path.as_posix()
            row["_score"] = str(score)

            prev = index.get(key)

            if (
                prev is None
                or score > int(prev.get("_score", "0"))
            ):
                index[key] = row

    return index


def find_manifest_row(
    index: dict[tuple[str, str], dict[str, str]],
    image_name: str,
    item_id: str,
) -> dict[str, str] | None:
    image_name = Path(
        image_name.replace("\\", "/")
    ).name

    item_id = str(item_id).strip()

    exact = index.get(
        (image_name, item_id)
    )

    if exact is not None:
        return exact

    prefix = item_id + "_"

    matches = [
        row
        for (img, gid), row in index.items()
        if img == image_name
        and (
            gid == item_id
            or gid.startswith(prefix)
        )
    ]

    if not matches:
        return None

    matches.sort(
        key=lambda x: int(
            x.get("_score", "0")
        ),
        reverse=True,
    )

    return matches[0]


def find_deepfashion_image(image_name: str) -> Path | None:
    known = [
        DATA_ROOT / "raw" / "train" / "train" / "image" / image_name,
        DATA_ROOT / "raw" / "validation" / "image" / image_name,
        DATA_ROOT / "raw" / "validation" / "validation" / "image" / image_name,
        DATA_ROOT / "raw" / "train" / "image" / image_name,
    ]

    for p in known:
        if valid_file(p):
            return p.resolve()

    # Last-resort targeted search only for this filename.
    roots = [
        DATA_ROOT / "raw" / "train",
        DATA_ROOT / "raw" / "validation",
    ]

    for root in roots:
        if not root.exists():
            continue

        found = next(
            (
                p for p in root.rglob(image_name)
                if valid_file(p)
            ),
            None,
        )

        if found is not None:
            return found.resolve()

    return None


def find_deepfashion_annotation(image_name: str) -> Path | None:
    stem = Path(image_name).stem
    filename = stem + ".json"

    known = [
        DATA_ROOT / "raw" / "train" / "train" / "annos" / filename,
        DATA_ROOT / "raw" / "train" / "train" / "annos" / "train" / filename,
        DATA_ROOT / "raw" / "validation" / "annos" / filename,
        DATA_ROOT / "raw" / "validation" / "validation" / "annos" / filename,
        DATA_ROOT / "raw" / "train" / "annos" / filename,
    ]

    for p in known:
        if valid_file(p):
            return p.resolve()

    roots = [
        DATA_ROOT / "raw" / "train",
        DATA_ROOT / "raw" / "validation",
    ]

    for root in roots:
        if not root.exists():
            continue

        found = next(
            (
                p for p in root.rglob(filename)
                if p.parent.name.lower().startswith("anno")
                and valid_file(p)
            ),
            None,
        )

        if found is not None:
            return found.resolve()

    return None


def bbox_from_segmentation(
    segmentation,
) -> tuple[int, int, int, int] | None:
    xs = []
    ys = []

    if not isinstance(segmentation, list):
        return None

    for poly in segmentation:
        if not isinstance(poly, list):
            continue

        for i in range(
            0,
            len(poly) - 1,
            2,
        ):
            try:
                xs.append(float(poly[i]))
                ys.append(float(poly[i + 1]))
            except Exception:
                pass

    if not xs or not ys:
        return None

    return (
        int(min(xs)),
        int(min(ys)),
        int(max(xs)) + 1,
        int(max(ys)) + 1,
    )


def rebuild_from_deepfashion_annotation(
    image_name: str,
    item_id: str,
    candidate_id: str,
    region: str,
) -> tuple[Path, str, str]:
    image_path = find_deepfashion_image(
        image_name
    )
    anno_path = find_deepfashion_annotation(
        image_name
    )

    if image_path is None:
        raise FileNotFoundError(
            f"DeepFashion2 source image not found for {image_name}"
        )

    if anno_path is None:
        raise FileNotFoundError(
            f"DeepFashion2 annotation JSON not found for {image_name}"
        )

    with anno_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    item = data.get(
        item_id
    )

    if not isinstance(item, dict):
        raise KeyError(
            f"{anno_path} has no item key {item_id!r}"
        )

    bbox = item.get(
        "bounding_box"
    )

    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
    ):
        bbox = bbox_from_segmentation(
            item.get(
                "segmentation"
            )
        )

    if bbox is None:
        raise ValueError(
            f"No usable bbox/segmentation in {anno_path} -> {item_id}"
        )

    x1, y1, x2, y2 = [
        int(round(float(v)))
        for v in bbox
    ]

    image = Image.open(
        image_path
    ).convert("RGB")

    w, h = image.size

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    crop = image.crop(
        (x1, y1, x2, y2)
    )

    out = (
        REBUILT_PARENT_DIR
        / region
        / f"{candidate_id}_{image_name}"
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop.save(
        out,
        quality=95,
    )

    return (
        out,
        "rebuilt_from_deepfashion_annotation",
        rel(anno_path),
    )


def recover_parent_crop(
    case: dict[str, str],
    manifest_index: dict[tuple[str, str], dict[str, str]],
) -> tuple[Path, str, str]:
    old = resolve_project_path(
        case.get(
            "candidate_crop_path",
            "",
        )
    )

    if valid_file(old):
        return (
            old,
            "existing_region_candidate_crop",
            "",
        )

    image_name = case.get(
        "image_name",
        "",
    ).strip()

    item_id = case.get(
        "item_id",
        "",
    ).strip()

    manifest = find_manifest_row(
        manifest_index,
        image_name,
        item_id,
    )

    if manifest is not None:
        manifest_name = manifest.get(
            "_manifest",
            "",
        )

        for key, method in (
            (
                "crop_path",
                "garment_manifest_crop_path",
            ),
            (
                "masked_preview_path",
                "garment_manifest_masked_preview",
            ),
        ):
            raw = str(
                manifest.get(
                    key,
                    "",
                )
            ).strip()

            if raw:
                p = resolve_project_path(raw)

                if valid_file(p):
                    return (
                        p,
                        method,
                        manifest_name,
                    )

        source_raw = str(
            manifest.get(
                "source_image",
                "",
            )
        ).strip()

        source = (
            resolve_project_path(source_raw)
            if source_raw
            else None
        )

        bbox_keys = (
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
        )

        if (
            source is not None
            and valid_file(source)
            and all(
                str(
                    manifest.get(
                        k,
                        "",
                    )
                ).strip()
                for k in bbox_keys
            )
        ):
            image = Image.open(
                source
            ).convert("RGB")

            w, h = image.size

            x1, y1, x2, y2 = [
                int(
                    float(
                        manifest[k]
                    )
                )
                for k in bbox_keys
            ]

            x1 = max(
                0,
                min(
                    w - 1,
                    x1,
                ),
            )
            y1 = max(
                0,
                min(
                    h - 1,
                    y1,
                ),
            )
            x2 = max(
                x1 + 1,
                min(
                    w,
                    x2,
                ),
            )
            y2 = max(
                y1 + 1,
                min(
                    h,
                    y2,
                ),
            )

            crop = image.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

            out = (
                REBUILT_PARENT_DIR
                / case["region"]
                / f"{case['candidate_id']}_{image_name}"
            )

            out.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            crop.save(
                out,
                quality=95,
            )

            return (
                out,
                "rebuilt_from_manifest_source_bbox",
                manifest_name,
            )

    # Final robust fallback: use original DeepFashion2 per-image JSON directly.
    return rebuild_from_deepfashion_annotation(
        image_name=image_name,
        item_id=item_id,
        candidate_id=case["candidate_id"],
        region=case["region"],
    )


def windows_for(region: str):
    if region == "shoulder":
        return [
            (
                "upper_left",
                0.00,
                0.00,
                0.62,
                0.48,
            ),
            (
                "upper_right",
                0.38,
                0.00,
                1.00,
                0.48,
            ),
        ]

    if region == "waist":
        return [
            (
                "middle_band",
                0.05,
                0.28,
                0.95,
                0.72,
            )
        ]

    if region == "pattern":
        return [
            (
                "garment_center",
                0.06,
                0.06,
                0.94,
                0.94,
            )
        ]

    raise ValueError(region)


def main() -> None:
    cases = [
        row
        for row in read_csv(
            INPUT_CASES
        )
        if row.get(
            "region",
            "",
        ).strip().lower()
        in TARGET_REGIONS
    ]

    if not cases:
        raise RuntimeError(
            "No shoulder/waist/pattern cases found."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_index = build_manifest_index()

    print(
        "Garment manifest index entries:",
        len(manifest_index),
    )

    refinement_rows = []
    resolution_rows = []
    preview_tiles = []

    for i, case in enumerate(
        cases,
        start=1,
    ):
        region = case[
            "region"
        ].strip().lower()

        print(
            f"[{i:02d}/{len(cases)}] "
            f"{case['candidate_id']} "
            f"{case['image_name']} "
            f"{case['item_id']} ..."
        )

        parent, method, source_ref = recover_parent_crop(
            case,
            manifest_index,
        )

        print(
            f"    parent={method}"
        )

        resolution_rows.append(
            {
                "candidate_id": case[
                    "candidate_id"
                ],
                "region": region,
                "image_name": case.get(
                    "image_name",
                    "",
                ),
                "item_id": case.get(
                    "item_id",
                    "",
                ),
                "original_candidate_crop_path": case.get(
                    "candidate_crop_path",
                    "",
                ),
                "resolved_parent_crop_path": rel(
                    parent
                ),
                "resolution_method": method,
                "source_reference": source_ref,
            }
        )

        image = Image.open(
            parent
        ).convert("RGB")

        w, h = image.size

        for (
            window_name,
            x1r,
            y1r,
            x2r,
            y2r,
        ) in windows_for(
            region
        ):
            x1 = max(
                0,
                min(
                    w - 1,
                    int(
                        round(
                            x1r * w
                        )
                    ),
                ),
            )
            y1 = max(
                0,
                min(
                    h - 1,
                    int(
                        round(
                            y1r * h
                        )
                    ),
                ),
            )
            x2 = max(
                x1 + 1,
                min(
                    w,
                    int(
                        round(
                            x2r * w
                        )
                    ),
                ),
            )
            y2 = max(
                y1 + 1,
                min(
                    h,
                    int(
                        round(
                            y2r * h
                        )
                    ),
                ),
            )

            crop = image.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

            scale = max(
                1.0,
                900.0 / max(
                    crop.size
                ),
            )

            if scale > 1.0:
                crop = crop.resize(
                    (
                        int(
                            round(
                                crop.width
                                * scale
                            )
                        ),
                        int(
                            round(
                                crop.height
                                * scale
                            )
                        ),
                    ),
                    Image.Resampling.LANCZOS,
                )

            out_path = (
                OUTPUT_DIR
                / region
                / (
                    f"{case['candidate_id']}_"
                    f"{window_name}.jpg"
                )
            )

            out_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            crop.save(
                out_path,
                quality=95,
            )

            refinement_rows.append(
                {
                    "candidate_id": case[
                        "candidate_id"
                    ],
                    "region": region,
                    "image_name": case.get(
                        "image_name",
                        "",
                    ),
                    "item_id": case.get(
                        "item_id",
                        "",
                    ),
                    "baseline_quality": case.get(
                        "localization_quality",
                        "",
                    ),
                    "baseline_detected": case.get(
                        "detected",
                        "",
                    ),
                    "baseline_top_score": case.get(
                        "top_score",
                        "",
                    ),
                    "baseline_top_box_area_ratio": case.get(
                        "top_box_area_ratio",
                        "",
                    ),
                    "prompt": case.get(
                        "prompt",
                        "",
                    ),
                    "threshold": case.get(
                        "threshold",
                        "",
                    ),
                    "window_name": window_name,
                    "parent_crop_path": rel(
                        parent
                    ),
                    "parent_resolution_method": method,
                    "window_x1": x1,
                    "window_y1": y1,
                    "window_x2": x2,
                    "window_y2": y2,
                    "window_area_ratio_parent": round(
                        (
                            (x2 - x1)
                            * (y2 - y1)
                        )
                        / float(
                            w * h
                        ),
                        6,
                    ),
                    "refinement_crop_path": rel(
                        out_path
                    ),
                    "refinement_detected": "",
                    "refinement_top_score": "",
                    "refinement_top_box_area_ratio_parent": "",
                    "refinement_quality": "",
                    "review_note": "",
                }
            )

            preview = image.copy()

            draw = ImageDraw.Draw(
                preview
            )

            draw.rectangle(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                outline=(
                    0,
                    220,
                    0,
                ),
                width=4,
            )

            draw.text(
                (5, 5),
                (
                    f"{case['candidate_id']} | "
                    f"{region} | "
                    f"{window_name}"
                ),
                fill=(
                    255,
                    0,
                    0,
                ),
                font=ImageFont.load_default(),
            )

            preview.thumbnail(
                (
                    380,
                    330,
                ),
                Image.Resampling.LANCZOS,
            )

            preview_tiles.append(
                preview
            )

    write_csv(
        MANIFEST_OUT,
        refinement_rows,
    )

    write_csv(
        RESOLUTION_LOG,
        resolution_rows,
    )

    cols = 4
    tile_w = 400
    tile_h = 360
    rows_n = math.ceil(
        len(
            preview_tiles
        )
        / cols
    )

    sheet = Image.new(
        "RGB",
        (
            cols * tile_w,
            rows_n * tile_h,
        ),
        "white",
    )

    for i, tile in enumerate(
        preview_tiles
    ):
        x = (
            (i % cols)
            * tile_w
            + (
                tile_w
                - tile.width
            )
            // 2
        )

        y = (
            (i // cols)
            * tile_h
            + 10
        )

        sheet.paste(
            tile,
            (
                x,
                y,
            ),
        )

    sheet.save(
        CONTACT_OUT,
        quality=92,
    )

    methods = {}

    for row in resolution_rows:
        method = row[
            "resolution_method"
        ]

        methods[method] = (
            methods.get(
                method,
                0,
            )
            + 1
        )

    print()
    print(
        "=== REFINEMENT INPUTS BUILT ==="
    )
    print(
        f"Source cases: {len(cases)}"
    )
    print(
        f"Refinement crops: {len(refinement_rows)}"
    )
    print(
        "Parent crop resolution:"
    )

    for method, count in sorted(
        methods.items()
    ):
        print(
            f"  {method}: {count}"
        )

    print()
    print(
        "Manifest:",
        MANIFEST_OUT.relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        "Contact :",
        CONTACT_OUT.relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        "Resolve :",
        RESOLUTION_LOG.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()
