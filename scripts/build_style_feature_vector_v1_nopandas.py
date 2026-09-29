"""Build PRD 3.1.3 unified continuous style feature vector v1 (no pandas).

This is a dependency-free replacement for build_style_feature_vector_v1.py.
It uses only the Python standard library (csv/json/pathlib), so it works in the
current .venv_attr environment without installing pandas.

Outputs
-------
reports/prd_attribute_extraction/style_feature_vector_v1/
    style_feature_vector_v1.csv
    style_feature_vector_v1.json
    source_coverage.csv
    run_info.txt
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "reports" / "prd_attribute_extraction"
OUT = BASE / "style_feature_vector_v1"

KEYS = ("source_image", "garment_id")

SOURCES = [
    {
        "name": "sleeve_semantic",
        "path": BASE / "continuous_v1" / "sleeve_continuous_predictions.csv",
        "keep": [
            "source_image",
            "garment_id",
            "garment_category",
            "sleeve_length_ratio",
            "normalized_entropy",
        ],
        "rename": {
            "normalized_entropy": "sleeve_entropy",
        },
    },
    {
        "name": "pants_length_semantic",
        "path": BASE / "pants_continuous_v1" / "pants_continuous_predictions.csv",
        "keep": [
            "source_image",
            "garment_id",
            "garment_category",
            "pants_length_ratio",
            "normalized_entropy",
        ],
        "rename": {
            "normalized_entropy": "pants_length_entropy",
        },
    },
    {
        "name": "skirt_dress_length_semantic",
        "path": BASE / "skirt_dress_length_v1" / "skirt_dress_length_predictions.csv",
        "keep": [
            "source_image",
            "garment_id",
            "garment_category",
            "lower_garment_length_score",
            "normalized_entropy",
        ],
        "rename": {
            "normalized_entropy": "lower_garment_length_entropy",
        },
    },
    {
        "name": "skirt_flare_geometry",
        "path": BASE / "skirt_flare_v1" / "skirt_flare_features.csv",
        "keep": [
            "source_image",
            "garment_id",
            "skirt_flare_score",
            "hem_to_waist_ratio",
        ],
        "rename": {
            "hem_to_waist_ratio": "skirt_hem_to_waist_ratio",
        },
    },
    {
        "name": "neckline_semantic",
        "path": BASE / "neckline_semantic_v2" / "neckline_semantic_predictions.csv",
        "keep": [
            "source_image",
            "garment_id",
            "garment_category",
            "neckline_depth_score",
            "neckline_width_score",
            "collar_height_score",
            "depth_entropy",
            "width_entropy",
            "collar_entropy",
        ],
        "rename": {},
    },
    {
        "name": "upper_silhouette_geometry",
        "path": BASE / "upper_silhouette_v2" / "upper_silhouette_features_v2.csv",
        "keep": [
            "source_image",
            "garment_id",
            "garment_category",
            "garment_length_proxy",
            "waist_to_chest_ratio",
            "hem_to_chest_ratio",
            "lower_taper_slope",
            "arm_contamination_ratio",
            "geometry_quality",
        ],
        "rename": {
            "geometry_quality": "upper_geometry_quality",
        },
    },
    {
        "name": "trouser_fit_geometry",
        "path": BASE / "trouser_fit_v2" / "trouser_fit_features_v2.csv",
        "keep": [
            "source_image",
            "garment_id",
            "garment_category",
            "thigh_width_ratio",
            "knee_width_ratio",
            "hem_width_ratio",
            "hem_to_thigh_ratio",
            "lower_leg_fullness",
            "lower_width_slope",
            "geometry_quality",
        ],
        "rename": {
            "thigh_width_ratio": "trouser_thigh_width_ratio",
            "knee_width_ratio": "trouser_knee_width_ratio",
            "hem_width_ratio": "trouser_hem_width_ratio",
            "hem_to_thigh_ratio": "trouser_hem_to_thigh_ratio",
            "lower_leg_fullness": "trouser_lower_leg_fullness",
            "lower_width_slope": "trouser_lower_width_slope",
            "geometry_quality": "trouser_geometry_quality",
        },
    },
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_value(value: str | None):
    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    lower = value.lower()

    if lower == "true":
        return True

    if lower == "false":
        return False

    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    merged: dict[tuple[str, str], dict] = {}
    coverage_rows: list[dict] = []
    all_feature_columns: set[str] = set()

    for source in SOURCES:
        name = source["name"]
        path: Path = source["path"]

        coverage = {
            "source": name,
            "relative_path": str(path.relative_to(PROJECT_ROOT)),
            "exists": path.exists(),
            "rows": 0,
            "unique_garments": 0,
            "duplicates_removed": 0,
            "status": "",
        }

        if not path.exists():
            coverage["status"] = "missing"
            coverage_rows.append(coverage)
            continue

        rows = read_csv_rows(path)
        coverage["rows"] = len(rows)

        if not rows:
            coverage["status"] = "empty"
            coverage_rows.append(coverage)
            continue

        columns = set(rows[0].keys())
        missing_keys = [key for key in KEYS if key not in columns]

        if missing_keys:
            coverage["status"] = (
                "skipped_missing_keys:" + ",".join(missing_keys)
            )
            coverage_rows.append(coverage)
            continue

        seen = set()
        unique_count = 0
        duplicate_count = 0

        for raw in rows:
            source_image = (raw.get("source_image") or "").strip()
            garment_id = (raw.get("garment_id") or "").strip()

            if not source_image or not garment_id:
                continue

            key = (source_image, garment_id)

            if key in seen:
                duplicate_count += 1
                continue

            seen.add(key)
            unique_count += 1

            if key not in merged:
                merged[key] = {
                    "source_image": source_image,
                    "garment_id": garment_id,
                }

            target = merged[key]

            for col in source["keep"]:
                if col not in raw:
                    continue

                new_col = source["rename"].get(col, col)
                value = normalize_value(raw.get(col))

                if value is None:
                    continue

                if new_col == "garment_category":
                    if not target.get("garment_category"):
                        target["garment_category"] = value
                    continue

                # Avoid silently overwriting a feature from another source.
                final_col = new_col

                if final_col in target and target[final_col] != value:
                    final_col = f"{new_col}__{name}"

                target[final_col] = value
                all_feature_columns.add(final_col)

        coverage["unique_garments"] = unique_count
        coverage["duplicates_removed"] = duplicate_count
        coverage["status"] = "loaded"
        coverage_rows.append(coverage)

    if not merged:
        raise FileNotFoundError(
            "No retained style-feature result files were found.\n"
            "Please check reports/prd_attribute_extraction/."
        )

    rows_out = list(merged.values())

    # Stable deterministic order.
    rows_out.sort(
        key=lambda row: (
            str(row.get("source_image", "")),
            str(row.get("garment_id", "")),
        )
    )

    first_cols = [
        "source_image",
        "garment_id",
        "garment_category",
    ]

    feature_cols = sorted(
        col
        for col in all_feature_columns
        if col not in first_cols
    )

    fieldnames = first_cols + feature_cols

    csv_path = OUT / "style_feature_vector_v1.csv"

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows_out:
            writer.writerow(
                {
                    col: row.get(col, "")
                    for col in fieldnames
                }
            )

    quality_keys = {
        "sleeve_entropy",
        "pants_length_entropy",
        "lower_garment_length_entropy",
        "depth_entropy",
        "width_entropy",
        "collar_entropy",
        "upper_geometry_quality",
        "trouser_geometry_quality",
        "arm_contamination_ratio",
    }

    json_instances = []

    for row in rows_out:
        style_features = {}
        quality = {}

        for col, value in row.items():
            if col in first_cols:
                continue

            if col in quality_keys or "quality" in col or col.endswith("_entropy"):
                quality[col] = value
            else:
                style_features[col] = value

        json_instances.append(
            {
                "source_image": row.get("source_image"),
                "garment_id": row.get("garment_id"),
                "garment_category": row.get("garment_category"),
                "style_features": style_features,
                "quality": quality,
            }
        )

    json_path = OUT / "style_feature_vector_v1.json"

    json_path.write_text(
        json.dumps(
            {
                "schema_version": "3.1.3-style-v1",
                "instances": json_instances,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    coverage_path = OUT / "source_coverage.csv"

    coverage_fields = [
        "source",
        "relative_path",
        "exists",
        "rows",
        "unique_garments",
        "duplicates_removed",
        "status",
    ]

    with coverage_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=coverage_fields,
        )
        writer.writeheader()
        writer.writerows(coverage_rows)

    loaded = sum(
        row["status"] == "loaded"
        for row in coverage_rows
    )

    missing = sum(
        row["status"] == "missing"
        for row in coverage_rows
    )

    run_info = (
        "PRD 3.1.3 unified style feature vector v1\n"
        f"merged_instances={len(rows_out)}\n"
        f"loaded_sources={loaded}\n"
        f"missing_sources={missing}\n"
        "implementation=python_standard_library_only\n"
        "merge_key=source_image+garment_id\n"
        "hard_style_labels=false\n"
        "business_thresholds=false\n"
        "material_and_craftsmanship=excluded\n"
    )

    (OUT / "run_info.txt").write_text(
        run_info,
        encoding="utf-8",
    )

    print("=== STYLE FEATURE VECTOR V1 ===")
    print(f"merged instances : {len(rows_out)}")
    print(f"loaded sources   : {loaded}")
    print(f"missing sources  : {missing}")
    print("")
    print("Source coverage:")

    for row in coverage_rows:
        print(
            f"- {row['source']:<28} "
            f"{row['status']}"
        )

    print("")
    print(
        "CSV  : reports/prd_attribute_extraction/"
        "style_feature_vector_v1/style_feature_vector_v1.csv"
    )
    print(
        "JSON : reports/prd_attribute_extraction/"
        "style_feature_vector_v1/style_feature_vector_v1.json"
    )
    print(
        "COVER: reports/prd_attribute_extraction/"
        "style_feature_vector_v1/source_coverage.csv"
    )
    print("================================")


if __name__ == "__main__":
    main()
