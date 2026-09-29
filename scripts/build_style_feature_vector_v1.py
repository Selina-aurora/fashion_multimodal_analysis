"""Build PRD 3.1.3 unified continuous style feature vector v1.

This script merges the retained style-feature experiment outputs into one
garment-level table and one JSON representation.

Expected inputs are read from PROJECT_ROOT/reports/prd_attribute_extraction/.
Missing experiment files are allowed; the script reports them and builds the
vector from whichever retained sources are currently available.

Output
------
reports/prd_attribute_extraction/style_feature_vector_v1/
    style_feature_vector_v1.csv
    style_feature_vector_v1.json
    source_coverage.csv
    run_info.txt

Recommended schema location
---------------------------
configs/style_feature_schema_v1.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "reports" / "prd_attribute_extraction"
OUT = BASE / "style_feature_vector_v1"

KEYS = ["source_image", "garment_id"]

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


def safe_read_source(source: dict) -> tuple[pd.DataFrame | None, dict]:
    path: Path = source["path"]

    coverage = {
        "source": source["name"],
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "exists": path.exists(),
        "rows": 0,
        "unique_garments": 0,
        "status": "",
    }

    if not path.exists():
        coverage["status"] = "missing"
        return None, coverage

    df = pd.read_csv(path, encoding="utf-8-sig")

    missing_keys = [
        key for key in KEYS
        if key not in df.columns
    ]

    if missing_keys:
        coverage["status"] = (
            "skipped_missing_keys:"
            + ",".join(missing_keys)
        )
        return None, coverage

    requested = [
        col for col in source["keep"]
        if col in df.columns
    ]

    df = df[requested].copy()
    df = df.rename(columns=source["rename"])

    # Enforce one row per garment key. If duplicates exist, keep first and
    # report this in coverage rather than silently multiplying rows on merge.
    before = len(df)
    df = df.drop_duplicates(
        subset=KEYS,
        keep="first",
    )

    coverage["rows"] = before
    coverage["unique_garments"] = len(df)
    coverage["status"] = (
        "loaded"
        if before == len(df)
        else f"loaded_deduplicated_{before-len(df)}"
    )

    return df, coverage


def coalesce_category(df: pd.DataFrame) -> pd.DataFrame:
    category_cols = [
        col for col in df.columns
        if col == "garment_category"
        or col.startswith("garment_category__")
    ]

    if not category_cols:
        return df

    category = None

    for col in category_cols:
        if category is None:
            category = df[col]
        else:
            category = category.combine_first(
                df[col]
            )

    df["garment_category"] = category

    drop_cols = [
        col for col in category_cols
        if col != "garment_category"
    ]

    return df.drop(
        columns=drop_cols,
        errors="ignore",
    )


def merge_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = None
    coverage_rows = []

    for source in SOURCES:
        df, coverage = safe_read_source(
            source
        )

        coverage_rows.append(
            coverage
        )

        if df is None:
            continue

        if merged is None:
            merged = df
            continue

        overlap = [
            col for col in df.columns
            if col in merged.columns
            and col not in KEYS
        ]

        rename_overlap = {
            col: f"{col}__{source['name']}"
            for col in overlap
        }

        df = df.rename(
            columns=rename_overlap
        )

        merged = merged.merge(
            df,
            on=KEYS,
            how="outer",
            validate="one_to_one",
        )

    if merged is None:
        raise FileNotFoundError(
            "None of the retained style-feature result files were found.\n"
            "Run the feature experiments first."
        )

    merged = coalesce_category(
        merged
    )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    return merged, coverage_df


def clean_for_json(value):
    if pd.isna(value):
        return None

    if isinstance(value, (
        str,
        bool,
        int,
        float,
    )):
        return value

    return str(value)


def build_json_records(
    df: pd.DataFrame,
) -> list[dict]:
    meta_cols = {
        "source_image",
        "garment_id",
        "garment_category",
    }

    quality_cols = {
        col for col in df.columns
        if (
            "quality" in col
            or col.endswith("_entropy")
            or col in {
                "arm_contamination_ratio",
                "depth_entropy",
                "width_entropy",
                "collar_entropy",
            }
        )
    }

    records = []

    for _, row in df.iterrows():
        features = {}
        quality = {}

        for col in df.columns:
            if col in meta_cols:
                continue

            value = clean_for_json(
                row[col]
            )

            if value is None:
                continue

            if col in quality_cols:
                quality[col] = value
            else:
                features[col] = value

        records.append(
            {
                "source_image": clean_for_json(
                    row.get("source_image")
                ),
                "garment_id": clean_for_json(
                    row.get("garment_id")
                ),
                "garment_category": clean_for_json(
                    row.get("garment_category")
                ),
                "style_features": features,
                "quality": quality,
            }
        )

    return records


def main() -> None:
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged, coverage = merge_sources()

    # Stable, readable column ordering.
    first = [
        col for col in [
            "source_image",
            "garment_id",
            "garment_category",
        ]
        if col in merged.columns
    ]

    rest = [
        col for col in merged.columns
        if col not in first
    ]

    merged = merged[
        first + sorted(rest)
    ]

    csv_path = (
        OUT
        / "style_feature_vector_v1.csv"
    )

    json_path = (
        OUT
        / "style_feature_vector_v1.json"
    )

    coverage_path = (
        OUT
        / "source_coverage.csv"
    )

    merged.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    records = build_json_records(
        merged
    )

    json_path.write_text(
        json.dumps(
            {
                "schema_version": "3.1.3-style-v1",
                "instances": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    coverage.to_csv(
        coverage_path,
        index=False,
        encoding="utf-8-sig",
    )

    loaded = coverage[
        coverage["status"].str.startswith(
            "loaded"
        )
    ]

    missing = coverage[
        coverage["status"] == "missing"
    ]

    run_info = (
        "PRD 3.1.3 unified style feature vector v1\n"
        f"merged_instances={len(merged)}\n"
        f"loaded_sources={len(loaded)}\n"
        f"missing_sources={len(missing)}\n"
        "merge_key=source_image+garment_id\n"
        "hard_style_labels=false\n"
        "business_thresholds=false\n"
        "material_and_craftsmanship=excluded\n"
    )

    (
        OUT / "run_info.txt"
    ).write_text(
        run_info,
        encoding="utf-8",
    )

    print("=== STYLE FEATURE VECTOR V1 ===")
    print(f"merged instances : {len(merged)}")
    print(f"loaded sources   : {len(loaded)}")
    print(f"missing sources  : {len(missing)}")
    print("")
    print("Source coverage:")
    for _, row in coverage.iterrows():
        print(
            f"- {row['source']:<28} "
            f"{row['status']}"
        )
    print("")
    print(
        "CSV  : "
        "reports/prd_attribute_extraction/style_feature_vector_v1/"
        "style_feature_vector_v1.csv"
    )
    print(
        "JSON : "
        "reports/prd_attribute_extraction/style_feature_vector_v1/"
        "style_feature_vector_v1.json"
    )
    print(
        "COVER: "
        "reports/prd_attribute_extraction/style_feature_vector_v1/"
        "source_coverage.csv"
    )
    print("================================")


if __name__ == "__main__":
    main()
