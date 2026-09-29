"""Apply validity/quality gating to PRD 3.1.3 style feature vector v1.

Default project paths:
  input CSV:
    reports/prd_attribute_extraction/style_feature_vector_v1/style_feature_vector_v1.csv
  input JSON:
    reports/prd_attribute_extraction/style_feature_vector_v1/style_feature_vector_v1.json

Outputs:
    style_feature_vector_v1_gated.csv
    style_feature_vector_v1_gated.json
    validity_gating_summary_v1.txt

Policy:
- upper_geometry_quality == weak:
    null/blank upper geometry features.
- trouser_geometry_quality == weak:
    null/blank trouser geometry features.
- good/usable:
    retain values and add a validity flag.
- semantic entropy:
    retained as metadata only; no threshold gating is applied because no
    calibrated entropy threshold has been established.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "style_feature_vector_v1"
)

CSV_IN = BASE / "style_feature_vector_v1.csv"
JSON_IN = BASE / "style_feature_vector_v1.json"

CSV_OUT = BASE / "style_feature_vector_v1_gated.csv"
JSON_OUT = BASE / "style_feature_vector_v1_gated.json"
SUMMARY_OUT = BASE / "validity_gating_summary_v1.txt"

UPPER_GEOMETRY_FEATURES = [
    "garment_length_proxy",
    "waist_to_chest_ratio",
    "hem_to_chest_ratio",
    "lower_taper_slope",
]

TROUSER_GEOMETRY_FEATURES = [
    "trouser_thigh_width_ratio",
    "trouser_knee_width_ratio",
    "trouser_hem_width_ratio",
    "trouser_hem_to_thigh_ratio",
    "trouser_lower_leg_fullness",
    "trouser_lower_width_slope",
]


def gate_csv() -> tuple[int, int, int, int, Counter, Counter]:
    with CSV_IN.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        original_fields = list(rows[0].keys())

    fieldnames = original_fields.copy()

    for col in [
        "upper_geometry_valid",
        "trouser_geometry_valid",
        "validity_gating_applied",
    ]:
        if col not in fieldnames:
            fieldnames.append(col)

    upper_gated = 0
    trouser_gated = 0
    upper_nulled = 0
    trouser_nulled = 0

    upper_counts = Counter()
    trouser_counts = Counter()

    for row in rows:
        reasons = []

        upper_q = (
            row.get("upper_geometry_quality", "")
            .strip()
            .lower()
        )

        trouser_q = (
            row.get("trouser_geometry_quality", "")
            .strip()
            .lower()
        )

        upper_counts[upper_q] += 1
        trouser_counts[trouser_q] += 1

        if upper_q:
            row["upper_geometry_valid"] = (
                "false" if upper_q == "weak" else "true"
            )
        else:
            row["upper_geometry_valid"] = ""

        if trouser_q:
            row["trouser_geometry_valid"] = (
                "false" if trouser_q == "weak" else "true"
            )
        else:
            row["trouser_geometry_valid"] = ""

        if upper_q == "weak":
            upper_gated += 1
            reasons.append("upper_geometry")

            for feature in UPPER_GEOMETRY_FEATURES:
                if row.get(feature, "").strip():
                    upper_nulled += 1
                row[feature] = ""

        if trouser_q == "weak":
            trouser_gated += 1
            reasons.append("trouser_geometry")

            for feature in TROUSER_GEOMETRY_FEATURES:
                if row.get(feature, "").strip():
                    trouser_nulled += 1
                row[feature] = ""

        row["validity_gating_applied"] = ";".join(reasons)

    with CSV_OUT.open(
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
        writer.writerows(rows)

    return (
        len(rows),
        upper_gated,
        trouser_gated,
        upper_nulled,
        trouser_nulled,
        upper_counts,
        trouser_counts,
    )


def gate_json() -> None:
    data = json.loads(
        JSON_IN.read_text(encoding="utf-8")
    )

    for inst in data.get("instances", []):
        features = inst.setdefault("style_features", {})
        quality = inst.setdefault("quality", {})

        reasons = []

        upper_q = str(
            quality.get("upper_geometry_quality", "") or ""
        ).lower()

        trouser_q = str(
            quality.get("trouser_geometry_quality", "") or ""
        ).lower()

        if upper_q:
            quality["upper_geometry_valid"] = (
                upper_q != "weak"
            )

        if trouser_q:
            quality["trouser_geometry_valid"] = (
                trouser_q != "weak"
            )

        if upper_q == "weak":
            reasons.append("upper_geometry")

            for feature in UPPER_GEOMETRY_FEATURES:
                if feature in features:
                    features[feature] = None

        if trouser_q == "weak":
            reasons.append("trouser_geometry")

            for feature in TROUSER_GEOMETRY_FEATURES:
                if feature in features:
                    features[feature] = None

        if reasons:
            quality["validity_gating_applied"] = reasons

    data["schema_version"] = "3.1.3-style-v1-gated"

    data["gating_policy"] = {
        "upper_geometry_quality": {
            "good": "retain",
            "usable": "retain_with_quality_flag",
            "weak": "set_upper_geometry_features_to_null",
            "features": UPPER_GEOMETRY_FEATURES,
        },
        "trouser_geometry_quality": {
            "good": "retain",
            "usable": "retain_with_quality_flag",
            "weak": "set_trouser_geometry_features_to_null",
            "features": TROUSER_GEOMETRY_FEATURES,
        },
        "semantic_entropy": (
            "retained_as_quality_metadata_only; "
            "no calibrated threshold established"
        ),
    }

    JSON_OUT.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    if not CSV_IN.exists():
        raise FileNotFoundError(CSV_IN)

    if not JSON_IN.exists():
        raise FileNotFoundError(JSON_IN)

    (
        n_rows,
        upper_gated,
        trouser_gated,
        upper_nulled,
        trouser_nulled,
        upper_counts,
        trouser_counts,
    ) = gate_csv()

    gate_json()

    summary = f"""PRD 3.1.3 style feature validity gating v1
==========================================
merged_instances={n_rows}

upper_good={upper_counts.get('good', 0)}
upper_usable={upper_counts.get('usable', 0)}
upper_weak={upper_counts.get('weak', 0)}
upper_not_applicable={upper_counts.get('', 0)}

trouser_good={trouser_counts.get('good', 0)}
trouser_usable={trouser_counts.get('usable', 0)}
trouser_weak={trouser_counts.get('weak', 0)}
trouser_not_applicable={trouser_counts.get('', 0)}

upper_instances_gated={upper_gated}
upper_feature_cells_nulled={upper_nulled}

trouser_instances_gated={trouser_gated}
trouser_feature_cells_nulled={trouser_nulled}

semantic_entropy_gating=false
business_thresholds=false
"""

    SUMMARY_OUT.write_text(
        summary,
        encoding="utf-8",
    )

    print("=== VALIDITY GATING V1 ===")
    print(f"instances              : {n_rows}")
    print(f"upper weak gated       : {upper_gated}")
    print(f"trouser weak gated     : {trouser_gated}")
    print(f"upper values nulled    : {upper_nulled}")
    print(f"trouser values nulled  : {trouser_nulled}")
    print("")
    print(
        "CSV  : reports/prd_attribute_extraction/"
        "style_feature_vector_v1/style_feature_vector_v1_gated.csv"
    )
    print(
        "JSON : reports/prd_attribute_extraction/"
        "style_feature_vector_v1/style_feature_vector_v1_gated.json"
    )
    print(
        "SUM  : reports/prd_attribute_extraction/"
        "style_feature_vector_v1/validity_gating_summary_v1.txt"
    )
    print("==========================")


if __name__ == "__main__":
    main()
