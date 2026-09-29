"""Repair PRD 3.1.3 design-label v1 outputs without rerunning CLIP.

This script fixes two integration issues in the first 9.17 run:
1. style_feature_vector_v1.json uses the root key "instances", so the first
   integration script failed to merge the existing continuous style features.
2. the first 40-row manual audit was not balanced across the four attributes.

It reuses the already-computed categorical predictions and only repairs the
integration / audit artifacts.

Run from the project root:
    python scripts/repair_design_labels_v1_outputs.py

Expected inputs:
    reports/prd_attribute_extraction/design_labels_v1/
        design_attribute_predictions_v1.json

    reports/prd_attribute_extraction/style_v1/
        style_feature_vector_v1.json

Outputs:
    reports/prd_attribute_extraction/design_labels_v1_fixed/
        design_attribute_predictions_v1_fixed.json
        design_attribute_predictions_v1_fixed.csv
        manual_audit_holdout_v1_fixed.csv
        summary_v1_fixed.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "design_labels_v1"
    / "design_attribute_predictions_v1.json"
)

DEFAULT_STYLE_JSON = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "style_v1"
    / "style_feature_vector_v1.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prd_attribute_extraction"
    / "design_labels_v1_fixed"
)

ATTRIBUTES = (
    "sleeve_length",
    "neckline",
    "silhouette_fit",
    "fashion_style",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDICTIONS.relative_to(PROJECT_ROOT)),
    )
    parser.add_argument(
        "--style-json",
        default=str(DEFAULT_STYLE_JSON.relative_to(PROJECT_ROOT)),
    )
    parser.add_argument("--audit-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    p = Path(raw).expanduser()

    candidates = [
        p,
        Path.cwd() / p,
        PROJECT_ROOT / p,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = [
        x.resolve()
        for x in PROJECT_ROOT.rglob(p.name)
        if x.is_file()
    ]
    if len(matches) == 1:
        print(
            f"[path fallback] {raw} -> "
            f"{matches[0].relative_to(PROJECT_ROOT)}"
        )
        return matches[0]

    raise FileNotFoundError(f"Could not resolve: {raw}")


def normalize_source_image(value: str) -> str:
    return str(value).replace("\\", "/").strip()


def read_style_index(path: Path) -> dict[tuple[str, str], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    # Current style_feature_vector_v1.json uses "instances".
    records = (
        payload.get("instances")
        or payload.get("garments")
        or payload.get("records")
        or []
    )

    if not isinstance(records, list):
        raise ValueError(
            "Style JSON must contain a list under "
            "'instances', 'garments', or 'records'."
        )

    index: dict[tuple[str, str], dict] = {}
    for item in records:
        key = (
            normalize_source_image(item.get("source_image", "")),
            str(item.get("garment_id", "")).strip(),
        )
        index[key] = item

    return index


def geometry_gate(style_item: dict) -> dict:
    quality = style_item.get(
        "quality",
        style_item.get("style_quality", {}),
    )

    upper_quality = quality.get("upper_geometry_quality")
    trouser_quality = quality.get("trouser_geometry_quality")

    return {
        "upper_geometry_quality": upper_quality,
        "upper_geometry_valid_for_downstream": (
            upper_quality in {"good", "usable"}
            if upper_quality is not None
            else None
        ),
        "trouser_geometry_quality": trouser_quality,
        "trouser_geometry_valid_for_downstream": (
            trouser_quality in {"good", "usable"}
            if trouser_quality is not None
            else None
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open(
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


def flatten_record(record: dict) -> dict:
    row = {
        "source_image": record["source_image"],
        "garment_id": record["garment_id"],
        "garment_category": record["garment_category"],
    }

    for attr_name in ATTRIBUTES:
        item = record["design_attributes"].get(attr_name)

        for field in [
            "label",
            "confidence",
            "top2_label",
            "top2_score",
            "margin",
            "entropy",
        ]:
            row[f"{attr_name}_{field}"] = (
                "" if item is None else item.get(field, "")
            )

    features = record.get(
        "continuous_style_features",
        {},
    )
    for key, value in features.items():
        row[key] = value

    gate = record.get("geometry_gate", {})
    row.update(gate)

    return row


def make_balanced_audit(
    records: list[dict],
    audit_size: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)

    pools: dict[str, list[dict]] = defaultdict(list)

    for record in records:
        for attr_name in ATTRIBUTES:
            item = record["design_attributes"].get(attr_name)
            if item is None:
                continue

            pools[attr_name].append(
                {
                    "source_image": record["source_image"],
                    "garment_id": record["garment_id"],
                    "garment_category": record["garment_category"],
                    "attribute": attr_name,
                    "predicted_label": item.get("label", ""),
                    "top1_score": item.get("confidence", ""),
                    "top2_label": item.get("top2_label", ""),
                    "top2_score": item.get("top2_score", ""),
                    "margin": item.get("margin", ""),
                    "entropy": item.get("entropy", ""),
                    "manual_label": "",
                    "manual_correct": "",
                    "ambiguous": "",
                    "error_type": "",
                    "notes": "",
                    "reviewer": "",
                }
            )

    available_attrs = [
        attr
        for attr in ATTRIBUTES
        if pools.get(attr)
    ]

    if not available_attrs:
        return []

    for attr in available_attrs:
        rng.shuffle(pools[attr])

    base = audit_size // len(available_attrs)
    remainder = audit_size % len(available_attrs)

    target = {
        attr: base + (1 if i < remainder else 0)
        for i, attr in enumerate(available_attrs)
    }

    selected: list[dict] = []

    for attr in available_attrs:
        take = min(target[attr], len(pools[attr]))
        selected.extend(pools[attr][:take])

    # In case one attribute did not have enough rows, fill from the others.
    if len(selected) < audit_size:
        used_keys = {
            (
                row["source_image"],
                row["garment_id"],
                row["attribute"],
            )
            for row in selected
        }

        remainder_pool: list[dict] = []
        for attr in available_attrs:
            for row in pools[attr]:
                key = (
                    row["source_image"],
                    row["garment_id"],
                    row["attribute"],
                )
                if key not in used_keys:
                    remainder_pool.append(row)

        rng.shuffle(remainder_pool)
        selected.extend(
            remainder_pool[: audit_size - len(selected)]
        )

    rng.shuffle(selected)
    return selected[:audit_size]


def derive_sleeve_weak_gt(garment_id: str) -> str | None:
    gid = garment_id.lower()

    if "vest" in gid or "sling" in gid:
        return "sleeveless"

    if "short_sleeve" in gid:
        return "short"

    if "long_sleeve" in gid:
        return "long"

    return None


def coarse_sleeve_prediction(label: str) -> str | None:
    if label == "sleeveless":
        return "sleeveless"

    if label in {"cap", "short"}:
        return "short"

    if label in {
        "elbow",
        "three_quarter",
        "long",
    }:
        return "long"

    return None


def main() -> None:
    args = parse_args()

    prediction_path = resolve_path(args.predictions)
    style_path = resolve_path(args.style_json)

    payload = json.loads(
        prediction_path.read_text(encoding="utf-8")
    )
    records = payload.get("records", [])

    if not isinstance(records, list) or not records:
        raise ValueError("Prediction JSON has no records.")

    style_index = read_style_index(style_path)

    matched = 0
    unmatched: list[tuple[str, str]] = []

    for record in records:
        source_image = normalize_source_image(
            record.get("source_image", "")
        )
        garment_id = str(
            record.get("garment_id", "")
        ).strip()

        record["source_image"] = source_image

        style_item = style_index.get(
            (source_image, garment_id)
        )

        if style_item is None:
            unmatched.append(
                (source_image, garment_id)
            )
            record[
                "continuous_style_features"
            ] = {}
            record["style_quality"] = {}
            record["geometry_gate"] = {}
            continue

        matched += 1

        record[
            "continuous_style_features"
        ] = style_item.get(
            "style_features",
            {},
        )

        record["style_quality"] = style_item.get(
            "quality",
            style_item.get(
                "style_quality",
                {},
            ),
        )

        record["geometry_gate"] = geometry_gate(
            style_item
        )

    if unmatched:
        sample = "\n".join(
            f"  - {src} | {gid}"
            for src, gid in unmatched[:10]
        )
        raise RuntimeError(
            "Style merge is incomplete.\n"
            f"matched={matched}/{len(records)}\n"
            f"unmatched={len(unmatched)}\n"
            f"Examples:\n{sample}"
        )

    payload[
        "schema_version"
    ] = "3.1.3-design-labels-v1-fixed"

    payload[
        "integration_note"
    ] = (
        "continuous style features repaired from "
        "style_feature_vector_v1.json; root key "
        "'instances' is supported"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fixed_json = (
        OUTPUT_DIR
        / "design_attribute_predictions_v1_fixed.json"
    )
    fixed_json.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    flat_rows = [
        flatten_record(record)
        for record in records
    ]
    write_csv(
        OUTPUT_DIR
        / "design_attribute_predictions_v1_fixed.csv",
        flat_rows,
    )

    audit_rows = make_balanced_audit(
        records,
        audit_size=args.audit_size,
        seed=args.seed,
    )
    write_csv(
        OUTPUT_DIR
        / "manual_audit_holdout_v1_fixed.csv",
        audit_rows,
    )

    # Weak-GT sleeve diagnostic from DeepFashion2 category names.
    sleeve_total = 0
    sleeve_correct = 0
    sleeve_confusion = Counter()

    for record in records:
        weak_gt = derive_sleeve_weak_gt(
            record["garment_id"]
        )
        item = record[
            "design_attributes"
        ].get("sleeve_length")

        if weak_gt is None or item is None:
            continue

        coarse_pred = coarse_sleeve_prediction(
            item.get("label", "")
        )
        if coarse_pred is None:
            continue

        sleeve_total += 1
        sleeve_correct += int(
            coarse_pred == weak_gt
        )
        sleeve_confusion[
            (weak_gt, coarse_pred)
        ] += 1

    audit_counts = Counter(
        row["attribute"]
        for row in audit_rows
    )

    lines = [
        "PRD 3.1.3 design attribute categorical baseline v1 — repaired integration",
        "=======================================================================",
        f"prediction_records={len(records)}",
        f"style_records={len(style_index)}",
        f"style_matches={matched}",
        f"style_unmatched={len(unmatched)}",
        f"audit_rows={len(audit_rows)}",
        f"audit_seed={args.seed}",
        "",
        "Balanced audit distribution",
        "---------------------------",
    ]

    for attr in ATTRIBUTES:
        lines.append(
            f"{attr}={audit_counts.get(attr, 0)}"
        )

    lines += [
        "",
        "Sleeve weak-GT diagnostic",
        "-------------------------",
        (
            "DeepFashion2 fine category names are used "
            "only as a coarse diagnostic, not as final "
            "PRD ground truth."
        ),
        f"diagnostic_rows={sleeve_total}",
        (
            "coarse_accuracy="
            f"{sleeve_correct / sleeve_total:.4f}"
            if sleeve_total
            else "coarse_accuracy=NA"
        ),
        "",
        "coarse_confusion (weak_gt -> prediction)",
    ]

    for (gt, pred), count in sorted(
        sleeve_confusion.items()
    ):
        lines.append(
            f"{gt} -> {pred}: {count}"
        )

    lines += [
        "",
        "Important",
        "---------",
        "- This repair does not rerun CLIP.",
        "- Existing categorical predictions are unchanged.",
        "- Continuous style features and quality fields are now restored.",
        "- Weak upper/trouser geometry remains preserved but is flagged for downstream gating.",
        "- The manual audit is now balanced across the four attributes.",
        "- Do not treat candidate-relative CLIP confidence as a calibrated probability.",
    ]

    (
        OUTPUT_DIR
        / "summary_v1_fixed.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n=== REPAIR COMPLETE ===")
    print(f"style matches: {matched}/{len(records)}")
    print(
        "audit distribution:",
        dict(audit_counts),
    )
    if sleeve_total:
        print(
            "sleeve coarse weak-GT accuracy:",
            f"{sleeve_correct / sleeve_total:.4f}",
        )

    print(
        "outputs:",
        OUTPUT_DIR.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
