"""
Audit the EXACT Fashionpedia -> PRD class mapping already used by this project.

Why
---
Before building the blind benchmark, we must keep the same class semantics
used by the existing 8-class training/validation data. This script does not
guess the mapping.

It reconstructs the mapping from:
1) existing configs/*.csv rows that reference Fashionpedia;
2) the raw Fashionpedia COCO-like annotation JSON;
3) existing project Python code containing Fashionpedia mapping logic.

Outputs
-------
benchmark/prd_3_1_v1/audit/fashionpedia_mapping_v1/
├── fashionpedia_used_annotation_mapping.csv
├── fashionpedia_mapping_summary.csv
├── code_mapping_hits.txt
└── fashionpedia_mapping_audit.txt

Run
---
cd /workspace/fashion_multimodal_analysis

python scripts/audit_fashionpedia_prd_mapping_v1.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
DATA_ROOT = PROJECT_ROOT.parent / "fashion_data"
FASHIONPEDIA_ROOT = DATA_ROOT / "raw" / "fashionpedia"

OUT_DIR = (
    PROJECT_ROOT
    / "benchmark"
    / "prd_3_1_v1"
    / "audit"
    / "fashionpedia_mapping_v1"
)

ANN_ID_RE = re.compile(r"(?:fashionpedia[_-]?)?ann[_-]?(\d+)", re.I)

SEARCH_TERMS = (
    "fashionpedia",
    "accessory",
    "shoe",
    "bag",
    "wallet",
    "belt",
)

PRD_CLASSES = {
    "top",
    "pants",
    "skirt",
    "outerwear",
    "dress",
    "shoe",
    "bag",
    "accessory",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def norm(v: Any) -> str:
    return str(v if v is not None else "").replace("\\", "/").strip()


def first(row: dict[str, str], names: tuple[str, ...]) -> str:
    for n in names:
        v = norm(row.get(n, ""))
        if v:
            return v
    return ""


def find_coco_json() -> Path:
    candidates = []

    for p in FASHIONPEDIA_ROOT.rglob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if (
            isinstance(data, dict)
            and isinstance(data.get("images"), list)
            and isinstance(data.get("annotations"), list)
            and isinstance(data.get("categories"), list)
        ):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No COCO-like Fashionpedia JSON found under {FASHIONPEDIA_ROOT}"
        )

    # Prefer the file with the most annotations.
    best = None
    best_n = -1
    for p in candidates:
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        n = len(d.get("annotations", []))
        if n > best_n:
            best = p
            best_n = n

    return best


def extract_ann_id(garment_id: str) -> int | None:
    m = ANN_ID_RE.search(str(garment_id))
    return int(m.group(1)) if m else None


def code_hits() -> str:
    blocks = []

    files = sorted(SCRIPT_DIR.glob("*.py"))

    for p in files:
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        hit_lines = set()

        for i, line in enumerate(lines):
            low = line.lower()
            if "fashionpedia" in low:
                for j in range(max(0, i - 5), min(len(lines), i + 10)):
                    hit_lines.add(j)

        # Also include likely mapping lines in files already containing Fashionpedia.
        if hit_lines:
            for i, line in enumerate(lines):
                low = line.lower()
                if any(term in low for term in SEARCH_TERMS[1:]):
                    for j in range(max(0, i - 2), min(len(lines), i + 3)):
                        hit_lines.add(j)

        if not hit_lines:
            continue

        blocks.append(
            f"\n===== {p.relative_to(PROJECT_ROOT)} ====="
        )

        previous = None
        for i in sorted(hit_lines):
            if previous is not None and i > previous + 1:
                blocks.append("...")
            blocks.append(f"{i+1:04d}: {lines[i]}")
            previous = i

    if not blocks:
        return "No Fashionpedia-related code hits found under scripts/.\n"

    return "\n".join(blocks) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    coco_path = find_coco_json()

    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {
        int(c["id"]): {
            "fashionpedia_category_name": norm(c.get("name", "")),
            "fashionpedia_supercategory": norm(c.get("supercategory", "")),
        }
        for c in coco["categories"]
    }

    ann_by_id = {
        int(a["id"]): a
        for a in coco["annotations"]
        if "id" in a
    }

    config_rows = []
    config_files_with_fp = set()
    unresolved = []

    for cfg in sorted(CONFIG_DIR.glob("*.csv")):
        try:
            rows = read_csv(cfg)
        except Exception:
            continue

        for row in rows:
            source_image = norm(row.get("source_image", ""))
            source_dataset = first(
                row,
                ("source_dataset", "dataset"),
            )

            if (
                "fashionpedia" not in source_image.lower()
                and "fashionpedia" not in source_dataset.lower()
            ):
                continue

            config_files_with_fp.add(
                str(cfg.relative_to(PROJECT_ROOT))
            )

            garment_id = norm(row.get("garment_id", ""))
            prd_class = first(
                row,
                ("garment_category", "category", "class_name"),
            ).lower()

            ann_id = extract_ann_id(garment_id)

            if ann_id is None or ann_id not in ann_by_id:
                unresolved.append({
                    "config_file": str(cfg.relative_to(PROJECT_ROOT)),
                    "source_image": source_image,
                    "garment_id": garment_id,
                    "prd_class": prd_class,
                    "reason": (
                        "annotation id not parseable from garment_id"
                        if ann_id is None
                        else f"annotation id {ann_id} not found in raw JSON"
                    ),
                })
                continue

            ann = ann_by_id[ann_id]
            cid = int(ann["category_id"])
            cat = categories.get(
                cid,
                {
                    "fashionpedia_category_name": "",
                    "fashionpedia_supercategory": "",
                },
            )

            config_rows.append({
                "config_file": str(cfg.relative_to(PROJECT_ROOT)),
                "source_image": source_image,
                "garment_id": garment_id,
                "prd_class": prd_class,
                "fashionpedia_annotation_id": ann_id,
                "fashionpedia_category_id": cid,
                "fashionpedia_category_name": cat[
                    "fashionpedia_category_name"
                ],
                "fashionpedia_supercategory": cat[
                    "fashionpedia_supercategory"
                ],
            })

    # Deduplicate the same garment repeated across several configs.
    dedup = {}
    for r in config_rows:
        key = (
            r["source_image"],
            r["garment_id"],
            r["prd_class"],
            r["fashionpedia_category_id"],
        )
        if key not in dedup:
            dedup[key] = dict(r)
            dedup[key]["source_configs"] = [r["config_file"]]
        else:
            dedup[key]["source_configs"].append(r["config_file"])

    used_rows = []
    for r in dedup.values():
        row = dict(r)
        row["source_configs"] = " | ".join(
            sorted(set(row.pop("source_configs")))
        )
        used_rows.append(row)

    used_rows.sort(
        key=lambda r: (
            r["prd_class"],
            int(r["fashionpedia_category_id"]),
            r["garment_id"],
        )
    )

    write_csv(
        OUT_DIR / "fashionpedia_used_annotation_mapping.csv",
        used_rows,
    )

    # Summary: empirical mapping actually used in existing data.
    grouped = defaultdict(Counter)
    source_examples = defaultdict(list)

    for r in used_rows:
        key = (
            int(r["fashionpedia_category_id"]),
            r["fashionpedia_category_name"],
            r["fashionpedia_supercategory"],
        )
        grouped[r["prd_class"]][key] += 1

        ekey = (r["prd_class"], key)
        if len(source_examples[ekey]) < 5:
            source_examples[ekey].append(r["garment_id"])

    summary_rows = []

    for prd_class in sorted(grouped):
        for (
            cid,
            name,
            supercat,
        ), n in grouped[prd_class].most_common():
            summary_rows.append({
                "prd_class": prd_class,
                "fashionpedia_category_id": cid,
                "fashionpedia_category_name": name,
                "fashionpedia_supercategory": supercat,
                "unique_used_garment_count": n,
                "example_garment_ids": " | ".join(
                    source_examples[
                        (
                            prd_class,
                            (cid, name, supercat),
                        )
                    ]
                ),
            })

    write_csv(
        OUT_DIR / "fashionpedia_mapping_summary.csv",
        summary_rows,
    )

    if unresolved:
        write_csv(
            OUT_DIR / "unresolved_fashionpedia_rows.csv",
            unresolved,
        )

    code_text = code_hits()
    (OUT_DIR / "code_mapping_hits.txt").write_text(
        code_text,
        encoding="utf-8",
    )

    target_summary = [
        r
        for r in summary_rows
        if r["prd_class"] in {"shoe", "bag", "accessory"}
    ]

    lines = [
        "PRD 3.1 Fashionpedia -> PRD Mapping Audit v1",
        "===========================================",
        "",
        f"raw_annotation_json={coco_path}",
        f"configs_with_fashionpedia_rows={len(config_files_with_fp)}",
        f"resolved_unique_fashionpedia_garments={len(used_rows)}",
        f"unresolved_rows={len(unresolved)}",
        "",
        "Empirical mapping used by existing project data",
        "-----------------------------------------------",
    ]

    if target_summary:
        for r in target_summary:
            lines.append(
                f"{r['prd_class']} <- "
                f"Fashionpedia id={r['fashionpedia_category_id']} "
                f"name={r['fashionpedia_category_name']} "
                f"supercategory={r['fashionpedia_supercategory']} "
                f"used_unique={r['unique_used_garment_count']}"
            )
    else:
        lines.append(
            "No resolved shoe/bag/accessory mapping found in existing configs."
        )

    lines += [
        "",
        "Consistency check",
        "-----------------",
    ]

    # Detect one PRD class mapping to multiple Fashionpedia categories,
    # and one Fashionpedia category mapping to multiple PRD classes.
    by_prd = defaultdict(set)
    by_fp = defaultdict(set)

    for r in summary_rows:
        by_prd[r["prd_class"]].add(
            (
                r["fashionpedia_category_id"],
                r["fashionpedia_category_name"],
            )
        )
        by_fp[
            (
                r["fashionpedia_category_id"],
                r["fashionpedia_category_name"],
            )
        ].add(r["prd_class"])

    for cls in ["shoe", "bag", "accessory"]:
        vals = by_prd.get(cls, set())
        lines.append(
            f"{cls}_fashionpedia_category_count={len(vals)}"
        )

    conflicts = [
        (fp, classes)
        for fp, classes in by_fp.items()
        if len(classes) > 1
    ]

    lines.append(
        f"fashionpedia_categories_mapped_to_multiple_prd_classes={len(conflicts)}"
    )

    if conflicts:
        for (cid, name), classes in conflicts:
            lines.append(
                f"CONFLICT id={cid} name={name}: {sorted(classes)}"
            )

    lines += [
        "",
        "Files",
        "-----",
        "- fashionpedia_used_annotation_mapping.csv: one row per previously used unique Fashionpedia garment.",
        "- fashionpedia_mapping_summary.csv: empirical source-category -> PRD-class mapping.",
        "- code_mapping_hits.txt: mapping-related code context from existing scripts.",
        "- unresolved_fashionpedia_rows.csv: only created if any rows cannot be reconstructed.",
        "",
        "Do not freeze benchmark mapping until this audit is reviewed.",
    ]

    (
        OUT_DIR / "fashionpedia_mapping_audit.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("FINISHED")
    for p in [
        OUT_DIR / "fashionpedia_mapping_audit.txt",
        OUT_DIR / "fashionpedia_mapping_summary.csv",
        OUT_DIR / "fashionpedia_used_annotation_mapping.csv",
        OUT_DIR / "code_mapping_hits.txt",
    ]:
        print(p.relative_to(PROJECT_ROOT))

    if unresolved:
        print(
            (OUT_DIR / "unresolved_fashionpedia_rows.csv").relative_to(
                PROJECT_ROOT
            )
        )


if __name__ == "__main__":
    main()
