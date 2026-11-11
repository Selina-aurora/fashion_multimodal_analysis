#!/usr/bin/env python3
"""Read-only DEV60 P1 evidence pack collector, standard library only.

Does not scan outside the explicit baseline marked_inputs directory.
Does not access holdout or run models. Outputs a new ZIP, never overwrites.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

BASE = Path('/workspace/PRD32_DEV60_QWEN_BASELINE_20261030_run01/marked_inputs')
FORBIDDEN = ('holdout', 'sealed', 'blind', 'independent_test')
FIELDS = ('case_id', 'instance_id', 'image_sha256', 'marked_baseline_image_path', 'review_priority')


def collect(worklist: Path, destination: Path, count: int) -> dict:
    root = BASE.resolve(strict=True)
    if any(t in str(root).lower() for t in FORBIDDEN):
        raise RuntimeError('Forbidden root')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    with worklist.open(encoding='utf-8-sig', newline='') as fp:
        reader = csv.DictReader(fp)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise RuntimeError('Missing required worklist fields')
        rows = [x for x in reader if x['review_priority'] == 'P1_BLOCKED_UNREVIEWED'][:count]
    if len(rows) != count:
        raise RuntimeError(f'Expected {count} P1 rows, found {len(rows)}')
    images = []
    names = set()
    for r in rows:
        p = Path(r['marked_baseline_image_path']).resolve(strict=True)
        if not p.is_file() or not p.is_relative_to(root):
            raise RuntimeError(f'Unsafe image path for {r["case_id"]}/{r["instance_id"]}: {p}')
        if any(w in str(p).lower() for w in FORBIDDEN):
            raise RuntimeError(f'Forbidden image path: {p}')
        suffix = p.suffix.lower()
        if suffix not in ('.jpg', '.jpeg', '.png', '.webp'):
            raise RuntimeError(f'Unexpected image format {suffix}')
        safe_id = f"{r['case_id']}__{r['instance_id']}"
        if safe_id in names:
            raise RuntimeError(f'Duplicate ID {safe_id}')
        names.add(safe_id)
        arcname = f'images/{safe_id}{suffix}'
        r['evidence_image_in_zip'] = arcname
        images.append((p, arcname, r))
    manifest = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, 'x', compression=ZIP_DEFLATED, compresslevel=6) as z:
        for p, arcname, r in images:
            blob = p.read_bytes()
            h = hashlib.sha256(blob).hexdigest()
            z.writestr(arcname, blob)
            manifest.append({'case_id': r['case_id'], 'instance_id': r['instance_id'],
                             'image_zip_path': arcname, 'image_sha256': h, 'size_bytes': len(blob)})
        b = io.StringIO(newline='')
        fields = list(rows[0].keys())
        writer = csv.DictWriter(b, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        z.writestr('review_batch_30.csv', b.getvalue().encode('utf-8-sig'))
        z.writestr('evidence_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr('README.txt', '30 DEV60 P1 marked images for AI PRE-REVIEW only. Not GT. No holdout. Do not publish.\n')
    with ZipFile(destination) as z:
        for m in manifest:
            assert hashlib.sha256(z.read(m['image_zip_path'])).hexdigest() == m['image_sha256']
    return {'output': str(destination), 'p1_images': len(images), 'missing': 0, 'bytes': destination.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worklist', type=Path, default=Path(__file__).with_name('review_worklist_171.csv'))
    parser.add_argument('--output', type=Path, default=Path('/workspace/PRD32_1109_P1_30_MARKED_EVIDENCE.zip'))
    parser.add_argument('--count', type=int, default=30)
    a = parser.parse_args()
    if a.count < 1 or a.count > 84:
        parser.error('--count must be 1..84')
    print(json.dumps(collect(a.worklist, a.output, a.count), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
