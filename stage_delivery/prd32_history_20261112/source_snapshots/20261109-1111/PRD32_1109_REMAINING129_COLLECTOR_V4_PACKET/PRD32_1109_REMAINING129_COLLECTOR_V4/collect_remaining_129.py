#!/usr/bin/env python3
"""Read-only DEV60 marked-image collector with log SHA-256 verification.

This script accesses only two user-specified DEV baseline paths, no file tree scans,
no model calls, no holdout, and refuses to overwrite outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

BASE_ROOT = Path('/workspace/PRD32_DEV60_QWEN_BASELINE_20261030_run01')
DEFAULT_MARKED = BASE_ROOT / 'marked_inputs'
DEFAULT_LOG = BASE_ROOT / 'qwen_results.jsonl'
FORBIDDEN = ('holdout', 'sealed', 'blind', 'independent_test')
TASKS = frozenset(('tags', 'vqa', 'copy'))
PRIO = {'P1_BLOCKED_UNREVIEWED': 0, 'P2_DRIFT_UNREVIEWED': 1, 'P3_OTHER_UNREVIEWED': 2}


def safe_path(path: Path, root: Path, file_expected: bool = True) -> Path:
    p = path.resolve(strict=True)
    root = root.resolve(strict=True)
    if any(t in str(p).lower() for t in FORBIDDEN):
        raise ValueError('Forbidden Holdout-like path: ' + str(p))
    if not p.is_relative_to(root):
        raise ValueError(f'Path escapes allowed root: {p}, root {root}')
    if file_expected and not p.is_file():
        raise ValueError('Not a file: ' + str(p))
    return p


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as fp:
        return list(csv.DictReader(fp))


def csv_blob(rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()), extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode('utf-8-sig')


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validated_rows(remaining: Path, worklist: Path, full_status: Path) -> list[dict[str, str]]:
    rem, work, full = csv_rows(remaining), csv_rows(worklist), csv_rows(full_status)
    if len(rem) != 129 or len(work) != 171 or len(full) != 171:
        raise ValueError('Unexpected DEV60 snapshot sizes; expected remaining=129, priority=171, status=171')

    def idx(rows: list[dict[str, str]], label: str) -> dict[tuple[str, str], dict[str, str]]:
        out = {}
        for r in rows:
            k = (r['case_id'], r['instance_id'])
            if k in out:
                raise ValueError(f'Duplicate instance in {label}: {k}')
            out[k] = r
        return out

    a, b, c = idx(rem, 'remaining'), idx(work, 'worklist'), idx(full, 'full status')
    if set(b) != set(c) or not set(a).issubset(b) or len(set(b)-set(a)) != 42:
        raise ValueError('Review snapshots disagree with expected DEV60 171/42/129 partition')
    for k in a:
        r, w, f = a[k], b[k], c[k]
        sha = r['image_sha256'].strip().lower()
        if any(other['image_sha256'].strip().lower() != sha for other in (w, f)):
            raise ValueError('Image digest disagreement for ' + str(k))
        if r['category_GT_status'] != 'UNREVIEWED' or f['category_GT_status'] != 'UNREVIEWED':
            raise ValueError('A confirmed instance was included: ' + str(k))
        if r['NEW_REVIEWER_CONFIRMED_YES_NO_TO_FILL'].upper() != 'NO':
            raise ValueError('Confirmed review was included: ' + str(k))
        if w['review_priority'] not in PRIO:
            raise ValueError('Unexpected priority: ' + str(k))
        r['review_priority'] = w['review_priority']
        r['frozen_guard_v5_policy'] = w['frozen_guard_v5_policy']
        r['baseline_focus_class_disagree'] = w['baseline_focus_class_disagree']
        r['focus_v1_category'] = w['focus_v1_category']
    for k in set(b)-set(a):
        if c[k].get('NEW_REVIEWER_CONFIRMED_YES_NO_TO_FILL', '').upper() != 'YES':
            raise ValueError('Completed partition has no user confirmation: ' + str(k))
    return sorted(a.values(), key=lambda r:(PRIO[r['review_priority']], r['case_id'], r['instance_id']))


def baseline_records(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    tasks: dict[tuple[str,str], dict[str, dict]] = defaultdict(dict)
    seen_request = set()
    with path.open('r', encoding='utf-8') as fp:
        for number, raw in enumerate(fp, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid Qwen JSONL line {number}: {e}') from e
            if row.get('model_variant') != 'baseline':
                raise ValueError(f'Unexpected model variant on line {number}')
            case, inst, task = row.get('case_id'), row.get('target_instance_id'), row.get('task')
            if not isinstance(case, str) or not isinstance(inst, str) or task not in TASKS:
                raise ValueError(f'Invalid case/instance/task on line {number}')
            key = (case, inst)
            request = row.get('request_id')
            if not request or request in seen_request:
                raise ValueError(f'Duplicate/missing Qwen request_id: {request}')
            seen_request.add(request)
            if task in tasks[key]:
                raise ValueError(f'Duplicate Qwen task {key}/{task}')
            tasks[key][task] = row
    if len(tasks) != 171 or sum(len(x) for x in tasks.values()) != 513:
        raise ValueError(f'Qwen log is not full DEV60: {len(tasks)} instances/{sum(len(x) for x in tasks.values())} tasks')
    joined = {}
    for key, trip in tasks.items():
        if set(trip) != TASKS:
            raise ValueError('Incomplete Qwen task triple: ' + str(key))
        fields = ('image_sha256', 'marked_image', 'marked_image_sha256')
        for field in fields:
            values = {str(row.get(field,'')) for row in trip.values()}
            if len(values) != 1 or not next(iter(values)):
                raise ValueError(f'Inconsistent {field} for {key}')
        first = trip['tags']
        joined[key] = {
            'image_sha256': first['image_sha256'].lower(),
            'marked_image': first['marked_image'],
            'marked_image_sha256': first['marked_image_sha256'].lower(),
            'task_request_ids': {t: trip[t]['request_id'] for t in sorted(TASKS)},
        }
    return joined


def collect(*, remaining: Path, worklist: Path, full_status: Path,
            marked_root: Path, baseline_log: Path, output: Path,
            count: int = 129) -> dict:
    marked_root = marked_root.resolve(strict=True)
    baseline_log = safe_path(baseline_log, marked_root.parent)
    if marked_root != DEFAULT_MARKED or baseline_log != DEFAULT_LOG:
        # Alternative locations only when explicitly selected and under the same DEV baseline root.
        if not (marked_root.is_relative_to(BASE_ROOT.resolve(strict=True)) and
                baseline_log.is_relative_to(BASE_ROOT.resolve(strict=True))):
            raise ValueError('Source paths must be within DEV60 baseline experiment root')
    if output.exists():
        raise FileExistsError(f'Refuse to overwrite existing output: {output}')

    rows = validated_rows(remaining, worklist, full_status)
    if count not in (30,129):
        raise ValueError('Use --count 30 for first batch or 129 for all unreviewed instances')
    selected = rows[:count]
    recs = baseline_records(baseline_log)
    manifest = []
    images = []
    for r in selected:
        key = (r['case_id'], r['instance_id'])
        rec = recs.get(key)
        if rec is None or rec['image_sha256'] != r['image_sha256'].lower():
            raise ValueError('Baseline record image identity mismatch: ' + str(key))
        recorded = Path(r['marked_baseline_image_path'])
        p = safe_path(recorded, marked_root)
        if p != safe_path(Path(rec['marked_image']), marked_root):
            raise ValueError('Marked image path mismatch: ' + str(key))
        if p.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
            raise ValueError('Unsupported image extension: ' + str(p))
        raw = p.read_bytes()
        digest = sha256_bytes(raw)
        if digest != rec['marked_image_sha256']:
            raise ValueError('Marked image SHA-256 mismatch: ' + str(key))
        target = f'images/{key[0]}__{key[1]}{p.suffix.lower()}'
        r['evidence_image_in_zip'] = target
        manifest.append({
            'case_id': key[0], 'instance_id': key[1],
            'original_image_sha256': r['image_sha256'].lower(),
            'marked_image_sha256': digest,
            'marked_image_in_zip': target,
            'byte_size': len(raw),
            'request_ids': rec['task_request_ids'],
            'review_priority': r['review_priority'],
        })
        images.append((target, raw))

    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, mode='x', compression=ZIP_DEFLATED, compresslevel=6) as z:
        for name, raw in images:
            z.writestr(name, raw)
        z.writestr('review_queue_selected.csv', csv_blob(selected))
        z.writestr('first_30_review_queue.csv', csv_blob(rows[:30]))
        z.writestr('marked_image_manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False).encode())
        z.writestr('collection_summary.json', json.dumps({
            'status': 'DEV60_MARKED_EVIDENCE_HASH_VERIFIED_NOT_GT',
            'collected': count, 'remaining_total': 129,
            'batch1_count': 30, 'dev60_total': 171,
            'user_confirmed_before_collection': 42,
            'count_by_priority': dict(Counter(r['review_priority'] for r in selected)),
            'holdout_accessed': False,
            'no_model_calls': True, 'no_gt_created': True,
            'output_dir_unchanged': True,
        }, indent=2, ensure_ascii=False).encode())
        z.writestr('README_SCOPE.txt', (
            'DEV60 Baseline marked images ONLY. Not originals or masks; image rectangle does NOT prove mask correctness.\n'
            'Images are hash-validated against three records in the original baseline Qwen JSONL.\n'
            'Prior 42 user-confirmed instances excluded. Remaining entries need AI pre-review and user confirmation.\n'
            'Not full PRD acceptance, no production release and no Holdout.\n').encode())
    with ZipFile(output, 'r') as z:
        if z.testzip() is not None:
            raise RuntimeError('ZIP CRC failure')
        for m in manifest:
            assert sha256_bytes(z.read(m['marked_image_in_zip'])) == m['marked_image_sha256']
    return {'status':'COLLECTED_AND_HASH_VERIFIED',
            'selected_count':len(selected), 'unreviewed_total':len(rows),
            'priority_counts':dict(Counter(r['review_priority'] for r in selected)),
            'zip':str(output), 'zip_bytes':output.stat().st_size,
            'holdout_accessed':False, 'model_calls':0}


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, choices=(30,129), default=129)
    p.add_argument('--remaining', type=Path, default=here/'remaining_129_review_queue.csv')
    p.add_argument('--worklist', type=Path, default=here/'priority_worklist_171.csv')
    p.add_argument('--full-status', type=Path, default=here/'dev60_171_review_status.csv')
    p.add_argument('--marked-root', type=Path, default=DEFAULT_MARKED)
    p.add_argument('--baseline-log', type=Path, default=DEFAULT_LOG)
    p.add_argument('--out', type=Path, default=Path('/workspace/PRD32_1109_REMAINING129_MARKED_EVIDENCE.zip'))
    a = p.parse_args()
    print(json.dumps(collect(remaining=a.remaining, worklist=a.worklist,
          full_status=a.full_status,marked_root=a.marked_root,
          baseline_log=a.baseline_log, output=a.out, count=a.count),
          ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
