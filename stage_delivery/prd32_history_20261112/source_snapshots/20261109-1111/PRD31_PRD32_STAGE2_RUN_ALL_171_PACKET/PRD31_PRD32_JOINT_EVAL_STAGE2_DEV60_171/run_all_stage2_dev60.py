#!/usr/bin/env python3
"""Execute the frozen PRD31+PRD32 DEV60 171-instance Stage2 benchmark in one command.

Runs eight already-defined chunks sequentially in separate processes, skipping only
archives which pass strict identity, 3-task, frozen-mask, Guard, and holdout audits.
Does NOT train, reconfigure models, enable release, read the holdout, or assert GT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

from joint_eval_stage2_dev60 import MANIFEST_SHA256, manifest_verified
from aggregate_stage2_8_chunks import inside, audit_zips

ROOT = Path(__file__).resolve().parent
PREFIX = 'PRD31_PRD32_STAGE2_CHUNK'
TASKS = {'tags', 'vqa', 'copy'}
GOOD_STATUS = 'PRD31_PRD32_STAGE2_CHUNK_COMPLETE_SHADOW_ONLY'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for buf in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def validate_chunk_zip(path: Path, expected_chunk: int, manifest: dict) -> dict:
    """Reject stale, partial, wrong-chunk or unsafe result archives; never mutate them."""
    expected = {(r['case_id'], r['instance_id']) for r in manifest['targets']
                if r['chunk_id'] == expected_chunk}
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError('corrupted ZIP member')
        s = json.loads(inside(z, 'summary.json'))
        if (s.get('status') != GOOD_STATUS or s.get('manifest_sha256') != MANIFEST_SHA256
                or s.get('chunk') != expected_chunk or s.get('target_count') != len(expected)):
            raise ValueError('wrong status, manifest, chunk or target count')
        if (s.get('all_guard_blocked') is not True or s.get('publish_allowed') is not False
                or s.get('holdout_accessed') is not False or
                s.get('all_frozen_mask_comparisons_passed') is not True or
                s.get('quality_gt') is not False):
            raise ValueError('unsafe or incomplete summary flags')
        rows = list(csv.DictReader(io.StringIO(inside(z, 'per_request_timing.csv').decode('utf-8-sig'))))
        if len(rows) != len(expected):
            raise ValueError('wrong measured request count')
        keys = set()
        for r in rows:
            key = (r['case_id'], r['instance_id'])
            if key in keys or key not in expected:
                raise ValueError('duplicate or unexpected measured target')
            keys.add(key)
            if (r['guard_status'] != 'REVIEW_REQUIRED_NOT_RELEASED'
                    or r['publish_allowed'].lower() != 'false'
                    or r['holdout_accessed'].lower() != 'false'
                    or float(r['frozen_mask_iou']) < 0.9999):
                raise ValueError('Guard, frozen mask or holdout contract failed')
        if keys != expected:
            raise ValueError('missing predicted target identities')
        calls = [json.loads(x) for x in inside(z, 'qwen_calls.jsonl').decode('utf-8').splitlines()
                 if x.strip()]
        if len(calls) != 3 * (len(expected) + 1) or s.get('measured_qwen_calls') != 3 * len(expected):
            raise ValueError('incomplete Qwen task calls')
        invocations = [c['invocation_uid'] for c in calls]
        if len(set(invocations)) != len(invocations):
            raise ValueError('duplicate invocation IDs')
        measured = [c for c in calls if str(c.get('cohort', '')).startswith('measured_t')]
        warmup = [c for c in calls if c.get('cohort') == 'warmup_excluded']
        if len(measured) != 3 * len(expected) or len(warmup) != 3:
            raise ValueError('measured/warmup separation failed')
        got_pairs = [(c['case_id'], c['instance_id'], c['task']) for c in measured]
        expected_pairs = {(a, b, t) for a, b in expected for t in TASKS}
        if set(got_pairs) != expected_pairs or len(set(got_pairs)) != len(got_pairs):
            raise ValueError('one of Tags, VQA, Copy is missing or duplicated')
        if any(c.get('publish_allowed') is not False for c in calls):
            raise ValueError('Qwen call publish flag unsafe')
        for req in ('preflight.json', 'co_residency.json'):
            inside(z, req)
    return {'chunk': expected_chunk, 'targets': len(expected), 'qwen_calls': len(calls),
            'archive': str(path), 'archive_sha256': sha256(path)}


def scan_existing(workspace: Path, run_dir: Path, manifest: dict) -> tuple[dict[int, Path], list[dict]]:
    found: dict[int, list[Path]] = {}
    invalid: list[dict] = []
    # Never scan all of /workspace recursively: no holdout and no unrelated files.
    for parent in (workspace, run_dir):
        if not parent.exists():
            continue
        for p in parent.glob('PRD31_PRD32_STAGE2_CHUNK??_*.zip'):
            m = re.match(r'^PRD31_PRD32_STAGE2_CHUNK(\d\d)_', p.name)
            if not m:
                continue
            ch = int(m.group(1))
            if ch not in range(1, 9):
                continue
            try:
                validate_chunk_zip(p, ch, manifest)
            except Exception as exc:
                invalid.append({'path': str(p), 'reason': str(exc)[:280]})
                continue
            found.setdefault(ch, []).append(p)
    chosen = {ch: max(set(paths), key=lambda p: (p.stat().st_mtime_ns, str(p)))
              for ch, paths in found.items()}
    return chosen, invalid


def run_logged(args: list[str], log_file: Path) -> None:
    print('\nCOMMAND:', ' '.join(args), flush=True)
    with log_file.open('a', encoding='utf-8') as lg:
        lg.write('\nCOMMAND: ' + ' '.join(args) + '\n')
        lg.flush()
        proc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                lg.write(line)
                lg.flush()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        code = proc.wait()
        if code:
            raise RuntimeError(f'child process failed with exit code {code}; prior chunks preserved')


def make_result_bundle(run_dir: Path, selected: dict[int, Path], combined: Path,
                       manifest: dict, log_file: Path) -> Path:
    # Strict complete cross-chunk verification comes from the UNMODIFIED aggregator.
    _, rows, grouped, calls = audit_zips([selected[n] for n in range(1, 9)])
    if len(rows) != 171 or len(grouped) != 8 or calls != 537:
        raise ValueError('complete 171-instance contract not met')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    bundle = run_dir.parent / f'PRD31_PRD32_STAGE2_ALL171_UPLOAD_THIS_{ts}_{os.getpid()}.zip'
    metadata = {
        'status': 'COMPLETE_171_PREDICTED_INSTANCES_BASELINE_NOT_OFFICIAL_ACCEPTANCE',
        'total_predicted_instances': 171, 'total_images': 60, 'chunks': 8,
        'measured_qwen_calls': 513, 'qwen_calls_including_warmups': 537,
        'holdout_used': False, 'production_release': False,
        'source_warning': 'V5-predicted instances, not exhaustive true garment GT; original 3.1.2/3.1.3 not certified in this runtime',
        'chunk_archives': [validate_chunk_zip(selected[i], i, manifest) for i in range(1, 9)],
    }
    with zipfile.ZipFile(bundle, 'w', allowZip64=True) as z:
        z.writestr('README_RESULT.txt',
            'PRD31+PRD32 fixed DEV60 Baseline, 171 V5-predicted instances across 60 images.\n'
            'Each of the 8 chunk ZIPs includes original real model outputs and masks.\n'
            'NOT a complete 3.1 GT recall benchmark, NOT a PRD 3.2 official accuracy test.\n'
            'No independent Holdout accessed. No production release.\n')
        z.writestr('RUN_ALL_METADATA.json', json.dumps(metadata, ensure_ascii=False, indent=2))
        z.write(combined, 'DEV60_STAGE2_171_COMBINED_SUMMARY.json', compress_type=zipfile.ZIP_DEFLATED)
        z.write(ROOT / 'full_dev60_171_manifest_v1.json', 'full_dev60_171_manifest_v1.json', compress_type=zipfile.ZIP_DEFLATED)
        if log_file.is_file():
            z.write(log_file, 'run_all.log', compress_type=zipfile.ZIP_DEFLATED)
        for i in range(1, 9):
            z.write(selected[i], 'CHUNK_ZIPS/' + selected[i].name, compress_type=zipfile.ZIP_STORED)
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:
            raise ValueError('final upload ZIP failed integrity check')
    return bundle


def main() -> int:
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--workspace', type=Path, default=Path('/workspace'),
                    help='GPU server directory containing existing chunk ZIPs')
    pa.add_argument('--dry-run', action='store_true', help='only verify and show pending chunks, no GPU')
    pa.add_argument('--execute-models', action='store_true', help='required for real model inference')
    args = pa.parse_args()
    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        pa.error(f'workspace not found: {workspace}')
    run_dir = workspace / 'PRD31_PRD32_STAGE2_RUN_ALL_171'
    manifest, _ = manifest_verified()
    chosen, invalid = scan_existing(workspace, run_dir, manifest)
    print('VERIFIED_COMPLETED_CHUNKS:', sorted(chosen), flush=True)
    print('PENDING_CHUNKS:', [n for n in range(1, 9) if n not in chosen], flush=True)
    for x in invalid:
        print('IGNORED_INCOMPLETE_OR_INVALID:', x['path'], x['reason'], flush=True)
    if args.dry_run:
        print('DRY_RUN_NO_GPU_NO_CHANGES')
        return 0
    if not args.execute_models:
        pa.error('real execution requires --execute-models; use --dry-run to inspect')
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / 'run_all.log'
    with log_file.open('a', encoding='utf-8') as lg:
        lg.write('\n=== BEGIN OR RESUME ' + datetime.now().isoformat() + ' ===\n')
        lg.write('Previously complete: ' + json.dumps(sorted(chosen)) + '\n')
    try:
        if len(chosen) < 8:
            first_missing = next(n for n in range(1, 9) if n not in chosen)
            preflight_out = run_dir / f'PREFLIGHT_CHUNK{first_missing:02d}_{int(time.time())}_{os.getpid()}'
            run_logged([sys.executable, '-u', str(ROOT / 'joint_eval_stage2_dev60.py'),
                        '--mode', 'preflight', '--chunk', str(first_missing),
                        '--out', str(preflight_out)], log_file)
        for i in range(1, 9):
            if i in chosen:
                print(f'SKIP_VERIFIED_CHUNK {i}: {chosen[i]}', flush=True)
                continue
            print(f'RUNNING_CHUNK {i}/8; one separate GPU process; no GT/holdout', flush=True)
            out = workspace / f'{PREFIX}{i:02d}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
            run_logged([sys.executable, '-u', str(ROOT / 'joint_eval_stage2_dev60.py'),
                        '--mode', 'live', '--chunk', str(i), '--execute-models', '--out', str(out)], log_file)
            zpath = out.with_suffix('.zip')
            record = validate_chunk_zip(zpath, i, manifest)
            chosen[i] = zpath
            print(f'VALIDATED_CHUNK {i}/8: {record["targets"]} targets; {record["qwen_calls"]} calls', flush=True)
        # One final audit across all 171, preserving original official aggregator contract.
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined = run_dir / f'DEV60_STAGE2_171_COMBINED_SUMMARY_{stamp}_{os.getpid()}.json'
        cmd = [sys.executable, '-u', str(ROOT / 'aggregate_stage2_8_chunks.py')]
        cmd.extend(str(chosen[i]) for i in range(1, 9))
        cmd.extend(['--out', str(combined)])
        run_logged(cmd, log_file)
        bundle = make_result_bundle(run_dir, chosen, combined, manifest, log_file)
        print('\nSTAGE2_ALL_171_COMPLETE_VERIFIED', flush=True)
        print('UPLOAD_THIS:', bundle, flush=True)
        print('NOTE: quality GT, missing V5 instances and formal 3.1.2/3.1.3 are not automatically validated', flush=True)
        return 0
    except KeyboardInterrupt:
        print('INTERRUPTED: completed chunk ZIPs remain intact. Rerun the same command to resume.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'RUN_ALL_STOPPED_SAFELY: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('Previous completed chunk ZIPs remain intact. Re-run after fixing failure, or upload logs for diagnosis.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
