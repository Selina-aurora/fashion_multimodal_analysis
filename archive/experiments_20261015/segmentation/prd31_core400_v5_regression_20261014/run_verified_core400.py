#!/usr/bin/env python3
"""Read-only provenance gate and Core400 segmentation regression runner.

Never changes model weights, manifests, or reports in the project repository.
Only runs the existing in-repository eval after all integrity gates pass.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

CORE_SHA = "c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89"
EXPECTED_TRAIN_SHA = "e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f"
EXPECTED_VAL_SHA = "39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93"
TRAIN_REL = "configs/prd_8class_train_v5_balanced150_20261008.csv"
VAL_REL = "configs/prd_8class_val_v2_full_targets.csv"
CORE_REL = "benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv"
CKPT_REL = "outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth"
EVAL_REL = "scripts/segmentation/evaluation/eval_prd31_core_regression_v2_balanced_v1.py"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(newline='', encoding='utf-8-sig') as f:
        return sum(1 for _ in csv.DictReader(f))


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def pack_result(report: Path, out_zip: Path):
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(report.rglob('*')):
            if p.is_file():
                z.write(p, str(p.relative_to(report)))
    with zipfile.ZipFile(out_zip) as z:
        corrupt = z.testzip()
        if corrupt:
            raise RuntimeError('zip integrity failure: '+corrupt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/workspace/fashion_multimodal_analysis')
    ap.add_argument('--out-dir', default='/workspace/prd31_regression_20261014')
    ap.add_argument('--warmup-runs', type=int, default=10)
    ap.add_argument('--timed-passes', type=int, default=5)
    ap.add_argument('--max-visualizations', type=int, default=12)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    base = Path(a.out_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / 'v5_balanced150_core400'
    # Do not overwrite any older result. Each invocation creates its own run directory.
    if run_dir.exists():
        run_dir = base / ('v5_balanced150_core400_' + time.strftime('%Y%m%d_%H%M%S'))
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {'status':'STARTED','root':str(root),'date':'2026-10-14','checkpoint':CKPT_REL,
              'train_manifest':TRAIN_REL,'val_manifest':VAL_REL,'core_manifest':CORE_REL,
              'regression':'segmentation_core400_only','timing_scope':'preprocess+forward+postprocess'}
    try:
        for rel in (TRAIN_REL,VAL_REL,CORE_REL,CKPT_REL,EVAL_REL):
            if not (root/rel).is_file():
                raise FileNotFoundError(str(root/rel))
        actual = {TRAIN_REL: digest(root/TRAIN_REL),VAL_REL:digest(root/VAL_REL),CORE_REL:digest(root/CORE_REL)}
        expected = {TRAIN_REL: EXPECTED_TRAIN_SHA,VAL_REL:EXPECTED_VAL_SHA,CORE_REL:CORE_SHA}
        report['input_sha256'] = actual
        report['input_rows'] = {rel:csv_rows(root/rel) for rel in (TRAIN_REL,VAL_REL,CORE_REL)}
        if actual != expected:
            raise RuntimeError('Manifest SHA256 mismatch: no evaluation permitted')
        if report['input_rows'][CORE_REL] != 400:
            raise RuntimeError('Core400 manifest row count differs from 400')
        import torch
        report['torch'] = torch.__version__
        report['cuda_available'] = bool(torch.cuda.is_available())
        report['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refuse to time a CPU run as GPU regression')
        # Checkpoints belong to user's own experiment. Full load is required for
        # older PyTorch checkpoint dictionaries storing NumPy RNG state.
        cp = torch.load(root/CKPT_REL, map_location='cpu', weights_only=False)
        if not isinstance(cp, dict):
            raise RuntimeError('Checkpoint is not a dictionary')
        report['checkpoint_provenance'] = {
            k:cp.get(k) for k in ('epoch','train_manifest_sha256','val_manifest_sha256','checkpoint_format','class_to_id')
        }
        saved = cp.get('args', {})
        report['checkpoint_args'] = {k: saved.get(k) for k in ('epochs','min_size','max_size','seed','no_balanced_sampler')}
        if cp.get('train_manifest_sha256') != actual[TRAIN_REL]:
            raise RuntimeError('Checkpoint train provenance mismatch')
        if cp.get('val_manifest_sha256') != actual[VAL_REL]:
            raise RuntimeError('Checkpoint val provenance mismatch')
        if cp.get('epoch') != 15:
            raise RuntimeError('Checkpoint is not final epoch 15')
        if not isinstance(cp.get('model_state_dict'), dict):
            raise RuntimeError('Checkpoint is missing model_state_dict')
        expected_classes = {'top':1,'pants':2,'skirt':3,'outerwear':4,'dress':5,'shoe':6,'bag':7,'accessory':8}
        if cp.get('class_to_id') != expected_classes:
            raise RuntimeError('Checkpoint class map mismatch')
        del cp
        gc.collect()
        report['gate'] = 'PROVENANCE_AND_CUDA_PASSED'
        save_json(run_dir/'gate_and_environment.json',report)
        command = [sys.executable,str(root/EVAL_REL),'--checkpoint',str(root/CKPT_REL),
                   '--val-csv',str(root/CORE_REL),'--device','cuda',
                   '--score-threshold','0.40','--mask-threshold','0.50',
                   '--match-bbox-iou','0.50','--warmup-runs',str(a.warmup_runs),
                   '--timed-passes',str(a.timed_passes),
                   '--max-visualizations',str(a.max_visualizations),
                   '--report-dir',str(run_dir/'evaluation')]
        report['eval_command'] = command
        save_json(run_dir/'gate_and_environment.json',report)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(root/'src') + os.pathsep + env.get('PYTHONPATH','')
        print('RUNNING_EXISTING_CORE400_EVALUATOR',flush=True)
        log_path = run_dir/'evaluator.log'
        with log_path.open('w',encoding='utf-8') as log:
            p = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, errors='replace')
            assert p.stdout is not None
            for line in p.stdout:
                sys.stdout.write(line)
                log.write(line)
            rc = p.wait()
        report['return_code'] = rc
        report['status'] = 'REGRESSION_COMPLETED' if rc == 0 else 'EVALUATOR_FAILED'
        if rc:
            report['error'] = 'Existing evaluation script returned nonzero; inspect evaluator.log'
    except Exception as e:
        report['status'] = 'GATE_FAILED_NO_EVALUATION'
        report['error'] = '%s: %s' % (type(e).__name__,e)
        (run_dir/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('GATE_FAILED_NO_EVALUATION',report['error'],flush=True)
    finally:
        save_json(run_dir/'gate_and_environment.json',report)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        out_zip = base/('prd31_core400_v5bal_results_'+stamp+'_UPLOAD_THIS.zip')
        pack_result(run_dir,out_zip)
        print('STATUS:', report['status'],flush=True)
        print('UPLOAD_THIS_ZIP:', out_zip,flush=True)
        print('PROJECT_FILES_UNMODIFIED: True',flush=True)
        if report['status']!='REGRESSION_COMPLETED':
            sys.exit(2)

if __name__=='__main__':
    main()
