"""Prepare an isolated V5 recovery run; training requires an explicit phase."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('phase', choices=['setup', 'prepare', 'check', 'train', 'backup'])
p.add_argument('--project', default='/workspace/fashion_multimodal_analysis')
p.add_argument('--work', default='/workspace/fashion_v5_recovery_20261008')
p.add_argument('--data', default='/workspace/fashion_data')
p.add_argument('--resume', action='store_true')
a = p.parse_args()
source = Path(a.project).resolve()
work = Path(a.work).resolve()
data = Path(a.data).resolve()
if source == work:
    raise SystemExit('Recovery work must differ from the original project.')
output = work / 'outputs/prd_instance_segmentation/maskrcnn_8class_v5_recovery_20261008'
report = work / 'reports/recovery_20261008/v5_training'
env = dict(os.environ, FASHION_PROJECT_ROOT=str(work), FASHION_DATA_ROOT=str(data))
env['PYTHONPATH'] = str(work / 'src') + os.pathsep + env.get('PYTHONPATH', '')

def run(args, log_name):
    log = work / 'recovery_logs' / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation keeps prior logs; reruns get a new numbered log.
    i = 1
    while log.exists():
        log = log.with_name(f'{Path(log_name).stem}_{i}.log')
        i += 1
    print('Log:', log, flush=True)
    with log.open('w', encoding='utf-8') as f:
        proc = subprocess.Popen(args, cwd=work, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            proc.wait()
            raise
    if code:
        raise SystemExit(f'Phase failed ({code}); inspect {log}.')

def trainer(dry=False):
    cmd = [sys.executable, '-u', '-m',
           'fashion_multimodal_analysis.segmentation.training.train_prd_8class_maskrcnn_v3_b1_dataexp',
           '--train-csv', 'configs/prd_8class_train_v5_full_targets.csv',
           '--val-csv', 'configs/prd_8class_val_v1.csv',
           '--output-dir', str(output), '--report-dir', str(report),
           '--experiment-name', 'v5-recovery-20261008', '--verify-isolation',
           '--epochs', '15', '--device', 'cuda', '--batch-size', '1',
           '--workers', '0', '--sampler-alpha', '0.5', '--save-every', '1']
    if dry:
        cmd.append('--dry-run')
    elif a.resume:
        checkpoint = output / 'checkpoint_last.pth'
        if not checkpoint.is_file():
            raise SystemExit('No checkpoint to resume.')
        cmd += ['--resume', str(checkpoint)]
    return cmd

if a.phase == 'setup':
    if work.exists():
        raise SystemExit('Work directory already exists; use subsequent phases or choose a new --work.')
    if not (source / 'pyproject.toml').is_file():
        raise SystemExit('Original project not found.')
    work.mkdir()
    for name in ['configs', 'benchmark', 'outputs']:
        shutil.copytree(source / name, work / name)
    # Reuse unchanged code while isolating all writable artifacts.
    (work / 'src').symlink_to(source / 'src', target_is_directory=True)
    shutil.copy2(source / 'pyproject.toml', work / 'pyproject.toml')
    (work / 'reports').mkdir()
    (work / 'recovery_source.json').write_text(json.dumps(
        {'project': str(source), 'work': str(work), 'data': str(data)}, indent=2))
    print('Setup complete. No training started:', work)
elif not (work / 'recovery_source.json').is_file():
    raise SystemExit('Run setup first.')
elif a.phase == 'prepare':
    run([sys.executable, '-u', '-m',
         'fashion_multimodal_analysis.segmentation.training.run_prd31_v5_full_targets',
         '--project', str(work), '--data-root', str(data), '--phase', 'prepare'], 'prepare.log')
    print('Prepared manifests and masks. Run check next; preparation does not train.')
elif a.phase == 'check':
    run(trainer(dry=True), 'check.log')
    print('Dry-run passed. Original reports remain untouched.')
elif a.phase == 'train':
    run(trainer(dry=True), 'pretrain_check.log')
    run(trainer(), 'train.log')
    print('Training completed. Run backup, then download the generated archive.')
elif a.phase == 'backup':
    checkpoints = sorted(output.glob('checkpoint*.pth'))
    if not checkpoints:
        raise SystemExit('No checkpoints found; backup not created.')
    dest = work / 'download_backups'
    dest.mkdir(exist_ok=True)
    archive = dest / 'v5_recovery_weights_and_records.zip'
    if archive.exists():
        raise SystemExit('Backup ZIP already exists; download it before creating another copy.')
    rows = []
    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''):
                h.update(block)
        return h.hexdigest()
    files = list(checkpoints)
    for folder in [report, work / 'configs', work / 'recovery_logs',
                   work / 'reports/reruns/recovery_v2/v5_full_targets']:
        if folder.exists():
            files += [q for q in folder.rglob('*') if q.is_file()]
    for q in files:
        rows.append({'path': str(q.relative_to(work)), 'size': q.stat().st_size, 'sha256': sha(q)})
    # ZIP_STORED avoids expensive recompression of model tensors.
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
        for q in files:
            z.write(q, q.relative_to(work).as_posix())
        z.writestr('backup_inventory.json', json.dumps(rows, indent=2))
    print('Backup archive:', archive)
    print('SHA256:', sha(archive))
    print('Download to your computer or persistent storage. This local copy is NOT off-instance protection.')
