"""安装前核对两个源文件版本，备份后写入补丁；失败自动回滚。"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    package = Path(__file__).resolve().parent
    for directory in ('src', 'scripts', 'configs'):
        if not (root / directory).is_dir():
            raise SystemExit(f'请从项目根目录运行，当前目录缺少 {directory}: {root}')
    semantic = root / 'src/fashion_multimodal_analysis/integration/region_semantics.py'
    if not semantic.is_file():
        raise SystemExit('缺少今天已有的 region_semantics.py；未写入任何补丁，请上传此文件/确认项目目录。')
    manifest = json.loads((package / 'patch_manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['expected_source_sha256'].items():
        target = root / relative
        if not target.is_file() or sha(target) != expected:
            raise SystemExit(f'源文件版本与本次上传不一致，未写入补丁：{relative}')
    files = [p for folder in ('src', 'scripts', 'configs', 'tests')
             for p in sorted((package / folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    backup = root / 'reports/patch_backups' / ('semantic_guard_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True, exist_ok=False)
    previous = {}
    # 所有已有文件先完成备份，再写入；原来的配置/权重/报告不改动。
    for source in files:
        relative = source.relative_to(package)
        target = root / relative
        previous[str(relative)] = target.is_file()
        if target.is_file():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
    (backup / 'previous_files.json').write_text(json.dumps(previous, indent=2))
    attempted = []
    try:
        for source in files:
            relative = source.relative_to(package)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            attempted.append(relative)
            shutil.copy2(source, target)
    except Exception:
        for relative in reversed(attempted):
            target = root / relative
            if previous[str(relative)]:
                shutil.copy2(backup / relative, target)
            else:
                target.unlink(missing_ok=True)
        raise
    print('PATCH_INSTALLED')
    print('备份目录:', backup)
    print('下一步：PYTHONPATH=src python scripts/run_prd31_review_regression_20261008.py --dry-run')


if __name__ == '__main__':
    main()
