#!/usr/bin/env python3
"""Safely add PRD3.2 output guard files to an existing project (no overwrites)."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

FILES = (
    'src/fashion_multimodal_analysis/integration/prd32_output_guard.py',
    'scripts/prd32_guard_service_outputs.py',
    'tests/test_prd32_output_guard.py',
)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--apply',action='store_true',help='actually install after dry run')
    args=ap.parse_args()
    project=args.project.resolve(strict=True)
    if not project.is_dir() or not (project/'src/fashion_multimodal_analysis/integration/pipeline.py').is_file():
        ap.error('Project does not look like the expected fashion_multimodal_analysis source tree')
    patch_root=Path(__file__).resolve().parent
    to_install=[]
    for name in FILES:
        source=patch_root/name
        destination=project/name
        if not source.is_file():
            raise FileNotFoundError(source)
        if not destination.resolve().is_relative_to(project):
            raise ValueError(f'unsafe destination symlink for {name}')
        data=source.read_bytes()
        if destination.exists() or destination.is_symlink():
            if not destination.is_file() or destination.read_bytes()!=data:
                raise FileExistsError(f'Existing different file preserved; no overwrite: {destination}')
            print('ALREADY_INSTALLED',name)
        else:
            print(('INSTALL' if args.apply else 'WOULD_INSTALL'),name,
                  hashlib.sha256(data).hexdigest()[:12])
            to_install.append((destination,data))
    if not args.apply:
        print('DRY_RUN_OK; use --apply to add the 3 missing code/test files')
        return
    for dst,data in to_install:
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as handle:handle.write(data)
    print('GUARD_INSTALL_OK; no existing source files were modified')

if __name__=='__main__':main()
