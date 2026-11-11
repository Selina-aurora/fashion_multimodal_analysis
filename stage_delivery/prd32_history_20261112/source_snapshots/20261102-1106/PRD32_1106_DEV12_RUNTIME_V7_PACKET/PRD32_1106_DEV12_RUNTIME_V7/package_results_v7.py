#!/usr/bin/env python3
"""Create a minimal provenance/results ZIP; supports incomplete blocked runs."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from producer_contract_v7 import refuse_restricted

PERMITTED = {'.json','.jsonl','.png','.jpg','.jpeg','.csv','.txt'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--producer-dir',type=Path,required=True)
    p.add_argument('--qwen-dir',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():raise FileExistsError('results ZIP already exists; no overwrite')
    roots=[a.producer_dir]+([a.qwen_dir] if a.qwen_dir else [])
    if not any(x.is_dir() for x in roots):raise FileNotFoundError('no output directories exist')
    manifest=[]
    with ZipFile(a.out,'x',ZIP_DEFLATED) as z:
        for root in roots:
            if not root.is_dir():continue
            refuse_restricted(root)
            for file in sorted(root.rglob('*')):
                if not file.is_file():continue
                refuse_restricted(file)
                if file.suffix.lower() not in PERMITTED:raise ValueError('unsupported output artifact '+str(file))
                if file.is_symlink() or not file.resolve().is_relative_to(root.resolve()):
                    raise ValueError('unsafe symlink in output')
                name=str(Path(root.name)/file.relative_to(root))
                z.write(file,name)
                manifest.append({'file':name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'size':file.stat().st_size})
        z.writestr('MANIFEST.json',json.dumps({'schema':'prd32_v7_gpu_run_bundle_v1','files':manifest,
                                              'manual_review_required':True,
                                              'publish_allowed':False},indent=2))
    print('V7_RESULTS_PACKED',a.out,'files',len(manifest))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('V7_PACK_BLOCKED',type(e).__name__,str(e),file=sys.stderr);sys.exit(2)
