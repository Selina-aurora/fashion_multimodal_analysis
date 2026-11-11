#!/usr/bin/env python3
"""Metadata-only inventory discovery for PRD 3.2 dev60 coverage.

Does not open sealed/holdout/test/val metadata, inspect corresponding images,
construct GT, update the frozen cohort, or run inference.
"""
import argparse
import csv
import json
import re
import os
from pathlib import Path

CLASSES = ('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
STOP = {'holdout','sealed','test','testing','testset','val','valid','validation',
        'eval','evaluation','devtest','__pycache__','.git','.venv','site-packages'}
CLASS_KEYS = ('garment_category','category','category_name','class_name','class','label','labels','categories')
PATH_KEYS = ('image_path','img_path','filepath','file_path','image_file','image_name','filename','file')
MAX_BYTES = 25 * 1024 * 1024
SENSITIVE_NAME = re.compile(r"(?:^|[_.-])(holdout|sealed|test|testing|testset|val|valid|validation|eval|evaluation|devtest)(?:$|[_.-])", re.I)

def is_blocked_part(part):
    return part.lower() in STOP or bool(SENSITIVE_NAME.search(part))


def safe_metadata_path(p):
    return not any(is_blocked_part(x) for x in p.parts) and p.suffix.lower() in {'.csv','.jsonl'}


def safe_train_image_path(p):
    parts = {x.lower() for x in p.parts}
    return 'train' in parts and not any(is_blocked_part(x) for x in p.parts) and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}


def parse_classes(value):
    """Only exact category tokens; avoids mapping coat/heel labels without human checking."""
    if isinstance(value, (list,tuple)):
        tokens = [str(x).lower() for x in value]
    else:
        tokens = re.split(r'[;,|/\s]+',str(value or '').lower())
    return [c for c in CLASSES if c in tokens]


def iter_rows(path, cap):
    try:
        with path.open('r',encoding='utf-8-sig',errors='replace',newline='') as f:
            if path.suffix.lower()=='.csv':
                rd=csv.DictReader(f)
                for n,row in enumerate(rd):
                    if n>=cap:break
                    if isinstance(row,dict):yield row
            else:
                for n,line in enumerate(f):
                    if n>=cap:break
                    try: row=json.loads(line)
                    except (ValueError,TypeError):continue
                    if isinstance(row,dict):yield row
    except (PermissionError,OSError,UnicodeError,csv.Error) as exc:
        return


def main():
    ap=argparse.ArgumentParser(description='Read permitted train metadata only; never read sealed/holdout/test files')
    ap.add_argument('--root',action='append',type=Path,help='Metadata root; repeatable')
    ap.add_argument('--output',type=Path,default=Path('/workspace/PRD32_DEV60_STAGE1B_METADATA_DISCOVERY'))
    ap.add_argument('--max-files',type=int,default=1500)
    ap.add_argument('--rows-per-file',type=int,default=200000)
    args=ap.parse_args()
    roots=args.root or [Path('/workspace/fashion_multimodal_analysis/configs'),Path('/workspace/fashion_data')]
    if args.output.exists():raise SystemExit('REFUSE_OVERWRITE:'+str(args.output))
    args.output.mkdir(parents=True)
    found=[]; unproven=[]; scanned=0; skipped_restricted=0; skipped_size=0
    selected=set()
    for root in roots:
        if not root.is_dir():continue
        for directory,dirnames,filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not is_blocked_part(d) and not d.startswith('.')]
            if scanned>=args.max_files:break
            for filename in filenames:
                if scanned>=args.max_files:break
                meta=Path(directory)/filename
                if meta.suffix.lower() not in {'.csv','.jsonl'}:continue
                if not safe_metadata_path(meta):
                    skipped_restricted+=1;continue
                if not meta.is_file():continue
                if meta.stat().st_size>MAX_BYTES:
                    skipped_size+=1;continue
                scanned+=1
                for row in iter_rows(meta,args.rows_per_file):
                    cats=[]
                    for key in CLASS_KEYS:
                        if key in row:cats.extend(parse_classes(row[key]))
                    cats=list(dict.fromkeys(cats))
                    if not cats:continue
                    ref=None
                    for key in PATH_KEYS:
                        if row.get(key):ref=str(row[key]).strip();break
                    if not ref:continue
                    raw=Path(ref)
                    choices=[raw] if raw.is_absolute() else [meta.parent/raw,Path('/workspace')/raw,Path('/workspace/fashion_data')/raw]
                    path=next((v for v in choices if safe_train_image_path(v) and v.is_file()),None)
                    if path is None:
                        unproven.append({'metadata':str(meta),'path_reference':ref[:300],'categories':';'.join(cats)})
                        continue
                    try:
                        resolved=path.resolve(strict=True)
                    except OSError:continue
                    if not safe_train_image_path(resolved):continue
                    for cat in cats:
                        k=(cat,str(resolved))
                        if k not in selected:
                            found.append({'category_hint_NOT_GT':cat,'image_path':str(resolved),'metadata_path':str(meta)})
                            selected.add(k)
        if scanned>=args.max_files:break
    with (args.output/'candidate_train_images_by_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['category_hint_NOT_GT','image_path','metadata_path']);w.writeheader();w.writerows(found)
    by_class={c:len({r['image_path'] for r in found if r['category_hint_NOT_GT']==c}) for c in CLASSES}
    info={'schema':'prd32_dev60_train_metadata_discovery_v1', 'scan_complete':scanned<args.max_files,
          'roots':[str(x) for x in roots], 'metadata_files_scanned':scanned,
          'restricted_metadata_skipped':skipped_restricted, 'oversize_metadata_skipped':skipped_size,
          'class_hint_candidate_counts_NOT_GT':by_class, 'unresolved_or_nontrain_path_rows':len(unproven),
          'proven_train_candidate_rows':len(found),
          'limitations':['No sealed/holdout/test/val metadata or images opened',
                         'Exact category token matching only; mapped classes may be missed',
                         'This is source metadata coverage, not visual GT or V5 recall',
                         'Does not verify holdout SHA overlap, cannot certify cohort',
                         'Original 60-image cohort and V1 images are unchanged']}
    (args.output/'discovery_summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TRAIN_METADATA_DISCOVERY_ONLY',json.dumps(by_class,ensure_ascii=False))
    print('FILES_SCANNED',scanned,'UNRESOLVED',len(unproven))
    print('OUTPUT',args.output)
    print('NO_FREEZE_NO_INFERENCE')

if __name__=='__main__':main()
