#!/usr/bin/env python3
"""PRD 3.2 train-only class-hint metadata audit. No image reads, GT reads or inference.
Known fix: source_image is a valid source manifest path field.
"""
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

CLASSES=('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
CLASS_KEYS=('garment_category','category','category_name','class_name','class','label','labels','categories')
PATH_KEYS=('source_image','image_path','img_path','filepath','file_path','image_file','image_name','filename','file')
BLOCK=re.compile(r'(^|[_.-])(holdout|sealed|test|testing|testset|val|valid|validation|eval|evaluation|devtest)($|[_.-])',re.I)
FORMATS={'.csv','.jsonl'}
EXTS={'.jpg','.jpeg','.png','.webp'}


def safe_parts(path):
    return all(not BLOCK.search(str(p)) for p in Path(path).parts)


def category_tokens(raw):
    if raw is None: return []
    if isinstance(raw,(list,tuple)):
        vals=[]
        for item in raw: vals.extend(category_tokens(item))
        return [x for x in CLASSES if x in vals]
    if isinstance(raw,dict): return []
    tokens=set(re.findall(r'[a-z]+',str(raw).lower()))
    return [x for x in CLASSES if x in tokens]


def iter_rows(path,limit):
    try:
        with path.open('r',encoding='utf-8-sig',errors='replace',newline='') as f:
            if path.suffix.lower()=='.csv':
                yield from (r for n,r in enumerate(csv.DictReader(f)) if n < limit and isinstance(r,dict))
            else:
                for n,line in enumerate(f):
                    if n>=limit: break
                    try: row=json.loads(line)
                    except ValueError: continue
                    if isinstance(row,dict): yield row
    except (OSError,UnicodeError,csv.Error):
        return


def resolve_train_image(raw,metadata,data_root,project_root):
    if not isinstance(raw,str) or not raw.strip(): return None,'EMPTY_IMAGE_PATH'
    path=Path(raw.strip())
    choices=[path] if path.is_absolute() else [project_root/path,data_root/path,data_root.parent/path,metadata.parent/path]
    failure='NOT_IN_APPROVED_TRAIN_ROOT_OR_MISSING'
    for p in choices:
        q=p.resolve(strict=False)
        if not safe_parts(q):
            failure='RESTRICTED_IMAGE_PATH';continue
        if not q.is_relative_to(data_root): continue
        if 'train' not in [a.lower() for a in q.parts]:
            failure='TRAIN_COMPONENT_MISSING';continue
        if q.suffix.lower() not in EXTS:
            failure='UNSUPPORTED_IMAGE_EXTENSION';continue
        if not q.is_file():
            failure='IMAGE_NOT_FOUND';continue
        return q,None
    return None,failure


def valid_metadata(p):
    return p.suffix.lower() in FORMATS and safe_parts(p) and p.is_file() and p.stat().st_size <= 30*1024*1024


def collect(root,limits):
    found=[]
    if not root.exists(): return found
    for d,dirs,files in os.walk(root):
        dirs[:]=sorted(x for x in dirs if safe_parts(Path(d)/x) and not x.startswith('.'))
        for f in sorted(files):
            p=Path(d)/f
            if valid_metadata(p):
                found.append(p)
                if len(found)>=limits: return found
    return found


def analyze(paths,data_root,project_root,rows_per_file):
    all_records=[]
    stats=Counter()
    file_stats=[]
    seen=set()
    for metadata in paths:
        st=Counter(); st['rows']=0
        for row in iter_rows(metadata,rows_per_file):
            st['rows']+=1
            cats=[]
            for k in CLASS_KEYS:
                if k in row: cats.extend(category_tokens(row[k]))
            cats=list(dict.fromkeys(cats))
            if not cats:
                st['rows_without_exact_class_hint']+=1;continue
            st['rows_with_recognized_class']+=1
            raw=next((row[k] for k in PATH_KEYS if isinstance(row.get(k),str) and row[k].strip()),None)
            if raw is None:
                st['rows_missing_known_path_field']+=1;continue
            st['rows_with_known_path_field']+=1
            img,reason=resolve_train_image(raw,metadata,data_root,project_root)
            if img is None:
                st['skipped_'+reason]+=1;continue
            st['rows_with_proven_train_image']+=1
            for c in cats:
                key=(c,str(img))
                if key in seen:continue
                seen.add(key)
                all_records.append({'category_hint_NOT_GT':c,'image_path':str(img),'metadata_path':str(metadata)})
        file_stats.append({'metadata_path':str(metadata),**dict(st)})
        stats.update(st)
    return all_records,stats,file_stats


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inventory',type=Path,action='append',help='Explicit approved train-source inventory CSV/JSONL, repeatable')
    ap.add_argument('--extra-root',type=Path,action='append',help='Optional additional TRAIN metadata root, repeatable')
    ap.add_argument('--data-root',type=Path,default=Path('/workspace/fashion_data'))
    ap.add_argument('--project-root',type=Path,default=Path('/workspace/fashion_multimodal_analysis'))
    ap.add_argument('--output',type=Path,default=Path('/workspace/PRD32_DEV60_STAGE1C_METADATA_V2'))
    ap.add_argument('--max-files',type=int,default=1000)
    ap.add_argument('--rows-per-file',type=int,default=200000)
    a=ap.parse_args()
    if a.output.exists():raise SystemExit('REFUSE_OVERWRITE:'+str(a.output))
    if a.max_files < 1 or a.rows_per_file < 1:raise SystemExit('INVALID_LIMITS')
    a.data_root=a.data_root.resolve()
    a.project_root=a.project_root.resolve()
    inventories=a.inventory or [a.project_root/'configs/prd_8class_unified_manifest_v2.csv']
    paths=[];skips=[]
    for raw in inventories:
        p=raw.resolve()
        if valid_metadata(p): paths.append(p)
        else:skips.append({'path':str(p),'reason':'NOT_FOUND_OR_RESTRICTED_OR_OVERSIZE'})
    for rt in a.extra_root or []:
        r=rt.resolve()
        if not safe_parts(r):
            skips.append({'path':str(r),'reason':'RESTRICTED_ROOT'});continue
        if not r.is_relative_to(a.project_root) and not r.is_relative_to(a.data_root):
            skips.append({'path':str(r),'reason':'ROOT_OUTSIDE_APPROVED_SOURCE'});continue
        paths.extend(collect(r,a.max_files))
    paths=list(dict.fromkeys(paths))[:a.max_files]
    rows,counts,by_file=analyze(paths,a.data_root,a.project_root,a.rows_per_file)
    a.output.mkdir(parents=True)
    out=a.output/'candidate_train_images_by_class.csv'
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,['category_hint_NOT_GT','image_path','metadata_path']);w.writeheader();w.writerows(rows)
    dist={c:len({r['image_path'] for r in rows if r['category_hint_NOT_GT']==c}) for c in CLASSES}
    summary={'schema':'prd32_dev60_train_metadata_discovery_v2','source_scope':'TRAIN_SOURCE_METADATA_ONLY',
             'status':'TRAIN_CANDIDATES_FOUND' if rows else 'NO_PROVEN_TRAIN_CANDIDATES',
             'metadata_files_scanned':len(paths),'metadata_files_skipped':skips,
             'diagnostic_counts':dict(counts),'class_hint_candidate_counts_NOT_GT':dist,
             'proven_train_candidate_pairs':len(rows),
             'candidate_unique_train_images':len({r['image_path'] for r in rows}),
             'confirmed_holdout_isolation':False,'v2_cohort_frozen':False,'inference_ran':False,
             'known_issue_fixed':'source_image is recognized as an image path field',
             'limitations':['Class hints come from training-source metadata, not human GT','No sealed/test/val/holdout data read','No image pixels read','Does not prove holdout exclusion','Does not replace current 60-image candidate cohort']}
    (a.output/'discovery_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'per_file_diagnostics.json').write_text(json.dumps(by_file,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TRAIN_METADATA_V2_OK',json.dumps(dist,ensure_ascii=False))
    print('PROVEN_TRAIN_IMAGES',summary['candidate_unique_train_images'])
    print('ROWS_WITH_RECOGNIZED_CLASS',counts['rows_with_recognized_class'])
    print('ROWS_MISSING_KNOWN_PATH_FIELD',counts['rows_missing_known_path_field'])
    print('NO_FREEZE_NO_HOLDOUT_READ_NO_INFERENCE')
    print('OUTPUT',a.output)

if __name__=='__main__':main()
