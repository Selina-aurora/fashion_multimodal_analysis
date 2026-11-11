#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a 60-image TRAIN-SOURCE development cohort without looking at sealed/test files.

Image-only freeze; no V5/Qwen inference, no use of GT bboxes, no PRD acceptance.
The optional exclusion registry contains ONLY pre-approved SHA-256 digests, not paths.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import html
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLASS_ORDER = ('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
QUOTAS = {'top':4,'pants':5,'skirt':7,'outerwear':8,'dress':4,'shoe':7,'bag':7,'accessory':6}
RESTRICTED = {'holdout','sealed','test','testing','testset','eval','evaluation','val','valid','validation','fullval','devtest'}
EXTS = {'.jpg','.jpeg','.png','.webp'}
HASH_RE = re.compile(r'^[a-f0-9]{64}$')
SEED = 'PRD32_FIXED_DEV_V2_60IMG_20261030'
SCHEMA = 'prd32_fixed_dev_v2_60_images_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def hstr(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def readj(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def readjl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def writej(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def writejl(path,rows):
    with Path(path).open('w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def restricted(part):
    v=part.lower()
    return v in RESTRICTED or any(v.startswith(q+'_') or v.endswith('_'+q) for q in RESTRICTED)


def safe_train_image(raw, project, data):
    """Only train files under approved root, including verified symlink destination."""
    if not isinstance(raw,str) or not raw.strip():
        return None,'EMPTY_PATH'
    p=Path(raw.strip())
    choices = ([p] if p.is_absolute() else [project/p, data/p, data.parent/p])
    for cand in choices:
        try:
            q=cand.resolve(strict=False)
            if not q.is_relative_to(data):
                continue
            rel=q.relative_to(data)
            if any(restricted(x) for x in rel.parts):
                return None,'RESTRICTED_PATH'
            if 'train' not in (x.lower() for x in rel.parts):
                return None,'TRAIN_COMPONENT_REQUIRED'
            if q.suffix.lower() not in EXTS:
                return None,'NOT_AN_IMAGE'
            if not q.is_file():
                return None,'FILE_MISSING'
            return q,''
        except (OSError,ValueError):
            continue
    return None,'OUTSIDE_TRAIN_ROOT_OR_MISSING'


def d_hash(path):
    from PIL import Image
    with Image.open(path) as im:
        if im.getexif().get(274,1) not in (1,None):
            raise ValueError('EXIF_ROTATION_UNSUPPORTED')
        if im.width < 16 or im.height < 16 or im.width*im.height > 45_000_000:
            raise ValueError('IMAGE_DIMENSIONS_UNSUPPORTED')
        # Fixed 64-bit difference hash for approximate duplicate detection.
        small=im.convert('L').resize((9,8),Image.Resampling.BILINEAR)
        pix=small.load()
        px=[pix[x,y] for y in range(8) for x in range(9)]
        n=0
        for y in range(8):
            for x in range(8):
                n=(n<<1)|(1 if px[y*9+x]>px[y*9+x+1] else 0)
        return n, [im.width,im.height]


def hamming(a,b):
    return (a^b).bit_count()


def read_approved_exclusions(path):
    """Read only a digest-only safe registry, never sealed manifests or images."""
    if path is None:
        return None
    p=Path(path).resolve()
    if any(restricted(x) for x in p.parts):
        raise ValueError('DO_NOT_READ_SEALED_HOLDOUT_RESOURCE; provide an already approved digest-only registry elsewhere')
    if not p.is_file():
        raise FileNotFoundError('APPROVED_HASH_REGISTRY_NOT_FOUND:'+str(p))
    if p.suffix.lower() not in ('.txt','.csv','.json'):
        raise ValueError('APPROVED_REGISTRY_MUST_BE_DIGEST_ONLY_TXT_CSV_JSON')
    if p.stat().st_size > 200_000:
        raise ValueError('APPROVED_REGISTRY_TOO_LARGE')
    content=p.read_text(encoding='utf-8-sig')
    if p.suffix.lower()=='.json':
        parsed=json.loads(content)
        if isinstance(parsed,dict):
            if set(parsed)!={'sha256'}:raise ValueError('REGISTRY_JSON_DIGEST_FIELD_ONLY')
            parsed=parsed['sha256']
        if not isinstance(parsed,list):raise ValueError('REGISTRY_LIST_REQUIRED')
        values=parsed
    elif p.suffix.lower()=='.csv':
        rows=list(csv.DictReader(io.StringIO(content)))
        if not rows or list(rows[0])!=['sha256']:raise ValueError('REGISTRY_CSV_ONLY_ONE_COLUMN_NAMED_SHA256')
        values=[r['sha256'] for r in rows]
    else:
        values=[x.strip() for x in content.splitlines() if x.strip()]
    if not values or any(not isinstance(v,str) or not HASH_RE.fullmatch(v.lower()) for v in values):
        raise ValueError('REGISTRY_MUST_CONTAIN_ONLY_SHA256_DIGESTS')
    result={v.lower() for v in values}
    if len(result)<8:
        raise ValueError('REGISTRY_AT_LEAST_EIGHT_APPROVED_HOLDOUT_DIGESTS_REQUIRED')
    return result


def load_v1(root):
    f=readj(root/'freeze.json')
    p=root/'frozen_v5_instances.jsonl'
    if f.get('schema')!='prd32_20261030_fixed_dev_freeze_v1':
        raise ValueError('ORIGINAL_V1_FREEZE_SCHEMA_MISMATCH')
    if sha(p)!=f.get('frozen_instance_manifest_sha256'):
        raise ValueError('ORIGINAL_V1_MANIFEST_SHA_MISMATCH')
    rows=readjl(p)
    ids=set(); by_case={}
    for r in rows:
        k=(r['case_id'],r['target_instance_id'])
        if k in ids:raise ValueError('V1_DUPLICATED_INSTANCE_ID')
        ids.add(k)
        cid=r['case_id']
        entry=(r['image_path'],r['image_sha256'])
        if cid in by_case and by_case[cid]!=entry:
            raise ValueError('V1_CASE_HAS_MIXED_IMAGES')
        by_case[cid]=entry
    if len(by_case)!=12 or len(rows)!=31 or f.get('images')!=12 or f.get('instances')!=31:
        raise ValueError('EXPECTED_ORIGINAL_12_IMAGES_31_INSTANCES')
    original=[]
    for cid in sorted(by_case):
        path,digest=by_case[cid]
        if not Path(path).is_file() or sha(path)!=digest:
            raise ValueError('V1_SOURCE_IMAGE_MISSING_OR_CHANGED:'+cid)
        original.append({'case_id':cid,'image_path':str(Path(path).resolve()),'image_sha256':digest,
                         'cohort_origin':'FROZEN_V1_UNCHANGED'})
    if [x['case_id'] for x in original]!=[f'fresh_dev_{i:03d}' for i in range(1,13)]:
        raise ValueError('UNEXPECTED_V1_CASE_NAMES')
    return original, f


def load_candidates(inventory,project,data,max_per_class):
    if not inventory.is_file():
        raise FileNotFoundError('SOURCE_INVENTORY_MISSING:'+str(inventory))
    by_path={};reject=Counter(); read_rows=0;path_cache={}
    with inventory.open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f)
        if not {'source_image','garment_category'}.issubset(rd.fieldnames or []):
            raise ValueError('INVENTORY_REQUIRES_source_image_AND_garment_category_COLUMNS')
        for r in rd:
            read_rows+=1
            cat=(r.get('garment_category') or '').strip().lower()
            if cat not in CLASS_ORDER:
                reject['INVALID_CLASS_HINT']+=1; continue
            raw_src=r.get('source_image','')
            if raw_src not in path_cache:
                path_cache[raw_src]=safe_train_image(raw_src,project,data)
            p,err=path_cache[raw_src]
            if err:
                reject[err]+=1;continue
            key=str(p)
            by_path.setdefault(key,set()).add(cat)
    # Limit hashing work per class deterministically; never pick a random filename by directory order.
    grouped={cat:[] for cat in CLASS_ORDER}
    for path,hints in by_path.items():
        for cat in hints:grouped[cat].append(path)
    reduced=set()
    for cat in CLASS_ORDER:
        order=sorted(grouped[cat],key=lambda x:hstr(SEED+'|'+x))
        reduced.update(order[:max_per_class])
    return ({p:by_path[p] for p in sorted(reduced)},
            {'inventory_records':read_rows,'valid_unique_train_paths':len(by_path),
             'considered_unique_train_paths':len(reduced),
             'raw_class_hint_image_counts':{c:len(grouped[c]) for c in CLASS_ORDER},
             'inventory_rejections':dict(reject)})


def select_new(pool,original,exclusions,num,threshold,quotas):
    old_sha={r['image_sha256'] for r in original}
    baseline_hashes=[]
    for r in original:
        dh,_=d_hash(r['image_path'])
        baseline_hashes.append(dh)
    records=[];blocked=Counter();digest_seen=set()
    for path,hints in pool.items():
        try:
            if Path(path).stat().st_size>30_000_000:
                blocked['IMAGE_TOO_LARGE']+=1;continue
            s=sha(path)
            if s in old_sha:
                blocked['MATCH_EXISTING_V1']+=1;continue
            if exclusions is not None and s in exclusions:
                blocked['MATCH_APPROVED_EXCLUSION']+=1;continue
            if s in digest_seen:
                blocked['DUPLICATE_SHA_IN_POOL']+=1;continue
            digest_seen.add(s)
            dh,dimensions=d_hash(path)
            if any(hamming(dh,x)<=threshold for x in baseline_hashes):
                blocked['NEAR_DUPLICATE_EXISTING_V1']+=1;continue
            records.append({'image_path':path,'image_sha256':s,'dhash64':dh,'image_size':dimensions,
                            'source_category_hints_SELECTION_ONLY':sorted(hints)})
        except Exception:
            blocked['BAD_IMAGE_OR_IO']+=1
    records.sort(key=lambda r:hstr(SEED+'|'+r['image_sha256']))
    chosen=[]; chosen_hashes=[]; assigned=Counter(); missing={}
    # Scarcity-first allocation: rare categories get a chance before abundant ones.
    categories=sorted(CLASS_ORDER,key=lambda c:(sum(c in x['source_category_hints_SELECTION_ONLY'] for x in records)/max(1,quotas[c]),CLASS_ORDER.index(c)))
    def add(r,c):
        if r['image_sha256'] in {x['image_sha256'] for x in chosen}:return False
        if any(hamming(r['dhash64'],dh)<=threshold for dh in chosen_hashes):return False
        chosen.append(dict(r,selection_stratum_NOT_GT=c))
        chosen_hashes.append(r['dhash64']);assigned[c]+=1
        return True
    for c in categories:
        target=quotas[c]
        for r in records:
            if assigned[c]>=target:break
            if c in r['source_category_hints_SELECTION_ONLY']:
                add(r,c)
        if assigned[c]<target: missing[c]=target-assigned[c]
    # Fall back to a deterministic fill if strata lack eligible samples.
    for r in records:
        if len(chosen)>=num:break
        if r['image_sha256'] in {x['image_sha256'] for x in chosen}:continue
        candidates=r['source_category_hints_SELECTION_ONLY']
        c=min(candidates,key=lambda x:(assigned[x]/max(1,quotas[x]),CLASS_ORDER.index(x)))
        add(r,c)
    if len(chosen)!=num:
        raise ValueError('INSUFFICIENT_DISTINCT_TRAIN_IMAGES: need '+str(num)+', found '+str(len(chosen))+
                         '; relax class/dHash limits ONLY after review of candidate pool')
    chosen.sort(key=lambda x:(CLASS_ORDER.index(x['selection_stratum_NOT_GT']),hstr(SEED+x['image_sha256'])))
    return chosen,{'pool_eligible_after_hash_and_v1_screen':len(records),'rejection_counts':dict(blocked),
                   'stratum_shortfalls_first_pass':missing,'selected_strata_counts':dict(assigned)}


def preview_page(rows, dest):
    from PIL import Image
    cards=[]
    for r in rows:
        with Image.open(r['image_path']) as im:
            image=im.convert('RGB')
            image.thumbnail((460,420))
            b=io.BytesIO()
            image.save(b,format='JPEG',quality=74,optimize=True)
            data=base64.b64encode(b.getvalue()).decode('ascii')
        cards.append('<article><b>'+html.escape(r['case_id'])+'</b><img alt="训练来源开发候选图片" src="data:image/jpeg;base64,'+data+""""/><p>仅供人工核对重复、场景和目标类别覆盖；预览不包含来源类别提示。</p></article>""")
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>PRD32 · 48 张新增开发候选图片</title><style>body{font:15px/1.5 system-ui,'Microsoft YaHei',sans-serif;background:#f3f5f9;color:#172b46;max-width:1320px;margin:auto;padding:24px}header{background:white;padding:16px;border-radius:12px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}article{background:white;border:1px solid #cbd5e1;border-radius:12px;padding:14px}img{width:100%;height:270px;object-fit:contain;background:#edf1f5}p{font-size:13px;color:#657184}</style><header><h1>PRD 3.2 48 张新增开发集候选图</h1><p>来自训练来源图像。只做人工核对；不展示类别提示，不代表自动标注完成。原有 12 张沿用旧冻结版本。</p></header><main>'''+''.join(cards)+'</main></html>'
    dest.write_text(page,encoding='utf-8')


def build(args):
    out=args.output.resolve(strict=False)
    if out.exists():raise FileExistsError('REFUSE_OVERWRITE:'+str(out))
    original,freeze=load_v1(args.v1)
    data=args.data_root.resolve()
    project=args.project_root.resolve()
    if not data.is_dir():raise FileNotFoundError('DATA_ROOT_NOT_FOUND:'+str(data))
    # Never inspect paths beneath a holdout or test directory, nor a sealed manifest.
    inventory=(args.inventory or project/'configs/prd_8class_unified_manifest_v2.csv').resolve()
    if any(restricted(x) for x in inventory.parts):
        raise ValueError('INVENTORY_MUST_NOT_BE_SEALED_OR_TEST')
    excl=read_approved_exclusions(args.exclusion_hashes)
    if excl and set(r['image_sha256'] for r in original)&excl:
        raise ValueError('ORIGINAL_DEV12_OVERLAPS_APPROVED_HOLDOUT_HASHES')
    pool,scan=load_candidates(inventory,project,data,args.max_per_class)
    selected,select_stats=select_new(pool,original,excl,args.total-12,args.dhash_distance,QUOTAS)
    new=[]
    for i,r in enumerate(selected,13):
        new.append(dict(r,case_id=f'fresh_dev_{i:03d}',cohort_origin='NEW_TRAIN_SOURCE'))
    full=original + new
    if len(full)!=args.total or len({r['image_sha256'] for r in full})!=args.total:
        raise ValueError('CROSS_COHORT_DUPLICATE_OR_WRONG_COUNT')
    # Verify all images immediately before creating immutable on-disk evidence.
    for r in full:
        if sha(r['image_path'])!=r['image_sha256']:
            raise ValueError('IMAGE_CHANGED_DURING_SCAN:'+r['case_id'])
    out.mkdir(parents=True,exist_ok=False)
    sanitized=[{'case_id':r['case_id'],'image_path':r['image_path'],
                'image_sha256':r['image_sha256'],'human_reference_withheld':True,
                'do_not_use_historical_gt_bbox':True} for r in full]
    # Model-facing input excludes all stratification metadata.
    writejl(out/'v5_image_only_inputs_60.jsonl',sanitized)
    writejl(out/'full_cohort_60_image_provenance.jsonl',[{'case_id':r['case_id'],
        'image_path':r['image_path'],'image_sha256':r['image_sha256'],
        'cohort_origin':r['cohort_origin']} for r in full])
    with (out/'new48_selection_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['case_id','image_sha256','image_path','stratum_hint_NOT_GT','all_category_hints_NOT_GT','width','height']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in new:
            w.writerow({'case_id':r['case_id'],'image_sha256':r['image_sha256'],
              'image_path':r['image_path'],'stratum_hint_NOT_GT':r['selection_stratum_NOT_GT'],
              'all_category_hints_NOT_GT':';'.join(r['source_category_hints_SELECTION_ONLY']),
              'width':r['image_size'][0],'height':r['image_size'][1]})
    preview_page(new,out/'new48_visual_review.html')
    coverage=Counter()
    for r in new:
        coverage.update(r['source_category_hints_SELECTION_ONLY'])
    report={'schema':SCHEMA+'_coverage', 'target_images':args.total, 'original_dev_images':12,
      'new_images':len(new),'new_selection_primary_strata_NOT_GT':select_stats['selected_strata_counts'],
      'new_category_hint_image_coverage_NOT_GT':{c:coverage[c] for c in CLASS_ORDER},
      'source_inventory':str(inventory),'source_inventory_sha256':sha(inventory),
      'inventory_scan':scan,'selection_diagnostics':select_stats,
      'near_duplicate_max_dhash_hamming':args.dhash_distance,
      'image_sha256_unique':len({r['image_sha256'] for r in full})==60,
      'holdout_digest_registry_provided':excl is not None,
      'holdout_digest_registry_sha256':sha(args.exclusion_hashes) if excl else None,
      'selection_hints_used_for_sampling_only':True,'selection_hints_in_model_input':False,
      'all_images_from_train_source':True,
      'warnings':['Stratification hints are NOT manually verified target GT',
         'Model may predict zero instances on some selected images',
         'Current 12 already reviewed images are development-only, not independent test',
         'Near-duplicate check is heuristic; manually inspect preview before acceptance',
         '60 images is an engineering target, NOT a statistical guarantee']}
    writej(out/'coverage_report.json',report)
    state='IMAGE_COHORT_FROZEN_WITH_APPROVED_HOLDOUT_HASH_EXCLUSION' if excl else 'PROVISIONAL_HOLDOUT_EXCLUSION_NOT_VERIFIED'
    if select_stats['stratum_shortfalls_first_pass']:
        state='PROVISIONAL_CLASS_STRATA_SHORTFALL' if excl else 'PROVISIONAL_HOLDOUT_AND_CLASS_COVERAGE_NOT_VERIFIED'
    manifest={'schema':SCHEMA,'status':state,'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
      'images':len(full),'original_images':12,'new_images':len(new),
      'case_ids':[r['case_id'] for r in full],
      'original_v1_freeze_sha256':sha(args.v1/'freeze.json'),
      'original_v1_instances_sha256':freeze['frozen_instance_manifest_sha256'],
      'model_image_only_manifest_sha256':sha(out/'v5_image_only_inputs_60.jsonl'),
      'provenance_manifest_sha256':sha(out/'full_cohort_60_image_provenance.jsonl'),
      'selection_audit_sha256':sha(out/'new48_selection_audit.csv'),
      'coverage_report_sha256':sha(out/'coverage_report.json'),
      'preview_sha256':sha(out/'new48_visual_review.html'),
      'approved_exclusion_registry_sha256':sha(args.exclusion_hashes) if excl else None,
      'human_gt_bbox_used_for_inference':False,'qwen_run_completed':False,
      'v5_inference_completed_for_60':False,'official_prd_acceptance':False,
      'publish_allowed':False,'main_pipeline_changed':False,
      'ready_for_model_inference':excl is not None and not bool(select_stats['stratum_shortfalls_first_pass'])}
    writej(out/'freeze_state.json',manifest)
    print('PRD32_DEV60_IMAGE_CANDIDATES_READY',len(full),'images',len(new),'new')
    print('FREEZE_STATE:',state)
    print('MODEL_INFERENCE_READY:',manifest['ready_for_model_inference'])
    print('SOURCE_CLASS_COVERAGE_HINTS_NOT_GT:',json.dumps(report['new_category_hint_image_coverage_NOT_GT'],ensure_ascii=False))
    print('OUTPUT:',out)
    return manifest


def verify(args):
    root=args.output.resolve()
    state=readj(root/'freeze_state.json')
    if state.get('schema')!=SCHEMA:raise ValueError('BAD_FREEZE_SCHEMA')
    for field,name in [('model_image_only_manifest_sha256','v5_image_only_inputs_60.jsonl'),
                       ('provenance_manifest_sha256','full_cohort_60_image_provenance.jsonl'),
                       ('selection_audit_sha256','new48_selection_audit.csv'),
                       ('coverage_report_sha256','coverage_report.json'),
                       ('preview_sha256','new48_visual_review.html')]:
        if sha(root/name)!=state[field]:raise ValueError('OUTPUT_CHANGED:'+name)
    original,f=load_v1(args.v1)
    if sha(args.v1/'freeze.json')!=state['original_v1_freeze_sha256']:
        raise ValueError('V1_FREEZE_CHANGED')
    rows=readjl(root/'v5_image_only_inputs_60.jsonl')
    if len(rows)!=60 or [r['case_id'] for r in rows]!=[f'fresh_dev_{i:03d}' for i in range(1,61)]:
        raise ValueError('CASE_IDENTITIES_CHANGED')
    if rows[:12]!=[{'case_id':r['case_id'],'image_path':r['image_path'],
                 'image_sha256':r['image_sha256'],'human_reference_withheld':True,
                 'do_not_use_historical_gt_bbox':True} for r in original]:
        raise ValueError('ORIGINAL_12_CASES_CHANGED')
    for r in rows:
        if sha(r['image_path'])!=r['image_sha256']:raise ValueError('IMAGE_HASH_MISMATCH:'+r['case_id'])
    print('PRD32_DEV60_VERIFY_OK',len(rows),'images; status:',state['status'])
    return state


def export(args):
    from zipfile import ZipFile,ZIP_DEFLATED
    verify(args)
    out=args.output/'PRD32_DEV60_STAGE1_REVIEW.zip'
    if out.exists():raise FileExistsError('REFUSE_OVERWRITE:'+str(out))
    names=('freeze_state.json','coverage_report.json','new48_selection_audit.csv',
           'v5_image_only_inputs_60.jsonl','new48_visual_review.html')
    with ZipFile(out,'w',ZIP_DEFLATED) as z:
        for name in names:z.write(args.output/name,name)
    print('PRD32_DEV60_REPORT_PACKAGE_OK:',out)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('build','verify','export'))
    parser.add_argument('--project-root',type=Path,default=Path('/workspace/fashion_multimodal_analysis'))
    parser.add_argument('--data-root',type=Path,default=Path('/workspace/fashion_data'))
    parser.add_argument('--v1',type=Path,default=Path('/workspace/PRD32_FIXED_DEV_FREEZE_20261030'))
    parser.add_argument('--inventory',type=Path,default=None)
    parser.add_argument('--exclusion-hashes',type=Path,default=None,
                        help='Optional approved, DIGEST-ONLY list of >=8 sealed holdout SHA256 values. Never open sealed file.')
    parser.add_argument('--output',type=Path,default=Path('/workspace/PRD32_FIXED_DEV_V2_60IMG_CANDIDATES_20261030'))
    parser.add_argument('--max-per-class',type=int,default=900)
    parser.add_argument('--dhash-distance',type=int,default=5)
    parser.add_argument('--total',type=int,default=60,help='Must be 60 for the official V2 tool')
    args=parser.parse_args()
    try:
        if args.total!=60 or args.max_per_class<48 or not (0<=args.dhash_distance<=12):
            raise ValueError('BAD_ARGUMENTS')
        if args.command=='build':build(args)
        elif args.command=='verify':verify(args)
        else:export(args)
    except Exception as exc:
        print('PRD32_DEV60_BLOCKED',type(exc).__name__+': '+str(exc),file=sys.stderr)
        raise SystemExit(2)

if __name__=='__main__':main()
