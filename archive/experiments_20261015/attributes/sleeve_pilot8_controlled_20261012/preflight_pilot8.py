#!/usr/bin/env python3
"""Read-only preflight: sleeve pilot8. Does not modify data or train a model."""
import argparse,csv,hashlib,json
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()

def cli():
    ap=argparse.ArgumentParser(description='Read-only pilot8 data preflight')
    ap.add_argument('--repo',default='/workspace/fashion_multimodal_analysis',help='fashion_multimodal_analysis repository root')
    ap.add_argument('--train-csv',help='Override fixed train CSV path')
    ap.add_argument('--val-csv',help='Override fixed val CSV path')
    ap.add_argument('--core400-hashes',help='Optional one-hash-per-line Core400 source image SHA256 list, or a JSON list of SHA256 strings; do not supply predictions instead')
    ap.add_argument('--output',default='pilot8_preflight_report.json')
    args=ap.parse_args()
    here=Path(__file__).resolve().parent
    qa=json.loads((here/'pilot8_candidates_FINAL.json').read_text(encoding='utf-8'))
    old=json.loads((here/'parent_split_index.json').read_text(encoding='utf-8'))
    repo=Path(args.repo).expanduser().resolve()
    train=Path(args.train_csv) if args.train_csv else repo/'configs/prd_8class_train_v5_balanced150_20261008.csv'
    val=Path(args.val_csv) if args.val_csv else repo/'configs/prd_8class_val_v2_full_targets.csv'
    expected={'train':qa['audit_train_csv_sha256'],'val':qa['audit_val_csv_sha256']}
    check={}
    for key,path in [('train',train),('val',val)]:
        found=sha(path) if path.is_file() else None
        check[f'{key}_csv']={'path':str(path),'found':found,'expected':expected[key],'pass':found==expected[key]}
    oldhash={r['source_image_sha256'] for r in old['records']}
    oldids={r['record_id'] for r in old['records']}
    qaids=[a['record_id'] for a in qa['annotations']]
    clean=len(qaids)==len(set(qaids))==8 and len(set(a['source_image_sha256'] for a in qa['annotations']))==8
    check['pilot8_unique']=clean
    check['old_split_record_disjoint']=not bool(set(qaids) & oldids)
    check['old_split_image_sha256_disjoint']=not bool({a['source_image_sha256'] for a in qa['annotations']} & oldhash)
    images=[]
    for a in qa['annotations']:
        # source_image is relative to the repository root, intentionally ../fashion_data.
        p=(repo/a['source_image']).resolve()
        found=sha(p) if p.is_file() else None
        good=found==a['source_image_sha256']
        images.append({'record_id':a['record_id'],'path':str(p),'sha256_ok':good,'exists':p.is_file()})
    check['all_source_images_verified']=all(v['sha256_ok'] for v in images)
    check['image_detail']=images
    check['no_unreviewed_records']=len(qa['annotations'])==8 and all(
        a['visual_qa_status']=='PASS_FOR_CONTROLLED_PILOT' and
        a['state']=='measurable' and a['current_class'] in ('cap','short','elbow') and
        a['review_status']=='confirmed' and a['same_arm_confirmed'] and
        a['class_confirmed'] and not a['sleeveless_confirmed'] and
        0 <= a['ratio'] <= 1 and set(a['points'])=={'shoulder','elbow','wrist','cuff'} and
        all(len(x)==2 and all(0<=v<=1 for v in x) for x in a['points'].values())
        for a in qa['annotations'])
    check['core400_overlap_checked']=False
    check['core400_overlap_free']=None
    if args.core400_hashes:
        p=Path(args.core400_hashes)
        if p.is_file():
            if p.suffix.lower()=='.json':
                v=json.loads(p.read_text(encoding='utf-8'))
                if not isinstance(v,list) or not all(isinstance(x,str) for x in v):
                    raise ValueError('Core400 JSON must be a list of SHA256 strings')
                ch=set(v)
            else:
                ch=set(p.read_text(encoding='utf-8').split())
            if len(ch) != 400:  # Require complete index, not an arbitrary partial sample.
                raise ValueError(f'Expected 400 distinct Core400 source image hashes; received {len(ch)}')
            if any(len(s)!=64 or any(c not in '0123456789abcdef' for c in s.lower()) for s in ch):
                raise ValueError('Invalid Core400 SHA256 value')
            check['core400_overlap_checked']=True
            check['core400_overlap_free']=not bool(ch & {a['source_image_sha256'] for a in qa['annotations']})
    required=['pilot8_unique','old_split_record_disjoint','old_split_image_sha256_disjoint','all_source_images_verified','no_unreviewed_records']
    structural=all(check[k] for k in required) and check['train_csv']['pass'] and check['val_csv']['pass']
    status='PREFLIGHT_PASSED_CORE400_PENDING' if structural and not check['core400_overlap_checked'] else (
        'PREFLIGHT_PASSED_NO_OVERLAP' if structural and check['core400_overlap_free'] else 'PREFLIGHT_FAILED')
    report={'status':status,'source':str(here),'checks':check,
        'safe_to_automerge':False,'training_executed':False,
        'note':'Passing these checks does not establish model gain, valid Core400 coverage or experimental reproducibility. No data were modified.'}
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PREFLIGHT_STATUS:',status)
    print('REPORT:',str(Path(args.output).resolve()))
    if status=='PREFLIGHT_FAILED':
        raise SystemExit(2)

if __name__=='__main__': cli()
