"""PRD3.2 historical V5 + Qwen result replay, no model calls or output publishing.

Command-line readers access only explicitly provided DEV60 files; no directory
scans. Every observed item is reconciled to frozen DEV60 Guard V5 policy.

What historical logs do NOT establish: ROI/Mask per-request source identity,
model-level response identity, independent GT, or actual end-to-end correction.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, re
from pathlib import Path
from collections import Counter,defaultdict

EXPECTED_INSTANCES=171
TASKS=('tags','vqa','copy')
VARIANTS=('baseline','focus_v1')
SHA=re.compile(r'^[a-f0-9]{64}$')
FORBIDDEN=('holdout','sealed','blind','independent_test')


def validate_path(path:Path, fixture:bool=False)->Path:
    p=path.expanduser().resolve()
    if any(x in str(p).lower() for x in FORBIDDEN):
        raise ValueError(f'Holdout-related path forbidden: {p}')
    if not fixture and ('DEV60' not in str(p).upper() or p.suffix.lower()!='.jsonl'):
        raise ValueError(f'Expected explicit DEV60 JSONL file: {p}')
    if not p.is_file(): raise FileNotFoundError(p)
    return p


def load_policy(path:Path)->dict:
    with path.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    if len(rows)!=EXPECTED_INSTANCES:raise ValueError(f'Frozen policy scope changed: {len(rows)}')
    result={}
    for r in rows:
        k=(r['case_id'],r['instance_id'])
        if k in result:raise ValueError(f'Duplicate policy key {k}')
        if not SHA.fullmatch(r['image_sha256']):raise ValueError(f'Bad policy image hash {k}')
        if r['v5_mainline_approved']!='NO' or r['v5_holdout_opened']!='NO':
            raise ValueError('Frozen safety gates changed')
        result[k]=r
    return result


def jsonlines(path:Path):
    with path.open(encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            if line.strip():
                obj=json.loads(line)
                if not isinstance(obj,dict):raise ValueError(f'JSONL object required: {path}:{n}')
                yield n,obj


def valid_bbox(box):
    if not isinstance(box,list) or len(box)!=4:return False
    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in box):return False
    return box[0]>=0 and box[1]>=0 and box[2]>box[0] and box[3]>box[1]


def compare_bbox(a,b,tol=0.002):
    return valid_bbox(a) and valid_bbox(b) and all(abs(x-y)<=tol for x,y in zip(a,b))


def validate_one(row:dict, expected_variant:str, frozen:dict)->list[str]:
    issues=[]
    case,inst,task=row.get('case_id'),row.get('target_instance_id'),row.get('task')
    key=(case,inst)
    if key not in frozen:
        return ['UNKNOWN_OR_UNFROZEN_INSTANCE_KEY']
    policy=frozen[key]
    if row.get('model_variant')!=expected_variant:issues.append('VARIANT_MISMATCH')
    if task not in TASKS:issues.append('UNKNOWN_TASK')
    if row.get('request_id')!=f'{case}__{inst}__{task}':issues.append('REQUEST_IDENTITY_OR_TASK_MISMATCH')
    if row.get('image_sha256')!=policy['image_sha256']:issues.append('SOURCE_IMAGE_SHA_MISMATCH')
    if row.get('v5_category_unverified')!=policy['v5_category_unverified']:
        issues.append('V5_CATEGORY_SOURCE_MISMATCH')
    if not valid_bbox(row.get('v5_predicted_bbox_xyxy')):issues.append('BAD_TARGET_BBOX')
    if row.get('publish_allowed') is not False:issues.append('SOURCE_UNAUTHORIZED_PUBLISH_FLAG')
    if row.get('status')!='GENERATED_REVIEW_REQUIRED':issues.append('UNEXPECTED_HISTORICAL_STATUS')
    if not isinstance(row.get('marked_image_sha256'),str) or not SHA.fullmatch(row['marked_image_sha256']):
        issues.append('INVALID_MARKED_IMAGE_DIGEST')
    if task=='tags':
        tag=row.get('tag_parse')
        if not isinstance(tag,dict) or tag.get('parseable') is not True or tag.get('schema_valid') is not True:
            issues.append('TAGS_PARSE_INVALID')
        else:
            out=tag.get('canonical_service_json') or {}
            if out.get('category') not in {'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}:
                issues.append('TAGS_CATEGORY_OUT_OF_TAXONOMY')
    if not isinstance(row.get('raw_text'),str):issues.append('MISSING_RAW_RESPONSE')
    for source_field in ('roi_instance_id','mask_instance_id','response_instance_id'):
        if source_field in row and row[source_field]!=inst:
            issues.append('CROSS_INSTANCE_'+source_field.upper())
    if row.get('__marked_image_verified__') is False:
        issues.append('MARKED_IMAGE_BYTES_SHA_MISMATCH_OR_MISSING')
    # Actual logs do not independently expose *source* ROI/Mask instance IDs.
    # Do not invent validation for these missing runtime provenance fields.
    return issues


def replay(policy:dict, records:dict[str,list[dict]], v5:dict|None=None)->tuple[list[dict],dict]:
    expected=set(policy)
    by_key=defaultdict(dict)
    raw_count=0
    for variant in VARIANTS:
        seen=set()
        for r in records[variant]:
            raw_count+=1
            k=(r.get('case_id'),r.get('target_instance_id'))
            identity=(k,r.get('task'))
            if identity in seen:raise ValueError(f'Duplicate Qwen task record: {variant}, {identity}')
            seen.add(identity)
            if k not in expected:raise ValueError(f'Qwen record outside frozen DEV60: {k}')
            by_key[k][(variant,r.get('task'))]=(r,validate_one(r,variant,policy))
    if not by_key:raise ValueError('No Qwen records loaded')
    rows=[]; counts=Counter(); issues_count=Counter();observed=0
    for k,p in policy.items():
        gathered=by_key.get(k,{})
        issues=[];flags=[]
        if gathered:
            observed+=1
            issues.extend(issue for _,i in gathered.values() for issue in i)
            missing=[(v,t) for v in VARIANTS for t in TASKS if (v,t) not in gathered]
            if missing:issues.append('INCOMPLETE_TASK_VARIANT_GRID')
            # Task metadata must agree on marked image & source bbox within each variant.
            for variant in VARIANTS:
                per=[entry[0] for (v,t),entry in gathered.items() if v==variant]
                if per:
                    shas={r.get('marked_image_sha256') for r in per}
                    if len(shas)!=1:issues.append('CROSS_TASK_MARKED_IMAGE_MISMATCH')
                    reference=per[0].get('v5_predicted_bbox_xyxy')
                    if any(not compare_bbox(reference,r.get('v5_predicted_bbox_xyxy')) for r in per):
                        issues.append('CROSS_TASK_TARGET_BBOX_MISMATCH')
            if ('baseline','tags') in gathered and ('focus_v1','tags') in gathered:
                b=gathered[('baseline','tags')][0];f=gathered[('focus_v1','tags')][0]
                if b.get('marked_image_sha256')!=f.get('marked_image_sha256'):
                    issues.append('CROSS_VARIANT_MARKED_IMAGE_MISMATCH')
                if not compare_bbox(b.get('v5_predicted_bbox_xyxy'),f.get('v5_predicted_bbox_xyxy')):
                    issues.append('CROSS_VARIANT_TARGET_BBOX_MISMATCH')
                bcat=(b.get('tag_parse') or {}).get('canonical_service_json') or {}
                fcat=(f.get('tag_parse') or {}).get('canonical_service_json') or {}
                if bcat.get('category')!=fcat.get('category'):
                    flags.append('QWEN_BASELINE_FOCUS_CLASS_DISAGREEMENT')
                if fcat.get('category')!=p.get('v5_category_unverified'):
                    flags.append('V5_QWEN_FOCUS_CLASS_DISAGREEMENT')
            if v5 is not None:
                if k not in v5: issues.append('V5_RAW_RECORD_MISSING')
                else:
                    raw=v5[k]
                    if raw.get('image_sha256') != p['image_sha256']:
                        issues.append('V5_RAW_IMAGE_SHA_MISMATCH')
                    if raw.get('source_image_path') is None:
                        issues.append('V5_RAW_SOURCE_PATH_MISSING')
                    # Full segmentation object structure is not standardized; do not
                    # infer per-instance bbox from a top-level segmentation list.
        else: flags.append('NO_QWEN_REPLAY_RECORD_IN_INPUT_SCOPE')
        if p['shadow_alias_candidate']=='YES':flags.append('ALIAS_QUARANTINE')
        if p['category_agreement']=='NO':flags.append('HISTORICAL_V5_QWEN_FOCUS_CONFLICT')
        if p.get('v5_review_status')=='USER_CONFIRMED_IN_CHAT':flags.append('CONFIRMED_REVIEW_ADVISORY')
        if p.get('v5_gt_determinate')=='NO':flags.append('CATEGORY_GT_UNRESOLVED')
        if p.get('v5_policy','').startswith('BLOCK_'):flags.append('FROZEN_V5_POLICY_BLOCK')
        if issues: status='BLOCK_METADATA_INTEGRITY'
        elif gathered:status='SHADOW_ONLY_RUNTIME_BINDING_UNVERIFIED'
        else:status='NO_QWEN_REPLAY_SCOPE'
        issues=list(dict.fromkeys(issues));flags=list(dict.fromkeys(flags))
        for v in issues:issues_count[v]+=1
        counts[status]+=1
        rows.append({
            'case_id':k[0],'instance_id':k[1],'image_sha256':p['image_sha256'],
            'qwen_task_records':len(gathered),'expected_task_records':6,
            'v5_category':p['v5_category_unverified'],
            'qwen_baseline_tags':((gathered.get(('baseline','tags')) or ({},[]))[0].get('tag_parse') or {}).get('canonical_service_json',{}).get('category',''),
            'qwen_focus_tags':((gathered.get(('focus_v1','tags')) or ({},[]))[0].get('tag_parse') or {}).get('canonical_service_json',{}).get('category',''),
            'frozen_guard_v5_policy':p.get('v5_policy',''),
            'replay_status':status,
            'metadata_violations':';'.join(issues),'risk_flags':';'.join(flags),
            'roi_instance_origin_verified':'NO_HISTORICAL_EVIDENCE',
            'mask_instance_origin_verified':'NO_HISTORICAL_EVIDENCE',
            'qwen_response_source_verified':'NO_INDEPENDENT_EVIDENCE',
            'model_outputs_modified':'NO','auto_category_override':'NO',
            'auto_alias_merge':'NO','publish_allowed':'NO','holdout_accessed':'NO',
        })
    summary={
        'schema':'prd32_v5_qwen_pipeline_shadow_replay_v1',
        'status':'SHADOW_REPLAY_COMPLETE_NOT_END_TO_END_APPROVED',
        'frozen_dev60_instances':len(policy),
        'instances_with_qwen_records':observed,
        'qwen_records_seen':raw_count,
        'metadata_status_counts':dict(counts),
        'metadata_violation_counts':dict(issues_count),
        'historical_roi_mask_response_source_proven':False,
        'frozen_alias_quarantine':sum(x['shadow_alias_candidate']=='YES' for x in policy.values()),
        'frozen_category_conflicts':sum(x['category_agreement']=='NO' for x in policy.values()),
        'holdout_opened':False,'model_called':False,'models_retrained':False,
        'source_predictions_modified':False,'category_auto_overrides':0,'alias_auto_merges':0,
        'production_approved':False,
        'meaning':'Historical request metadata reconciliation only; GPU live source instrumentation still needed.',
    }
    return rows,summary


def load_v5(path:Path)->dict:
    result={}
    for _,r in jsonlines(path):
        k=(r.get('case_id'),)
        if not k[0]:raise ValueError('V5 raw record missing case_id')
        if k in result:raise ValueError(f'Duplicate V5 raw per image: {k}')
        result[k]=r
    # Expand per-image provenance lazily during reconcile
    return result


def write_result(rows,summary,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'shadow_decisions_171.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'shadow_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'shadow_risk_queue.csv').open('w',encoding='utf-8-sig',newline='') as f:
        r=[x for x in rows if x['qwen_task_records'] and (x['metadata_violations'] or x['risk_flags'])]
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(r)


def verify_marked_images(records:dict[str,list[dict]])->dict:
    cache={};ok=0;bad=0
    for variant in VARIANTS:
        required_root=Path('/workspace/PRD32_DEV60_QWEN_BASELINE_20261030_run01/marked_inputs' if variant=='baseline' else '/workspace/PRD32_DEV60_QWEN_FOCUS_V1_20261030_run01/marked_inputs').resolve()
        for row in records[variant]:
            filename=row.get('marked_image')
            path=Path(filename).resolve() if isinstance(filename,str) else None
            if path is None or required_root not in path.parents or not path.is_file():
                row['__marked_image_verified__']=False;bad+=1;continue
            if path not in cache:
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
                cache[path]=h.hexdigest()
            row['__marked_image_verified__']=(cache[path]==row.get('marked_image_sha256'))
            if row['__marked_image_verified__']:ok+=1
            else:bad+=1
    return {'marked_image_record_sha_matches':ok,'marked_image_record_sha_mismatches':bad,'unique_files_verified':len(cache)}

def run(*,baseline:Path|None=None,focus:Path|None=None,fixture:Path|None=None,out:Path, v5_raw:Path|None=None,verify_marked:bool=False):
    root=Path(__file__).resolve().parent
    policy=load_policy(root/'inputs/v5_dev60_policy_171.csv')
    if fixture:
        p=validate_path(fixture,True);allr=[r for _,r in jsonlines(p)]
        records={v:[x for x in allr if x.get('model_variant')==v] for v in VARIANTS}
    else:
        if baseline is None or focus is None:raise ValueError('Provide both baseline and focus JSONL inputs')
        b=validate_path(baseline); f=validate_path(focus)
        if b==f:raise ValueError('Baseline and Focus input paths must be distinct')
        records={'baseline':[r for _,r in jsonlines(b)], 'focus_v1':[r for _,r in jsonlines(f)]}
    marked_verification=verify_marked_images(records) if verify_marked else None
    if fixture and verify_marked:raise ValueError('Fixture mode does not have full GPU marked-image files')
    raw=None
    if v5_raw is not None:
        vp=validate_path(v5_raw)
        raw_per_case=load_v5(vp)
        raw={(case,inst):raw_per_case.get((case,),{}) for case,inst in policy}
    rows,summary=replay(policy,records,raw)
    summary['input_kind']='FIXTURE_32_SELECTED' if fixture else 'GPU_DEV60_HISTORICAL'
    summary['qwen_input_sha256']={v:hashlib.sha256((fixture if fixture else baseline if v=='baseline' else focus).read_bytes()).hexdigest() for v in VARIANTS}
    summary['guard_v5_policy_sha256']=hashlib.sha256((root/'inputs/v5_dev60_policy_171.csv').read_bytes()).hexdigest()
    summary['v5_raw_used']=v5_raw is not None
    summary['marked_image_file_bytes_verified']=bool(verify_marked)
    summary['marked_image_verification']=marked_verification
    write_result(rows,summary,out)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture',action='store_true',help='Replay included 32 targeted DEV instances only')
    ap.add_argument('--baseline',type=Path,default=Path('/workspace/PRD32_DEV60_QWEN_BASELINE_20261030_run01/qwen_results.jsonl'))
    ap.add_argument('--focus',type=Path,default=Path('/workspace/PRD32_DEV60_QWEN_FOCUS_V1_20261030_run01/qwen_results.jsonl'))
    ap.add_argument('--v5-raw',type=Path,default=None,help='Explicit DEV60 V5 raw provenance file (optional)')
    ap.add_argument('--verify-marked-images',action='store_true',help='Hash only the exact marked DEV60 file paths recorded by Qwen logs')
    ap.add_argument('--out',type=Path,default=Path('/workspace/PRD32_1106_PIPELINE_SHADOW_RESULTS'))
    args=ap.parse_args()
    summary=run(fixture=(Path(__file__).resolve().parent/'inputs/qwen_prior32_192.jsonl' if args.fixture else None),baseline=args.baseline,focus=args.focus,out=args.out,v5_raw=args.v5_raw,verify_marked=args.verify_marked_images)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
