#!/usr/bin/env python3
"""Metadata-only historical bridge audit plus hash-verified fresh TRAIN image-only cohort.
NEVER loads V5, Qwen, old validation/holdout images, or human GT labels into inference.
"""
import argparse, hashlib, json, re, sys
from pathlib import Path

BANNED = {'holdout','sealed','fullval','val','valid','validation','test','testing','testset','eval','evaluation','devtest'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for v in iter(lambda:f.read(1<<20),b''):h.update(v)
    return h.hexdigest()

def jread(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def jlread(path):
    rows=[]
    for i,line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():continue
        obj=json.loads(line)
        if not isinstance(obj,dict):raise ValueError(f'{path}:{i} not object')
        rows.append(obj)
    return rows

def save_json(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def is_banned_name(name):
    s=name.lower()
    return any(s==b or s.startswith(b+'_') or s.endswith('_'+b) for b in BANNED)

def path_policy(path,root):
    """No access to path, only lexical/resolve metadata; never open unless allowed."""
    if not isinstance(path,str) or not path.strip():return 'MISSING_PATH'
    p=Path(path).expanduser(); parts=p.parts
    if not p.is_absolute():return 'NOT_ABSOLUTE'
    if any(is_banned_name(x) for x in parts):return 'RESTRICTED_SPLIT'
    rp=p.resolve(strict=False); rr=Path(root).resolve(strict=False)
    if any(is_banned_name(x) for x in rp.parts):return 'RESTRICTED_RESOLVED_SPLIT'
    if not rp.is_relative_to(rr):return 'OUTSIDE_APPROVED_DATA_ROOT'
    if 'train' not in {x.lower() for x in rp.parts}:return 'MISSING_TRAIN_SOURCE_COMPONENT'
    if rp.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:return 'NOT_SUPPORTED_IMAGE_EXT'
    return 'TRAIN_PATH_ELIGIBLE'

def run(a):
    project=Path(a.project).resolve();freeze=Path(a.freeze_dir).resolve();fnew=Path(a.newdev_frozen).resolve()
    output=Path(a.output).resolve()
    if output.exists():raise ValueError('Refusing to overwrite output: '+str(output))
    summary=jread(freeze/'summary.json');record=jread(freeze/'freeze_record.json')
    if summary.get('status')!='PRD31_STAGE_FREEZE_READY' or summary.get('phase_freeze_ready') is not True:raise ValueError('PRD31 freeze stage is not READY')
    if record.get('frozen') is not True or record.get('candidate_id')!=summary.get('candidate_id'):raise ValueError('Frozen candidate identity mismatch')
    if summary.get('checkpoint_check',{}).get('passed') is not True:raise ValueError('Historical frozen checkpoint check failed')
    expected=record.get('checkpoint_sha256')
    original=summary['checkpoint_check']
    if expected!=original.get('expected_sha256') or expected!=original.get('actual_sha256'):raise ValueError('Frozen record and summary checkpoint SHA mismatch')
    cp=(project/'outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth').resolve()
    frozen_path=Path(original['path']).resolve()
    if cp!=frozen_path:raise ValueError('Checkpoint path not same as frozen stage: '+str(frozen_path))
    checkpoint={'path':str(cp),'expected_sha256':expected,'exists':cp.is_file(),'hash_verified_now':False,'actual_sha256':None}
    if cp.is_file() and not a.skip_checkpoint_hash:
        actual=sha(cp);checkpoint['actual_sha256']=actual;checkpoint['hash_verified_now']=actual==expected
        if actual!=expected:raise ValueError('V5 checkpoint changed since freeze')
    bridge_path=freeze/'bridge_inputs.jsonl'
    bridge=jlread(bridge_path)
    seen=set();source=[]
    for row in bridge:
        cid=row.get('case_id');
        if cid in seen or not isinstance(cid,str):raise ValueError('Duplicate or missing bridge case ID')
        seen.add(cid)
        decision=path_policy(row.get('image_path'),a.data_root)
        source.append({'case_id':cid,'task':row.get('task'),'path_policy':decision,'available_for_new_inference':False,
                       'reason':'HISTORICAL_BRIDGE_REPLAY_PROVENANCE_NOT_AUTOMATICALLY_APPROVED'})
    # New 12 from a *different*, explicitly TRAIN-only cohort, using the frozen input plan but NOT its bbox or human labels.
    cohort=jread(fnew/'freeze.json')
    input_file=fnew/'frozen_inference_plan.jsonl'
    reference_file=fnew/'frozen_human_reference.jsonl'
    if cohort.get('status')!='FROZEN_DEV_NOT_INDEPENDENT':raise ValueError('Unexpected new-dev freeze schema/status')
    if sha(input_file)!=cohort.get('plan_sha256') or sha(reference_file)!=cohort.get('human_reference_sha256'):
        raise ValueError('New-dev frozen hashes changed; fail closed')
    rows=jlread(input_file)
    if len(rows)!=cohort.get('total') or len(rows)!=12:raise ValueError('Expected 12 new-dev image identities')
    candidates=[];rejected=[];seenid=set();seenhash=set()
    for row in rows:
        cid=row.get('case_id');p=row.get('image_path');image_sha=row.get('image_sha256')
        if cid in seenid or image_sha in seenhash or not cid or not image_sha:raise ValueError('Duplicate/empty identity in new cohort')
        seenid.add(cid);seenhash.add(image_sha)
        decision=path_policy(p,a.data_root)
        if decision!='TRAIN_PATH_ELIGIBLE':
            rejected.append({'case_id':cid,'reason':decision});continue
        # Only paths successfully whitelisted by split, resolved root and suffix can EVER be opened.
        rp=Path(p).resolve(strict=False)
        if not rp.is_file():rejected.append({'case_id':cid,'reason':'ALLOWED_TRAIN_IMAGE_MISSING'});continue
        actual=sha(rp)
        if actual!=image_sha:rejected.append({'case_id':cid,'reason':'TRAIN_IMAGE_HASH_MISMATCH'});continue
        if row.get('bbox_origin')!='historical_inventory_gt_pilot_NOT_V5_prediction':raise ValueError('Unexpected historical input identity')
        # Absolutely no GT, labels, bbox, source category or prior predictions in V5 inference manifest.
        candidates.append({'case_id':cid,'image_path':str(rp),'image_sha256':image_sha,
          'requested_output':'PRD31_V5_RAW_PREDICTIONS_ONLY','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
          'do_not_use_historical_gt_bbox':True,'human_reference_withheld':True})
    if rejected:raise ValueError('New-dev cohort image identity/source invalid: '+str(rejected))
    report={'schema':'prd32_v5_live_preflight_v1','stage':'CHECKPOINT_AND_SOURCE_PROVENANCE_ONLY',
      'prd31_freeze_ready':True,'bridge_cases_checked':len(source),
      'bridge_source_warnings':[x for x in source if x['path_policy']!='TRAIN_PATH_ELIGIBLE'],
      'historical_bridge_not_automatically_approved_for_live_v5':True,
      'new_train_source_candidates_verified':len(candidates),'checkpoint':checkpoint,
      'ready_for_live_v5_invocation': bool(checkpoint['hash_verified_now'] and len(candidates)==12),
      'v5_inference_performed':False,'qwen_inference_performed':False,'image_reads_from_restricted_splits':0,
      'read_approved_train_images_for_sha':len(candidates),'human_gt_used_as_v5_bbox':False,
      'official_prd_acceptance':False,'note':'Inputs are explicit train-source diagnostic samples, not independent tests. Run original V5 inference only after verifying its invocation code and output schema.'}
    output.mkdir(parents=True)
    save_json(output/'v5_live_preflight_summary.json',report)
    with (output/'v5_image_only_inputs.jsonl').open('w',encoding='utf8') as f:
        for obj in candidates:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
    with (output/'bridge_metadata_provenance_audit.jsonl').open('w',encoding='utf8') as f:
        for obj in source:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
    print('V5_PREFLIGHT_DONE',json.dumps({'train_inputs':len(candidates),'bridge_path_warnings':len(report['bridge_source_warnings']),
       'checkpoint_verified':checkpoint['hash_verified_now'],'ready_for_next_stage':report['ready_for_live_v5_invocation']},ensure_ascii=False))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='/workspace/fashion_multimodal_analysis')
    p.add_argument('--freeze-dir',default='/workspace/prd31_stage_freeze_20261026')
    p.add_argument('--newdev-frozen',default='/workspace/PRD32_NEWCohort_FROZEN_20261029_run01')
    p.add_argument('--data-root',default='/workspace/fashion_data')
    p.add_argument('--output',required=True)
    p.add_argument('--skip-checkpoint-hash',action='store_true',help='Diagnostic only. Will never mark ready true.')
    try:run(p.parse_args())
    except Exception as e:
        print('V5_PREFLIGHT_BLOCKED',str(e),file=sys.stderr);sys.exit(2)
if __name__=='__main__':main()
