#!/usr/bin/env python3
# coding: utf-8
"""ID-V1 controlled prompt-only study on existing 60/171 DEV cohort.

Only experiment delta: append instance identity instruction to locked Focus V1 prompt.
Same original marked images, V5 predictions, model weights, request IDs and model settings.
No human GT / evaluation files are imported. Development-only, NOT PRD acceptance.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
ROOT = Path('/workspace')
TASKS = ('tags','vqa','copy')
VARIANT = 'focus_v1_plus_instance_identity_id_v1'
SCOPED = 'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY'
PINNED_LEGACY_SHA = '98a833c429da4c39849942a81f1067239273258a8605eac9265d0aeb9defc6f3'
PINNED_SMOKE_SHA = 'f4037719bf42c64a98eca6cfda44591c34159bf7bd9186cca97282379000dd6e'

def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def text_sha(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def jlread(p):
    return [json.loads(line) for line in Path(p).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
def jload(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def jwrite(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def jappend(p,x):
    with Path(p).open('a',encoding='utf-8') as f:
        f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())

def legacy_module():
    p=HERE/'reference_dev60_qwen_ab.py'
    if file_sha(p)!=PINNED_LEGACY_SHA: raise ValueError('REFERENCE_RUNNER_SHA_CHANGED')
    spec=importlib.util.spec_from_file_location('prd32_1104_idv1_readonly_legacy',p)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def original_args(args):
    return argparse.Namespace(raw=args.raw,input=args.input,source=args.source,
      focus_script=args.focus_script,model_dir=args.model_dir,
      snapshot=args.snapshot,focus=args.original_focus)

def canonical_inputs(args):
    # Revalidate the source checkpoint, V5 predictions, image bytes, prompt hashes and original 513 results.
    old=legacy_module()
    ori=original_args(args)
    rows,core,helper,focus=old.prompt_check(ori)
    original=old.completed_run(args.original_focus,'focus_v1',old.expected_keys(rows),rows,ori)
    manifest=jload(HERE/'fixed_manifest.json')
    for name,expected in manifest['pinned_files'].items():
        if file_sha(HERE/name)!=expected: raise ValueError('FIXED_INPUT_FILE_SHA_CHANGED:'+name)
    frozen=jlread(HERE/'instances171_input_only.jsonl')
    req=jlread(HERE/'requests513_input_only.jsonl')
    if frozen!=rows or len(rows)!=171: raise ValueError('CANONICAL_171_MISMATCH')
    exp=[(x['case_id'],x['target_instance_id'],t) for x in rows for t in TASKS]
    got=[(r['case_id'],r['target_instance_id'],r['task']) for r in req]
    if got!=exp or any(r['request_id']!='__'.join(k) for r,k in zip(req,exp)):
        raise ValueError('CANONICAL_513_MISMATCH')
    if old.CKPT!=manifest['V5_checkpoint_sha256'] or old.V5_HASH!=manifest['canonical_V5_predictions_sha256']:
        raise ValueError('V5_CHECKPOINT_OR_PREDICTION_NOT_PINNED')
    if old.PROMPT_SHA['focus_v1']!=manifest['focus_v1_original_prompt_sha256']:
        raise ValueError('FOCUS_V1_PROMPT_NOT_PINNED')
    smoke=jlread(HERE/'smoke24_input_only.jsonl')
    if file_sha(HERE/'smoke24_input_only.jsonl')!=PINNED_SMOKE_SHA: raise ValueError('SMOKE24_SHA_CHANGED')
    keys={(r['case_id'],r['target_instance_id']) for r in rows}
    skeys={(r['case_id'],r['target_instance_id']) for r in smoke}
    if len(smoke)!=24 or len(skeys)!=24 or not skeys.issubset(keys):raise ValueError('SMOKE24_COHORT_CHANGED')
    # Identity delta must be exactly the output of the canonical generator. No human correction may enter.
    deltas=jlread(args.delta)
    spec=jload(HERE/'candidate_addenda.json')
    suffix=spec['ID-V1']
    if len(deltas)!=513:raise ValueError('DELTA_COUNT_NOT_513')
    for rec,delta in zip(req,deltas):
        identity=rec['target_instance_id']
        expected_text='\n\n'+suffix.format(instance_id=identity)
        for k in ['request_id','case_id','target_instance_id','task']:
            if delta.get(k)!=rec.get(k):raise ValueError('DELTA_KEY_OR_ORDER_WRONG:'+k)
        if delta.get('variant')!='ID-V1' or delta.get('human_reference_used') is not False:
            raise ValueError('DELTA_VERSION_OR_HUMAN_INFO_UNSAFE')
        if delta.get('append_to_original_focus_v1_prompt')!=expected_text:
            raise ValueError('DELTA_TEXT_MODIFIED')
        if delta.get('suffix_sha256')!=text_sha(expected_text.lstrip('\n')):
            raise ValueError('DELTA_TEXT_HASH_MISMATCH')
    frozen_image_map={(r['case_id'],r['target_instance_id']):r['image_sha256'] for r in rows}
    for x,r in zip(original,req):
        if (x['case_id'],x['target_instance_id'],x['task'])!=(r['case_id'],r['target_instance_id'],r['task']):
            raise ValueError('ORIGINAL_REQUEST_MISMATCH')
        if x['image_sha256']!=frozen_image_map[(r['case_id'],r['target_instance_id'])]:
            raise ValueError('ORIGINAL_SOURCE_HASH_MISMATCH')
        marked=Path(x['marked_image'])
        if not marked.is_file() or file_sha(marked)!=x['marked_image_sha256']:
            raise ValueError('ORIGINAL_MARKED_IMAGE_MISSING_OR_CHANGED:'+str(marked))
    return old,rows,core,helper,focus,original,req,deltas,skeys

def scope_keys(scope,rows,skeys):
    return [(r['case_id'],r['target_instance_id'],t) for r in rows
            if scope=='full' or (r['case_id'],r['target_instance_id']) in skeys for t in TASKS]

def provenance(args,old,scope):
    return {'schema':'prd32_1104_idv1_prompt_only_run_v1','scope':SCOPED,
       'model_variant':VARIANT,'run_scope':scope,'images':60,'instances':171,
       'request_count':72 if scope=='smoke' else 513,
       'canonical_171_sha256':file_sha(HERE/'instances171_input_only.jsonl'),
       'canonical_513_sha256':file_sha(HERE/'requests513_input_only.jsonl'),
       'smoke24_sha256':PINNED_SMOKE_SHA,'delta_sha256':file_sha(args.delta),
       'original_focus_results_sha256':file_sha(args.original_focus/'qwen_results.jsonl'),
       'original_focus_prompt_sha256':old.PROMPT_SHA['focus_v1'],
       'v5_predictions_sha256':old.V5_HASH,'v5_checkpoint_sha256':old.CKPT,
       'unchanged_original_marked_images':True,'single_variable':'append_instance_identity_prompt_only',
       'task_limits':{'tags':170,'vqa':128,'copy':120},
       'human_labels_used_for_model':False,'holdout_used':False,
       'new_results_human_verified':False,'official_prd_acceptance':False,'publish_allowed':False}

def output_dir(args): return args.smoke_output if args.scope=='smoke' else args.full_output

def check_existing_prefix(prior,keys,origmap,deltamap,old,expected_variant=VARIANT):
    if len(prior)>len(keys):raise ValueError('RESUME_TOO_MANY_ROWS')
    for rec,key in zip(prior,keys):
        k='__'.join(key);o=origmap[key];d=deltamap[key]
        if rec.get('request_id')!=k or rec.get('model_variant')!=expected_variant or rec.get('status')!='GENERATED_REVIEW_REQUIRED':
            raise ValueError('RESUME_ID_OR_VARIANT_CHANGED')
        if rec.get('image_sha256')!=o.get('image_sha256') or rec.get('marked_image_sha256')!=o.get('marked_image_sha256'):
            raise ValueError('RESUME_VISUAL_CHANGED')
        if rec.get('base_prompt_sha256')!=old.PROMPT_SHA['focus_v1'][key[2]]:
            raise ValueError('RESUME_BASE_PROMPT_CHANGED')
        if ('original_prompt' not in rec or
            text_sha(rec['original_prompt'])!=old.PROMPT_SHA['focus_v1'][key[2]] or
            rec.get('final_prompt_sha256')!=text_sha(rec['original_prompt']+d['append_to_original_focus_v1_prompt'])):
            raise ValueError('RESUME_FINAL_PROMPT_HASH_INVALID')
        if rec.get('human_label_or_gt_bbox_used') is not False or rec.get('publish_allowed') is not False:
            raise ValueError('RESUME_LABEL_LEAK')
        if not isinstance(rec.get('raw_text'),str) or not rec['raw_text'].strip():
            raise ValueError('RESUME_EMPTY_RESULT')

def write_progress(out,scope,completed,total,finished,load_seconds=None):
    jwrite(out/'run_summary.json',{'schema':'prd32_1104_idv1_run_summary_v1',
      'scope':scope,'status':('COMPLETE_72_FOR_REVIEW' if scope=='smoke' else 'COMPLETE_513_FOR_REVIEW') if finished else 'INCOMPLETE_RESUMABLE',
      'successful':completed,'expected':total,'remaining':total-completed,
      'model_load_seconds_session':load_seconds,'new_results_human_verified':False,
      'official_prd_acceptance':False,'publish_allowed':False})

def run(args):
    old,rows,core,helper,focus,original,req,deltas,skeys=canonical_inputs(args)
    keys=scope_keys(args.scope,rows,skeys)
    oldmap={(x['case_id'],x['target_instance_id'],x['task']):x for x in original}
    delta_map={(x['case_id'],x['target_instance_id'],x['task']):x for x in deltas}
    rowmap={(r['case_id'],r['target_instance_id']):r for r in rows}
    out=output_dir(args)
    prov=provenance(args,old,args.scope)
    if args.scope=='full':
        smoke_summary=args.smoke_output/'run_summary.json'
        if not smoke_summary.is_file() or jload(smoke_summary).get('status')!='COMPLETE_72_FOR_REVIEW':
            raise ValueError('SMOKE72_REQUIRED_FIRST')
    if out.exists():
        if not args.resume:raise FileExistsError('OUTPUT_EXISTS_USE_RESUME')
        if jload(out/'provenance.json')!=prov:raise ValueError('PROVENANCE_CHANGED')
        prior=jlread(out/'qwen_results.jsonl')
        check_existing_prefix(prior,keys,oldmap,delta_map,old)
    else:
        if args.resume:raise ValueError('RESUME_REQUIRES_EXISTING_DIRECTORY')
        out.mkdir(parents=True)
        jwrite(out/'provenance.json',prov)
        (out/'qwen_results.jsonl').touch()
        prior=[]
    if len(prior)==len(keys):
        write_progress(out,args.scope,len(prior),len(keys),True)
        print('IDV1_ALREADY_COMPLETE',args.scope,len(prior)); return
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    if torch.cuda.mem_get_info()[0]/1024**3<18:raise RuntimeError('GPU_FREE_MEMORY_BELOW_18GiB')
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    start=time.perf_counter()
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.model_dir),local_files_only=True,
      device_map='auto',torch_dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,max_pixels=512*28*28)
    load_s=round(time.perf_counter()-start,2)
    for n in range(len(prior),len(keys)):
        key=keys[n];cid,iid,task=key
        row=rowmap[cid,iid];base=oldmap[key];delta=delta_map[key]
        qrec={'image_path':row['image_path'],'image_sha256':row['image_sha256'],
          'target_bbox_xyxy':row['target_bbox_xyxy'],'target_instance_id':iid,
          'task':task,'query':old.QUESTION,'length_chars':[20,80],'predicted_attribute_context':[]}
        prompt=core.build_prompt(helper,focus,qrec,task,'focus_v1')
        if text_sha(prompt)!=old.PROMPT_SHA['focus_v1'][task]: raise ValueError('FOCUS_V1_BASE_PROMPT_CHANGED')
        final_prompt=prompt+delta['append_to_original_focus_v1_prompt']
        marked=Path(base['marked_image'])
        if file_sha(marked)!=base['marked_image_sha256']:raise ValueError('ORIGINAL_MARKED_IMAGE_CHANGED_DURING_RUN')
        output={'case_id':cid,'target_instance_id':iid,'task':task,'request_id':'__'.join(key),
          'status':'GENERATED_REVIEW_REQUIRED','model_variant':VARIANT,
          'image_sha256':row['image_sha256'],'v5_predicted_bbox_xyxy':row['target_bbox_xyxy'],
          'v5_category_unverified':row['predicted_category_v5_unverified'],
          'marked_image_sha256':base['marked_image_sha256'],'marked_image':str(marked),
          'base_prompt_sha256':text_sha(prompt),'final_prompt_sha256':text_sha(final_prompt),
          'delta_sha256':text_sha(delta['append_to_original_focus_v1_prompt']),
          'visual_change_only':False,'prompt_change_only':True,'human_label_or_gt_bbox_used':False,
          'quality_human_verified':False,'publish_allowed':False,'official_prd_acceptance':False,
          'original_prompt':prompt, # for resume integrity only; not human labels
          'question_sha256':old.tsha(old.QUESTION) if task=='vqa' else None}
        try:
            raw,timing=helper.run_model_case(model,processor,qrec,marked,final_prompt,
               task_limits={'tags':170,'vqa':128,'copy':120})
            if not isinstance(raw,str) or not raw.strip():raise ValueError('EMPTY_GENERATION')
            output.update(raw_text=raw,timing=timing)
            if task=='tags':
                parsed=helper.parse_tag_output(raw);output['tag_parse']=parsed
                if isinstance(parsed,dict) and parsed.get('canonical_service_json') is not None:
                    output['tag_bridge_candidate']=helper.bridge_tags_adapter(parsed['canonical_service_json'])
            if task=='vqa':output['vqa_visible_abstention_marker']=bool(old.ABSTAIN.search(raw))
            if task=='copy':output['copy_review_flags']=['DEBUG_MARKER_IN_COPY'] if old.DEBUG.search(raw) else []
            jappend(out/'qwen_results.jsonl',output)
            prior.append(output)
            print('IDV1_REQUEST',args.scope,n+1,'/',len(keys),output['request_id'],'OK',flush=True)
            if (n+1)%12==0:write_progress(out,args.scope,len(prior),len(keys),False,load_s)
        except Exception as exc:
            jwrite(out/'last_error.json',{'request_index':n+1,'request_id':output['request_id'],
               'error_type':type(exc).__name__,'message':str(exc)[:400], 'safe_to_resume':True})
            write_progress(out,args.scope,len(prior),len(keys),False,load_s)
            raise
    write_progress(out,args.scope,len(prior),len(keys),True,load_s)
    print('IDV1_RUN_COMPLETE',args.scope,len(prior),'/',len(keys))

def export(args):
    old,rows,core,helper,focus,original,req,deltas,skeys=canonical_inputs(args)
    keys=scope_keys(args.scope,rows,skeys)
    out=output_dir(args)
    if jload(out/'run_summary.json').get('status')!=('COMPLETE_72_FOR_REVIEW' if args.scope=='smoke' else 'COMPLETE_513_FOR_REVIEW'):
        raise ValueError('RUN_NOT_COMPLETE')
    results=jlread(out/'qwen_results.jsonl')
    if len(results)!=len(keys):raise ValueError('INCOMPLETE_RESULTS')
    oldmap={(x['case_id'],x['target_instance_id'],x['task']):x for x in original}
    delta_map={(x['case_id'],x['target_instance_id'],x['task']):x for x in deltas}
    check_existing_prefix(results,keys,oldmap,delta_map,old)
    pairs=[];stats=Counter()
    for key,r in zip(keys,results):
        b=oldmap[key]
        if r['marked_image_sha256']!=b['marked_image_sha256'] or r['image_sha256']!=b['image_sha256']:
            raise ValueError('PAIRED_IMAGE_CHANGED')
        same=r['raw_text']==b['raw_text']
        stats['unchanged' if same else 'changed']+=1
        stats[f'{key[2]}_'+('unchanged' if same else 'changed')]+=1
        pairs.append({'request_id':'__'.join(key),'case_id':key[0],
          'target_instance_id':key[1],'task':key[2],
          'focus_v1_raw':b['raw_text'],'id_v1_raw':r['raw_text'],
          'text_unchanged':same,'marked_image_sha256':r['marked_image_sha256'],
          'base_prompt_sha256':r['base_prompt_sha256'],
          'id_v1_prompt_sha256':r['final_prompt_sha256']})
    if args.export.exists():raise FileExistsError('REFUSE_OVERWRITE_EXPORT')
    with ZipFile(args.export,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for fn in ('provenance.json','run_summary.json','qwen_results.jsonl'):
            z.write(out/fn,arcname=f'{out.name}/{fn}')
        z.write(HERE/'fixed_manifest.json',arcname='fixed_manifest.json')
        z.write(HERE/'candidate_addenda.json',arcname='candidate_addenda.json')
        z.writestr('paired_text_rows.jsonl',''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pairs))
        z.writestr('paired_diagnostic_NOT_ACCURACY.json',json.dumps({
            'schema':'idv1_paired_text_diagnostic', 'scope':args.scope,
            'cases':len(keys),'counts_TEXT_CHANGE_NOT_ACCURACY':dict(stats),
            'human_review_of_new_outputs_required':True,'holdout_used':False,
            'official_prd_acceptance':False},ensure_ascii=False,indent=2))
    print('IDV1_UPLOAD_THIS',args.export,round(args.export.stat().st_size/1024**2,3),'MiB')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','run','export'])
    p.add_argument('--scope',choices=['smoke','full'],default='smoke')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--raw',type=Path,default=ROOT/'PRD32_DEV60_V5_REAL_20261030_run01')
    p.add_argument('--input',type=Path,default=ROOT/'PRD32_FIXED_DEV_V2_60IMG_CANDIDATES_20261030/v5_image_only_inputs_60.jsonl')
    p.add_argument('--source',type=Path,default=ROOT/'PRD32_20261029_NextBatch_GTFree')
    p.add_argument('--focus-script',type=Path,default=ROOT/'PRD32_FOCUS_AB_PATCH_20261029/prd32_focus_ab.py')
    p.add_argument('--model-dir',type=Path,default=ROOT/'models/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--snapshot',type=Path,default=ROOT/'PRD32_DEV60_QWEN_DIAGNOSTIC_SNAPSHOT_20261030')
    p.add_argument('--original-focus',type=Path,default=ROOT/'PRD32_DEV60_QWEN_FOCUS_V1_20261030_run01')
    p.add_argument('--delta',type=Path,default=ROOT/'PRD32_IDV1_delta.jsonl')
    p.add_argument('--smoke-output',type=Path,default=ROOT/'PRD32_1104_IDV1_SMOKE_RUN01')
    p.add_argument('--full-output',type=Path,default=ROOT/'PRD32_1104_IDV1_FULL_RUN01')
    p.add_argument('--export',type=Path)
    args=p.parse_args()
    if args.export is None:
        args.export=ROOT/f'PRD32_1104_IDV1_{args.scope.upper()}_UPLOAD_THIS.zip'
    try:
        if args.command=='preflight':
            old,rows,core,helper,focus,original,req,deltas,skeys=canonical_inputs(args)
            print('IDV1_RUNTIME_PREFLIGHT_PASS 60 images 171 instances 513 tasks; smoke24')
            print('ORIGINAL_MARKED_INPUTS_LOCKED; ONLY_PROMPT_SUFFIX_CHANGED; GPU_NOT_LOADED')
        elif args.command=='run':run(args)
        else:export(args)
    except Exception as e:
        print('IDV1_BLOCKED',type(e).__name__,str(e),file=sys.stderr)
        raise SystemExit(2)

if __name__=='__main__':main()
