#!/usr/bin/env python3
"""DEV60-only 4090 resident Qwen/V5 architecture A/B; SHADOW-ONLY.

Baseline: V5 + Tags + VQA + Copy + unchanged existing Guard.
VQA route: V5 + VQA only (not quality-equivalent to all-3 baseline).
Merchant route: V5 + Tags + Copy only.
Template candidate: V5 + Tags + CPU rule-based Copy (HUMAN REVIEW REQUIRED).

This does not train, alter weights, change baseline prompts, access Holdout, or publish.
Partial arms CANNOT call the full-task Guard contract and remain blocked by design.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,json,os,statistics,sys,time,traceback,zipfile
from datetime import datetime
from pathlib import Path
from arch_ab_common import make_copy,stats,sha_text
from stage1_verified_runtime import (WORKSPACE,REPO,V5_RAW,IMAGE_ROOT,CKPT,RUNNER,QWEN_MODEL,
    v5_preflight,verified_runner,load_guard,refuse_restricted,run_one,
    digest_file,sync,dump_jsonl,write_json,zip_results,check_producer_match,
    canonical_mask_digest)
from joint_eval_stage2_dev60 import manifest_verified,load_all_targets

ROOT=Path(__file__).resolve().parent
SELECTION=ROOT/'AB16_SELECTED_TARGETS.json'
SELECTION_SHA='914aca4f9d928d8771ac6aeb6441efe6c8c303174a97ce6b4b6a45c003b426c6'
ARMS=('baseline_three','vqa_only','merchant_tags_copy','merchant_tags_template')
TASKS_BY_ARM={'vqa_only':('vqa',),'merchant_tags_copy':('tags','copy'),'merchant_tags_template':('tags',)}
LIMITS={'tags':170,'vqa':128,'copy':120}

def select_keys(count):
    if digest_file(SELECTION)!=SELECTION_SHA:raise ValueError('AB target selection modified')
    m=json.loads(SELECTION.read_text('utf8'))
    rows=m['targets']
    if m.get('schema')!='prd32_architecture_ab_v1_perf_stratified_16' or len(rows)!=16:
        raise ValueError('invalid AB16 selection')
    entire,_=manifest_verified()
    allmap={(x['case_id'],x['instance_id']):x for x in entire['targets']}
    bycat=collections.Counter(r['v5_frozen_category'] for r in rows)
    if bycat!={c:2 for c in ('top','pants','skirt','outerwear','dress','shoe','bag','accessory')}:
        raise ValueError('need 2 selected per each of 8 V5 classes')
    for r in rows:
        k=(r['case_id'],r['instance_id'])
        if k not in allmap or r['source_image_sha256']!=allmap[k]['source_image_sha256']:
            raise ValueError('selection not based on verified frozen DEV60')
        if r['reference_status']!='DETERMINATE' or r['target_identity_review']!='匹配':
            raise ValueError('unexpected category reference or target identity')
    if count==8: rows=rows[::2]
    if len({(r['case_id'],r['instance_id']) for r in rows})!=count:
        raise ValueError('duplicate targets')
    return rows,[(r['case_id'],r['instance_id']) for r in rows],entire

def prepare(args,out):
    for p in (args.repo,args.raw_dir,args.image_root,args.checkpoint,args.runner,args.model_dir,SELECTION,out):refuse_restricted(p)
    selected,keys,manifest=select_keys(args.cases)
    pinned,module_hashes=v5_preflight(args.raw_dir,args.image_root,args.repo,args.checkpoint)
    if len(pinned)!=12:raise ValueError('pinned V5 12 contract drift')
    allkeys=[(r['case_id'],r['instance_id']) for r in manifest['targets']]
    targets=load_all_targets(args.raw_dir,args.image_root,manifest,allkeys)
    runner=verified_runner(args.runner)
    valid,core,helper,focus=runner.prompt_check(argparse.Namespace(
        snapshot=runner.DEFAULT_SNAP,source=runner.DEFAULT_SOURCE,
        focus_script=runner.DEFAULT_FOCUS,raw=runner.DEFAULT_RAW,input=runner.DEFAULT_INPUT))
    if len(valid)!=171:raise ValueError('baseline Qwen prompt snapshot drift')
    guardfile=args.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py'
    guard=load_guard(guardfile)
    if not (args.model_dir/'config.json').is_file():raise FileNotFoundError('Qwen model dir missing config')
    write_json(out/'preflight.json',{'status':'ARCH_AB_PRECHECK_VALID','run_mode':args.mode,'targets':len(keys),
       'categories_covered':list(dict.fromkeys(r['v5_frozen_category'] for r in selected)),
       'selection_sha256':SELECTION_SHA,'original_dev60_171_manifest_checked':True,
       'v5_source_hashes':module_hashes,'v5_checkpoint_sha256':digest_file(args.checkpoint),
       'qwen_runner_sha256':digest_file(args.runner),'guard_sha256':digest_file(guardfile),
       'quality_reference':'chat-confirmed developer reference, NOT independent blind GT',
       'holdout_used':False,'weights_or_baseline_prompts_modified':False,'release_allowed':False,
       'gpu_calls_per_target':{'baseline_three':3,'vqa_only':1,'merchant_tags_copy':2,'merchant_tags_template':1},
       'total_measured_gpu_calls':len(keys)*7*args.repeats,
       'additional_warmup_gpu_calls':3})
    return selected,keys,targets,runner,core,helper,focus,guard

def run_routed(*,arm,label,key,targets,v5_model,qwen,processor,runner,core,helper,focus,device,torch,np,Image,ID_TO_CLASS,out):
    """Real GPU request using frozen V5/Qwen helper and unchanged baseline prompt.

    Partial routes do not call existing full-task guard. They are forced SHADOW BLOCKED.
    """
    if arm not in TASKS_BY_ARM:raise ValueError('wrong arm')
    case,inst=key;frozen=targets[key]
    t_start=time.perf_counter()
    sync(torch);t_vstart=time.perf_counter()
    with Image.open(frozen['image_path']) as f: img=f.convert('RGB')
    from fashion_multimodal_analysis.evaluation.runtime import predict_timed
    boxes,labels,scores,masks,v5_inner=predict_timed(img,v5_model,device,score_threshold=.4,mask_threshold=.5)
    sync(torch);t_vend=time.perf_counter()
    idx=int(inst.split('_')[-1]);
    if idx>=len(labels):raise ValueError('V5 missing frozen predicted instance '+str(key))
    cat=ID_TO_CLASS.get(int(labels[idx]));newmask=(np.asarray(masks[idx])>0).astype(np.uint8)
    with Image.open(frozen['frozen_mask_path']) as f: oldmask=np.asarray(f.convert('L'))>127
    checked=check_producer_match(target=frozen,ins={'instance_id':inst,'category':cat,
          'score':float(scores[idx]),'bbox':[float(b) for b in boxes[idx]]},
          reference_mask=oldmask,produced_mask=newmask)
    if checked['mask_iou']<.9999:raise ValueError('V5 output not frozen-compatible')
    request={'case_id':case,'target_instance_id':inst,'image_path':frozen['image_path'],
        'image_sha256':frozen['image_sha256'],'target_bbox_xyxy':[float(b) for b in boxes[idx]],
        'predicted_category_v5_unverified':cat}
    marked_dir=out/label/'marked_inputs';marked_dir.mkdir(parents=True,exist_ok=False)
    times={};raw_by_task={};token_by_task={}
    for task in TASKS_BY_ARM[arm]:
        qrec={'image_path':request['image_path'],'image_sha256':request['image_sha256'],
            'target_bbox_xyxy':request['target_bbox_xyxy'],'target_instance_id':inst,'task':task,
            'query':runner.QUESTION,'length_chars':[20,80],'predicted_attribute_context':[]}
        sync(torch);pstart=time.perf_counter()
        marked=helper.make_marked_image(qrec,marked_dir,'marked_full_image',.05)
        if not Path(marked).resolve(strict=True).is_relative_to(marked_dir.resolve(strict=True)):
            raise ValueError('marked target escaped output directory')
        prompt=core.build_prompt(helper,focus,qrec,task,'baseline')
        psha=sha_text(prompt)
        if psha!=runner.PROMPT_SHA['baseline'][task]:raise ValueError('frozen baseline prompt drift')
        raw,timing=helper.run_model_case(qwen,processor,qrec,marked,prompt,task_limits=LIMITS)
        sync(torch);pend=time.perf_counter()
        raw_by_task[task]=raw; token_by_task[task]=timing.get('generated_tokens')
        times[task]=round((pend-pstart)*1000,3)
        dump_jsonl(out/'alternative_qwen_calls.jsonl',{
            'arm':arm,'cohort':label,'case_id':case,'instance_id':inst,
            'task':task,'baseline_prompt_sha256':psha,
            'marked_input_sha256':digest_file(marked),'response':raw,
            'generated_tokens':timing.get('generated_tokens'),
            'model_preprocess_ms':timing.get('preprocess_ms'),
            'model_generation_ms':timing.get('generation_ms'),
            'host_task_ms':times[task],'publish_allowed':False})
    t_tpl=0.0; template_result=None
    if arm=='merchant_tags_template':
        ts=time.perf_counter();template_result=make_copy(raw_by_task['tags'],style='客观',min_chars=20,max_chars=80)
        t_tpl=(time.perf_counter()-ts)*1000
        raw_by_task['copy_template']=template_result['text']
    sync(torch);t_end=time.perf_counter()
    # The original Guard requires full Tags+VQA+Copy. Never fake a Guard PASS.
    result={'cohort':label,'arm':arm,'case_id':case,'instance_id':inst,
        'request_scope':'attribute_vqa' if arm=='vqa_only' else 'merchant_content',
        'total_host_wall_ms':round((t_end-t_start)*1000,3),
        'v5_host_wall_ms':round((t_vend-t_vstart)*1000,3),
        'tags_task_host_wall_ms':times.get('tags'), 'vqa_task_host_wall_ms':times.get('vqa'),
        'copy_task_host_wall_ms':times.get('copy'), 'template_cpu_ms':round(t_tpl,4),
        'task_calls_made':list(TASKS_BY_ARM[arm]),'qwen_calls':len(TASKS_BY_ARM[arm]),
        'outputs':raw_by_task,'output_tokens':token_by_task,
        'template_status':template_result['status'] if template_result else 'NOT_APPLICABLE',
        'guard_status':'NOT_CALLED_PARTIAL_TASK_SET_FORCED_SHADOW_BLOCK',
        'guard_contract_validated_for_this_arm':False,
        'frozen_mask_iou':checked['mask_iou'],'v5_category':cat,
        'source_mask_supplied_as_qwen_pixels':False,'production_approved':False,
        'publish_allowed':False,'holdout_used':False}
    dump_jsonl(out/'alternative_events.jsonl',result)
    return result

def reconcile(baseline,alternatives,baseline_calls):
    # Baseline labels are 'ab_r#_t#_baseline_three', calls can be associated by cohort.
    bmap={c['cohort']:(c['case_id'],c['instance_id']) for c in baseline_calls if c['cohort']!='warmup_excluded'}
    calls=collections.defaultdict(dict)
    for c in baseline_calls:
        if c['cohort']=='warmup_excluded':continue
        k=(c['case_id'],c['instance_id'],c['cohort']);calls[k][c['task']]=c['response']
    brow={};arm_rows=collections.defaultdict(dict)
    for r in baseline:
        k=(r['case_id'],r['instance_id'],r['cohort'].split('_baseline_three')[0]);brow[k]=r
    for r in alternatives:
        k=(r['case_id'],r['instance_id'],r['cohort'].rsplit('_'+r['arm'],1)[0]);arm_rows[k][r['arm']]=r
    comparisons=[]
    for k,br in brow.items():
        if len(arm_rows[k])!=3:raise ValueError('missing arm '+str(k))
        prefix=br['cohort']
        base_out=calls[(k[0],k[1],prefix)];
        if set(base_out)!={'tags','vqa','copy'}:raise ValueError('incomplete baseline task set')
        p={'case_id':k[0],'instance_id':k[1],'repeat_key':k[2],
           'baseline_full_ms':br['total_host_wall_ms'],
           'baseline_guard_status':br['guard_status']}
        for arm in ('vqa_only','merchant_tags_copy','merchant_tags_template'):
            r=arm_rows[k][arm];p[arm+'_ms']=r['total_host_wall_ms']
            p[arm+'_qwen_calls']=r['qwen_calls']
            for task in TASKS_BY_ARM[arm]:
                p[arm+'_'+task+'_equals_baseline']=r['outputs'][task]==base_out[task]
            if arm=='merchant_tags_template':
                p['template_status']=r['template_status'];p['template_copy']=r['outputs']['copy_template'];p['original_copy']=base_out['copy']
        comparisons.append(p)
    if len(comparisons)!=len(baseline):raise ValueError('baseline to arms comparison lost rows')
    return comparisons

def run(args,out):
    selected,keys,targets,runner,core,helper,focus,guard=prepare(args,out)
    if args.mode=='preflight':
        print('ARCH_AB_V1_PREFLIGHT_PASS targets='+str(len(keys))+' 8 V5 categories, holdout sealed; NO_GPU_FORWARD',flush=True);return
    if not args.execute_models:raise ValueError('live requires --execute-models')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch,numpy as np
    from PIL import Image
    if not torch.cuda.is_available():raise RuntimeError('CUDA missing')
    free=torch.cuda.mem_get_info()[0]/1024**3
    if free<18:raise RuntimeError(f'need 18 GiB free at startup, only {free:.2f} available')
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    tt=time.perf_counter()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    qwen=Qwen2_5_VLForConditionalGeneration.from_pretrained(
       str(args.model_dir),local_files_only=True,device_map='auto',torch_dtype=dtype,
       attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,max_pixels=512*28*28)
    sync(torch)
    dm=getattr(qwen,'hf_device_map',None)
    if isinstance(dm,dict) and any(str(v).lower() in ('cpu','disk') for v in dm.values()):
        raise ValueError('Qwen CPU/DISK offload violates comparison contract')
    sys.path.insert(0,str(args.repo/'src'))
    from fashion_multimodal_analysis.common.schema import CLASS_TO_ID,ID_TO_CLASS
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if cp.get('class_to_id')!=CLASS_TO_ID or cp.get('epoch')!=15 or cp.get('args',{}).get('epochs')!=15:
        raise ValueError('V5 checkpoint class/epoch drift')
    kw=cp.get('args',{})
    v5=build_evaluation_model(int(kw.get('min_size',640)),int(kw.get('max_size',1024)))
    v5.load_state_dict(cp['model_state_dict'],strict=True)
    device=torch.device('cuda');v5.to(device).eval();del cp
    sync(torch)
    free_after=torch.cuda.mem_get_info()[0]/1024**3
    if free_after<1:raise ValueError('both models loaded but <1 GiB GPU free')
    write_json(out/'co_resident.json',{'qwen_v5_warm_loaded_same_process':True,
        'load_ms_excluded_from_request_latency':round((time.perf_counter()-tt)*1000,1),
        'gpu_free_after_gib':round(free_after,2),'dtype':str(dtype),'attn_implementation':'sdpa'})
    torch.cuda.reset_peak_memory_stats()
    warm=run_one(key=keys[0],label='warmup_excluded',targets=targets,v5_model=v5,
        qwen=qwen,processor=processor,runner=runner,core=core,helper=helper,focus=focus,
        guard=guard,device=device,torch=torch,np=np,Image=Image,ID_TO_CLASS=ID_TO_CLASS,out=out)
    baseline=[];alternatives=[]
    for repeat in range(1,args.repeats+1):
        for idx,key in enumerate(keys,1):
            prefix=f'ab_r{repeat:02d}_t{idx:02d}'
            arm_order=list(ARMS)
            # deterministic rotation of arm order across 16 keys to reduce thermal/order confounding
            shift=(idx-1)%4;arm_order=arm_order[shift:]+arm_order[:shift]
            for arm in arm_order:
                label=prefix+'_'+arm
                if arm=='baseline_three':
                    r=run_one(key=key,label=label,targets=targets,v5_model=v5,qwen=qwen,
                      processor=processor,runner=runner,core=core,helper=helper,focus=focus,
                      guard=guard,device=device,torch=torch,np=np,Image=Image,ID_TO_CLASS=ID_TO_CLASS,out=out)
                    if r['guard_status']!='REVIEW_REQUIRED_NOT_RELEASED' or r['publish_allowed'] is not False:
                        raise ValueError('baseline Guard fail closed missing')
                    baseline.append(r)
                else:
                    r=run_routed(arm=arm,label=label,key=key,targets=targets,v5_model=v5,
                      qwen=qwen,processor=processor,runner=runner,core=core,helper=helper,
                      focus=focus,device=device,torch=torch,np=np,Image=Image,ID_TO_CLASS=ID_TO_CLASS,out=out)
                    alternatives.append(r)
                print(f'ARCH_AB_EVENT repeat={repeat} target={idx}/{len(keys)} arm={arm} '
                    f'latency_ms={r["total_host_wall_ms"]} release=false',flush=True)
    basecalls=[json.loads(x) for x in (out/'qwen_calls.jsonl').read_text('utf8').splitlines()]
    if len(basecalls)!=(1+len(baseline))*3:raise ValueError('missing baseline Qwen calls')
    altcalls=[json.loads(x) for x in (out/'alternative_qwen_calls.jsonl').read_text('utf8').splitlines()]
    if len(altcalls)!=len(baseline)*4:raise ValueError('missing alternative Qwen calls')
    comparisons=reconcile(baseline,alternatives,basecalls)
    fields=list(comparisons[0].keys())
    with (out/'PAIRED_AB_RESULTS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(comparisons)
    matched={}
    for arm in ('vqa_only','merchant_tags_copy','merchant_tags_template'):
        cols=[c for c in fields if c.startswith(arm+'_') and c.endswith('_equals_baseline')]
        matched[arm]={c:sum(bool(r[c]) for r in comparisons) for c in cols}
    quality={'template_review_required_count':sum(r['template_status']=='TEMPLATE_DRAFT_NEEDS_HUMAN_REVIEW' for r in comparisons),
        'template_blocked_count':sum(r['template_status']!='TEMPLATE_DRAFT_NEEDS_HUMAN_REVIEW' for r in comparisons),
        'needs_human_copy_quality':True,'baseline_existing_guard_blocked_count':len(baseline),
        'alternative_guard_not_evaluated_due_to_partial_tasks':True}
    summary={'schema':'PRD32_ARCHITECTURE_AB_V1_SHADOW','status':'AB_COMPLETED_PENDING_USER_REVIEW',
      'gpu_name':torch.cuda.get_device_name(0),'unique_dev60_targets':len(keys),'repeats':args.repeats,
      'baseline_full_three_tasks':stats([x['total_host_wall_ms'] for x in baseline]),
      'vqa_only':stats([x['total_host_wall_ms'] for x in alternatives if x['arm']=='vqa_only']),
      'merchant_tags_copy':stats([x['total_host_wall_ms'] for x in alternatives if x['arm']=='merchant_tags_copy']),
      'merchant_tags_template':stats([x['total_host_wall_ms'] for x in alternatives if x['arm']=='merchant_tags_template']),
      'measured_qwen_calls':len(baseline)*7,'warmup_qwen_calls':3,
      'comparable_unmodified_task_responses':matched,'quality_review':quality,
      'request_scope_warning':'VQA-only and merchant routes do not produce all three task outputs; not equivalent to baseline full request. No all-three PRD claim.',
      'template_warning':'Rule Copy uses predicted Tags: inherits tag errors, cannot replace original Copy without human quality scoring and new Guard contract.',
      'original_guard_changed':False,'all_candidate_outputs_shadow_blocked':True,
      'models_and_baseline_prompts_unchanged':True,'holdout_accessed':False,
      'production_release':False,'official_prd_acceptance':False}
    write_json(out/'ARCH_AB_SUMMARY.json',summary)
    print('ARCH_AB_V1_RUN_COMPLETE targets='+str(len(keys))+' responses compared; pending human template quality review',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('preflight','live'),default='preflight')
    p.add_argument('--execute-models',action='store_true')
    p.add_argument('--cases',type=int,choices=(8,16),default=16)
    p.add_argument('--repeats',type=int,choices=(1,2),default=1)
    p.add_argument('--repo',type=Path,default=REPO)
    p.add_argument('--raw-dir',type=Path,default=V5_RAW)
    p.add_argument('--image-root',type=Path,default=IMAGE_ROOT)
    p.add_argument('--checkpoint',type=Path,default=CKPT)
    p.add_argument('--runner',type=Path,default=RUNNER)
    p.add_argument('--model-dir',type=Path,default=QWEN_MODEL)
    args=p.parse_args()
    out=WORKSPACE/f'PRD32_ARCH_AB_V1_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    out.mkdir(exist_ok=False)
    status=0
    try:run(args,out)
    except BaseException as exc:
        status=2
        write_json(out/'FAILED_CLOSED.json',{'status':'INCOMPLETE_NOT_COMPARABLE',
            'exception_type':type(exc).__name__,'message':str(exc),
            'traceback':traceback.format_exc()[-20000:],
            'publish_allowed':False,'holdout_used':False})
        print('ARCH_AB_BLOCKED',type(exc).__name__,str(exc),file=sys.stderr)
    finally:
        z=out.with_name(out.name+'_UPLOAD_THIS.zip')
        with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as archive:
            for f in sorted(out.rglob('*')):
                if f.is_file():archive.write(f,f.relative_to(out).as_posix())
        print('UPLOAD_THIS:',z,flush=True)
    return status
if __name__=='__main__':sys.exit(main())
