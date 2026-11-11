#!/usr/bin/env python3
"""Qwen2.5-VL-7B isolated per-token streaming timing (DEV60 only, read-only).

Does not train, alter model/PRD files, use Holdout, or send releases.
The streamer synchronizes at each completed token; it is an *instrumented*
measurement, not a proof of CUDA kernel-level decode latency.
"""
from __future__ import annotations
import argparse,hashlib,json,os,statistics,sys,time,zipfile
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUNNER=Path('/workspace/PRD32_DEV60_QWEN_AB_20261030/dev60_qwen_ab.py')
STAGE2=Path('/workspace/PRD31_PRD32_JOINT_EVAL_STAGE2_DEV60_171')
MODEL=Path('/workspace/models/Qwen2.5-VL-7B-Instruct')
EXPECTED_RUNNER_SHA='98a833c429da4c39849942a81f1067239273258a8605eac9265d0aeb9defc6f3'
LIMITS={'tags':170,'vqa':128,'copy':120}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def htext(x):return hashlib.sha256(x.encode('utf8')).hexdigest()
def pct(arr,p):
    a=sorted(arr); k=(len(a)-1)*p/100;i=int(k);j=min(i+1,len(a)-1)
    return a[i]*(j-k)+a[j]*(k-i)

def write_json(path,d):Path(path).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')

def prepare_cases():
    rows=json.loads((ROOT/'PROBE_CASES.json').read_text('utf8'))
    assert len(rows)==3
    for r in rows:
        image=(ROOT/r['marked_image']).resolve()
        if not image.is_relative_to(ROOT.resolve()): raise ValueError('image escaped local probe assets')
        if sha(image)!=r['marked_sha256']:raise ValueError('marked image SHA mismatch')
        assert set(r['tasks'])==set(LIMITS)
    return rows

def get_frozen_prompts(rows):
    if sha(RUNNER)!=EXPECTED_RUNNER_SHA:raise ValueError('original Qwen runner changed / missing')
    sys.path.insert(0,str(STAGE2))
    from run_live_probe_v4 import verified_runner
    runner=verified_runner(RUNNER)
    # Original Stage2 prompt check, read-only. This does NOT load the model.
    valid,core,helper,focus=runner.prompt_check(argparse.Namespace(
        snapshot=runner.DEFAULT_SNAP,source=runner.DEFAULT_SOURCE,
        focus_script=runner.DEFAULT_FOCUS,raw=runner.DEFAULT_RAW,input=runner.DEFAULT_INPUT))
    if len(valid)!=171:raise ValueError('DEV60 prompt snapshot changed')
    prompts={}
    for r in rows:
        for task in LIMITS:
            qrec={'image_path':str(ROOT/r['marked_image']),'image_sha256':r['marked_sha256'],
                  'target_bbox_xyxy':[0,0,1,1], 'target_instance_id':r['instance_id'],
                  'task':task,'query':runner.QUESTION,'length_chars':[20,80],
                  'predicted_attribute_context':[]}
            prompt=core.build_prompt(helper,focus,qrec,task,'baseline')
            if htext(prompt)!=r['tasks'][task]['prompt_sha256']:
                raise ValueError('prompt SHA mismatch: '+r['case_id']+'/'+task)
            prompts[(r['case_id'],r['instance_id'],task)]=prompt
    return prompts

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('preflight','live'),default='preflight')
    p.add_argument('--execute-models',action='store_true')
    p.add_argument('--repeats',type=int,choices=(1,2,3),default=2)
    p.add_argument('--model-dir',type=Path,default=MODEL)
    args=p.parse_args()
    rows=prepare_cases(); prompts=get_frozen_prompts(rows)
    print('DECODE_PROBE_PREFLIGHT_PASS; cases=3; prompts=9; input_and_prompt_SHA_checked; no Holdout')
    if args.mode=='preflight':return 0
    if not args.execute_models:raise ValueError('live requires --execute-models')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from transformers.generation.streamers import BaseStreamer
    from qwen_vl_utils import process_vision_info
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable')
    free_before=torch.cuda.mem_get_info()[0]/1024**3
    if free_before<18.0:raise RuntimeError('need >=18 GiB GPU free before Qwen model load')
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir),local_files_only=True,device_map='auto',
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,max_pixels=512*28*28)
    if any(str(d).lower() in ('cpu','disk') for d in getattr(model,'hf_device_map',{}).values()):
        raise RuntimeError('model offloaded; invalid comparative timing')
    torch.cuda.synchronize()
    free_after=torch.cuda.mem_get_info()[0]/1024**3
    class Timestamps(BaseStreamer):
        def __init__(self):self.stamps=[]
        def put(self, tokens):
            torch.cuda.synchronize()
            self.stamps.append(time.perf_counter())
        def end(self):pass
    def one(r,task,cohort):
        prompt=prompts[(r['case_id'],r['instance_id'],task)]
        msg=[{'role':'user','content':[{'type':'image','image':(ROOT/r['marked_image']).resolve().as_uri()},
                                      {'type':'text','text':prompt}]}]
        torch.cuda.synchronize(); pstart=time.perf_counter()
        txt=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        pics,vid=process_vision_info(msg)
        inputs=processor(text=[txt],images=pics,videos=vid,padding=True,return_tensors='pt').to(model.device)
        torch.cuda.synchronize(); pend=time.perf_counter()
        stream=Timestamps()
        torch.cuda.synchronize(); t0=time.perf_counter()
        with torch.inference_mode():
            outputs=model.generate(**inputs,do_sample=False,max_new_tokens=LIMITS[task],streamer=stream)
        torch.cuda.synchronize(); t1=time.perf_counter()
        gen=outputs[:,inputs.input_ids.shape[1]:]
        raw=processor.batch_decode(gen,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        n=int(gen.shape[-1]); stamps=stream.stamps
        gaps=[(stamps[k]-stamps[k-1])*1000 for k in range(2,len(stamps))]
        return {'cohort':cohort,'case_id':r['case_id'],'instance_id':r['instance_id'],'task':task,
            'output_tokens':n,'prompt_tokens_including_visual':int(inputs.input_ids.shape[-1]),
            'preprocess_ms':round((pend-pstart)*1000,2),
            'generate_ms':round((t1-t0)*1000,2),
            'streamer_put_count':len(stamps), 'streamer_count_matches_tokens':len(stamps)==n+1,
            'first_token_ms_including_prefill':round((stamps[1]-t0)*1000,2) if len(stamps)>=2 else None,
            'decode_intervals_n':len(gaps),
            'decode_intervals_mean_ms':round(statistics.mean(gaps),2) if gaps else None,
            'decode_intervals_median_ms':round(statistics.median(gaps),2) if gaps else None,
            'decode_intervals_p95_ms':round(pct(gaps,95),2) if gaps else None,
            'response_matches_frozen':htext(raw)==r['tasks'][task]['response_sha256'],
            'original_generated_tokens':r['tasks'][task]['generated_tokens'],
            'original_generate_ms_including_prefill':r['tasks'][task]['generation_ms_including_prefill'],
            'token_intervals_ms':[round(x,2) for x in gaps]}
    # A true warmup is excluded; do not mutate frozen weights/prompts.
    for t in LIMITS:one(rows[0],t,'warmup_excluded')
    results=[]
    for rep in range(1,args.repeats+1):
        for r in rows:
            for t in LIMITS:
                obs=one(r,t,f'repeat_{rep}'); results.append(obs)
                print(f'{obs["case_id"]} {t} output_tokens={obs["output_tokens"]} '
                      f'prefill_first={obs["first_token_ms_including_prefill"]}ms '
                      f'decode_gap_med={obs["decode_intervals_median_ms"]}ms '
                      f'output_match={obs["response_matches_frozen"]}',flush=True)
    valid=[x for x in results if x['streamer_count_matches_tokens'] and x['decode_intervals_n']>0]
    all_gaps=[v for x in valid for v in x['token_intervals_ms']]
    summary={'schema':'QWEN25VL_GPU_DECODE_STREAMER_PROBE_V1',
       'benchmark_kind':'instrumented_sync_per_token_not_uninstrumented_pure_cuda_kernel_floor',
       'model':'Qwen2.5-VL-7B-Instruct','dtype':str(next(model.parameters()).dtype),'attn_impl':'sdpa',
       'gpu_name':torch.cuda.get_device_name(0),'gpu_free_before_gib':round(free_before,2),
       'gpu_free_after_model_load_gib':round(free_after,2),
       'repeats':args.repeats,'measured_task_calls':len(results),
       'output_match_count':sum(x['response_matches_frozen'] for x in results),
       'decode_gaps_n':len(all_gaps),'decode_gap_mean_ms':round(statistics.mean(all_gaps),2) if all_gaps else None,
       'decode_gap_median_ms':round(statistics.median(all_gaps),2) if all_gaps else None,
       'decode_gap_p95_ms':round(pct(all_gaps,95),2) if all_gaps else None,
       'median_steady_tokens_per_s':round(1000/statistics.median(all_gaps),2) if all_gaps else None,
       'holdout_used':False,'model_weights_changed':False,'production_release':False,
       'not_equivalent_to_stage2':'Qwen-only isolated decode; Stage2 co-loads V5; stream synchronization adds observer overhead'}
    out=Path('/workspace')/('PRD32_DECODE_GPU_PROBE_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    write_json(out/'summary.json',summary);write_json(out/'per_call.json',results)
    zipout=out.parent/(out.name+'_UPLOAD_THIS.zip')
    with zipfile.ZipFile(zipout,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for fn in out.iterdir():z.write(fn,fn.name)
    print('DECODE_PROBE_COMPLETE; UPLOAD_THIS:',zipout)
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        import traceback
        print('DECODE_PROBE_BLOCKED',type(e).__name__,str(e),file=sys.stderr)
        traceback.print_exc();sys.exit(2)
