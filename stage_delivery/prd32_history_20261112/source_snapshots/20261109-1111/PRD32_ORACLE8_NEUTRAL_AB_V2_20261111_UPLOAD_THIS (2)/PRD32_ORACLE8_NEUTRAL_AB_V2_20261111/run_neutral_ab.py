#!/usr/bin/env python3
"""Run five strict pixel-controlled red/cyan comparisons with neutral prompts.
No images or answers are fabricated; produces recorded model output only on GPU.
"""
import argparse,csv,hashlib,json,random,sys,time,datetime,os
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='/workspace/models/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--output',default='outputs/neutral_results.jsonl')
    ap.add_argument('--max-new-tokens',type=int,default=96)
    ap.add_argument('--seed',type=int,default=20261111)
    ap.add_argument('--no-resume',action='store_true')
    args=ap.parse_args()
    here=Path(__file__).resolve().parent
    with (here/'manifest.csv').open(encoding='utf-8-sig',newline='') as f: cases=list(csv.DictReader(f))
    assert len(cases)==5
    tasks=[]
    for c in cases:
        assert '红框' not in c['prompt'] and '青框' not in c['prompt']
        for arm in ('red','cyan'):
            p=here/c[f'{arm}_image']
            if not p.is_file() or sha(p)!=c[f'{arm}_sha256']:
                raise RuntimeError(f'Image missing / hash mismatch: {p}')
            tasks.append((c,arm,p))
    out=here/args.output if not Path(args.output).is_absolute() else Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    done={}
    if out.exists() and not args.no_resume:
        for line in out.read_text(encoding='utf-8').splitlines():
            if line.strip():
                o=json.loads(line)
                done[(o['case_id'],o['arm'])]=o
        for c,arm,p in tasks:
            prev=done.get((c['case_id'],arm))
            if prev and (prev.get('image_sha256')!=sha(p) or prev.get('prompt')!=c['prompt'] or prev.get('model')!=args.model or prev.get('max_new_tokens')!=args.max_new_tokens):
                raise RuntimeError('Existing output has different config. Use a fresh --output path.')
    if args.no_resume and out.exists():
        raise RuntimeError('To start a separate run, use a new --output path; existing results never overwritten.')
    tasks=[t for t in tasks if (t[0]['case_id'],t[1]) not in done]
    print('queued:',len(tasks),'of 10; original confounded outputs are NOT reused')
    if not tasks: return
    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise SystemExit('Missing Qwen2.5-VL dependency: '+str(e)+'\nSee README_FIRST.md')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: this run requires the original GPU model environment.')
    random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,torch_dtype='auto',device_map='auto',attn_implementation='sdpa')
    model.eval()
    processor=AutoProcessor.from_pretrained(args.model)
    print('model loaded:',args.model)
    for k,(c,arm,p) in enumerate(tasks,1):
        messages=[{'role':'user','content':[{'type':'image','image':p.resolve().as_uri()},{'type':'text','text':c['prompt']}]}]
        txt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        imgs,vids=process_vision_info(messages)
        inputs=processor(text=[txt],images=imgs,videos=vids,padding=True,return_tensors='pt').to(model.device)
        torch.cuda.synchronize()
        st=time.perf_counter()
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,num_beams=1)
        torch.cuda.synchronize()
        elapsed=(time.perf_counter()-st)*1000
        answer_ids=[a[len(b):] for a,b in zip(generated,inputs.input_ids)]
        answer=processor.batch_decode(answer_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
        obj={'schema':'prd32_oracle8_neutral_ab_v2','status':'SHADOW_BLOCKED','test_set':'DEV_EXPLORATORY_ORACLE8_NOT_BLIND_GT',
          'case_id':c['case_id'],'target':c['target'],'question_id':c['question_id'],'question_type':c['question_type'],
          'arm':arm,'prompt':c['prompt'],'image':str(p.relative_to(here)),'image_sha256':sha(p),
          'answer':answer,'duration_ms':round(elapsed,3),'generated_tokens':sum(int(x.numel()) for x in answer_ids),
          'model':args.model,'max_new_tokens':args.max_new_tokens,'seed':args.seed,'do_sample':False,
          'torch_version':torch.__version__,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'human_review':'PENDING_USER_REVIEW','independent_blind_GT':False,'production_release_allowed':False}
        with out.open('a',encoding='utf-8') as f:
            f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        print(f'[{k}/{len(tasks)}] {c["case_id"]} {arm} {elapsed:.1f} ms {answer[:120]}',flush=True)
    print('DONE:',out)
if __name__=='__main__':main()
