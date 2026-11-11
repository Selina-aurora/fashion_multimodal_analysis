#!/usr/bin/env python3
'''16 confirmed DEV60 targets x 2 natural-language VQA questions. Shadow-only, no holdout.

All source/code/V5/Qwen checked against pinned frozen DEV60 lineage.
Existing color question is *not* reused as a substitute for new questions.
'''
from __future__ import annotations
import argparse,csv,hashlib,html,base64,json,os,statistics,time,sys,traceback,zipfile,collections
from pathlib import Path
from datetime import datetime
from stage1_verified_runtime import (WORKSPACE,REPO,V5_RAW,IMAGE_ROOT,CKPT,RUNNER,QWEN_MODEL,
    v5_preflight,verified_runner,load_guard,refuse_restricted,write_json,sync,digest_file,check_producer_match)
from joint_eval_stage2_dev60 import manifest_verified,load_all_targets
from run_vqa_concise_ab_v2 import load_models,sha_text,percent
HERE=Path(__file__).resolve().parent
PLAN=HERE/'FIXED_16_TARGETS_32_QUESTIONS.json'
PLAN_SHA=HERE/'FIXED_SELECTION_SHA256.txt'
TYPES={'attribute_visible','design_visible','material_process_boundary','styling_advice'}
STATUS='SHADOW_BLOCKED_VQA_MULTITYPE_NO_APPROVED_PRODUCTION_GUARD'
# A fresh, task-neutral prompt is REQUIRED for multi-question VQA.
# The original locked DEV60 `core.build_prompt()` is exclusively for the
# frozen color Baseline verification and the excluded warmup.  It ignores
# qrec['query'], so it MUST NEVER be used for the new 32 questions.
MULTITYPE_PROMPT_VERSION = 'vqa_multitype_question_passthrough_v2'
PROMPT_PREFIX = (
    '你是电商服饰图片问答助手。输入为一张带红色矩形目标标记的商品整图。\n'
    '请先确认红框真正指向哪一件商品，只围绕该商品回答用户当前的问题；'
    '不要描述背景或其他衣物，也不要把标记框自身的颜色算作商品颜色。\n'
    '仅根据图片中实际可见、能够可靠确认的特征回答。'
    '当目标不清楚、目标被遮挡或无法从图片核实某项信息时，请明确说明“无法确定”，不要猜测。'
    '不能仅凭外观断言纤维成分、比例、实际制造工艺或商品未展示的信息。'
    '穿搭建议应以可见颜色和款式为依据，表述为建议而非产品事实。\n'
    '不要提及红框、标注、模型或推理过程；回答可以完整但应简洁。\n'
    '用户当前问题：'
)

def build_multitype_prompt(question: str, typ: str) -> str:
    """Construct a new question-bearing prompt, not a color-baseline clone."""
    if typ not in TYPES:
        raise ValueError('unknown question type: '+repr(typ))
    q=str(question).strip()
    if not q or '\x00' in q or len(q)>1200:
        raise ValueError('invalid user question')
    result = PROMPT_PREFIX + q + '\n请直接回答这一个问题：'
    if result.count(q)!=1:
        raise ValueError('question duplicated in prompt')
    return result

def validate_question_prompts(questions, original_question):
    """Fail closed for missing, duplicated, stale-color or identical prompts."""
    prompts=[]
    for q in questions:
        question=q['question']
        p=build_multitype_prompt(question,q['type'])
        if p.count(question)!=1 or not p.endswith('请直接回答这一个问题：'):
            raise ValueError('QUESTION_NOT_IN_PROMPT')
        # The locked color question is permitted ONLY if the new question
        # deliberately contains it; do not let it leak as a second question.
        if original_question in p and original_question not in question:
            raise ValueError('ORIGINAL_COLOR_QUESTION_REMAINS')
        prompts.append(p)
    if len(prompts)!=32 or len(set(prompts))<4:
        raise ValueError('multi-question input diversity insufficient')
    return prompts

def read_plan():
    if digest_file(PLAN)!=PLAN_SHA.read_text('ascii').strip():raise ValueError('selection hash changed')
    obj=json.loads(PLAN.read_text(encoding='utf8'))
    if obj['schema']!='prd32_vqa_multitype_dev60_fixed_16targets_32questions_v1' or len(obj['targets'])!=16 or len(obj['questions'])!=32:
        raise ValueError('wrong fixed plan')
    if collections.Counter(q['type'] for q in obj['questions'])!={x:8 for x in TYPES}:raise ValueError('wrong question coverage')
    if collections.Counter(t['assisted_reference_category'] for t in obj['targets'])!={x:2 for x in ('top','pants','skirt','outerwear','dress','shoe','bag','accessory')}:
        raise ValueError('wrong class coverage')
    keys={(t['case_id'],t['instance_id']) for t in obj['targets']}
    if len(keys)!=16 or len({q['question_id'] for q in obj['questions']})!=32:raise ValueError('duplicate target/question')
    if { (q['case_id'],q['instance_id']) for q in obj['questions']}!=keys:raise ValueError('query target mismatch')
    if any(q['no_independent_answer_gt'] is not True or not q['question'].strip() or t['development_reference_only'] is not True for q in obj['questions'] for t in []):
        raise ValueError('unsupported reference')
    if any(q['no_independent_answer_gt'] is not True for q in obj['questions']):raise ValueError('false GT status')
    if obj['holdout_used'] is not False or obj['production_release_allowed'] is not False:raise ValueError('unsafe metadata')
    return obj

def percentile(a,p=95):
    v=sorted(float(x) for x in a)
    if not v:return None
    pos=(len(v)-1)*p/100; i=int(pos);j=min(i+1,len(v)-1)
    return round(v[i]+(v[j]-v[i])*(pos-i),3)

def preflight(args,out=None):
    obj=read_plan();manifest,_=manifest_verified()
    mm={(r['case_id'],r['instance_id']):r for r in manifest['targets']}
    keys=[(t['case_id'],t['instance_id']) for t in obj['targets']]
    for t in obj['targets']:
        k=t['case_id'],t['instance_id'];r=mm.get(k)
        if r is None or r['source_image_sha256']!=t['source_image_sha256'] or r['v5_frozen_category']!=t['v5_frozen_category']:
            raise ValueError('DEV60 frozen target hash/category drift: '+str(k))
    for path in (args.repo,args.raw_dir,args.image_root,args.checkpoint,args.runner,args.model_dir):refuse_restricted(path)
    _,hashes=v5_preflight(args.raw_dir,args.image_root,args.repo,args.checkpoint)
    targets=load_all_targets(args.raw_dir,args.image_root,manifest,keys)
    runner=verified_runner(args.runner)
    valid,core,helper,focus=runner.prompt_check(argparse.Namespace(
        snapshot=runner.DEFAULT_SNAP,source=runner.DEFAULT_SOURCE,
        focus_script=runner.DEFAULT_FOCUS,raw=runner.DEFAULT_RAW,input=runner.DEFAULT_INPUT))
    if len(valid)!=171:raise ValueError('171 baseline prompts did not verify')
    load_guard(args.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py') # source only; NOT called
    if not (args.model_dir/'config.json').is_file():raise FileNotFoundError('missing Qwen config')
    sample=obj['questions'][0];k=(sample['case_id'],sample['instance_id']);t=targets[k]
    qrec={'image_path':t['image_path'],'image_sha256':t['image_sha256'],
      'target_bbox_xyxy':t['expected_bbox'],'target_instance_id':k[1],
      'task':'vqa','query':runner.QUESTION,'length_chars':[20,80],'predicted_attribute_context':[]}
    baseline=core.build_prompt(helper,focus,qrec,'vqa','baseline')
    if sha_text(baseline)!=runner.PROMPT_SHA['baseline']['vqa']:
        raise ValueError('original baseline VQA prompt hash mismatch')
    qrec['query']=sample['question']
    # Legacy core.build_prompt ignores `query`: DO NOT use it for multitype Q&A.
    validate_question_prompts(obj['questions'],runner.QUESTION)
    novel=build_multitype_prompt(sample['question'],sample['type'])
    if novel==baseline:
        raise ValueError('multitype prompt unexpectedly identical to frozen color Baseline')
    if out:
        write_json(out/'PREFLIGHT.json',{'status':'MULTITYPE_PILOT_PREFLIGHT_PASS','n_targets':16,'n_queries':32,
          'source_hashes':hashes,'fixed_plan_sha256':digest_file(PLAN),'original_baseline_prompt_sha256':sha_text(baseline),
          'example_new_prompt_sha256':sha_text(novel),'new_question_present_in_prompt':True,
          'multitype_prompt_version':MULTITYPE_PROMPT_VERSION,
          'all_32_questions_prompt_checked':True,
          'original_baseline_used_only_for_warmup':True,
          'old_question_removed':True,'holdout_used':False,'production_release':False,
          'three_task_guard_not_invoked':True,'independent_gt':False})
    return obj,targets,runner,core,helper,focus

def run_live(args,out):
    obj,targets,runner,core,helper,focus=preflight(args,out)
    if not args.execute_models:raise ValueError('--execute-models is required for live')
    import torch,numpy as np
    from PIL import Image
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    free=torch.cuda.mem_get_info()[0]/2**30
    if free<18:raise RuntimeError(f'at least 18GiB free GPU needed, got {free:.2f}')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    loaded=load_models(args,out)
    torch,np,Image,qwen,processor,v5,ID_TO_CLASS=loaded
    from fashion_multimodal_analysis.evaluation.runtime import predict_timed
    all_results=[]
    query_by_target=collections.defaultdict(list)
    for q in obj['questions']:query_by_target[q['case_id'],q['instance_id']].append(q)
    for ti,trow in enumerate(obj['targets'],1):
        case,inst=trow['case_id'],trow['instance_id'];t=targets[case,inst]
        sync(torch);tv=time.perf_counter()
        with Image.open(t['image_path']) as im:img=im.convert('RGB')
        boxes,labels,scores,masks,_=predict_timed(img,v5,torch.device('cuda'),score_threshold=.4,mask_threshold=.5)
        sync(torch);v5_ms=(time.perf_counter()-tv)*1000
        idx=int(inst.split('_')[-1]);
        if idx>=len(labels):raise ValueError('V5 target missing: '+case+'/'+inst)
        cat=ID_TO_CLASS.get(int(labels[idx]))
        newmask=(np.asarray(masks[idx])>0).astype(np.uint8)
        with Image.open(t['frozen_mask_path']) as im:oldmask=np.asarray(im.convert('L'))>127
        ck=check_producer_match(target=t,ins={'instance_id':inst,'category':cat,'score':float(scores[idx]),'bbox':[float(v) for v in boxes[idx]]},
             reference_mask=oldmask,produced_mask=newmask)
        if ck['mask_iou']<.9999:raise ValueError('frozen V5 output drift: '+case+'/'+inst)
        qrec={'image_path':t['image_path'],'image_sha256':t['image_sha256'],
           'target_bbox_xyxy':[float(x) for x in boxes[idx]],'target_instance_id':inst,
           'task':'vqa','query':runner.QUESTION,'length_chars':[20,80],'predicted_attribute_context':[]}
        mdir=out/'_working_marked'/f'{case}__{inst}';mdir.mkdir(parents=True,exist_ok=False)
        tm=time.perf_counter(); marked=Path(helper.make_marked_image(qrec,mdir,'marked_full_image',.05)).resolve(strict=True)
        mark_ms=(time.perf_counter()-tm)*1000
        if not marked.is_relative_to(mdir.resolve()):raise ValueError('marked file escaped output folder')
        rel=f'marked_inputs/{case}__{inst}.jpg'
        (out/'marked_inputs').mkdir(exist_ok=True)
        with Image.open(marked) as im:im.convert('RGB').save(out/rel,quality=85,optimize=True)
        original_prompt=core.build_prompt(helper,focus,qrec,'vqa','baseline')
        if sha_text(original_prompt)!=runner.PROMPT_SHA['baseline']['vqa']:
            raise ValueError('original prompt drift')
        if ti==1:
            warm_ans,warm_info=helper.run_model_case(qwen,processor,qrec,marked,original_prompt,task_limits={'vqa':128})
            warm_ref=json.loads((HERE/'VQA_FROZEN_BASELINE_171.json').read_text(encoding='utf8'))['targets'][case+'/'+inst]
            if sha_text(warm_ans)!=warm_ref['vqa_response_sha256']:
                raise ValueError('excluded warmup baseline VQA response drift')
            write_json(out/'WARMUP_EXCLUDED.json',{'case_id':case,'instance_id':inst,'task':'vqa',
                'answer_sha256':sha_text(warm_ans),'generated_tokens':warm_info['generated_tokens'],
                'included_in_32_measured_calls':False,'publish_allowed':False})
        for query in query_by_target[case,inst]:
            qrec['query']=query['question']
            prompt=build_multitype_prompt(query['question'],query['type'])
            if prompt.count(query['question'])!=1:
                raise ValueError('new question is not isolated in VQA prompt')
            if runner.QUESTION in prompt and runner.QUESTION not in query['question']:
                raise ValueError('legacy color question leaked into multi-question prompt')
            sync(torch);st=time.perf_counter()
            response,timing=helper.run_model_case(qwen,processor,qrec,marked,prompt,task_limits={'vqa':128})
            sync(torch);host_ms=round((time.perf_counter()-st)*1000,3)
            if not isinstance(response,str) or not response.strip():raise ValueError('empty answer')
            rr={**query,'source_image_sha256':t['image_sha256'],'marked_image_sha256':digest_file(marked),
              'marked_image_rel':rel,'v5_frozen_category':cat,'v5_mask_iou_against_frozen':ck['mask_iou'],
              'assisted_reference_category':trow['assisted_reference_category'],
              'assisted_identity':trow['assisted_target_identity'],'assisted_mask':trow['assisted_mask_review'],
              'answer':response,'answer_sha256':sha_text(response),'prompt_sha256':sha_text(prompt),
              'prompt_version':MULTITYPE_PROMPT_VERSION,'question_in_prompt':True,
              'generated_tokens':timing['generated_tokens'],'max_new_tokens':128,
              'token_cap_reached':timing['generated_tokens']>=128,'qwen_host_ms':host_ms,
              'model_preprocess_ms':timing['preprocess_ms'],'model_generation_ms':timing['generation_ms'],
              'v5_host_ms_shared_two_questions':round(v5_ms,3),'marker_ms_shared_two_questions':round(mark_ms,3),
              'per_question_v5_plus_mark_plus_qwen_estimated_ms':round(v5_ms+mark_ms+host_ms,3),
              'review_status':'PENDING_NEW_MANUAL_REVIEW','guard_status':STATUS,'production_release':False,
              'original_full_task_guard_invoked':False,'independent_gt':False,'holdout_used':False}
            all_results.append(rr)
            with (out/'PER_QUESTION_RESULTS.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(rr,ensure_ascii=False)+'\n')
            print('MULTITYPE_RESULT',len(all_results),'/32',query['question_id'],query['type'],
                  'qwen_ms',host_ms,'tokens',timing['generated_tokens'],'release=false',flush=True)
    if len(all_results)!=32 or len({r['question_id'] for r in all_results})!=32 or any(r['production_release'] is not False for r in all_results):
        raise ValueError('incomplete or unsafe question set')
    csvcols=['question_id','case_id','instance_id','type','question','answer','assisted_reference_category',
      'assisted_identity','assisted_mask','v5_frozen_category','generated_tokens','token_cap_reached',
      'qwen_host_ms','per_question_v5_plus_mark_plus_qwen_estimated_ms','review_status','human_rating',
      'human_evidence_note','error_type','marked_image_rel','production_release']
    with (out/'QUESTION_HUMAN_REVIEW_PENDING.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=csvcols,extrasaction='ignore');w.writeheader();w.writerows(all_results)
    bytype={}
    for typ in sorted(TYPES):
        ss=[r for r in all_results if r['type']==typ]
        bytype[typ]={'n':len(ss),'mean_qwen_host_ms':round(statistics.mean(r['qwen_host_ms'] for r in ss),3),
                     'p95_qwen_host_ms':percentile([r['qwen_host_ms'] for r in ss]),
                     'mean_output_tokens':round(statistics.mean(r['generated_tokens'] for r in ss),2),
                     'hit_token_cap_count':sum(r['token_cap_reached'] for r in ss)}
    write_json(out/'SUMMARY.json',{'status':'MULTITYPE_VQA_PILOT_V1_COMPLETED_SHADOW_ONLY',
      'qwen_calls_measured':32,'warmup_qwen_calls_excluded':1,'target_count':16,'question_types':bytype,
      'overall_mean_qwen_host_ms':round(statistics.mean(r['qwen_host_ms'] for r in all_results),3),
      'original_guard_invoked':False,'production_release':False,'no_holdout':True,
      'multitype_prompt_version':MULTITYPE_PROMPT_VERSION,
      'independent_gt':False,'official_prd_accuracy_evaluated':False,
      'note':'Exploratory DEV60 multitype pilot; do not compare directly to fixed-color VQA accuracy or production E2E SLA.'})
    make_html(all_results,out)
    print('VQA_MULTITYPE_PILOT_V1_COMPLETE_SHADOW_ONLY',flush=True)

def make_html(records,out):
    blocks=[]
    grouped=collections.defaultdict(list)
    for r in records:grouped[r['case_id'],r['instance_id']].append(r)
    for (case,inst),qs in grouped.items():
        im=out/qs[0]['marked_image_rel']
        encoded=base64.b64encode(im.read_bytes()).decode('ascii')
        content=''.join(f'<section><p class="type">{html.escape(r["question_id"])} · {html.escape(r["type"])}</p>'
            f'<p><b>问题：</b>{html.escape(r["question"])}</p>'
            f'<p><b>回答：</b>{html.escape(r["answer"])}</p>'
            f'<p>Qwen {r["qwen_host_ms"]:.1f}ms · {r["generated_tokens"]} tokens · Shadow 阻断</p>'
            f'<label>人工判断<select data-q="{html.escape(r["question_id"])}"><option value="待审核">待审核</option>'
            f'<option>正确有据</option><option>部分正确</option><option>合理拒答</option><option>编造或跨目标</option><option>目标证据不足</option></select></label>'
            f'<textarea data-note="{html.escape(r["question_id"])}" placeholder="填写具体图像证据与质量问题"></textarea></section>' for r in qs)
        blocks.append(f'<article><h2>{html.escape(case)}/{html.escape(inst)} · DEV参考类别 {html.escape(qs[0]["assisted_reference_category"])}</h2>'
            f'<p>目标身份 {html.escape(qs[0]["assisted_identity"])} · Mask {html.escape(qs[0]["assisted_mask"])}（开发集辅助确认，非独立GT）</p>'
            f'<img src="data:image/jpeg;base64,{encoded}" alt="目标红框图">{content}</article>')
    page='''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>PRD32 VQA多问题类型审核</title><style>
    body{font:15px/1.5 system-ui,'Microsoft YaHei',sans-serif;background:#f5f7fa;color:#1f2937;max-width:1000px;margin:auto;padding:20px}article{background:white;border:1px solid #ddd;border-radius:12px;padding:16px;margin:16px 0}img{width:100%;max-width:520px;max-height:400px;object-fit:contain}section{border-top:1px solid #e7eaf0;padding:12px 0}select,textarea{display:block;width:90%;max-width:780px;padding:7px;margin-top:5px}textarea{min-height:45px}.type{color:#075985;font-weight:600}button{background:#1d4ed8;border:0;border-radius:8px;color:white;padding:10px 16px;cursor:pointer}
    </style></head><body><h1>VQA多题型 DEV60 试验 · 16目标/32题</h1><p>所有结果禁止发布。问答需要人工核对，尤其是目标不确定、材料工艺编造和跨目标描述。不得直接算正式PRD验收。</p><button id="export">导出人工审核CSV</button>'''+''.join(blocks)+'''<script>
    function cell(s){return '"'+String(s).replaceAll('"','""')+'"'}
    document.getElementById('export').addEventListener('click',()=>{let rows=[['question_id','human_review','human_note']];document.querySelectorAll('select[data-q]').forEach(e=>{let id=e.dataset.q;let note=document.querySelector('textarea[data-note="'+id+'"]');rows.push([id,e.value,note?.value??''])});let b=new Blob(['\\ufeff'+rows.map(r=>r.map(cell).join(',')).join('\\r\\n')],{type:'text/csv;charset=utf-8'});let u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='VQA_MULTITYPE_HUMAN_REVIEW.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)});
    </script></body></html>'''
    # literal JS escaped sequence must be actual js escape, not backslash doubled
    page=page.replace("'\\\\ufeff'", "'\\ufeff'").replace("'\\\\r\\\\n'", "'\\r\\n'")
    (out/'VQA_MULTITYPE_32_VISUAL_REVIEW.html').write_text(page,encoding='utf8')

def archive(out):
    dest=out.with_suffix('.zip')
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and '_working_marked' not in p.parts:z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None:raise ValueError('corrupt result archive')
    print('UPLOAD_THIS:',dest,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('package-check','preflight','live'),default='package-check')
    p.add_argument('--execute-models',action='store_true')
    p.add_argument('--repo',type=Path,default=REPO);p.add_argument('--raw-dir',type=Path,default=V5_RAW)
    p.add_argument('--image-root',type=Path,default=IMAGE_ROOT);p.add_argument('--checkpoint',type=Path,default=CKPT)
    p.add_argument('--runner',type=Path,default=RUNNER);p.add_argument('--model-dir',type=Path,default=QWEN_MODEL)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.mode=='package-check':
        obj=read_plan();manifest,_=manifest_verified()
        mm={(r['case_id'],r['instance_id']):r for r in manifest['targets']}
        for t in obj['targets']:
            k=t['case_id'],t['instance_id']
            if k not in mm or mm[k]['source_image_sha256']!=t['source_image_sha256']:raise ValueError('manifest mismatch')
        print('VQA_MULTITYPE_PACKAGE_CHECK_PASS 16 targets 32 questions; no GPU');return
    out=args.out or WORKSPACE/f'PRD32_VQA_MULTITYPE_PILOT_V1_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}_UPLOAD_THIS'
    if out.exists():raise SystemExit('Refusing overwrite: '+str(out))
    out.mkdir(parents=True)
    try:
        if args.mode=='preflight':
            preflight(args,out);print('VQA_MULTITYPE_PREFLIGHT_PASS; no GPU inference',flush=True)
        else:run_live(args,out)
    except BaseException as e:
        write_json(out/'ERROR.json',{'error_type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),
                                    'production_release':False})
        archive(out);raise
    archive(out)
if __name__=='__main__':main()
