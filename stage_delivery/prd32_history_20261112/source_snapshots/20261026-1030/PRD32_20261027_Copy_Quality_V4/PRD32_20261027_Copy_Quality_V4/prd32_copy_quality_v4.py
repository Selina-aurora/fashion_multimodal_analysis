#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD 3.2 evidence-first copy experiment V4 (DEV, not acceptance).

One visual model call per frozen COPY case -> 7-field provisional evidence JSON
-> strict parser -> conservative template draft -> human review. The evidence is
NEVER called ground truth. Old human GT/grades are NEVER supplied to the model.

Uses V3 full-batch guarded input screening; copies of existing code are bundled
for reproducibility. No frozen PRD 3.1 files are written. No sealed holdout.
"""
from __future__ import annotations
import argparse, base64, collections, csv, datetime, hashlib, html, importlib.util
import json, re, statistics, sys, time
from pathlib import Path

from prd32_resilient_runner_v3 import load_v1, select_with_guard
from prd32_defensive_gate import read_jsonl_lenient, restricted, sha256_file

FROZEN = Path('/workspace/prd31_stage_freeze_20261026')
MODEL = Path('/workspace/models/Qwen2.5-VL-7B-Instruct')
OUTPUT = Path('/workspace/PRD32_COPY_QUALITY_V4_20261027')
TASK='copy'
CLASSES={'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}
PATTERNS={'纯色','条纹','格纹','人字纹','波点','花卉印花','文字印花','字母印花','几何印花','动物纹','拼色'}
SUBTYPES={'背带裙','吊带裙','连衣裙','半身裙','T恤','衬衫','卫衣','针织衫','夹克','西装外套','大衣','长裤','短裤','运动鞋','皮鞋','背包','手提包','配饰'}
STRUCTURES={'背带','交叉肩带','肩带','露背','圆领','V领','翻领','长袖','短袖','无袖','门襟','纽扣','纽扣装饰','口袋','拼接','黑白条纹下摆','条纹下摆','宽松裙摆','褶皱','拉链','抽绳','蝴蝶结','刺绣','字母印花'}
ERROR_TERMS=('红框','红色框','框内','图片里','模特','100%','纯棉','真丝','面料成分','保证','绝对','显瘦','高品质','蕾丝面料')
PROMPT = '''请只观察图片中红色矩形框对应的【单件目标服饰】，忽略内搭、包、路人及背景。完成可见事实提取，不写广告文案。
务必区分：背带裙与普通无袖连衣裙；字母/文字印花与数字/波点；纽扣与波点；服饰裙摆和周围其他衣物。只有确实能辨认时才填写；不能推断时使用真正的JSON null。
不要把口袋默认看作装饰，纽扣即使有功能也可描述为可见扣件。不要推断面料成分、产地、品牌、用途、身材效果、工艺或质量。
只输出一个JSON对象，不要解释，字段严格为：
{"category":null,"primary_color":null,"pattern":null,"subtype":null,"visible_structures":[],"distinct_decorations":[],"uncertainty":[]}
category 必须从 top,pants,skirt,outerwear,dress,shoe,bag,accessory 中选或null。
pattern 仅可为 纯色、条纹、格纹、人字纹、波点、花卉印花、文字印花、字母印花、几何印花、动物纹、拼色 或null；蕾丝等材质/工艺不能当图案；看不到图案≠已证实纯色。
subtype仅选可辨认的具体款式，如背带裙、吊带裙、连衣裙、半身裙、T恤、衬衫、卫衣、针织衫、夹克、西装外套、大衣、长裤、短裤、运动鞋、皮鞋、背包、手提包、配饰，或null。
visible_structures 与 distinct_decorations 都是能明确目视的短词列表；不确定的细节不写入。
uncertainty 列出不能确定的属性名称，例如pattern、subtype；不要编造。'''
KEYS={'category','primary_color','pattern','subtype','visible_structures','distinct_decorations','uncertainty'}


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_json(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_jsonl(p,objs):
 with p.open('w',encoding='utf-8') as f:
  for a in objs:f.write(json.dumps(a,ensure_ascii=False,allow_nan=False)+'\n')


def parse_facts(raw):
    """Fail closed on bad shapes; unknown values abstain rather than coercing."""
    issues=[]
    if not isinstance(raw,str) or len(raw)>12000:return None,['BAD_RAW_TEXT']
    s=raw.strip()
    if s.startswith('```'):
        s=re.sub(r'^```(?:json)?\s*','',s,flags=re.I)
        s=re.sub(r'\s*```\s*$','',s)
    try: obj=json.loads(s,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError,TypeError):return None,['JSON_PARSE_FAILURE']
    if not isinstance(obj,dict) or set(obj)!=KEYS:return None,['JSON_SCHEMA_KEYS_INVALID']
    result={}
    for key,allowed in [('category',CLASSES),('pattern',PATTERNS),('subtype',SUBTYPES)]:
        v=obj[key]
        if v is not None and (not isinstance(v,str) or v not in allowed):
            issues.append('INVALID_ENUM_'+key.upper());v=None
        result[key]=v
    v=obj['primary_color']
    if v is not None and (not isinstance(v,str) or not 1<=len(v.strip())<=12 or any(w in v for w in ERROR_TERMS)):
        issues.append('INVALID_COLOR');v=None
    result['primary_color']=v.strip() if isinstance(v,str) else None
    for key in ('visible_structures','distinct_decorations','uncertainty'):
        v=obj[key]
        if not isinstance(v,list):
            issues.append('INVALID_LIST_'+key.upper());v=[]
        clean=[]
        for item in v[:16]:
            if not isinstance(item,str) or not item.strip():issues.append('INVALID_ITEM_'+key.upper());continue
            item=item.strip()
            if len(item)>20 or any(w in item for w in ERROR_TERMS):issues.append('UNSAFE_ITEM_'+key.upper());continue
            if key=='visible_structures' and item not in STRUCTURES:
                issues.append('UNKNOWN_STRUCTURE_ABSTAINED');continue
            if key=='distinct_decorations' and item not in STRUCTURES:
                issues.append('UNKNOWN_DECORATION_ABSTAINED');continue
            if item not in clean: clean.append(item)
        result[key]=clean
    if result['subtype']=='背带裙' and result['category'] not in (None,'dress'):
        issues.append('SUBTYPE_CLASS_CONFLICT');result['subtype']=None
    if result['subtype']=='吊带裙' and result['category'] not in (None,'dress'):
        issues.append('SUBTYPE_CLASS_CONFLICT');result['subtype']=None
    if result['pattern'] is None and 'pattern' not in result['uncertainty']: result['uncertainty'].append('pattern')
    if result['subtype'] is None and 'subtype' not in result['uncertainty']: result['uncertainty'].append('subtype')
    return result,issues


def draft_from_facts(facts,max_chars=80):
    """Unverified DRAFT only, NEVER approved facts. No unsupported inferring."""
    if not isinstance(facts,dict) or facts.get('category') not in CLASSES:
        return None,['NO_VALID_CATEGORY']
    if not facts.get('primary_color'):
        return None,['NO_RELIABLE_COLOR']
    names={'top':'上衣','pants':'裤装','skirt':'半身裙','outerwear':'外套','dress':'连衣裙','shoe':'鞋','bag':'包','accessory':'配饰'}
    garment=facts.get('subtype') or names[facts['category']]
    noun=facts['primary_color']+garment
    clauses=[]
    p=facts.get('pattern')
    if p=='纯色': clauses.append('裙身无明显印花' if garment in ('连衣裙','背带裙','吊带裙') else '表面以纯色为主')
    elif p is not None:clauses.append('可见'+p+'图案')
    s=[x for x in facts.get('visible_structures',[]) if x in STRUCTURES]
    d=[x for x in facts.get('distinct_decorations',[]) if x in STRUCTURES]
    for x in s + d:
        desc='带有'+x+'设计'
        if desc not in clauses:clauses.append(desc)
    text='，'.join([noun]+clauses)+'。'
    issues=[]
    if any(w in text for w in ERROR_TERMS):return None,['UNSAFE_COPY_TERMS']
    if len(text)>max_chars:
        while clauses and len('，'.join([noun]+clauses)+'。')>max_chars: clauses.pop()
        text='，'.join([noun]+clauses)+'。'
        issues.append('TRUNCATED_TO_LIMIT')
    if len(text)>max_chars:return None,['COPY_LENGTH_INVALID']
    return text,issues


def validate_all(freeze_dir, v1):
    bridge=freeze_dir/'bridge_inputs.jsonl'
    freeze_info=v1.validate_freeze(freeze_dir,bridge)
    raw=read_jsonl_lenient(bridge)
    audited=select_with_guard(raw)
    return bridge, freeze_info, audited


def baseline_predictions(path):
    if path is None:return {}
    p=Path(path)
    if not p.is_file():raise FileNotFoundError('baseline missing: '+str(p))
    result={}
    for r in read_jsonl_lenient(p):
        if isinstance(r,dict) and r.get('task')=='copy' and isinstance(r.get('case_id'),str):
            result[r['case_id']]=r.get('raw_text')
    return result


def html_review(out, rows, original):
    """Static-first review, no JS needed to read case/image/output."""
    cards=[]
    for r in rows:
        cid=r['case_id']; old=original.get(cid,'（未提供旧版结果）')
        image=r.get('marked_image')
        img=''
        if image and Path(image).is_file():
            data=base64.b64encode(Path(image).read_bytes()).decode()
            img=f'<img style="width:280px;max-width:100%;height:auto" alt="marked target" src="data:image/jpeg;base64,{data}">'
        st=f'<pre>{html.escape(json.dumps(r.get("facts"),ensure_ascii=False,indent=2))}</pre>'
        candidate=r.get('draft') or '〔安全拦截：未生成候选文案〕'
        cards.append(f'''<section class="card"><div><h2>{html.escape(cid)}</h2>{img}<p>{html.escape(r.get('image_id',''))} / {html.escape(r.get('target_instance_id',''))}</p></div>
        <div><b>旧版</b><p>{html.escape(str(old))}</p><b>V4 视觉候选</b><p>{html.escape(candidate)}</p>
        <small>状态：{html.escape(r['status'])}；问题：{html.escape(', '.join(r.get('issues',[])))}</small>
        <details><summary>展开模型提取的候选属性（未经人工核验）</summary>{st}</details>
        <label>新版文案人工满意度：<select data-case="{html.escape(cid)}"><option value="PENDING">待审核</option><option value="SATISFIED">满意</option><option value="NOT_SATISFIED">不满意</option><option value="BLOCKED_REVIEW">模型拒发，需要人工处理</option></select></label>
        <label>错误说明：<textarea data-note="{html.escape(cid)}" rows="2" placeholder="如类别、颜色、背带、字母、纽扣或下摆有错误，请具体写明"></textarea></label></div></section>''')
    document='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>PRD 3.2 商品文案 V4 对照审核</title>
    <style>body{font:16px system-ui,"Microsoft Yahei",sans-serif;background:#f5f7fa;color:#1c2b3a;margin:28px auto;max-width:1140px}h1{margin:0}header{padding:22px;background:#182741;color:white;border-radius:12px;margin-bottom:20px}.card{display:grid;grid-template-columns:310px 1fr;gap:25px;padding:22px;border:1px solid #d6dfe9;border-radius:12px;margin:15px 0;background:white}.card label{display:block;margin:12px 0}.card textarea{display:block;width:97%;padding:8px}button{padding:10px 18px;background:#2358ad;color:white;border:0;border-radius:8px;cursor:pointer}details{font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:650px){.card{grid-template-columns:1fr}}</style></head><body>
    <header><h1>PRD 3.2 商品文案质量 V4</h1><p>旧版 V2 vs V4 新候选。均为开发集，非盲测试；新版未使用人工GT。目标红框仅用于定位，不作为文案事实。</p></header>
    <button onclick="exportReview()">导出评分 JSON</button><p>先查看图片、比较旧版和新版，再审核。被拦截的结果不得算满意。</p>
    <main>'''+''.join(cards)+'''</main><script>
    function exportReview(){const results=[...document.querySelectorAll('select[data-case]')].map(s=>({case_id:s.dataset.case,verdict:s.value,notes:document.querySelector('textarea[data-note="'+s.dataset.case+'"]').value}));const blob=new Blob([JSON.stringify({schema:'prd32_copy_v4_review_dev_v1',results},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='PRD32_COPY_V4_human_review.json';document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(a.href)}
    </script></body></html>'''
    (out/'COPY_V4_compare_review.html').write_text(document,encoding='utf-8')


def run(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-dir',default=str(FROZEN))
    p.add_argument('--model-id',default=str(MODEL))
    p.add_argument('--v1-script',default=str(Path(__file__).with_name('prd32_integrated_pipeline_20261027.py')))
    p.add_argument('--output',default=str(OUTPUT))
    p.add_argument('--baseline-file',default='/workspace/PRD32_INTEGRATED_ALL12_20261027/predictions.jsonl')
    p.add_argument('--max-pixels',type=int,default=512*28*28)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args(argv)
    if args.self_test: self_test();return 0
    paths=(args.freeze_dir,args.v1_script,args.output,args.model_id)
    if any(restricted(x) for x in paths):p.error('sealed holdout prohibited')
    v1=load_v1(args.v1_script)
    freeze_dir=Path(args.freeze_dir)
    bridge,freeze_info,decisions=validate_all(freeze_dir,v1)
    all_records=read_jsonl_lenient(bridge)
    copy_positions=[i for i,r in enumerate(all_records) if isinstance(r,dict) and r.get('task')=='copy']
    copy_decisions=[(all_records[i],decisions[i]) for i in copy_positions]
    pass_count=sum(x['decision']!='BLOCKED' for _,x in copy_decisions)
    print(f'PRECHECK copy_cases={len(copy_decisions)} eligible={pass_count} blocked={len(copy_decisions)-pass_count}',flush=True)
    if args.dry_run:
        for rec,g in copy_decisions:print(rec.get('case_id'),g['decision'],[x['code'] for x in g.get('issues',[])])
        print('DRY_RUN_OK: source frozen, model not loaded, no files written');return 0
    out=Path(args.output)
    if out.exists():p.error(f'output exists; choose another --output: {out}')
    if not Path(args.model_id,'config.json').is_file():p.error('model directory not found, no network fetch')
    # Prevent accidental complete stop on an odd copy record: one failure -> explicit null.
    old=baseline_predictions(args.baseline_file) if args.baseline_file else {}
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():p.error('GPU unavailable')
    free_gib=torch.cuda.mem_get_info()[0]/1024**3
    if free_gib<18:p.error(f'Insufficient available GPU memory: {free_gib:.2f}GiB (<18GiB). Check other GPU processes.')
    out.mkdir(parents=True,exist_ok=False)
    imgs=out/'marked_images';imgs.mkdir()
    write_json(out/'source_run_config.json',{'time':now(),'model':args.model_id,'freeze_candidate':freeze_info,
        'bridge_sha256':sha256_file(bridge),'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
        'copy_case_ids':[r.get('case_id') for r,g in copy_decisions], 'human_gt_passed_to_model':False,
        'sealed_holdout_used':False,'prd31_modified':False,'mode':'DEV_NOT_ACCEPTED'})
    (out/'prompt_visible_facts.txt').write_text(PROMPT,encoding='utf-8')
    start=time.perf_counter()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id,local_files_only=True,device_map='auto',dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(args.model_id,local_files_only=True,max_pixels=args.max_pixels)
    load_time=round(time.perf_counter()-start,2)
    outputs=[];times=[]
    for idx,(rec,guard) in enumerate(copy_decisions,1):
        identity=guard.get('identity',{})
        result={'case_id':identity.get('case_id') or f'__line_{idx}',
            'task':'copy','image_id':identity.get('image_id',''),'target_instance_id':identity.get('target_instance_id',''),
            'status':'BLOCKED_INVALID_INPUT','draft':None,'facts':None,'raw_model_output':None,
            'issues':[x['code'] for x in guard.get('issues',[])],'inference_performed':False,
            'human_verified':False,'model_backend':'real_qwen','latency_ms':None,'marked_image':None}
        if guard['decision']!='BLOCKED':
            try:
                safe=guard['guarded_record']
                image=v1.make_marked_image(safe,imgs,'marked_full_image',0.05)
                result['marked_image']=str(image)
                # Don't rewrite the task: V1 GPU helper selects token cap by task='copy'.
                raw,timing=v1.run_model_case(model,processor,safe,image,PROMPT,task_limits={'copy':240})
                result['raw_model_output']=raw;result['timing']=timing
                result['latency_ms']=timing['end_to_end_ms'];times.append(timing['end_to_end_ms'])
                result['inference_performed']=True
                facts,issues=parse_facts(raw)
                result['facts']=facts;result['issues']+=issues
                candidate,e=draft_from_facts(facts)
                result['issues']+=e;result['draft']=candidate
                result['status']='DRAFT_REVIEW_REQUIRED' if candidate else 'HOLD_REVIEW_REQUIRED'
            except Exception as exc:
                result['issues'].append('CASE_EXCEPTION_'+type(exc).__name__)
                result['status']='HOLD_RUNTIME_ERROR';result['draft']=None
        outputs.append(result)
        print(f'[{idx}/{len(copy_decisions)}] {result["case_id"]}: {result["status"]}; {result["issues"]}',flush=True)
        # Write after each case to avoid losing earlier results on terminal/GPU failure.
        write_jsonl(out/'candidate_drafts.jsonl',outputs)
    html_review(out,outputs,old)
    with (out/'human_review_blank.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['case_id','image_id','baseline_copy','candidate_copy','v4_human_verdict','error_note'])
        for r in outputs:w.writerow([r['case_id'],r['image_id'],old.get(r['case_id'],''),r.get('draft') or '', '', ''])
    counts=collections.Counter(r['status'] for r in outputs)
    summary={'schema':'prd32_copy_quality_v4_dev','status':'DEV_COPY_EXPERIMENT_COMPLETED_NOT_ACCEPTED',
        'source_cases':len(copy_decisions),'eligible':pass_count,'inference_performed':sum(r['inference_performed'] for r in outputs),
        'status_counts':dict(counts),'candidate_count':sum(r['draft'] is not None for r in outputs),
        'median_one_call_inference_ms':round(statistics.median(times),2) if times else None,
        'model_load_seconds':load_time,'human_v4_reviewed':False,'visual_facts_are_unverified':True,
        'human_gt_passed_to_model':False,'frozen_prd31_not_modified':True,'holdout_used':False,
        'full_prd_acceptance':False}
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return 0 if counts.get('HOLD_RUNTIME_ERROR',0)==0 else 2


def self_test():
    one='''```json
{"category":"dress","primary_color":"黑色","pattern":null,"subtype":"背带裙","visible_structures":["背带"],"distinct_decorations":[],"uncertainty":["pattern"]}
```'''
    facts,issues=parse_facts(one)
    assert facts['subtype']=='背带裙' and not issues,(facts,issues)
    copy,issues=draft_from_facts(facts)
    assert '黑色背带裙' in copy and '材质' not in copy
    f2,issues=parse_facts('{"category":"shirt","primary_color":"白色","pattern":"蕾丝","subtype":null,"visible_structures":[],"distinct_decorations":[],"uncertainty":[]}')
    assert f2['category'] is None and f2['pattern'] is None
    assert draft_from_facts(f2)[0] is None
    assert parse_facts('{"category":"dress"}')[0] is None
    f3,issues=parse_facts('{"category":"top","primary_color":"白色","pattern":"字母印花","subtype":null,"visible_structures":["门襟","纽扣","条纹下摆"],"distinct_decorations":[],"uncertainty":[]}')
    c3,_=draft_from_facts(f3)
    assert '字母印花' in c3 and '纽扣' in c3 and '条纹下摆' in c3
    # Verify no word from human review notes is ever loaded by the prompt logic.
    assert '连衣裙是白色的' not in PROMPT
    print('V4_SELF_TEST_PASS: JSON, invalid enums, conservative refusal, visible details, no-GT-leak')

if __name__=='__main__':
    try:sys.exit(run())
    except KeyboardInterrupt:print('INTERRUPTED; inspect partial JSONL',file=sys.stderr);sys.exit(130)
