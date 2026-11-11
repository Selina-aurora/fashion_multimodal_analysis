#!/usr/bin/env python3
"""Matched Qwen VQA answer-format A/B on fixed DEV60 slow17 + control17.

Baseline: V5 + frozen baseline color VQA prompt.
Candidate: identical V5, marked RGB image, query, model and decoding limits;
           only appends a concise-answer instruction.

SHADOW-ONLY. Original 3-task Guard is source-checked but deliberately NOT
invoked on a partial-task result. Never auto-publish. No holdout access.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time
import traceback
import zipfile
from datetime import datetime

from stage1_verified_runtime import (
    WORKSPACE, REPO, V5_RAW, IMAGE_ROOT, CKPT, RUNNER, QWEN_MODEL,
    v5_preflight, verified_runner, load_guard, refuse_restricted,
    digest_file, sync, write_json, check_producer_match,
)
from joint_eval_stage2_dev60 import manifest_verified, load_all_targets

HERE = Path(__file__).resolve().parent
SELECTION = HERE / 'FIXED_SELECTION_34.json'
SELECTION_HASH = HERE / 'FIXED_SELECTION_SHA256.txt'
BASELINE_SNAPSHOT = HERE / 'VQA_FROZEN_BASELINE_171.json'
BASELINE_SHA = HERE / 'SOURCE_SNAPSHOT_SHA256.txt'
PROMPT_SUFFIX = (
    '\n\n【仅对本次回答的格式约束】请只输出目标服饰的主要可见颜色信息，不写完整句子。'
    '单色只写颜色词（如“蓝色”）；双色保留两种颜色（如“蓝色和白色”）；'
    '条纹或图案可见且影响颜色判断时，保留其颜色与图案（如“黑白条纹”）。'
    '若无法可靠判断，只写“无法确定”。'
    '不要提“红框”“被框选”“目标服饰”等图像标记措辞，不要增加未看到的颜色、材质或工艺。'
    '不要省略真实存在的第二种颜色；直接给出短答案，不要解释。'
)
MAX_NEW_TOKENS = 128  # unchanged original VQA cap; do NOT truncate output
NO_RELEASE = 'SHADOW_BLOCKED_FORMAT_AB_NO_APPROVED_GUARD'


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def percent(vals, pct=95):
    arr = sorted(float(x) for x in vals)
    if not arr:
        return None
    pos = (len(arr) - 1) * pct / 100
    lo = int(pos)
    hi = min(lo + 1, len(arr) - 1)
    return round(arr[lo] + (arr[hi] - arr[lo]) * (pos - lo), 3)


def dist(vals):
    a = [float(x) for x in vals]
    if not a:
        return None
    return {'n':len(a),'mean':round(statistics.mean(a),3),
            'median':round(statistics.median(a),3),'p95':percent(a),
            'max':round(max(a),3)}


def canonical_color_text(raw: str) -> str:
    """Heuristic formatting-only comparison, NOT ground-truth correctness."""
    x = str(raw).strip()
    x = re.sub(r'^\s*(?:红框内|被框选的|框内|图中|图片中)?目标服饰的主要可见颜色是\s*', '', x)
    x = re.sub(r'^\s*(?:红框内)?(?:目标服饰的)?主要可见颜色是\s*', '', x)
    x = re.sub(r'^\s*(?:颜色是|主要颜色是|可见颜色是)\s*', '', x)
    x = re.sub(r'[\s。．，,；;!！]+$', '', x)
    return re.sub(r'\s+', '', x)


def inspection_issues(candidate: str):
    flags = []
    if not candidate.strip():
        flags.append('EMPTY_ANSWER')
    if any(term in candidate for term in ('红框','标记框','检测框','被框选','目标服饰','bounding box')):
        flags.append('DEBUG_MARKER_TEXT')
    if any(term in candidate for term in ('棉','羊毛','真丝','涤纶','刺绣','水洗工艺')):
        flags.append('IRRELEVANT_MATERIAL_CRAFT')
    if len(candidate) > 14:
        flags.append('ANSWER_LONGER_THAN_14_CHARS')
    return flags


def load_selection():
    expected = SELECTION_HASH.read_text('ascii').strip()
    if digest_file(SELECTION) != expected:
        raise ValueError('fixed 34-case selection SHA changed')
    content = json.loads(SELECTION.read_text('utf8'))
    rows = content.get('targets', [])
    if content.get('schema') != 'prd32_vqa_concise_ab_v1_fixed_slow17_control17' or len(rows) != 34:
        raise ValueError('wrong selection schema/length')
    if collections.Counter(x['cohort'] for x in rows) != {'SLOW17':17,'CONTROL17':17}:
        raise ValueError('selection cohort counts drift')
    ids = [(r['case_id'],r['instance_id']) for r in rows]
    if len(set(ids)) != 34:
        raise ValueError('duplicate selected case')
    for x in rows:
        if sha_text(x['frozen_vqa_answer']) != x['frozen_vqa_sha256']:
            raise ValueError('reference VQA content mutated')
        if x['is_independent_gt'] is not False:
            raise ValueError('development reference falsely marked independent')
    return rows


def local_check():
    rows = load_selection()
    manifest, _ = manifest_verified()
    mm = {(x['case_id'],x['instance_id']):x for x in manifest['targets']}
    s = json.loads(BASELINE_SNAPSHOT.read_text('utf8'))
    if digest_file(BASELINE_SNAPSHOT) != BASELINE_SHA.read_text('ascii').strip():
        raise ValueError('original 171 reference SHA changed')
    if len(s.get('targets',{})) != 171:
        raise ValueError('reference not 171')
    for x in rows:
        key = (x['case_id'],x['instance_id'])
        if key not in mm or mm[key]['source_image_sha256'] != x['source_image_sha256']:
            raise ValueError('frozen target/source mismatch '+str(key))
        if s['targets'][key[0]+'/'+key[1]]['vqa_response_sha256'] != x['frozen_vqa_sha256']:
            raise ValueError('frozen baseline answer hash drift '+str(key))
    return rows,manifest


def source_preflight(args,out):
    rows,manifest = local_check()
    for p in (args.repo,args.raw_dir,args.image_root,args.checkpoint,args.runner,args.model_dir,out):
        refuse_restricted(p)
    _,source_hashes = v5_preflight(args.raw_dir,args.image_root,args.repo,args.checkpoint)
    keys = [(r['case_id'],r['instance_id']) for r in rows]
    targets = load_all_targets(args.raw_dir,args.image_root,manifest,keys)
    runner = verified_runner(args.runner)
    valid,core,helper,focus=runner.prompt_check(argparse.Namespace(
        snapshot=runner.DEFAULT_SNAP, source=runner.DEFAULT_SOURCE,
        focus_script=runner.DEFAULT_FOCUS, raw=runner.DEFAULT_RAW, input=runner.DEFAULT_INPUT))
    if len(valid)!=171:
        raise ValueError('original Qwen Baseline 171-case prompt snapshot changed')
    guard_file = args.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py'
    load_guard(guard_file)  # source audit only. DO NOT invoke original three-task Guard
    if not (args.model_dir/'config.json').is_file():
        raise FileNotFoundError('Qwen config missing: '+str(args.model_dir))
    write_json(out/'PREFLIGHT.json', {
        'status':'CONCISE_AB_PREFLIGHT_PASS','selection_sha256':digest_file(SELECTION),
        'selection_count':34,'slow_count':17,'control_count':17,
        'v5_checkpoint_sha256':digest_file(args.checkpoint),'v5_source_hashes':source_hashes,
        'qwen_runner_sha256':digest_file(args.runner),'guard_source_sha256':digest_file(guard_file),
        'frozen_qwen_reference_sha256':digest_file(BASELINE_SNAPSHOT),
        'max_new_tokens_both_arms':MAX_NEW_TOKENS,'candidate_suffix_sha256':sha_text(PROMPT_SUFFIX),
        'two_arms_one_identical_marked_image_per_target':True,
        'no_holdout':True,'original_guard_invoked':False,'production_release_allowed':False,
        'metadata_is_development_assisted_review_not_independent_blind_gt':True})
    return rows, targets, runner,core,helper,focus


def load_models(args,out):
    import torch
    import numpy as np
    from PIL import Image
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not found')
    free = torch.cuda.mem_get_info()[0] / 2**30
    if free < 18:
        raise RuntimeError(f'need at least 18 GiB free before Qwen/V5 load; found {free:.2f} GiB')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    t0=time.perf_counter()
    qwen=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir),local_files_only=True,device_map='auto',
        torch_dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(
        str(args.model_dir),local_files_only=True,max_pixels=512*28*28)
    sync(torch)
    dm=getattr(qwen,'hf_device_map',None)
    if isinstance(dm,dict) and any(str(v).lower() in {'cpu','disk'} for v in dm.values()):
        raise RuntimeError('Qwen offloaded to CPU/disk; reject timing')
    sys.path.insert(0,str(args.repo/'src'))
    from fashion_multimodal_analysis.common.schema import ID_TO_CLASS,CLASS_TO_ID
    from fashion_multimodal_analysis.segmentation.modeling import build_evaluation_model
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if cp.get('class_to_id')!=CLASS_TO_ID or cp.get('epoch')!=15 or cp.get('args',{}).get('epochs')!=15:
        raise ValueError('V5 checkpoint provenance mismatch')
    kw=cp.get('args',{})
    v5=build_evaluation_model(int(kw.get('min_size',640)),int(kw.get('max_size',1024)))
    v5.load_state_dict(cp['model_state_dict'],strict=True)
    v5.to(torch.device('cuda')).eval()
    del cp
    sync(torch)
    write_json(out/'MODELS_LOADED.json', {
        'model':'Qwen2.5-VL-7B-Instruct','qwen_dtype':str(dtype),'attention':'sdpa',
        'load_ms_excluded':round((time.perf_counter()-t0)*1000,3),
        'gpu_free_after_both_loaded_gib':round(torch.cuda.mem_get_info()[0]/2**30,3),
        'model_device_map':dm,'production_release_allowed':False})
    return torch,np,Image,qwen,processor,v5,ID_TO_CLASS


def make_review_page(results,out):
    rows=[]
    for r in results:
        ref=html.escape(r['frozen_vqa_answer'])
        a=html.escape(r['baseline']['answer'])
        b=html.escape(r['concise']['answer'])
        k=html.escape(r['case_id']+'/'+r['instance_id'])
        img=html.escape(r['marked_image_rel'])
        note=html.escape(','.join(r['candidate_flags']) or '无自动格式标志')
        rows.append(f'<article><h3>{k} <small>{r["cohort"]}</small></h3>'
            f'<img loading="lazy" src="{img}" alt="目标红框图" />'
            f'<p>冻结参考：{ref}</p><p>本次 Baseline：{a}</p><p><b>格式候选：{b}</b></p>'
            f'<p>规范化语义一致：{r["normalized_answer_matches"]}；标志：{note}</p>'
            '<label>人工判断：<select><option>待确认</option><option>语义保持</option>'
            '<option>语义退化</option><option>目标不可靠</option><option>无法判断</option></select></label>'
            '<textarea placeholder="审核备注"></textarea></article>')
    css='''body{font-family:system-ui,"Microsoft Yahei",sans-serif;max-width:1120px;margin:auto;padding:20px;background:#f6f8fb;color:#172339}article{background:white;border:1px solid #dbe2ed;border-radius:12px;padding:16px;margin:12px 0;display:grid;gap:7px}article img{width:min(100%,550px);max-height:430px;object-fit:contain}h3{margin:0}p{margin:0}small{font-size:13px;color:#66768e}select,textarea{font:inherit;padding:6px}textarea{display:block;width:90%;min-height:45px}button{padding:9px;border-radius:8px;cursor:pointer}'''
    js='''function exportReview(){let out=[['case_id','instance_id','human_review','human_note']];document.querySelectorAll('article').forEach(a=>{let s=a.querySelector('h3').textContent.trim().split(' ')[0].split('/');out.push([s[0],s[1],a.querySelector('select').value,a.querySelector('textarea').value])});let csv=out.map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\\r\\n');let blob=new Blob(['\\ufeff'+csv],{type:'text/csv;charset=utf-8'});let u=URL.createObjectURL(blob);let b=document.createElement('a');b.href=u;b.download='VQA_CONCISE_AB_HUMAN_REVIEW.csv';b.click();setTimeout(()=>URL.revokeObjectURL(u),1500)}'''
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>VQA格式A/B审核（Shadow）</title><style>'+css+'</style><body><h1>VQA-only 回答格式 A/B 审核</h1><p>34个DEV60目标；原图片目标框，所有判断待人工确认；不是独立GT，绝不发布。</p><button onclick="exportReview()">导出审核CSV</button>'+''.join(rows)+'<script>'+js+'</script></body></html>'
    (out/'REVIEW_PENDING_CONFIRMATION.html').write_text(page,encoding='utf-8')


def perform_pair(index, row, targets, runner, core, helper, focus, loaded, out, warmup=False):
    torch,np,Image,qwen,processor,v5,ID_TO_CLASS = loaded
    key=(row['case_id'],row['instance_id'])
    frozen=targets[key]
    case,inst=key
    sync(torch)
    t0=time.perf_counter()
    from fashion_multimodal_analysis.evaluation.runtime import predict_timed
    with Image.open(frozen['image_path']) as f: img=f.convert('RGB')
    boxes,labels,scores,masks,v5_internal=predict_timed(
        img,v5,torch.device('cuda'),score_threshold=.4,mask_threshold=.5)
    sync(torch)
    v5_ms=(time.perf_counter()-t0)*1000
    target_idx=int(inst.split('_')[-1])
    if target_idx>=len(labels):
        raise ValueError('V5 lost predicted target '+str(key))
    cat=ID_TO_CLASS.get(int(labels[target_idx]))
    newmask=(np.asarray(masks[target_idx])>0).astype(np.uint8)
    with Image.open(frozen['frozen_mask_path']) as im:
        oldmask=np.asarray(im.convert('L'))>127
    chk=check_producer_match(target=frozen,ins={
        'instance_id':inst,'category':cat,'score':float(scores[target_idx]),
        'bbox':[float(b) for b in boxes[target_idx]]},
        reference_mask=oldmask,produced_mask=newmask)
    if chk['mask_iou']<.9999:
        raise ValueError('new V5 output differs from frozen baseline')
    qrec={'image_path':frozen['image_path'],'image_sha256':frozen['image_sha256'],
          'target_bbox_xyxy':[float(b) for b in boxes[target_idx]],
          'target_instance_id':inst,'task':'vqa','query':runner.QUESTION,
          'length_chars':[20,80],'predicted_attribute_context':[]}
    markfolder=out/'_working_marked'/f'{"warmup__" if warmup else "measured__"}{case}__{inst}'
    markfolder.mkdir(parents=True,exist_ok=False)
    t_mark=time.perf_counter()
    marked=helper.make_marked_image(qrec,markfolder,'marked_full_image',.05)
    mark_ms=(time.perf_counter()-t_mark)*1000
    marked=Path(marked).resolve(strict=True)
    if not marked.is_relative_to(markfolder.resolve(strict=True)):
        raise ValueError('marked image escaped output folder')
    marked_sha=digest_file(marked)
    review_rel=f'marked_inputs/{case}__{inst}.jpg'
    if not warmup:
        (out/'marked_inputs').mkdir(exist_ok=True)
        with Image.open(marked) as f:
            f.convert('RGB').save(out/review_rel,quality=88,optimize=True)
    base_prompt=core.build_prompt(helper,focus,qrec,'vqa','baseline')
    if sha_text(base_prompt)!=runner.PROMPT_SHA['baseline']['vqa']:
        raise ValueError('original baseline prompt changed')
    concise_prompt=base_prompt + PROMPT_SUFFIX
    prompts={'baseline':base_prompt,'concise':concise_prompt}
    # Alternate order to reduce one-direction warm/caching bias.
    order=('baseline','concise') if index%2==0 else ('concise','baseline')
    arms={}
    for arm in order:
        sync(torch)
        t_run=time.perf_counter()
        raw,timing=helper.run_model_case(qwen,processor,qrec,marked,prompts[arm],
                                          task_limits={'vqa':MAX_NEW_TOKENS})
        sync(torch)
        wall=(time.perf_counter()-t_run)*1000
        arms[arm]={'answer':raw,'answer_sha256':sha_text(raw),
            'prompt_sha256':sha_text(prompts[arm]),
            'marked_input_sha256':marked_sha,'host_qwen_ms':round(wall,3),
            'model_preprocess_ms':timing['preprocess_ms'],
            'model_generation_ms':timing['generation_ms'],
            'generated_tokens':timing['generated_tokens'],
            'v5_plus_mark_plus_qwen_ms':round(v5_ms+mark_ms+wall,3),
            'publish_allowed':False,'approved_guard_invoked':False}
    if arms['baseline']['answer_sha256']!=row['frozen_vqa_sha256']:
        raise ValueError('original baseline VQA output drift: '+case+'/'+inst)
    if arms['baseline']['marked_input_sha256']!=arms['concise']['marked_input_sha256']:
        raise ValueError('different images used in paired comparison')
    if warmup:
        return None
    flags=inspection_issues(arms['concise']['answer'])
    norm_eq=(canonical_color_text(arms['baseline']['answer'])==canonical_color_text(arms['concise']['answer']))
    if not norm_eq:
        flags.append('ANSWER_SEMANTICS_DIFFER_REQUIRE_REVIEW')
    result={'case_id':case,'instance_id':inst,'cohort':row['cohort'],
        'v5_category':cat,'developer_review_identity':row['developer_review_identity'],
        'developer_review_mask':row['developer_review_mask'],
        'developer_review_vqa':row['developer_review_vqa'],
        'source_image_sha256':frozen['image_sha256'],'marked_image_sha256':marked_sha,
        'marked_image_rel':review_rel,'v5_host_ms':round(v5_ms,3),
        'mark_host_ms':round(mark_ms,3),'frozen_mask_iou':chk['mask_iou'],
        'frozen_vqa_answer':row['frozen_vqa_answer'],'previous_vqa_e2e_ms':row['frozen_vqa_e2e_ms'],
        'order':list(order),'baseline':arms['baseline'],'concise':arms['concise'],
        'normalized_answer_matches':norm_eq,'candidate_flags':flags,
        'human_verified_new_candidate':False,'independent_gt':False,
        'original_guard_invoked':False,'production_release':False,'status':NO_RELEASE}
    return result


def summarize(results):
    out={'status':'COMPLETE_SHADOW_ONLY','schema':'prd32_vqa_concise_ab_v1',
         'selected_targets':len(results),'generation_limit_both_arms':MAX_NEW_TOKENS,
         'two_arms_same_image_and_weights':True,
         'same_sitting_paired_comparison':True,
         'original_guard_invoked':False,'publish_allowed':False,
         'quality_comparisons_not_independent_gt':True,
         'prompt_suffix_sha256':sha_text(PROMPT_SUFFIX),'cohorts':{}}
    for cohort in ['SLOW17','CONTROL17','ALL34']:
        subset=results if cohort=='ALL34' else [r for r in results if r['cohort']==cohort]
        b=[r['baseline']['v5_plus_mark_plus_qwen_ms'] for r in subset]
        c=[r['concise']['v5_plus_mark_plus_qwen_ms'] for r in subset]
        out['cohorts'][cohort]={
            'n':len(subset),'baseline_e2e_ms':dist(b),'candidate_e2e_ms':dist(c),
            'paired_mean_speedup_pct':round(100*(1-statistics.mean(c)/statistics.mean(b)),3),
            'baseline_under_400ms':sum(v<=400 for v in b),
            'candidate_under_400ms':sum(v<=400 for v in c),
            'baseline_host_qwen_ms':dist([r['baseline']['host_qwen_ms'] for r in subset]),
            'candidate_host_qwen_ms':dist([r['concise']['host_qwen_ms'] for r in subset]),
            'baseline_generated_tokens':dist([r['baseline']['generated_tokens'] for r in subset]),
            'candidate_generated_tokens':dist([r['concise']['generated_tokens'] for r in subset]),
            'candidate_normalized_matches_baseline':sum(r['normalized_answer_matches'] for r in subset),
            'candidate_flagged_for_review':sum(bool(r['candidate_flags']) for r in subset),
        }
    out['decision']='NO_AUTO_ADOPTION_REVIEW_RESULTS_AND_DEV60_REGRESSION_FIRST'
    out['notes']=[
        'Strict matched A/B: same V5 image, bbox, mark, Qwen weights, processor, decoding max_new_tokens; only prompt appended.',
        'E2E here = shared V5 + image marking + arm Qwen. No Guard, service/network/queue overhead.',
        'Comparison includes selected 17 >400ms cases and 17 ordinary short-answer controls: intentional stratified selection, not full DEV60 distribution.',
        'Canonical normalized answer match is only an automated string check, not semantic correctness; all candidate answers require human visual review.',
        'The original 3-task Guard was not called; all candidates are Shadow-only. No independent Holdout used.'
    ]
    return out


def write_rows(results,out):
    fields=['case_id','instance_id','cohort','v5_category','developer_review_identity',
            'developer_review_mask','developer_review_vqa','frozen_vqa_answer',
            'baseline_answer','candidate_answer','baseline_generated_tokens','candidate_generated_tokens',
            'baseline_qwen_ms','candidate_qwen_ms','baseline_e2e_ms','candidate_e2e_ms',
            'normalized_answer_matches','candidate_flags','human_review','review_note']
    with (out/'REVIEW_PENDING_USER_CONFIRMATION.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in results:
            w.writerow({'case_id':r['case_id'],'instance_id':r['instance_id'],'cohort':r['cohort'],
                'v5_category':r['v5_category'],'developer_review_identity':r['developer_review_identity'],
                'developer_review_mask':r['developer_review_mask'],'developer_review_vqa':r['developer_review_vqa'],
                'frozen_vqa_answer':r['frozen_vqa_answer'],
                'baseline_answer':r['baseline']['answer'],'candidate_answer':r['concise']['answer'],
                'baseline_generated_tokens':r['baseline']['generated_tokens'],
                'candidate_generated_tokens':r['concise']['generated_tokens'],
                'baseline_qwen_ms':r['baseline']['host_qwen_ms'],
                'candidate_qwen_ms':r['concise']['host_qwen_ms'],
                'baseline_e2e_ms':r['baseline']['v5_plus_mark_plus_qwen_ms'],
                'candidate_e2e_ms':r['concise']['v5_plus_mark_plus_qwen_ms'],
                'normalized_answer_matches':r['normalized_answer_matches'],
                'candidate_flags':';'.join(r['candidate_flags']),
                'human_review':'待确认','review_note':''})


def make_output_archive(out):
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and '_working_marked' not in p.parts:
                z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('archive integrity test failed')
    print('UPLOAD_THIS:',archive,flush=True)
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['package-check','preflight','live'],default='package-check')
    p.add_argument('--execute-models',action='store_true')
    p.add_argument('--out',type=Path)
    p.add_argument('--repo',type=Path,default=REPO)
    p.add_argument('--raw-dir',type=Path,default=V5_RAW)
    p.add_argument('--image-root',type=Path,default=IMAGE_ROOT)
    p.add_argument('--checkpoint',type=Path,default=CKPT)
    p.add_argument('--runner',type=Path,default=RUNNER)
    p.add_argument('--model-dir',type=Path,default=QWEN_MODEL)
    args=p.parse_args()
    if args.mode=='package-check':
        rows,_=local_check()
        print('CONCISE_AB_PACKAGE_CHECK_PASS',len(rows),'fixed cases','NO_GPU');return 0
    out=args.out or WORKSPACE/f'PRD32_VQA_CONCISE_AB_V1_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
    if out.exists():
        raise SystemExit('refuse overwrite existing output '+str(out))
    out.mkdir(parents=True)
    status=0
    try:
        rows,targets,runner,core,helper,focus=source_preflight(args,out)
        if args.mode=='preflight':
            print('CONCISE_AB_PREFLIGHT_PASS',len(rows),'DEV60 ONLY NO_GPU',flush=True)
            return 0
        if not args.execute_models:
            raise ValueError('GPU run requires --execute-models')
        loaded=load_models(args,out)
        # One warmup target: NOT included in measured 34. Both arms warm up.
        warm=next(x for x in rows if x['cohort']=='CONTROL17')
        perform_pair(0,warm,targets,runner,core,helper,focus,loaded,out,warmup=True)
        results=[]
        for i,row in enumerate(rows):
            r=perform_pair(i,row,targets,runner,core,helper,focus,loaded,out)
            results.append(r)
            with (out/'PER_CASE_RESULTS.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
            print('CONCISE_AB_PAIR',i+1,'/',len(rows),r['case_id'],r['instance_id'],
                'baseline_ms',r['baseline']['v5_plus_mark_plus_qwen_ms'],
                'candidate_ms',r['concise']['v5_plus_mark_plus_qwen_ms'],
                'candidate_tokens',r['concise']['generated_tokens'],
                'normalized_match',r['normalized_answer_matches'],'RELEASE_FALSE',flush=True)
        if len(results)!=34 or len({r['case_id']+'/'+r['instance_id'] for r in results})!=34:
            raise ValueError('run incomplete or duplicated')
        if any(r['status']!=NO_RELEASE or r['production_release'] is not False or r['original_guard_invoked'] is not False for r in results):
            raise ValueError('Shadow safety boundary violated')
        write_json(out/'AB_SUMMARY.json',summarize(results))
        write_rows(results,out)
        make_review_page(results,out)
        (out/'README_RESULT.txt').write_text(
            '34 DEV60 targets matched baseline vs concise color-VQA formatting A/B.\n'
            'Not independent GT. Every candidate under Shadow BLOCK.\n'
            'A/B time: shared V5 + mark + one Qwen VQA call, excluding Guard/service.\n'
            'Review_pending_user_confirmation.csv and REVIEW_PENDING_CONFIRMATION.html require manual check.\n',encoding='utf8')
        print('VQA_CONCISE_AB_V1_COMPLETE_SHADOW_ONLY',flush=True)
    except BaseException as exc:
        status=2
        write_json(out/'FAILED_CLOSED.json',{
            'status':'FAILED_CLOSED_NO_RELEASE','type':type(exc).__name__,
            'error':str(exc),'traceback':traceback.format_exc()[-16000:],
            'production_release':False,'holdout_used':False})
        print('VQA_CONCISE_AB_V1_FAILED_CLOSED',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
    finally:
        make_output_archive(out)
    return status

if __name__=='__main__':
    sys.exit(main())
