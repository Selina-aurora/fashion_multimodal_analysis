#!/usr/bin/env python3
"""Matched 3-arm Qwen VQA answer-format experiment on fixed DEV60 slow17 + control17.

Three arms: original frozen prompt, V1 concise prompt, V2 evidence-first concise prompt.
Same V5, same marked image, Qwen weights and decoding cap; only prompt differs.

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
# V2 is a NEW candidate: semantic guardrails before brevity, not forced guessing.
PROMPT_SUFFIX_V2 = (
    '\n\n【本次仅调整回答格式；安全与真实性优先】'
    '问题仍是：只判断当前红框所指的同一件服饰的主要可见颜色。'
    '先检查目标是否可辨、是否仅有背景或多件衣物混入；'
    '如果目标身份、可见范围或颜色不足以可靠确认，直接回答“无法确定”，绝对不要猜颜色。'
    '不要把旁边衣物、皮肤、背景或红框本身的颜色写成目标颜色。'
    '若可靠可见，答案只写颜色词：保持原本可辨的深浅色（浅蓝色不得省成蓝色）；'
    '两种及以上主色要分别保留，不得把双色省成单色。'
    '题目只问颜色，避免把“条纹”“格纹”等图案名称替代颜色词；'
    '例如黑白条纹答“黑色和白色”，橙白条纹答“橙色和白色”。'
    '单色示例“黑色”，双色示例“浅蓝色和白色”，证据不足示例“无法确定”。'
    '不写“红框”“目标服饰”“图片中”等长句；不写材料、工艺、设计。'
    '即使要求简短，也不能改变拒答为猜测；只输出答案。'
)
V1_GPU_REFERENCE = HERE / 'PREVIOUS_V1_GPU_34.json'
V1_GPU_REFERENCE_SHA = HERE / 'PREVIOUS_V1_GPU_34_SHA256.txt'
V1_USER_CONFIRMED_REVIEW = HERE / 'V1_34_REVIEW_USER_CONFIRMED_ASSISTED.csv'

NO_RELEASE = 'SHADOW_BLOCKED_V2_THREE_ARM_NO_APPROVED_GUARD'


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
    refs=json.loads(V1_GPU_REFERENCE.read_text(encoding='utf8'))
    if digest_file(V1_GPU_REFERENCE) != V1_GPU_REFERENCE_SHA.read_text('ascii').strip():
        raise ValueError('V1 GPU answer reference SHA changed')
    if len(refs)!=34 or set(refs)!={x['case_id']+'/'+x['instance_id'] for x in rows}:
        raise ValueError('previous V1 34 reference mismatch')
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
        'status':'CONCISE_V2_PREFLIGHT_PASS','selection_sha256':digest_file(SELECTION),
        'selection_count':34,'slow_count':17,'control_count':17,
        'v5_checkpoint_sha256':digest_file(args.checkpoint),'v5_source_hashes':source_hashes,
        'qwen_runner_sha256':digest_file(args.runner),'guard_source_sha256':digest_file(guard_file),
        'frozen_qwen_reference_sha256':digest_file(BASELINE_SNAPSHOT),
        'max_new_tokens_both_arms':MAX_NEW_TOKENS,'v1_suffix_sha256':sha_text(PROMPT_SUFFIX),'v2_suffix_sha256':sha_text(PROMPT_SUFFIX_V2),
        'previous_v1_gpu_sha256':digest_file(V1_GPU_REFERENCE),
        'three_arms_one_identical_marked_image_per_target':True,
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
    cards=[]
    for r in results:
        k=html.escape(r['case_id']+'/'+r['instance_id'])
        b=html.escape(r['baseline']['answer']); v1=html.escape(r['concise']['answer'])
        v2=html.escape(r['guarded_v2']['answer'])
        notes=html.escape(', '.join(r['candidate_flags']) or '无自动风险标志')
        previous=html.escape(r['previous_user_confirmed_v1_review'])
        img=html.escape(r['marked_image_rel'])
        cards.append(f'<article data-key="{k}"><h3>{k}（{r["cohort"]}）</h3>'
          f'<img src="{img}" loading="lazy" alt="目标图"><p>原始 Baseline：{b}</p>'
          f'<p>V1 简短：{v1} · 既往用户确认意见：{previous}</p>'
          f'<p><strong>V2 约束：{v2}</strong></p><p>自动提示：{notes}</p>'
          '<label>V2人工审核：<select><option>待确认</option><option>准确且保留细节</option>'
          '<option>答案与目标不符</option><option>颜色细节丢失</option>'
          '<option>拒答转猜测</option><option>目标证据不足</option><option>其他</option></select></label>'
          '<textarea placeholder="可填写判断依据"></textarea></article>')
    css="""body{font-family:system-ui,'Microsoft Yahei',sans-serif;max-width:1080px;padding:22px;margin:auto;background:#f6f7fb;color:#182436}article{background:#fff;border:1px solid #dde2eb;border-radius:12px;padding:16px;margin:14px 0;display:grid;gap:8px}img{max-width:540px;width:100%;max-height:460px;object-fit:contain}p,h3{margin:0}select,textarea{font:inherit;padding:8px;border:1px solid #ccd4e2;border-radius:6px}textarea{min-height:45px;width:94%}button{padding:10px;border-radius:8px;background:#1f4d83;color:white;border:0;cursor:pointer}"""
    js="""function exportReview(){let out=[['case_id','instance_id','v2_human_review','v2_human_note']];document.querySelectorAll('article').forEach(a=>{let s=a.dataset.key.split('/');out.push([s[0],s[1],a.querySelector('select').value,a.querySelector('textarea').value])});let csv=out.map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n');let b=new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8'});let u=URL.createObjectURL(b);let a=document.createElement('a');a.href=u;a.download='VQA_CONCISE_AB_V2_HUMAN_REVIEW.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}"""
    page='<html lang="zh-CN"><head><meta charset="utf-8"><title>VQA V2受控实验</title><style>'+css+'</style></head><body><h1>VQA 原版 / V1 / V2 同场对照</h1><p>34条开发集目标，全部 Shadow 阻断，AI对照不等于人工真值；审核后导出CSV。</p><button onclick="exportReview()">导出审核CSV</button>'+''.join(cards)+'<script>'+js+'</script></body></html>'
    (out/'V2_REVIEW_PENDING_CONFIRMATION.html').write_text(page,encoding='utf-8')


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
    v2_prompt=base_prompt + PROMPT_SUFFIX_V2
    prompts={'baseline':base_prompt,'concise':concise_prompt,'guarded_v2':v2_prompt}
    # Rotate arm order to reduce systematic warming/position bias.
    permutations=[('baseline','concise','guarded_v2'),
                  ('concise','guarded_v2','baseline'),
                  ('guarded_v2','baseline','concise')]
    order=permutations[index%3]
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
    if len({v['marked_input_sha256'] for v in arms.values()}) != 1:
        raise ValueError('different images used in paired comparison')
    previous=json.loads(V1_GPU_REFERENCE.read_text(encoding='utf8'))[case+'/'+inst]
    if arms['concise']['answer_sha256'] != previous['v1_concise_answer_sha256']:
        raise ValueError('V1 concise answer changed since prior GPU experiment: '+case+'/'+inst)
    if warmup:
        return None
    flags=inspection_issues(arms['guarded_v2']['answer'])
    norm_eq=(canonical_color_text(arms['baseline']['answer'])==canonical_color_text(arms['guarded_v2']['answer']))
    if not norm_eq:
        flags.append('V2_DIFFERS_FROM_BASELINE_REVIEW')
    norm_eq_v1=(canonical_color_text(arms['concise']['answer'])==canonical_color_text(arms['guarded_v2']['answer']))
    if not norm_eq_v1:
        flags.append('V2_DIFFERS_FROM_V1_REVIEW')
    if '无法确定' in arms['baseline']['answer'] and '无法确定' not in arms['guarded_v2']['answer']:
        flags.append('BASELINE_REFUSAL_BECAME_DEFINITE_HIGH_RISK')
    if '无法确定' in arms['concise']['answer'] and '无法确定' not in arms['guarded_v2']['answer']:
        flags.append('V1_REFUSAL_BECAME_DEFINITE_HIGH_RISK')
    if '浅' in arms['baseline']['answer'] and '浅' not in arms['guarded_v2']['answer']:
        flags.append('COLOR_SHADE_LOST_REVIEW')
    if '深' in arms['baseline']['answer'] and '深' not in arms['guarded_v2']['answer']:
        flags.append('COLOR_SHADE_LOST_REVIEW')
    result={'case_id':case,'instance_id':inst,'cohort':row['cohort'],
        'v5_category':cat,'developer_review_identity':row['developer_review_identity'],
        'developer_review_mask':row['developer_review_mask'],
        'developer_review_vqa':row['developer_review_vqa'],
        'source_image_sha256':frozen['image_sha256'],'marked_image_sha256':marked_sha,
        'marked_image_rel':review_rel,'v5_host_ms':round(v5_ms,3),
        'mark_host_ms':round(mark_ms,3),'frozen_mask_iou':chk['mask_iou'],
        'frozen_vqa_answer':row['frozen_vqa_answer'],'previous_vqa_e2e_ms':row['frozen_vqa_e2e_ms'],
        'order':list(order),'baseline':arms['baseline'],'concise':arms['concise'],
        'guarded_v2':arms['guarded_v2'],
        'normalized_answer_matches':norm_eq,'v2_matches_v1':norm_eq_v1,
        'previous_user_confirmed_v1_review':previous['user_confirmed_review'],
        'candidate_flags':flags,
        'human_verified_new_candidate':False,'independent_gt':False,
        'original_guard_invoked':False,'production_release':False,'status':NO_RELEASE}
    return result


def summarize(results):
    out={'schema':'prd32_vqa_concise_ab_v2_three_arm',
         'status':'COMPLETE_SHADOW_ONLY','selected_targets':len(results),
         'original_guard_invoked':False,'publish_allowed':False,
         'independent_gt':False,'model_weights_unchanged':True,
         'all_arms_max_new_tokens':MAX_NEW_TOKENS,
         'v1_suffix_sha256':sha_text(PROMPT_SUFFIX),
         'v2_suffix_sha256':sha_text(PROMPT_SUFFIX_V2),'cohorts':{}}
    for cohort in ['SLOW17','CONTROL17','ALL34']:
        subset=results if cohort=='ALL34' else [r for r in results if r['cohort']==cohort]
        v1=[r['concise']['v5_plus_mark_plus_qwen_ms'] for r in subset]
        v2=[r['guarded_v2']['v5_plus_mark_plus_qwen_ms'] for r in subset]
        base=[r['baseline']['v5_plus_mark_plus_qwen_ms'] for r in subset]
        out['cohorts'][cohort]={
          'n':len(subset),
          'baseline_e2e_ms':dist(base),'v1_e2e_ms':dist(v1),'v2_e2e_ms':dist(v2),
          'v2_vs_v1_mean_latency_change_pct':round((statistics.mean(v2)/statistics.mean(v1)-1)*100,3),
          'v2_vs_baseline_mean_reduction_pct':round((1-statistics.mean(v2)/statistics.mean(base))*100,3),
          'baseline_under_400ms':sum(t<=400 for t in base),
          'v1_under_400ms':sum(t<=400 for t in v1),
          'v2_under_400ms':sum(t<=400 for t in v2),
          'v1_generated_tokens':dist([r['concise']['generated_tokens'] for r in subset]),
          'v2_generated_tokens':dist([r['guarded_v2']['generated_tokens'] for r in subset]),
          'v2_normalized_matches_baseline':sum(r['normalized_answer_matches'] for r in subset),
          'v2_normalized_matches_v1':sum(r['v2_matches_v1'] for r in subset),
          'v2_flagged_review_count':sum(bool(r['candidate_flags']) for r in subset),
          'v2_baseline_refusal_became_definite':sum('BASELINE_REFUSAL_BECAME_DEFINITE_HIGH_RISK' in r['candidate_flags'] for r in subset),
          'v2_v1_refusal_became_definite':sum('V1_REFUSAL_BECAME_DEFINITE_HIGH_RISK' in r['candidate_flags'] for r in subset),
        }
    out['decision']='DO_NOT_MERGE_OR_PUBLISH_AWAIT_HUMAN_REVIEW_AND_FULL_DEV60_REGRESSION'
    out['notes']=[
       'All three arms use the same marked image; only the VQA prompt differs.',
       'Baseline and V1 output hashes must reproduce frozen previous GPU results.',
       'No actual standalone VQA guard approval; all remain shadow-blocked.',
       'Selective 34-case DEV60 sample is intentionally enriched for slow cases; not representative of full DEV60.',
       'V2 quality assessment needs review; baseline text or previous chat-approved review is not independent blind GT.',
       'Reported latency = shared V5 + marked image + one Qwen call; excludes service, network, routing and Guard.'
    ]
    return out


def write_rows(results,out):
    fields=['case_id','instance_id','cohort','v5_category','identity_reference','mask_reference',
      'v1_user_confirmed_review','original_answer','v1_answer','v2_answer',
      'baseline_tokens','v1_tokens','v2_tokens',
      'baseline_e2e_ms','v1_e2e_ms','v2_e2e_ms',
      'v2_matches_baseline_text','v2_matches_v1_text','auto_flags','review_state','human_review','human_note']
    with (out/'V2_REVIEW_PENDING_USER_CONFIRMATION.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in results:
            w.writerow(dict(case_id=r['case_id'],instance_id=r['instance_id'],cohort=r['cohort'],
              v5_category=r['v5_category'],identity_reference=r['developer_review_identity'],
              mask_reference=r['developer_review_mask'],v1_user_confirmed_review=r['previous_user_confirmed_v1_review'],
              original_answer=r['baseline']['answer'],v1_answer=r['concise']['answer'],
              v2_answer=r['guarded_v2']['answer'],baseline_tokens=r['baseline']['generated_tokens'],
              v1_tokens=r['concise']['generated_tokens'],v2_tokens=r['guarded_v2']['generated_tokens'],
              baseline_e2e_ms=r['baseline']['v5_plus_mark_plus_qwen_ms'],
              v1_e2e_ms=r['concise']['v5_plus_mark_plus_qwen_ms'],
              v2_e2e_ms=r['guarded_v2']['v5_plus_mark_plus_qwen_ms'],
              v2_matches_baseline_text=r['normalized_answer_matches'],
              v2_matches_v1_text=r['v2_matches_v1'],auto_flags=';'.join(r['candidate_flags']),
              review_state='AI_OUTPUT_UNREVIEWED',human_review='待确认',human_note=''))


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
        print('CONCISE_V2_PACKAGE_CHECK_PASS',len(rows),'fixed cases','NO_GPU');return 0
    out=args.out or WORKSPACE/f'PRD32_VQA_CONCISE_AB_V2_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}_UPLOAD_THIS'
    if out.exists():
        raise SystemExit('refuse overwrite existing output '+str(out))
    out.mkdir(parents=True)
    status=0
    try:
        rows,targets,runner,core,helper,focus=source_preflight(args,out)
        if args.mode=='preflight':
            print('CONCISE_V2_PREFLIGHT_PASS',len(rows),'DEV60 ONLY NO_GPU',flush=True)
            return 0
        if not args.execute_models:
            raise ValueError('GPU run requires --execute-models')
        loaded=load_models(args,out)
        # One warmup target: NOT included in measured 34. All three arms warm up.
        warm=next(x for x in rows if x['cohort']=='CONTROL17')
        perform_pair(0,warm,targets,runner,core,helper,focus,loaded,out,warmup=True)
        results=[]
        for i,row in enumerate(rows):
            r=perform_pair(i,row,targets,runner,core,helper,focus,loaded,out)
            results.append(r)
            with (out/'PER_CASE_RESULTS.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
            print('CONCISE_V2_TRIPLE',i+1,'/',len(rows),r['case_id'],r['instance_id'],
                'v1_ms',r['concise']['v5_plus_mark_plus_qwen_ms'],
                'v2_ms',r['guarded_v2']['v5_plus_mark_plus_qwen_ms'],
                'v2_tokens',r['guarded_v2']['generated_tokens'],
                'normalized_match',r['normalized_answer_matches'],'RELEASE_FALSE',flush=True)
        if len(results)!=34 or len({r['case_id']+'/'+r['instance_id'] for r in results})!=34:
            raise ValueError('run incomplete or duplicated')
        if any(r['status']!=NO_RELEASE or r['production_release'] is not False or r['original_guard_invoked'] is not False for r in results):
            raise ValueError('Shadow safety boundary violated')
        write_json(out/'AB_SUMMARY.json',summarize(results))
        write_rows(results,out)
        make_review_page(results,out)
        (out/'README_RESULT.txt').write_text(
            '34 DEV60 targets matched original vs concise V1 vs grounded concise V2.\n'
            '102 formally measured Qwen calls plus 3 warmup. No independent GT.\n'
            'All outputs Shadow BLOCK; no Guard execution and no publish.\n'
            'Latency: V5+mark+one Qwen call; excludes Guard/service.\n'
            'V2_REVIEW_PENDING_USER_CONFIRMATION.csv and V2_REVIEW_PENDING_CONFIRMATION.html require review.\n',encoding='utf8')
        print('VQA_CONCISE_AB_V2_COMPLETE_SHADOW_ONLY',flush=True)
    except BaseException as exc:
        status=2
        write_json(out/'FAILED_CLOSED.json',{
            'status':'FAILED_CLOSED_NO_RELEASE','type':type(exc).__name__,
            'error':str(exc),'traceback':traceback.format_exc()[-16000:],
            'production_release':False,'holdout_used':False})
        print('VQA_CONCISE_AB_V2_FAILED_CLOSED',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
    finally:
        make_output_archive(out)
    return status

if __name__=='__main__':
    sys.exit(main())
