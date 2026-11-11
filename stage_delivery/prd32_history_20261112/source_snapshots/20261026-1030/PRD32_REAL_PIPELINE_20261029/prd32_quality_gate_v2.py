#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD 3.2 quality gate v2 for frozen-3.1 offline replay.

Read only: analyzes already-produced Qwen predictions and service outputs.
No inference, no 3.1 changes, no sealed holdout, no GT inference.
Builds a provenance-linked v2 *draft* contract, semantic flags, manual QA/copy
review dashboard with optional embedded image files, and reproducibility report.
A candidate contract is NOT a verified answer, merchandising fact or PRD acceptance.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ALLOWED_TASKS = {'vqa', 'tags', 'copy'}
ALLOWED_CLASS = {'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}
BAD_PATTERN = {'无','没有','无图案','无明显图案','none','unknown','无法确定','不确定','null','蕾丝'}
CANONICAL_PATTERN = {'纯色','条纹','格纹','人字纹','波点','花卉','花卉印花','文字印花','字母印花','印花','几何','动物纹','拼色'}
BAD_DESIGN = {'无','没有','无设计','无明显设计','无特殊设计','无装饰','none','unknown','无法确定','不确定','null'}
MATERIAL_CLAIMS = ('蕾丝','真丝','纯棉','羊毛成分','涤纶','皮革材质','亚麻材质','丝绸材质')
DEBUG_WORDS = ('红框','红色框','框内','框中','标注框')
REQUIRED_PRD = {'qa_accuracy_min':0.85,'qa_generation_ms_max':200,'tag_accuracy_min':0.90,
                'copy_satisfaction_min':0.85,'single_product_generation_ms_max':100}


def rows_jsonl(path):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(p)
    out=[]
    for lineno,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        try: r=json.loads(line)
        except Exception as e: raise ValueError(f'{p}:{lineno}: JSON error: {e}') from e
        if not isinstance(r,dict): raise ValueError(f'{p}:{lineno}: expected object')
        out.append(r)
    return out


def unique_map(rows, file_label):
    ans={}
    for row in rows:
        cid=row.get('case_id')
        if not isinstance(cid,str) or not cid: raise ValueError(f'{file_label}: missing case_id')
        if cid in ans: raise ValueError(f'{file_label}: duplicate case_id {cid}')
        ans[cid]=row
    return ans


def image_data_uri(pathstr):
    if not isinstance(pathstr,str) or not pathstr: return None
    path=Path(pathstr)
    if any(x in str(path).lower() for x in ('sealed_do_not_open','holdout')): return None
    if not path.is_file(): return None
    try:
        from PIL import Image
        with Image.open(path) as src:
            im=src.convert('RGB')
        im.thumbnail((750,850))
        buff=io.BytesIO(); im.save(buff,format='JPEG',quality=75,optimize=True)
        return 'data:image/jpeg;base64,' + base64.b64encode(buff.getvalue()).decode('ascii')
    except (OSError,ValueError,ImportError):
        return None


def normalized_tag(service, prediction):
    """Strict type + conservative semantic validation; never promote to human truth."""
    flags=[]; errors=[]
    obj=service.get('canonical_attributes_v2')
    if not isinstance(obj,dict):
        obj={}; errors.append('canonical_attributes_missing')
    cats=obj.get('category')
    if cats is not None and (not isinstance(cats,str) or cats not in ALLOWED_CLASS):
        errors.append('category_not_eight_class_enum'); cats=None
    col=obj.get('primary_color')
    if col is not None and (not isinstance(col,str) or not col.strip()):
        errors.append('primary_color_wrong_type'); col=None
    pat=obj.get('pattern')
    if pat is not None and not isinstance(pat,str):
        errors.append('pattern_wrong_type'); pat=None
    if isinstance(pat,str):
        p=pat.strip()
        if p.lower() in BAD_PATTERN:
            flags.append('pattern_nonstandard_abstained'); pat=None
        elif p not in CANONICAL_PATTERN:
            flags.append('pattern_not_in_reviewed_ontology_abstained'); pat=None
        else: pat=p
    design=obj.get('visible_design')
    if design is not None and not isinstance(design,str):
        errors.append('design_wrong_type'); design=None
    if isinstance(design,str) and design.strip().lower() in BAD_DESIGN:
        flags.append('design_nonstructural_abstained'); design=None
    if design is not None: flags.append('design_semantics_not_human_verified')
    # The bridge decoration field has no verified signal from V2 model; abstain.
    if obj.get('visible_design') and isinstance(service.get('payload'),dict) and service['payload'].get('visible_decoration') is not None:
        flags.append('decoration_candidate_requires_review')
    unc=[k for k,v in {'category':cats,'primary_color':col,'pattern':pat,'visible_design':design,'visible_decoration':None}.items() if v is None]
    if not prediction.get('raw_native_json',False): flags.append('model_raw_not_native_json')
    return {'category':cats,'primary_color':col,'pattern':pat,'visible_design':design,
            'visible_decoration':None,'unconfirmed':unc},flags,errors


def review_flags_for_text(task,text):
    flags=[]
    if any(w in text for w in DEBUG_WORDS): flags.append('debug_bbox_language')
    if task=='copy':
        if any(w in text for w in MATERIAL_CLAIMS): flags.append('fabric_or_material_claim_needs_evidence')
        if len(text)>80: flags.append('copy_exceeds_80_zh_chars')
        if '适合日常' in text or '百搭' in text or '优雅' in text: flags.append('subjective_claim_needs_human_review')
    return flags


def run(pred_path, service_path, output_dir, *, dry_run=False):
    predictions=rows_jsonl(pred_path); services=rows_jsonl(service_path)
    pm=unique_map(predictions,'predictions'); sm=unique_map(services,'service')
    if set(pm)!=set(sm): raise ValueError(f'case IDs differ: predictions_only={sorted(set(pm)-set(sm))}, service_only={sorted(set(sm)-set(pm))}')
    assert len(pm)>0
    if dry_run:
        print(json.dumps({'status':'DRY_RUN_OK','case_count':len(pm),'task_counts':dict(Counter(x.get('task') for x in predictions)),
                          'provenance_complete':all(p.get('image_id') and p.get('target_instance_id') for p in predictions)},ensure_ascii=False,indent=2));return
    out=Path(output_dir)
    if out.exists(): raise FileExistsError('Will not overwrite existing output: '+str(out))
    out.mkdir(parents=True,exist_ok=False)
    audit=[]; contracts=[]; review=[]; source_images={}
    tag_candidates={}; structured_bad=0
    for prediction in predictions:
        cid=prediction['case_id']; service=sm[cid]; task=prediction.get('task'); errors=[]; flags=[]
        if task not in ALLOWED_TASKS or service.get('task')!=task: errors.append('task_mismatch')
        if prediction.get('bbox_origin')!='prd31_frozen_prediction': errors.append('bbox_not_from_frozen_prediction')
        image_id=prediction.get('image_id');instance_id=prediction.get('target_instance_id')
        if not isinstance(image_id,str) or not isinstance(instance_id,str) or not image_id or not instance_id:
            errors.append('missing_target_identity')
        if task=='tags':
            candidate, tagflags, tagerrors=normalized_tag(service,prediction)
            flags+=tagflags; errors+=tagerrors
            tag_candidates[(image_id,instance_id)]=candidate
            payload=candidate
        else:
            payload=service.get('payload')
            if not isinstance(payload,str) or not payload.strip():
                errors.append('empty_or_nontext_response');payload=''
            flags+=review_flags_for_text(task,payload)
            if task=='vqa' and payload.replace('。','').strip() in ('无法确定','看不清','不确定'):
                flags.append('abstention_requires_human_grading')
        flags.append('content_not_human_verified')
        record={'schema':'prd32_service_review_contract_v2','case_id':cid,'task':task,
                'target':{'image_id':image_id,'target_instance_id':instance_id,'bbox_origin':prediction.get('bbox_origin'),
                          'input_mode':prediction.get('input_mode')},
                'status':'DRAFT_REVIEW_REQUIRED' if not errors else 'REJECTED_CONTRACT_INVALID',
                'payload':payload,'human_verified':False,'publishable':False,'source_3_1_attributes_are_unverified':True,
                'raw_model_response_sha256':hashlib.sha256(str(prediction.get('raw_text','')).encode('utf-8')).hexdigest(),
                'review_flags':sorted(set(flags)),'contract_errors':sorted(set(errors))}
        contracts.append(record)
        if errors: structured_bad+=1
        audit.append({'case_id':cid,'task':task,'image_id':image_id,'target_instance_id':instance_id,
                      'contract_valid':not errors,'errors':';'.join(sorted(set(errors))),
                      'flags':';'.join(sorted(set(flags))), 'latency_ms':prediction.get('end_to_end_ms')})
        if task in ('vqa','copy'):
            key=(image_id,instance_id)
            if key not in source_images: source_images[key]=image_data_uri(prediction.get('input_sent'))
            review.append({'case_id':cid,'task':task,'image_id':image_id,'instance_id':instance_id,
                           'prompt':prediction.get('prompt',''), 'answer':payload,
                           'latency_ms':prediction.get('end_to_end_ms'),
                           'input_image_embedded':bool(source_images[key]),
                           'image':source_images[key] or '',
                           'flags':sorted(set(flags))})
    # Cross-output consistency applies only when the target is the same, never by image alone.
    for a, c in zip(audit,contracts):
        if c['task']!='copy': continue
        key=(c['target']['image_id'],c['target']['target_instance_id'])
        tag=tag_candidates.get(key)
        if tag and tag.get('primary_color')=='白色' and ('浅灰色' in c['payload'] or '灰色连衣裙' in c['payload']):
            a['flags'] += ';copy_vs_unverified_tag_color_disagreement'
            c['review_flags'].append('copy_vs_unverified_tag_color_disagreement')
    counts=Counter(f for c in contracts for f in c['review_flags'])
    issues=Counter(e for c in contracts for e in c['contract_errors'])
    lats=[p['end_to_end_ms'] for p in predictions if isinstance(p.get('end_to_end_ms'),(int,float))]
    task_lats={}
    for task in sorted(ALLOWED_TASKS):
        vals=[p['end_to_end_ms'] for p in predictions if p.get('task')==task and isinstance(p.get('end_to_end_ms'),(int,float))]
        task_lats[task]={'n':len(vals),'median_ms':round(median(vals),2) if vals else None}
    summary={'schema':'prd32_quality_gate_v2','status':'DEV_CONTRACT_READY_FOR_REVIEW_NOT_PRD_ACCEPTED' if not structured_bad else 'DEV_CONTRACT_NEEDS_FIX',
             'source_pred_file':str(pred_path),'source_service_file':str(service_path),
             'total_requests':len(contracts),'contract_valid_count':len(contracts)-structured_bad,
             'raw_model_native_json_tags':sum(bool(p.get('raw_native_json')) for p in predictions if p.get('task')=='tags'),
             'tag_request_count':sum(p.get('task')=='tags' for p in predictions),
             'qa_review_items':sum(p.get('task')=='vqa' for p in predictions),
             'copy_review_items':sum(p.get('task')=='copy' for p in predictions),
             'embedded_image_count':sum(x['input_image_embedded'] for x in review),
             'review_flags':dict(counts),'hard_errors':dict(issues),
             'task_latency_observed':task_lats,'prd_targets':REQUIRED_PRD,
             'human_accuracy_or_satisfaction_scored':False,'full_prd_acceptance':False,
             'mode':'offline_frozen_prd31_replay','sealed_holdout_used':False,'frozen_prd31_modified':False,
             'warnings':['No sample is automatically approved for publication.','Unknown pattern/design values abstain, never become verified GT.',
                         'Task medians are single-session offline timing, not production P95/QPS.','QA and copy reviewer needs actual visible image and objective criteria.',
                         'Existing 7-image dev metrics cannot establish eight-class 90% tag accuracy.']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'service_contract_v2.jsonl').open('w',encoding='utf-8') as fp:
        for x in contracts: fp.write(json.dumps(x,ensure_ascii=False)+'\n')
    with (out/'audit_per_case.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    (out/'QA_COPY_review.html').write_text(make_html(review),encoding='utf-8')
    (out/'review_items_source.json').write_text(json.dumps([{k:v for k,v in item.items() if k!='image'} for item in review],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('RESULT_DIR:',out)


def make_html(items):
    safe_data=json.dumps(items,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    t='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRD 3.2 · 问答与商品文案人工审核</title><style>
:root{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#192434;background:#f4f7fb}*{box-sizing:border-box}body{margin:0}.wrap{max-width:1150px;margin:auto;padding:26px}.hero{background:#13223b;color:#fff;padding:25px 30px;border-radius:15px}h1{margin:0 0 8px;font-size:25px}h2{font-size:18px;margin:12px 0}.sub{color:#bbc9e0;font-size:14px}button,select,input,textarea{font:inherit}.tool{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}.tool button{border:1px solid #cbd5e1;border-radius:8px;background:#fff;padding:10px 14px;cursor:pointer}.tool button.primary{background:#255dbc;color:#fff;border-color:#255dbc}.status{background:#fff;border-radius:10px;padding:12px 16px;margin:12px 0;border:1px solid #dae3ef}.card{display:grid;grid-template-columns:320px 1fr;gap:20px;background:#fff;padding:22px;margin:18px 0;border-radius:14px;border:1px solid #dae3ef}.image img{max-width:100%;border-radius:8px;border:1px solid #dbe2ee}.missing{background:#fff4df;padding:16px;border-radius:10px;color:#70471b}.hint{font-size:12px;color:#68768a}.question{padding:10px 14px;background:#f1f5fb;border-radius:9px;white-space:pre-wrap}.answer{border-left:4px solid #255dbc;padding:10px 15px;margin:12px 0;white-space:pre-wrap}.form{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.field{display:flex;flex-direction:column;gap:5px;margin:10px 0}.field label{font-size:13px;font-weight:650;color:#34445a}.field select,.field textarea{width:100%;padding:8px;border:1px solid #bdc9d8;border-radius:7px;background:#fff}.field textarea{min-height:70px}.chips{font-size:12px;color:#7a4530}.pill{display:inline-block;background:#eaf2fc;border-radius:999px;padding:4px 9px;margin-right:8px;font-size:12px}.foot{font-size:13px;color:#64748b;margin:25px 0}@media(max-width:760px){.card{grid-template-columns:1fr}.form{grid-template-columns:1fr}.wrap{padding:12px}}</style></head><body><div class="wrap">
<header class="hero"><h1>PRD 3.2 · QA / 商品文案人工审核</h1><div class="sub">10.27 冻结桥接回放 · 研发评估，不是 PRD 正式验收 | 6 条 QA + 3 条文案</div></header>
<p>请查看图片原目标框、问题及回答后再评分。<b>无法看到图片时，不应填写“正确”或“满意”。</b> 本页自动本地暂存，完成后务必导出 JSON 备份和 CSV。原始模型预测保持不变。</p>
<div class="tool"><button class="primary" onclick="exportCSV()">导出人工审核 CSV</button><button onclick="exportJSON()">导出 JSON 备份</button><button onclick="document.getElementById('importFile').click()">导入 JSON 备份</button><input type="file" id="importFile" accept=".json" style="display:none" onchange="importJSON(this)"></div>
<div class="status" id="status"></div><main id="cards"></main><div class="foot">注：标签正确率需要独立 GT；QA 正确率与文案满意度需人工评分，9 条开发案例不足以宣称正式验收。图像只取自当前冻结开发回放产物，不包含 sealed holdout。</div></div>
<script>const cases = __DATA__;const key='PRD32_REVIEW_20261027_'+cases.map(x=>x.case_id).join('_');let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
const choices={qa:['','CORRECT','INCORRECT','JUSTIFIED_ABSTENTION','UNJUSTIFIED_ABSTENTION','UNCERTAIN'],copy:['','SATISFIED','NOT_SATISFIED','UNCERTAIN']};
function escapeHTML(t){return String(t??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function option(v,sel){return '<option value="'+v+'"'+(sel===v?' selected':'')+'>'+ (v||'待审核') +'</option>'}
function view(){document.getElementById('cards').innerHTML=cases.map((x,i)=>{let r=saved[x.case_id]||{};let img=x.image?'<img alt="'+escapeHTML(x.image_id)+' 目标框图" src="'+x.image+'">':'<div class="missing">当前 HTML 没有内嵌这张图。请在 AutoDL 运行脚本后重新打开生成的 HTML，确认原图确实可见，再评分。</div>';return '<section class="card"><div class="image"><span class="pill">'+escapeHTML(x.task.toUpperCase())+'</span><span class="pill">'+escapeHTML(x.image_id)+'</span><h2>'+escapeHTML(x.case_id)+'</h2>'+img+'<p class="hint">目标实例：'+escapeHTML(x.instance_id)+' · '+escapeHTML(x.latency_ms)+' ms</p></div><div><h2>输入问题/生成要求</h2><div class="question">'+escapeHTML(x.prompt)+'</div><h2>模型原始输出（待人工审核）</h2><div class="answer">'+escapeHTML(x.answer)+'</div><div class="chips">自动警告：'+escapeHTML(x.flags.join('; ')||'无')+'</div><div class="form"><div class="field"><label>结果判定（QA 正确性 / 文案满意度）</label><select data-c="'+i+'" data-k="verdict">'+choices[x.task].map(v=>option(v,r.verdict||'')).join('')+'</select></div><div class="field"><label>目标实例是否匹配</label><select data-c="'+i+'" data-k="target_binding">'+['','YES','NO','UNCERTAIN'].map(v=>option(v,r.target_binding||'')).join('')+'</select></div><div class="field"><label>是否有可见图像依据支持</label><select data-c="'+i+'" data-k="visually_grounded">'+['','YES','NO','UNCERTAIN'].map(v=>option(v,r.visually_grounded||'')).join('')+'</select></div><div class="field"><label>审核状态</label><select data-c="'+i+'" data-k="review_status">'+['PENDING','VERIFIED','UNCERTAIN'].map(v=>option(v,r.review_status||'PENDING')).join('')+'</select></div></div><div class="field"><label>人工备注 / 错误归因</label><textarea data-c="'+i+'" data-k="notes">'+escapeHTML(r.notes||'')+'</textarea></div></div></section>'}).join('');document.querySelectorAll('[data-c]').forEach(el=>{el.addEventListener('change',change);el.addEventListener('input',change)});update()}
function change(e){let el=e.target,x=cases[Number(el.dataset.c)];saved[x.case_id]??={};saved[x.case_id][el.dataset.k]=el.value;localStorage.setItem(key,JSON.stringify(saved));update()}
function update(){let done=cases.filter(x=>(saved[x.case_id]||{}).review_status==='VERIFIED' && (saved[x.case_id]||{}).verdict).length;document.getElementById('status').textContent='已确认 '+done+' / '+cases.length+' 条。全部确认前不得计算正式准确率。'}
function payload(){return {schema:'prd32_qa_copy_review_v1',scope:'dev_offline_not_official',results:cases.map(x=>({case_id:x.case_id,task:x.task,image_id:x.image_id,target_instance_id:x.instance_id,...(saved[x.case_id]||{})}))}}
function download(name,txt,type){let b=new Blob([txt],{type:type||'text/plain;charset=utf-8'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download=name;a.click();URL.revokeObjectURL(u)}
function exportJSON(){download('PRD32_QA_COPY_review.json',JSON.stringify(payload(),null,2),'application/json')}
function exportCSV(){let rows=payload().results;let fields=['case_id','task','image_id','target_instance_id','verdict','target_binding','visually_grounded','review_status','notes'];let esc=v=>'"'+String(v??'').replace(/"/g,'""')+'"';download('PRD32_QA_COPY_review.csv','\ufeff'+[fields.join(','),...rows.map(x=>fields.map(k=>esc(x[k])).join(','))].join('\r\n'),'text/csv;charset=utf-8')}
function importJSON(input){let f=input.files?.[0];if(!f)return;let rd=new FileReader();rd.onload=()=>{try{let obj=JSON.parse(rd.result);if(obj.schema!=='prd32_qa_copy_review_v1')throw Error('Schema 不匹配');for(let r of obj.results||[])if(cases.some(x=>x.case_id===r.case_id))saved[r.case_id]=r;localStorage.setItem(key,JSON.stringify(saved));view()}catch(e){alert('导入失败: '+e.message)}};rd.readAsText(f)}view();</script></body></html>'''
    return t.replace('__DATA__',safe_data)


def score_review(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    if obj.get('schema')!='prd32_qa_copy_review_v1': raise ValueError('Wrong review schema')
    grouped=defaultdict(list)
    for item in obj.get('results',[]):
        if item.get('review_status')=='VERIFIED' and item.get('verdict'):
            grouped[item.get('task')].append(item)
    qa=grouped['vqa']; copy=grouped['copy']
    qa_correct=sum(x['verdict']=='CORRECT' for x in qa)
    qa_justified=sum(x['verdict']=='JUSTIFIED_ABSTENTION' for x in qa)
    copy_sat=sum(x['verdict']=='SATISFIED' for x in copy)
    scored={'status':'DEV_HUMAN_REVIEW_PARTIAL_NOT_PRD_ACCEPTED',
           'qa_reviewed':len(qa),'qa_correct':qa_correct,'qa_justified_abstentions':qa_justified,
           'qa_exact_fraction':qa_correct/len(qa) if qa else None,
           'copy_reviewed':len(copy),'copy_satisfied':copy_sat,
           'copy_satisfaction_fraction':copy_sat/len(copy) if copy else None,
           'gt_scope':'small non-blind development batch; adjudication needed for ambiguous cases',
           'formal_prd_acceptance':False}
    print(json.dumps(scored,ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',type=Path,default=Path('/workspace/PRD32_INTEGRATED_ALL12_20261027/predictions.jsonl'))
    ap.add_argument('--service',type=Path,default=Path('/workspace/PRD32_INTEGRATED_ALL12_20261027/service_outputs.jsonl'))
    ap.add_argument('--output',type=Path,default=Path('/workspace/PRD32_QUALITY_GATE_V2_20261027'))
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--score-review',type=Path,help='Optional: score human export JSON, no model used')
    args=ap.parse_args()
    if args.score_review: score_review(args.score_review); return
    for p in (args.predictions,args.service):
        if any(x in str(p).lower() for x in ('sealed_do_not_open','holdout')): ap.error('Sealed inputs prohibited')
    run(args.predictions,args.service,args.output,dry_run=args.dry_run)

if __name__=='__main__': main()
