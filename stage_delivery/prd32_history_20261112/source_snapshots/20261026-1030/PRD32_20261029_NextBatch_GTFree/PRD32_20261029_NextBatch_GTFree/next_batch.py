#!/usr/bin/env python3
# coding: utf-8
"""PRD 3.2 additional GT-free V5->Qwen development batch.

preflight: validate frozen V5 data and choose NEW images using case IDs only;
           audit bbox + saved mask overlap (no GT and no GPU).
gpu: rerun unchanged frozen Qwen pipeline on every predicted instance in the chosen images.
score: postprocess immutable outputs using standalone reviewed5 rule set, then build
       a review-only service envelope and editable human QA page. Never publish.
"""
from __future__ import annotations
import argparse,base64,csv,hashlib,html,json,math,statistics,sys
from collections import Counter,defaultdict
from pathlib import Path

import v5_to_qwen_bridge as bridge
import prd32_guard as guard

HERE=Path(__file__).resolve().parent
RAW=Path(bridge.DEFAULT_RAW)
GATE=Path(bridge.DEFAULT_GATE)
DATA=Path(bridge.DEFAULT_DATA)
PREP=Path(bridge.DEFAULT_PREP)
PLAN=Path('/workspace/PRD32_NEXTBATCH_PREP_20261029_run01')
GPU=Path('/workspace/PRD32_NEXTBATCH_GPU_20261029_run01')
SCORE=Path('/workspace/PRD32_NEXTBATCH_SCORE_20261029_run01')
PREVIOUS_CASES={'fresh_dev_001','fresh_dev_004'}
TASKS=['tags','vqa','copy']
ENUM={
  'instance_binding':['UNREVIEWED','CORRECT','WRONG_INSTANCE','UNCERTAIN'],
  'duplicate_assessment':['UNREVIEWED','DUPLICATE','DISTINCT','UNCERTAIN'],
  'tags_quality':['UNREVIEWED','CORRECT','INCORRECT','UNCERTAIN'],
  'vqa_quality':['UNREVIEWED','CORRECT','INCORRECT','ABSTENTION_REASONABLE','UNCERTAIN'],
  'copy_quality':['UNREVIEWED','CORRECT','WRONG_INSTANCE','DEBUG_TEXT','UNSUPPORTED_CLAIM','OTHER_ERROR','UNCERTAIN'],
}

def writej(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def readj(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):return bridge.sha(Path(path))

def image_counts(records):
    counts=Counter(r['case_id'] for r in records)
    return dict(sorted(counts.items()))

def choose_new_cases(records,max_images=3,max_instances=12):
    """Deterministic by ID only: does NOT use category/GT/Qwen/error history."""
    counts=image_counts(records)
    chosen=[];total=0
    for case in sorted(counts):
        if case in PREVIOUS_CASES:continue
        if len(chosen)>=max_images:break
        if counts[case]<=max_instances-total:
            chosen.append(case);total+=counts[case]
    if not chosen:raise ValueError('NO_NEW_IMAGE_CAN_FIT_BATCH')
    return chosen,total

def mask_overlap(path_a,path_b,wh):
    from PIL import Image
    import numpy as np
    masks=[]
    for path in (path_a,path_b):
        with Image.open(path) as image:
            if image.size!=tuple(wh):raise ValueError('PREDICTED_MASK_NOT_ORIGINAL_IMAGE_SIZE')
            masks.append(np.asarray(image.convert('L'))>127)
    a,b=masks
    inter=int(np.count_nonzero(a & b))
    union=int(np.count_nonzero(a | b))
    aa=int(np.count_nonzero(a));bb=int(np.count_nonzero(b))
    return {'mask_iou':round(inter/union,6) if union else None,
            'mask_intersection_over_smaller':round(inter/min(aa,bb),6) if min(aa,bb)>0 else None,
            'mask_area_pixels_a':aa,'mask_area_pixels_b':bb}

def all_predicted_duplicate_audit(raw_root,threshold=.8):
    """Candidate discovery with V5 predictions only; no human reference."""
    root=Path(raw_root).resolve()
    rows=bridge.readjl(root/'v5_raw_predictions.jsonl')
    out=[];stats={'images':len(rows),'instances':0,'mask_pairs_checked':0,'mask_pairs_unavailable':0}
    for image in rows:
        ins=image['instances'];stats['instances']+=len(ins)
        for i,a in enumerate(ins):
            for b in ins[i+1:]:
                bbox_iou=guard.iou(a['bbox_xyxy'],b['bbox_xyxy'])
                if bbox_iou<threshold:continue
                pair={'case_id':image['case_id'],'instance_a':a['instance_id'],'instance_b':b['instance_id'],
                      'v5_category_a':a['predicted_category'],'v5_category_b':b['predicted_category'],
                      'v5_score_a':a['score'],'v5_score_b':b['score'],
                      'bbox_iou':round(bbox_iou,6),'mask_evidence':'NOT_SAVED',
                      'status':'POSSIBLE_DUPLICATE_REVIEW_ONLY','auto_merged':False}
                paths=[]
                for item in (a,b):
                    mask=item.get('mask_path')
                    if mask is None:break
                    if not isinstance(mask,str) or not mask.endswith('.png') or Path(mask).name!=mask:raise ValueError('UNSAFE_MASK_FILENAME')
                    safe=(root/mask).resolve()
                    if safe.parent!=root or not safe.is_file():raise ValueError('MASK_MISSING_OR_UNSAFE_SYMLINK')
                    paths.append(safe)
                if len(paths)==2:
                    pair.update(mask_overlap(*paths,image['image_size']))
                    pair['mask_evidence']='COMPUTED_FROM_V5_MASKS';stats['mask_pairs_checked']+=1
                else:stats['mask_pairs_unavailable']+=1
                out.append(pair)
    stats['bbox_iou_candidate_pairs']=len(out)
    return out,stats

def load_verified(prep,raw,gate,data_root):
    records,s=bridge.validated(raw,gate,data_root)
    prior=readj(Path(prep)/'bridge_preparation_summary.json')
    prepared=bridge.readjl(Path(prep)/'gt_free_all_instances.jsonl')
    if prior.get('status')!='READY_FOR_DEVELOPMENT_SMOKE':raise ValueError('OLD_BRIDGE_NOT_READY')
    if prior.get('human_gt_or_posthoc_bbox_used') is not False:raise ValueError('HUMAN_GT_IN_PREPARATION')
    if prior.get('v5_raw_predictions_sha256')!=s['v5_predictions_sha256']:raise ValueError('RAW_V5_HASH_MISMATCH')
    if prior.get('all_instances_manifest_sha256')!=digest(Path(prep)/'gt_free_all_instances.jsonl'):raise ValueError('BRIDGE_SHA_MISMATCH')
    if records!=prepared:raise ValueError('APPROVED_GT_FREE_PREPARATION_CHANGED')
    return records,s

def preflight(a):
    if a.plan.exists():raise FileExistsError('Refusing to overwrite '+str(a.plan))
    records,s=load_verified(a.prep,a.raw,a.gate,a.data_root)
    chosen,total=choose_new_cases(records,max_images=3,max_instances=12)
    pairs,dups=all_predicted_duplicate_audit(a.raw,threshold=.8)
    plan={'schema':'prd32_nextbatch_gtfree_plan_v1','scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
          'case_ids':chosen,'previously_processed_cases':sorted(PREVIOUS_CASES),
          'n_new_images':len(chosen),'n_new_v5_instances':total,'n_planned_qwen_requests':total*3,
          'tasks':TASKS,'selection_method':'FIRST_UP_TO_THREE_UNSEEN_CASE_IDS_NO_HUMAN_LABELS',
          'instance_policy':'EVERY_V5_PREDICTION_IN_CHOSEN_IMAGES',
          'v5_raw_sha256':s['v5_predictions_sha256'],
          'prepared_bridge_sha256':digest(Path(a.prep)/'gt_free_all_instances.jsonl'),
          'frozen_prompt_helper_sha256':digest(HERE/'frozen_helpers/prd32_integrated_pipeline_20261027.py'),
          'human_gt_used':False,'publish_allowed':False,'official_PRD_acceptance':False}
    a.plan.mkdir(parents=True)
    writej(a.plan/'next_batch_plan.json',plan)
    writej(a.plan/'v5_duplicate_bbox_mask_audit.json',{'schema':'prd32_v5_bbox_mask_audit_v1','scope':'TRAIN_SOURCE_DEV_ONLY',
        'status':'REVIEW_ONLY_NO_AUTO_MERGE','metrics':dups,'pairs':pairs,'human_reference_used':False})
    with (a.plan/'v5_duplicate_bbox_mask_audit.csv').open('w',newline='',encoding='utf-8-sig') as file:
        fields=['case_id','instance_a','instance_b','v5_category_a','v5_category_b','v5_score_a','v5_score_b',
                'bbox_iou','mask_iou','mask_intersection_over_smaller','mask_evidence','status']
        w=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for item in pairs:w.writerow(item)
    print('PRD32_NEW_BATCH_PREFLIGHT_OK',json.dumps({'images':chosen,'instances':total,'qwen_requests':total*3,
          'v5_duplicate_bbox_candidates_all12':dups['bbox_iou_candidate_pairs'],
          'mask_pairs_computed':dups['mask_pairs_checked']},ensure_ascii=False),flush=True)

def gpu(a):
    plan=readj(a.plan/'next_batch_plan.json')
    if plan['schema']!='prd32_nextbatch_gtfree_plan_v1':raise ValueError('PLAN_SCHEMA_MISMATCH')
    rec,s=load_verified(a.prep,a.raw,a.gate,a.data_root)
    chosen,total=choose_new_cases(rec,max_images=3,max_instances=12)
    if (plan['case_ids']!=chosen or plan['n_new_v5_instances']!=total or
        plan['v5_raw_sha256']!=s['v5_predictions_sha256'] or
        plan['prepared_bridge_sha256']!=digest(Path(a.prep)/'gt_free_all_instances.jsonl') or
        plan['frozen_prompt_helper_sha256']!=digest(HERE/'frozen_helpers/prd32_integrated_pipeline_20261027.py')):
        raise ValueError('PLAN_BINDINGS_CHANGED')
    if a.gpu_out.exists():raise FileExistsError('Refusing overwrite '+str(a.gpu_out))
    bridge.gpu(a.prep,a.raw,a.gate,a.data_root,a.gpu_out,a.model_dir,','.join(chosen),','.join(TASKS))

def base64_img(path,allowed_root):
    if not path:return None
    path=Path(path).resolve(strict=False)
    root=Path(allowed_root).resolve()
    if not path.is_file():return None
    if not path.is_relative_to(root):raise ValueError('MARKED_IMAGE_OUTSIDE_APPROVED_OUTPUT')
    if path.stat().st_size>6_000_000:raise ValueError('MARKED_IMAGE_TOO_LARGE')
    if path.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:raise ValueError('UNSUPPORTED_MARKED_IMAGE')
    mime={'.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png','.webp':'image/webp'}[path.suffix.lower()]
    data=base64.b64encode(path.read_bytes()).decode('ascii')
    return 'data:'+mime+';base64,'+data

def make_review_page(output,qa_rows,gate_rows,html_file):
    groups=defaultdict(dict)
    for row in qa_rows:groups[(row['case_id'],row['target_instance_id'])][row['task']]=row
    gmap={(r['case_id'],r['instance_id']):r for r in gate_rows}
    cards=[]
    for (case,inst),tasks in sorted(groups.items()):
        rid=case+'__'+inst
        t=tasks['tags']; g=gmap[(case,inst)]
        photo=base64_img(t.get('marked_image',''),Path(output)/'marked_inputs')
        im=f'<img class="photo" src="{photo}" alt="Predicted bbox highlighted" />' if photo else '<p>图片未附，请用 AutoDL 原始红框图复核</p>'
        content=''.join(f'<div><strong>{key.upper()}</strong><pre>{html.escape(tasks[key].get("raw_text", ""))}</pre></div>' for key in TASKS)
        fields=''.join('<label>'+html.escape(fld)+'<select data-field="'+html.escape(fld)+'">'+''.join('<option value="'+e+'">'+e+'</option>' for e in opts)+'</select></label>' for fld,opts in ENUM.items())
        flags=', '.join(g['flags']) or 'NONE'
        cards.append(f'<section class="case" data-case="{html.escape(case)}" data-inst="{html.escape(inst)}"><h2>{html.escape(rid)}</h2>'+
           f'<p>V5: {html.escape(str(g["v5_category"]))} / Qwen: {html.escape(str(g["qwen_tag_category"]))}; '+
           f'Duplicate candidate: {g["duplicate_candidate"]}; Copy: {html.escape(g["copy_disposition"])}; Flags: {html.escape(flags)}</p>'+
           f'{im}<div class="tri">{content}</div><div class="fields">{fields}<label class="note">审核备注<textarea data-field="reviewer_note" rows="3" placeholder="请填写实际核查依据"></textarea></label></div></section>')
    js=r'''function exportReview(){const sections=[...document.querySelectorAll('.case')];
 const rows=sections.map(s=>{const x={case_id:s.dataset.case,instance_id:s.dataset.inst};
   s.querySelectorAll('[data-field]').forEach(e=>x[e.dataset.field]=e.value);return x});
 const checked=document.getElementById('attested').checked;
 const missing=rows.filter(r=>r.instance_binding==='UNREVIEWED'||r.duplicate_assessment==='UNREVIEWED'||
       r.tags_quality==='UNREVIEWED'||r.vqa_quality==='UNREVIEWED'||r.copy_quality==='UNREVIEWED'||!r.reviewer_note.trim());
 if(checked&&missing.length){alert('还有 '+missing.length+' 个实例未完整审核，不能标记为人工确认');return;}
 const obj={schema:'prd32_v5_qwen_human_review_v1',scope:'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',reviewed_by_human:checked,rows};
 const blob=new Blob([JSON.stringify(obj,null,2)],{type:'application/json'});
 const u=URL.createObjectURL(blob); const a=document.createElement('a');a.href=u;a.download='PRD32_NextBatch_HumanReview.json';a.click();URL.revokeObjectURL(u);
 document.getElementById('status').textContent='已导出 '+rows.length+' 个实例；人工确认='+checked;
}'''
    hdoc='''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>PRD 3.2 新开发实例审核</title><style>
 body{font-family:system-ui,'Microsoft YaHei',sans-serif;background:#f6f7fb;color:#242934;max-width:1280px;margin:20px auto;padding:0 15px}
 section{background:white;border:1px solid #dce0e6;border-radius:12px;padding:18px;margin:17px 0}
 .photo{display:block;max-height:600px;max-width:min(100%,780px);object-fit:contain;margin:14px 0}
 .tri{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.tri div{background:#f1f3f8;padding:9px;border-radius:8px}
 pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
 .fields label{display:grid;gap:4px}.note{grid-column:1/-1}select,textarea{font:inherit;padding:7px}button{border:0;border-radius:7px;background:#193668;color:white;padding:12px 17px;cursor:pointer}
 @media(max-width:760px){.tri,.fields{grid-template-columns:1fr}}
 </style></head><body><h1>PRD 3.2 新批次 · 全预测实例逐图审核</h1>
 <p>原始 V5 预测与 Qwen 输出均未经人工质量审核；本页的选项默认 UNREVIEWED。蓝框/红框来源为模型预测，不是人工目标框。</p>
 <p>不使用旧 5 例审核标签来判断新实例。已检出的类别冲突、重复候选和文案拦截只是风险提示；不能自动认定为真值。</p>
 <p><label><input type="checkbox" id="attested"/> 我已亲自逐条核对红框图与输出，并填写完整审核选项及依据</label></p>
 <button onclick="exportReview()">导出审核 JSON</button><p id="status"></p>'''+''.join(cards)+'<script>'+js+'</script></body></html>'
    Path(html_file).write_text(hdoc,encoding='utf-8')

def score(a):
    if a.score_out.exists():raise FileExistsError('Refusing overwrite '+str(a.score_out))
    plan=readj(a.plan/'next_batch_plan.json')
    s=readj(a.gpu_out/'gt_free_qwen_smoke_summary.json')
    provenance=readj(a.gpu_out/'smoke_provenance.json')
    qwen_path=a.gpu_out/'qwen_results.jsonl';qwen=bridge.readjl(qwen_path)
    if (s['status']!='COMPLETED_FOR_REVIEW' or s['failed_outputs']!=0 or
      s['n_requests']!=plan['n_planned_qwen_requests'] or s['n_images']!=plan['n_new_images'] or
      s['n_predicted_instances_processed']!=plan['n_new_v5_instances'] or
      s['v5_raw_sha256']!=plan['v5_raw_sha256']):raise ValueError('QWEN_BATCH_INCOMPLETE_OR_PLAN_MISMATCH')
    if (provenance['helper_sha256']!=plan['frozen_prompt_helper_sha256'] or
        provenance['GT_was_not_loaded'] is not True or
        provenance['case_ids']!=plan['case_ids'] or
        provenance['instance_policy']!='ALL_INSTANCES_IN_SELECTED_IMAGES'):
        raise ValueError('GPU_PROVENANCE_OR_PROMPT_CHANGED')
    if len(qwen)!=plan['n_planned_qwen_requests'] or any(r['case_id'] not in plan['case_ids'] for r in qwen):raise ValueError('REQUEST_SET_NOT_NEW_BATCH')
    result=guard.runtime_gate(qwen,threshold=.8)
    if result['images']!=plan['n_new_images'] or result['instances']!=plan['n_new_v5_instances']:
        raise ValueError('RUNTIME_INSTANCE_IDENTITY_MISMATCH')
    a.score_out.mkdir(parents=True)
    writej(a.score_out/'runtime_gate.json',result)
    writej(a.score_out/'duplicate_candidates.json',result['duplicate_candidate_pairs'])
    by={(x['case_id'],x['target_instance_id']):{} for x in qwen}
    for x in qwen:by[(x['case_id'],x['target_instance_id'])][x['task']]=x
    rows=[]
    for r in result['rows']:
        key=(r['case_id'],r['instance_id']);qs=by[key]
        entry={'schema':'prd32_service_envelope_development_v1','case_id':r['case_id'],'instance_id':r['instance_id'],
            'provenance':'PRD31_FROZEN_V5_LIVE_PREDICTION','v5_unverified_category':r['v5_category'],
            'v5_unverified_score':r['v5_score'],'v5_bbox_xyxy':r['bbox'],'qwen_unverified_category':r['qwen_tag_category'],
            'tags_candidate':(qs['tags'].get('tag_bridge_candidate') or None),
            'vqa_raw_unverified':qs['vqa']['raw_text'],'copy_raw_unverified':qs['copy']['raw_text'],
            'copy_disposition':r['copy_disposition'],'vqa_disposition':r['vqa_disposition'],
            'duplicate_candidate':r['duplicate_candidate'],'flags':r['flags'],'copy_block_reasons':r['copy_block_reasons'],
            'service_disposition':r['service_disposition'],'publish_allowed':False,
            'human_reference_used':False,'needs_manual_quality_review':True}
        rows.append(entry)
    bridge.writejl(a.score_out/'service_candidate_envelopes.jsonl',rows)
    make_review_page(a.gpu_out,qwen,result['rows'],a.score_out/'Review_NewInstances.html')
    summary={'schema':'prd32_gtfree_newbatch_service_gate_summary_v1','scope':'TRAIN_SOURCE_DEV_ONLY_NOT_INDEPENDENT',
         'status':'SERVICE_CANDIDATES_GENERATED_REVIEW_REQUIRED','case_ids':plan['case_ids'],
         'images':result['images'],'instances':result['instances'],'qwen_requests':len(qwen),
         'runtime_counts':result['counts'],'gt_free_raw_predictions_sha256':plan['v5_raw_sha256'],
         'qwen_results_sha256':digest(qwen_path),'publish_allowed':False,'human_review_completed':False,
         'automatically_deduplicated_instances':0,'official_PRD_acceptance':False,
         'note':'Original 5 reviewed images excluded. No independent accuracy; all marked service envelopes are review-only.'}
    writej(a.score_out/'service_gate_summary.json',summary)
    with (a.score_out/'service_gate_rows.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['case_id','instance_id','v5_category','qwen_tag_category','duplicate_candidate','copy_disposition','copy_block_reasons','vqa_disposition','service_disposition','flags','publish_allowed']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in result['rows']:
            row={k:r.get(k) for k in fields};row['copy_block_reasons']='|'.join(r['copy_block_reasons']);row['flags']='|'.join(r['flags']);w.writerow(row)
    print('PRD32_NEXTBATCH_GATE_DONE',json.dumps({'images':summary['images'],'instances':summary['instances'],
        'requests':summary['qwen_requests'],'counts':summary['runtime_counts'],
        'review_html':str(a.score_out/'Review_NewInstances.html')},ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='cmd',required=True)
    for cmd in ('preflight','gpu','score'):
        a=sp.add_parser(cmd)
        a.add_argument('--raw',type=Path,default=RAW);a.add_argument('--gate',type=Path,default=GATE)
        a.add_argument('--data-root',type=Path,default=DATA);a.add_argument('--prep',type=Path,default=PREP)
        a.add_argument('--plan',type=Path,default=PLAN);a.add_argument('--gpu-out',type=Path,default=GPU)
        a.add_argument('--score-out',type=Path,default=SCORE);a.add_argument('--model-dir',type=Path,default=Path(bridge.DEFAULT_MODEL))
    a=p.parse_args()
    try:
        {'preflight':preflight,'gpu':gpu,'score':score}[a.cmd](a)
    except Exception as e:
        print('PRD32_NEXTBATCH_BLOCKED',type(e).__name__+': '+str(e),file=sys.stderr,flush=True)
        raise SystemExit(2)
if __name__=='__main__':main()
