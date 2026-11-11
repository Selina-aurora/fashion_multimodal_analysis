#!/usr/bin/env python3
"""DEV60 Oracle8 chat-approved assisted geometry & marker color paired VQA study.

Two separate comparisons, NOT pooled: bbox 2 targets vs color-marker 5 targets.
One overlapping target excluded. No independent GT, no holdout, no release.
"""
from __future__ import annotations
import argparse,base64,collections,csv,hashlib,html,json,os,shutil,statistics,sys,time,traceback,zipfile
from pathlib import Path
from datetime import datetime
from stage1_verified_runtime import (WORKSPACE,REPO,V5_RAW,IMAGE_ROOT,CKPT,RUNNER,QWEN_MODEL,
    v5_preflight,verified_runner,load_guard,refuse_restricted,write_json,sync,digest_file,check_producer_match)
from joint_eval_stage2_dev60 import manifest_verified,load_all_targets
from run_vqa_concise_ab_v2 import load_models,sha_text
from run_vqa_multitype_pilot_v1 import build_multitype_prompt

HERE=Path(__file__).resolve().parent
AUDIT=HERE/'ORACLE8_CHAT_CONFIRMED_ASSISTED_DEV_NOT_BLIND_GT.json'
QUESTION_PLAN=HERE/'FIXED_16_TARGETS_32_QUESTIONS.json'
QUESTION_SHA=HERE/'FIXED_SELECTION_SHA256.txt'
PRIOR=HERE/'ORIGINAL_MULTITYPE_8_ANSWERS_DEV_REFERENCE.json'
STATUS='SHADOW_BLOCKED_ORACLE8_NO_PRODUCTION_GUARD'
SHA=(HERE/'ORACLE8_INPUT_HASHES.json')


def load_selection():
    d=json.loads(AUDIT.read_text(encoding='utf8'))
    if d['schema']!='prd32_oracle8_user_chat_confirmed_assisted_20261111_v1' or not d['human_confirmed'] or d['independent_blind_gt'] or d['holdout_used']:
        raise ValueError('assisted DEV review provenance invalid')
    rows=d['annotations']
    if len(rows)!=8 or len({(x['case_id'],x['instance_id']) for x in rows})!=8:raise ValueError('not 8 unique cases')
    count=collections.Counter(x['ai_review_decision'] for x in rows)
    if count != {'bbox_correction_candidate':2,'same_bbox_control':4,'marker_style_control':1,'needs_mask_or_target_review':1}:
        raise ValueError('unexpected group size/types: '+str(count))
    for r in rows:
        if not r['user_chat_confirmed_ai_assisted_review'] or not r['human_approved'] or r['holdout_used'] or r['production_release'] or r['independent_blind_gt']:
            raise ValueError('unconfirmed or unsafe record')
        b=r['v5_bbox_xyxy'];w,h=r['source_image_size']; check_bbox(b,w,h)
        if r['ai_review_decision']=='bbox_correction_candidate':
            if not r['eligible_for_oracle_comparison'] or r['oracle_bbox_xyxy']!=r['ai_proposed_bbox_xyxy'] or b==r['oracle_bbox_xyxy']:
                raise ValueError('changed bbox missing or degenerate')
            check_bbox(r['oracle_bbox_xyxy'],w,h)
        else:
            if r['eligible_for_oracle_comparison'] or r['oracle_bbox_xyxy'] is not None:raise ValueError('control incorrectly treated as Oracle')
    plan=json.loads(QUESTION_PLAN.read_text(encoding='utf8'))
    if digest_file(QUESTION_PLAN)!=QUESTION_SHA.read_text().strip():raise ValueError('question-plan hash changed')
    qq={q['question_id']:q for q in plan['questions']}
    prior=json.loads(PRIOR.read_text(encoding='utf8'))
    if len(prior)!=8:raise ValueError('historical reference incomplete')
    for r in rows:
        q=qq[r['question_id']];pr=prior[r['question_id']]
        if (q['case_id'],q['instance_id'])!=(r['case_id'],r['instance_id']) or (pr['case_id'],pr['instance_id'])!=(r['case_id'],r['instance_id']):
            raise ValueError('question/target wrong pairing')
        if q['question']!=pr['question'] or q['type']!=pr['type'] or pr['source_image_sha256']!=r['source_image_sha256']:
            raise ValueError('question or image drift')
    # Don't let expected input files be silently changed post-packing.
    pinned=json.loads(SHA.read_text(encoding='utf8'))
    for name,sha in pinned.items():
        if digest_file(HERE/name)!=sha:raise ValueError('packaged input hash mismatch: '+name)
    return rows,qq,prior


def check_bbox(b,w,h):
    if len(b)!=4 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in b):raise ValueError('invalid bbox components')
    x1,y1,x2,y2=b
    if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('bbox outside image bounds')


def neutralize_prompt(q,kind):
    # The model sees the same neutral wording in both RED and CYAN marker arms.
    src=q['question']; neutral=src.replace('红色矩形框','标记框').replace('红色框','标记框').replace('红框','标记框')
    prompt=build_multitype_prompt(neutral,q['type'])
    prompt=prompt.replace('带红色矩形目标标记','带矩形目标标记').replace('红框','标记框')
    if '红框' in prompt or '带红色矩形目标标记' in prompt or neutral not in prompt:
        raise ValueError('non-neutral question/prompt')
    return prompt


def draw_color_border(image,bbox,color,out,stroke=5):
    from PIL import Image,ImageDraw
    if color not in ('red','cyan'):raise ValueError('unsupported marker color')
    palette={'red':(230,30,40),'cyan':(0,190,215)}
    with Image.open(image) as f:base=f.convert('RGB')
    x1,y1,x2,y2=bbox
    w,h=base.size;check_bbox(bbox,w,h)
    im=base.copy();d=ImageDraw.Draw(im)
    rect=[round(x1),round(y1),min(w-1,round(x2)),min(h-1,round(y2))]
    d.rectangle(rect,outline=palette[color],width=stroke)
    im.save(out,format='PNG') # no JPEG recompression; actual model input is packaged exactly
    return out


def preflight(args,out=None):
    rows,qs,prior=load_selection()
    for p in (args.repo,args.raw_dir,args.image_root,args.checkpoint,args.runner,args.model_dir):refuse_restricted(p)
    manifest,_=manifest_verified()
    mm={(r['case_id'],r['instance_id']):r for r in manifest['targets']}
    active=[r for r in rows if r['ai_review_decision']!='needs_mask_or_target_review']
    keys=[(r['case_id'],r['instance_id']) for r in active]
    for r in rows:
        key=(r['case_id'],r['instance_id'])
        if key not in mm or mm[key]['source_image_sha256']!=r['source_image_sha256']:
            raise ValueError('source lineage invalid: '+str(key))
    _,modules=v5_preflight(args.raw_dir,args.image_root,args.repo,args.checkpoint)
    targets=load_all_targets(args.raw_dir,args.image_root,manifest,keys)
    for r in active:
        t=targets[r['case_id'],r['instance_id']]
        if t['image_sha256']!=r['source_image_sha256']:raise ValueError('source SHA mismatch')
        if any(abs(float(a)-float(b))>0.20 for a,b in zip(t['expected_bbox'],r['v5_bbox_xyxy'])):
            raise ValueError('original frozen bbox drift')
    runner=verified_runner(args.runner)
    valid,core,helper,focus=runner.prompt_check(argparse.Namespace(
        snapshot=runner.DEFAULT_SNAP,source=runner.DEFAULT_SOURCE,
        focus_script=runner.DEFAULT_FOCUS,raw=runner.DEFAULT_RAW,input=runner.DEFAULT_INPUT))
    if len(valid)!=171:raise ValueError('Baseline 171 prompt source drift')
    load_guard(args.repo/'src/fashion_multimodal_analysis/integration/prd32_output_guard.py') # source read only
    if not (args.model_dir/'config.json').is_file():raise FileNotFoundError('Qwen config missing')
    for r in rows:
        q=qs[r['question_id']]
        p=build_multitype_prompt(q['question'],q['type'])
        if sha_text(p)!=prior[r['question_id']]['prompt_sha256']:raise ValueError('original multitype prompt mismatch')
        if r['ai_review_decision'] in ('same_bbox_control','marker_style_control'):
            p2=neutralize_prompt(q,'marker')
            if p2==p:raise ValueError('marker prompt must be neutral and distinct')
    if out:
        write_json(out/'PREFLIGHT.json',{'status':'ORACLE8_AB_PREFLIGHT_PASS','n_cases':8,'active_cases':7,
            'geometry_pairs':2,'marker_pairs':5,'gpu_model_calls_measured_expected':14,
            'verified_source_images':7,'source_hashes':modules,
            'audit_sha256':digest_file(AUDIT),'questions_sha256':digest_file(QUESTION_PLAN),
            'three_task_guard_invoked':False,'production_release':False,'holdout_used':False,
            'source_record_provenance':'user_chat_confirmed_AI_assisted_DEV_not_blind_GT'})
    return rows,qs,prior,targets,runner,core,helper,focus


def median(xs):return round(statistics.median(xs),3) if xs else None


def make_review_page(results,out):
    blocks=[]
    byid=collections.defaultdict(dict)
    for r in results:byid[r['case_id']+'/'+r['instance_id']][r['arm']]=r
    for key,arm_map in byid.items():
        rr=list(arm_map.values())
        tag=rr[0]['experiment'];i=rr[0]['question_id']
        rows=[]
        for arm in ('A','B'):
            r=arm_map[arm];encoded=base64.b64encode((out/r['model_input_image_rel']).read_bytes()).decode('ascii')
            mime='image/png' if r['model_input_image_rel'].endswith('png') else 'image/jpeg'
            rows.append(f'<div class="arm"><b>{html.escape(r["arm_label"])} · {r["qwen_host_ms"]:.1f}ms</b>'
                f'<img src="data:{mime};base64,{encoded}" alt="模型真实输入">'
                f'<p>{html.escape(r["answer"])}</p><small>{r["generated_tokens"]} tokens · SHA {html.escape(r["model_input_image_sha256"][:12])}</small></div>')
        blocks.append(f'<section><h2>{html.escape(key)} · {html.escape(i)} · {html.escape(tag)}</h2>'
            f'<p><b>问题：</b>{html.escape(rr[0]["question"])}</p><div class="pair">'+''.join(rows)+'</div>'
            '<label>开发集对照审核：<select><option>待人工核对</option><option>A更可靠</option><option>B更可靠</option>'
            '<option>质量相同</option><option>均有问题</option><option>目标证据不足</option></select></label>'
            '<textarea placeholder="说明图片证据，不要只按答案文本差异判断"></textarea></section>')
    js=r"""function esc(s){return '"'+String(s).replaceAll('"','""')+'"'}; document.querySelector('button').addEventListener('click',()=>{let rows=[['case','review','reason']];document.querySelectorAll('section').forEach(s=>{rows.push([s.querySelector('h2').innerText.split(' · ')[0],s.querySelector('select').value,s.querySelector('textarea').value])});let v='\ufeff'+rows.map(r=>r.map(esc).join(',')).join('\r\n');let blob=new Blob([v],{type:'text/csv;charset=utf-8'}),u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='ORACLE8_AB_PAIR_QUALITY_REVIEW.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)});"""
    htmls='<html lang="zh-CN"><head><meta charset="utf-8"><title>Oracle8 配对 VQA 开发集复核</title><style>body{font:15px system-ui,Microsoft Yahei,sans-serif;background:#f7f8fa;color:#1b2937;max-width:1100px;padding:24px;margin:auto}section{background:white;padding:20px;margin:20px 0;border:1px solid #e2e6ed;border-radius:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.arm{padding:12px;background:#fafafa;border-radius:8px}img{width:100%;height:350px;object-fit:contain}textarea,select{display:block;width:95%;padding:9px;margin:8px 0}textarea{min-height:56px}button{padding:10px 16px;background:#1d4ed8;border:0;border-radius:6px;color:white}</style></head><body><h1>PRD32 Oracle8 开发集目标对照</h1><p>两类实验分别统计：2组框坐标比较、5组红色/青色标记比较。不是独立 GT / 正式 PRD 验收。所有结果 Shadow 阻断。</p><button>导出审核CSV</button>'+''.join(blocks)+'<script>'+js+'</script></body></html>'
    (out/'ORACLE8_AB_7_PAIRS_VISUAL_REVIEW.html').write_text(htmls,encoding='utf8')


def live(args,out):
    rows,qs,prior,targets,runner,core,helper,focus=preflight(args,out)
    if not args.execute_models:raise ValueError('live requires explicit --execute-models')
    import torch,numpy as np
    from PIL import Image
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if torch.cuda.mem_get_info()[0]/2**30<18:raise RuntimeError('need >=18 GiB free GPU at start')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    loaded=load_models(args,out)
    torch,np,Image,qwen,processor,v5,ID_TO_CLASS=loaded
    from fashion_multimodal_analysis.evaluation.runtime import predict_timed
    original_input=out/'_work_original';original_input.mkdir(exist_ok=True)
    generated=out/'model_inputs_exact';generated.mkdir(exist_ok=True)
    qa=[];pairs=[];warmup_done=False
    for i,r in enumerate(rows):
        if r['ai_review_decision']=='needs_mask_or_target_review':continue
        case,inst=r['case_id'],r['instance_id'];t=targets[case,inst]
        with Image.open(t['image_path']) as file:img=file.convert('RGB')
        if img.size!=tuple(r['source_image_size']):raise ValueError('image dims differ')
        sync(torch);tv=time.perf_counter()
        boxes,labels,scores,masks,_=predict_timed(img,v5,torch.device('cuda'),score_threshold=.4,mask_threshold=.5)
        sync(torch);v5ms=(time.perf_counter()-tv)*1000
        idx=int(inst.split('_')[-1]);
        if idx>=len(labels):raise ValueError('missing predicted instance')
        cat=ID_TO_CLASS.get(int(labels[idx]));newmask=(np.asarray(masks[idx])>0).astype(np.uint8)
        with Image.open(t['frozen_mask_path']) as f:oldmask=np.asarray(f.convert('L'))>127
        ck=check_producer_match(target=t,ins={'instance_id':inst,'category':cat,'score':float(scores[idx]),'bbox':[float(v) for v in boxes[idx]]},
              reference_mask=oldmask,produced_mask=newmask)
        if ck['mask_iou']<.9999:raise ValueError('live V5 frozen mask drift')
        if any(abs(float(a)-float(b))>.20 for a,b in zip(boxes[idx],r['v5_bbox_xyxy'])):raise ValueError('V5 box drift')
        q=qs[r['question_id']]
        qrec={'image_path':t['image_path'],'image_sha256':t['image_sha256'],
              'target_bbox_xyxy':list(map(float,boxes[idx])),'target_instance_id':inst,
              'task':'vqa','query':q['question'],'length_chars':[20,80],'predicted_attribute_context':[]}
        name=case+'__'+inst
        if r['eligible_for_oracle_comparison']:
            exp='geometry_only'; prompt=build_multitype_prompt(q['question'],q['type'])
            if sha_text(prompt)!=prior[r['question_id']]['prompt_sha256']:raise ValueError('original multiQA prompt drift')
            arm_specs=[('A','V5原始框',list(map(float,boxes[idx]))),('B','辅助审核修正框',list(map(float,r['oracle_bbox_xyxy'])))]
            armimgs={}
            for arm,lab,bbox in arm_specs:
                rec={**qrec,'target_bbox_xyxy':bbox}
                p=original_input/f'{name}__{arm}';p.mkdir(exist_ok=True)
                src=Path(helper.make_marked_image(rec,p,'marked_full_image',.05)).resolve(strict=True)
                dest=generated/f'{name}__{arm}{src.suffix.lower()}'
                shutil.copy2(src,dest) # exact model input bytes are exported; no preview recompression
                armimgs[arm]=(dest,lab,rec)
        else:
            exp='marker_color_only';prompt=neutralize_prompt(q,'marker')
            armimgs={}
            for arm,color,lab in [('A','red','同框红标记'),('B','cyan','同框青标记')]:
                dest=generated/f'{name}__{arm}.png'
                draw_color_border(t['image_path'],r['v5_bbox_xyxy'],color,dest)
                armimgs[arm]=(dest,lab,qrec)
            # Pixel changes can occur only at the marker border.
            with Image.open(armimgs['A'][0]) as f:a=np.asarray(f.convert('RGB'))
            with Image.open(armimgs['B'][0]) as f:b=np.asarray(f.convert('RGB'))
            changed=np.any(a!=b,axis=2)
            if not changed.any() or changed.sum()>0.12*a.shape[0]*a.shape[1]:raise ValueError('marker differs across too much of image')
        if sha_text(prompt)==sha_text(core.build_prompt(helper,focus,qrec,'vqa','baseline')):raise ValueError('fixed-color question leaked')
        if not warmup_done:
            wp=armimgs['A'][0];warm,_=helper.run_model_case(qwen,processor,qrec,wp,prompt,task_limits={'vqa':128})
            write_json(out/'WARMUP_EXCLUDED.json',{'target':name,'sha256':sha_text(warm),'included_in_measured_calls':False,'publish_allowed':False})
            warmup_done=True
        # Alternate arm order per target, but retain arm labels.
        order=['A','B'] if i%2==0 else ['B','A']
        pair=[]
        for arm in order:
            dest,lab,rec=armimgs[arm]
            # Mandatory before & after checksum to catch accidental image alteration.
            sha=digest_file(dest)
            sync(torch);t0=time.perf_counter()
            ans,timing=helper.run_model_case(qwen,processor,rec,dest,prompt,task_limits={'vqa':128})
            sync(torch);hostms=(time.perf_counter()-t0)*1000
            if digest_file(dest)!=sha or not isinstance(ans,str) or not ans.strip():raise ValueError('input changed or empty response')
            item={'case_id':case,'instance_id':inst,'question_id':r['question_id'],'question':q['question'],
                  'question_type':q['type'],'experiment':exp,'arm':arm,'arm_label':lab,'order_in_pair':order.index(arm)+1,
                  'model_input_image_rel':str(dest.relative_to(out)),'model_input_image_sha256':sha,
                  'input_prompt_sha256':sha_text(prompt),'source_image_sha256':r['source_image_sha256'],
                  'v5_bbox_xyxy':r['v5_bbox_xyxy'],'used_bbox_xyxy':rec['target_bbox_xyxy'],
                  'v5_category':cat,'mask_iou_vs_frozen_prediction_not_gt':ck['mask_iou'],
                  'qwen_host_ms':round(hostms,3),'model_preprocess_ms':timing['preprocess_ms'],
                  'model_generation_ms':timing['generation_ms'],'generated_tokens':timing['generated_tokens'],
                  'answer':ans,'answer_sha256':sha_text(ans),'v5_host_ms_shared':round(v5ms,3),
                  'chat_confirmed_ai_assisted_bbox_not_blind_gt':True,'guard_status':STATUS,
                  'production_release':False,'original_three_task_guard_invoked':False,
                  'holdout_used':False,'independent_blind_gt':False,
                  'quality_review_status':'PENDING_PAIR_QUALITY_REVIEW'}
            if exp=='marker_color_only':item['marker_prompt_note']='neutral marker terminology in both arms'
            qa.append(item);pair.append(item)
            with (out/'PER_ARM_OUTPUTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
            print('ORACLE8_AB',name,exp,arm,'ms',round(hostms,2),'tok',timing['generated_tokens'],'release=false',flush=True)
        pair=sorted(pair,key=lambda x:x['arm'])
        if pair[0]['input_prompt_sha256']!=pair[1]['input_prompt_sha256']:raise ValueError('paired prompts unequal')
        if exp=='geometry_only' and pair[0]['used_bbox_xyxy']==pair[1]['used_bbox_xyxy']:raise ValueError('no box delta in correction pair')
        if exp=='marker_color_only' and pair[0]['used_bbox_xyxy']!=pair[1]['used_bbox_xyxy']:raise ValueError('box changed in marker pair')
        pairs.append({'case_id':case,'instance_id':inst,'question_id':r['question_id'],'experiment':exp,
             'qwen_mean_arm_a_ms':pair[0]['qwen_host_ms'],'qwen_mean_arm_b_ms':pair[1]['qwen_host_ms'],
             'same_answer':pair[0]['answer']==pair[1]['answer'],'review_status':'PENDING_HUMAN_QUALITY_REVIEW',
             'is_independent_oracle_ground_truth':False,'publish_allowed':False})
    if len(qa)!=14 or len(pairs)!=7 or len({(x['case_id'],x['instance_id'],x['arm']) for x in qa})!=14:
        raise ValueError('incomplete controlled experiment')
    if any(x['production_release'] or x['holdout_used'] or x['independent_blind_gt'] for x in qa):raise ValueError('unsafe release')
    with (out/'PAIRED_7_CASES_REVIEW_PENDING.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    summ={'status':'ORACLE8_AB_V1_COMPLETED_SHADOW_ONLY','n_source_targets':8,'n_used_targets':7,
        'excluded_target':'fresh_dev_040/pred_0000','n_geometry_pairs':2,'n_marker_color_pairs':5,
        'n_measured_qwen_calls':14,'warmup_calls_excluded':1,
        'all_model_input_images_byte_identical_to_packaged_preview':True,
        'mean_qwen_host_ms':round(statistics.mean(x['qwen_host_ms'] for x in qa),3),
        'n_exact_same_answer_pairs':sum(x['same_answer'] for x in pairs),
        'geometry_exact_same_answer_pairs':sum(x['same_answer'] for x in pairs if x['experiment']=='geometry_only'),
        'marker_exact_same_answer_pairs':sum(x['same_answer'] for x in pairs if x['experiment']=='marker_color_only'),
        'holdout_used':False,'independent_ground_truth':False,'official_quality_metric':False,
        'three_task_guard_invoked':False,'production_release':False,
        'note':'The two factor types MUST NOT be pooled. Comparisons remain exploratory DEV-only and human-review required.'}
    write_json(out/'SUMMARY.json',summ)
    make_review_page(qa,out)
    print('ORACLE8_AB_V1_COMPLETE_SHADOW_ONLY',flush=True)


def archive(out):
    target=out.with_suffix('.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and '_work_original' not in p.parts:
                z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(target) as z:
        if z.testzip():raise ValueError('broken ZIP')
    print('UPLOAD_THIS:',str(target),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['package-check','preflight','live'],default='package-check')
    p.add_argument('--execute-models',action='store_true')
    for nm,default in [('repo',REPO),('raw-dir',V5_RAW),('image-root',IMAGE_ROOT),('checkpoint',CKPT),('runner',RUNNER),('model-dir',QWEN_MODEL)]:p.add_argument('--'+nm,type=Path,default=default)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.mode=='package-check':
        rows,qq,prior=load_selection()
        print('ORACLE8_AB_PACKAGE_CHECK_PASS 2 geometry pairs + 5 marker-color pairs; 1 excluded, no GPU')
        return
    out=args.out or WORKSPACE/f'PRD32_ORACLE8_AB_V1_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}_UPLOAD_THIS'
    if out.exists():raise SystemExit('Refusing overwrite of existing result directory')
    out.mkdir(parents=True)
    try:
        if args.mode=='preflight':
            preflight(args,out);print('ORACLE8_AB_PREFLIGHT_PASS; no GPU inference',flush=True)
        else:live(args,out)
    except BaseException as e:
        write_json(out/'ERROR.json',{'error_type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),
         'production_release':False})
        archive(out);raise
    archive(out)
if __name__=='__main__':main()
