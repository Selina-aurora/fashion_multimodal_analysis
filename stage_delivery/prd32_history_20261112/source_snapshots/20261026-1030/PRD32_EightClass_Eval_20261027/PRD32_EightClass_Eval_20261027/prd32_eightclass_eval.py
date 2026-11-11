#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD 3.2: eight-category DEVELOPMENT coverage and evaluation.

Commands: prepare, run, review, score, self-test. No default access to sealed holdout.
- prepare never alters PRD 3.1; reads CSV/JSONL/JSON of existing prediction boxes.
- run calls local Qwen only on input-validated dev cases, one request per instance.
- review creates static-first embedded-thumb human GT annotation page.
- score reports per-field exact-match DEV numbers (not formal acceptance).
"""
from __future__ import annotations
import argparse, base64, collections, csv, hashlib, html, json, math, os, re, statistics, sys, time
from pathlib import Path

CATS = ['top','pants','skirt','outerwear','dress','shoe','bag','accessory']
PATTERNS = {'纯色','条纹','格纹','人字纹','波点','花卉印花','文字印花','字母印花','几何印花','动物纹','拼色'}
EXCLTOKENS = ('holdout','sealed','test_private','private_test','do_not_open')
PROVENANCE_TRAIN = {'train','training','fit','trainval','v5_training'}
HISTORY_IDS = {'new1022_0003','new1022_0004','new1022_0011','new1022_0018',
               'dev_02','dev_04','dev_05','dev_06','dev_08','dev_09','dev_12'}


def safe(p):
    s=str(p).lower().replace('\\','/')
    return not any(t in s for t in EXCLTOKENS)


def shasum(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()


def dump(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def records(path):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(p)
    if not safe(p):raise ValueError('Sealed/holdout source prohibited')
    if p.suffix.lower()=='.csv':
        with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
    if p.suffix.lower()=='.jsonl':
        vals=[]
        for ln,s in enumerate(p.read_text(encoding='utf-8-sig').splitlines(),1):
            if s.strip():
                try: vals.append(json.loads(s))
                except json.JSONDecodeError as e: vals.append({'_invalid_json_line':ln,'_reason':str(e)})
        return vals
    v=json.loads(p.read_text(encoding='utf-8-sig'))
    if isinstance(v,list):return v
    for k in ('predictions','instances','records','items','data','results'):
        if isinstance(v.get(k),list):return v[k]
    raise ValueError('Input JSON needs an array or an array under predictions/instances/records')


def jsonl(path,ls):
    with Path(path).open('w',encoding='utf-8') as f:
        for x in ls:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')


def parsebox(b):
    if isinstance(b,str):
        try:b=json.loads(b)
        except (TypeError,ValueError):return None
    if not isinstance(b,(list,tuple)) or len(b)!=4:return None
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in b):return None
    x1,y1,x2,y2=b
    if not (0<=x1<x2 and 0<=y1<y2):return None
    return [float(x) for x in b]


def normalize(r,source,index):
    """Do not equate a predicted label with independent human GT."""
    if not isinstance(r,dict):return None,['NOT_OBJECT']
    if '_invalid_json_line' in r:return None,['INVALID_JSON']
    keys=lambda *xs: next((r[x] for x in xs if x in r and r[x] not in (None,'')),None)
    iid=keys('target_instance_id','instance_id','object_id','pred_instance_id')
    ip=keys('image_path','file_path','path','image_file')
    imageid=keys('image_id','img_id','file_name') or (Path(str(ip)).stem if ip else None)
    bbox=parsebox(keys('target_bbox_xyxy','predicted_target_bbox_xyxy','bbox_xyxy','bbox','box'))
    imgsha=keys('image_sha256','sha256','image_hash')
    cohort=str(keys('source_split','split','provenance') or 'unverified').strip().lower()
    origin=str(keys('target_bbox_origin','bbox_origin') or 'NOT_RECORDED').strip()
    predicted=keys('predicted_category','predicted_class','category_hint','category')
    caseid=str(keys('case_id') or f'ec_{index:05d}')
    errors=[]
    if not iid or not isinstance(iid,(str,int)):errors.append('INSTANCE_ID_MISSING')
    if not ip or not isinstance(ip,str):errors.append('IMAGE_PATH_MISSING')
    if not imageid or not isinstance(imageid,(str,int)):errors.append('IMAGE_ID_MISSING')
    if not bbox:errors.append('BBOX_INVALID')
    if origin not in ('prd31_frozen_prediction','prd31_v5_prediction'):
        errors.append('BBOX_PROVENANCE_UNVERIFIED')
    if predicted is not None and predicted not in CATS:errors.append('PREDICTED_CLASS_INVALID')
    if cohort in PROVENANCE_TRAIN or cohort.startswith('train'):errors.append('TRAIN_SOURCE_EXCLUDED')
    if not safe(ip or '') or not safe(source):errors.append('RESTRICTED_SOURCE')
    # Old dev files are allowed only as illustrative DEV cases, not fresh coverage.
    old=str(imageid) in HISTORY_IDS or str(imageid).startswith('new1022_')
    if old:errors.append('PREVIOUSLY_USED_DEV_IMAGE')
    return dict(case_id=caseid,task='tags',image_id=str(imageid),target_instance_id=str(iid) if iid else None,
         image_path=str(ip) if ip else None,image_sha256=str(imgsha).lower() if imgsha else None,
         target_bbox_xyxy=bbox,target_bbox_origin=origin,predicted_category=predicted,
         source_split=cohort,previous_dev=old,source_input=str(source)),errors


def validate_img(o, *, inspect=True):
    issues=[]
    if o['image_path'] is None or not safe(o['image_path']):return ['RESTRICTED_OR_EMPTY_IMAGE_PATH']
    p=Path(o['image_path'])
    if p.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:issues.append('IMAGE_TYPE_UNSUPPORTED')
    if not p.is_file():return issues+['IMAGE_NOT_FOUND']
    if inspect:
        try:
            from PIL import Image
            if p.stat().st_size>45_000_000:issues.append('IMAGE_FILE_TOO_LARGE')
            else:
                with Image.open(p) as a:
                    w,h=a.size
                    if w*h>45_000_000:issues.append('IMAGE_PIXELS_TOO_LARGE')
                    b=o.get('target_bbox_xyxy')
                    if b and (b[2]>w or b[3]>h):issues.append('BBOX_OUTSIDE_ACTUAL_IMAGE')
                    a.verify()
                if o.get('image_sha256') and shasum(p)!=o['image_sha256']:
                    issues.append('IMAGE_SHA_MISMATCH')
        except (OSError,ValueError) as e:issues.append('IMAGE_UNREADABLE_'+type(e).__name__)
    return issues


def prepare(args):
    if Path(args.output).exists():raise FileExistsError(args.output)
    inp=records(args.input); out=Path(args.output);out.mkdir(parents=True)
    excluded_shas=set()
    for ep in (args.exclude or []):
        if not safe(ep):raise ValueError('Restricted exclusion file')
        for v in records(ep):
            if isinstance(v,dict):
                h=v.get('image_sha256') or v.get('sha256');excluded_shas.add(h)
    allrows=[]; candidates=[]; seen=set(); seen_caseids=set(); counts=collections.Counter()
    for idx,r in enumerate(inp,1):
        obj,why=normalize(r,args.input,idx)
        if not obj:
            allrows.append({'source_row':idx,'case_id':None,'decision':'BLOCKED','reasons':why});continue
        # Multiple tasks referring to same target are not independent examples.
        key=(obj['image_id'],obj['target_instance_id'])
        if key in seen:why.append('DUPLICATED_INSTANCE')
        seen.add(key)
        if obj['case_id'] in seen_caseids:why.append('DUPLICATE_CASE_ID')
        seen_caseids.add(obj['case_id'])
        if obj.get('image_sha256') in excluded_shas:why.append('IMAGE_IN_EXCLUSION_LIST')
        if not args.skip_image_checks and not any('RESTRICTED' in x for x in why):why+=validate_img(obj)
        # Historical data may be audited, but never counted as NEW evaluation coverage.
        if 'PREVIOUSLY_USED_DEV_IMAGE' in why and args.allow_historical_dev:
            why.remove('PREVIOUSLY_USED_DEV_IMAGE');obj['source_split']='historical_dev'
        decision='BLOCKED' if why else 'ELIGIBLE_DEV'
        allrows.append({'source_row':idx, 'case_id':obj['case_id'], 'image_id':obj['image_id'],
                        'instance_id':obj['target_instance_id'],'predicted_category':obj['predicted_category'],
                        'source_split':obj['source_split'],'decision':decision,'reasons':why})
        counts.update([decision])
        if not why:candidates.append(obj)
    # Do not claim 8-class completeness on V5 predicted categories alone.
    bypred=collections.Counter(x['predicted_category'] for x in candidates if x['predicted_category'])
    catcoverage={c:bypred[c] for c in CATS}
    jsonl(out/'candidate_decisions.jsonl',allrows)
    jsonl(out/'eligible_dev.jsonl',candidates)
    summary={'schema':'prd32_eightclass_prepare_dev_v1','input_count':len(inp),'eligible':len(candidates),
         'blocked':len(inp)-len(candidates),'predicted_category_coverage_NOT_GT':catcoverage,
         'predicted_8class_presence':all(v>0 for v in catcoverage.values()),
         'independent_eval_status':'NOT_ESTABLISHED','human_gt_completed':False,
         'holdout_used':False,'frozen_prd31_modified':False,'output_dir':str(out.resolve()),
         'note':'Per-category counts use UNVERIFIED upstream predicted labels. Do not treat as ground truth.'}
    dump(out/'summary.json',summary)
    print('PREPARE_DONE:',json.dumps(summary,ensure_ascii=False));return 0


def load_helper():
    import prd32_integrated_pipeline_20261027 as v1
    return v1


def run(args):
    if not safe(args.input) or not safe(args.output):raise ValueError('Holdout path prohibited')
    inp=records(args.input)
    if not inp:raise ValueError('No inputs found')
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    # Re-validate all before loading model; one bad input cannot poison others.
    checked=[]; identities=set(); cases_seen=set()
    for idx,rec in enumerate(inp,1):
        if not isinstance(rec,dict):continue
        obj,why=normalize(rec,args.input,idx)
        if obj:
            why=[x for x in why if x!='DUPLICATED_INSTANCE' and not (x=='PREVIOUSLY_USED_DEV_IMAGE' and args.allow_historical_dev)]
            identity=(obj['image_id'],obj['target_instance_id'])
            if identity in identities:why.append('DUPLICATE_INSTANCE')
            identities.add(identity)
            if obj['case_id'] in cases_seen:why.append('DUPLICATE_CASE_ID')
            cases_seen.add(obj['case_id'])
            if not any('RESTRICTED' in x for x in why):why+=validate_img(obj)
        checked.append({'obj':obj,'issues':why,'source':rec.get('case_id',f'row_{idx}')})
    valid=sum(bool(q['obj']) and not q['issues'] for q in checked)
    print(f'INPUT_PREFLIGHT: {len(checked)} requests; eligible={valid}; blocked={len(checked)-valid}')
    if args.dry_run:
        for q in checked:print(q['source'], 'PASS' if not q['issues'] else 'BLOCKED',q['issues'])
        print('DRY_RUN_OK; GPU not loaded and no output files written');return 0
    from PIL import Image
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    modeldir=Path(args.model_id)
    if not (modeldir/'config.json').is_file():raise FileNotFoundError(modeldir/'config.json')
    if torch.cuda.mem_get_info()[0] < 18*1024**3:raise RuntimeError('Less than 18 GiB free GPU; inspect GPU processes')
    v1=load_helper()
    out.mkdir(parents=True);imgroot=out/'marked_images';imgroot.mkdir()
    dump(out/'run_config.json',{'model':str(modeldir),'input_sha256':shasum(args.input),
          'input':args.input,'prompt_sha256':hashlib.sha256(v1.TAG_PROMPT.encode()).hexdigest(),
          'max_new_tokens':args.max_new_tokens,'human_gt_used_for_inference':False,
          'inference_mode':'offline_dev_predicted_bbox','prd31_modified':False,'holdout_used':False})
    t0=time.perf_counter()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(modeldir),local_files_only=True,device_map='auto',dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    processor=AutoProcessor.from_pretrained(str(modeldir),local_files_only=True,max_pixels=args.max_pixels)
    load_sec=round(time.perf_counter()-t0,2)
    pred=[]; goodtimes=[];stats=collections.Counter()
    for n,q in enumerate(checked,1):
        o=q['obj']; result={'case_id':o['case_id'] if o else q['source'],'image_id':o.get('image_id') if o else None,
            'target_instance_id':o.get('target_instance_id') if o else None,'task':'tags',
            'status':'BLOCKED_INPUT' if q['issues'] else 'PENDING','input_issues':q['issues'],
            'raw_model_output':None,'raw_native_json':False,'schema_valid':False,
            'canonical_tags':None,'issues':[],'timing':None,'inference_performed':False}
        if o and not q['issues']:
            try:
                # Only facts from image in prompt; never pass predicted class or human GT.
                rec={'task':'tags','case_id':o['case_id'],'image_path':o['image_path'],
                     'image_sha256':o.get('image_sha256') or shasum(o['image_path']),
                     'target_instance_id':o['target_instance_id'],'target_bbox_xyxy':o['target_bbox_xyxy'],
                     'query':'visible tags'}
                marked=v1.make_marked_image(rec,imgroot,'marked_full_image',0.05)
                raw,t=v1.run_model_case(model,processor,rec,marked,v1.TAG_PROMPT,task_limits={'tags':args.max_new_tokens})
                parsed=v1.parse_tag_output(raw)
                can=parsed['canonical_service_json']
                if can:
                    p=can.get('pattern')
                    if p is not None and p not in PATTERNS:
                        can['pattern']=None;parsed['issues'].append('PATTERN_NONSTANDARD_ABSTAIN')
                result.update(status='GENERATED_UNVERIFIED',raw_model_output=raw,raw_native_json=parsed['raw_native_json'],
                              schema_valid=parsed['schema_valid'],canonical_tags=can,
                              issues=parsed['issues'],timing=t,inference_performed=True,
                              marked_image=str(marked))
                goodtimes.append(t['end_to_end_ms'])
            except Exception as e:
                result.update(status='RUNTIME_FAILED',issues=['CASE_EXCEPTION_'+type(e).__name__])
        stats[result['status']]+=1;pred.append(result)
        jsonl(out/'predictions.jsonl',pred)
        print(f'[{n}/{len(checked)}] {result["case_id"]}: {result["status"]}',flush=True)
    summary={'schema':'prd32_eightclass_attribute_run_dev_v1','status_counts':dict(stats),'inputs':len(checked),
         'model_load_seconds':load_sec,'median_ms':round(statistics.median(goodtimes),2) if goodtimes else None,
         'p95_ms':round(sorted(goodtimes)[max(0,math.ceil(.95*len(goodtimes))-1)],2) if len(goodtimes)>=20 else None,
         'p95_interpretation':'Observed session percentile, not service SLO' if len(goodtimes)>=20 else 'insufficient n (<20)',
         'accuracy':None,'human_gt_evaluated':False,'raw_model_json_native_count':sum(x['raw_native_json'] for x in pred),
         'strict_schema_valid_count':sum(x['schema_valid'] for x in pred),
         'frozen_prd31_modified':False,'holdout_used':False,'full_prd_acceptance':False}
    dump(out/'summary.json',summary);print('RUN_DONE:',json.dumps(summary,ensure_ascii=False));return 0


def review(args):
    data=records(args.predictions);origin=records(args.manifest)
    find={(r.get('case_id')):r for r in origin if isinstance(r,dict)}
    imagechunks=[];gtrows=[]
    for p in data:
        if not isinstance(p,dict):continue
        cid=p.get('case_id');record=find.get(cid) or {}
        if not record or not safe(record.get('image_path','')):continue
        image_url='';src=record.get('image_path')
        if src and Path(src).is_file():
            from PIL import Image,ImageDraw,ImageOps
            from io import BytesIO
            im=Image.open(src).convert('RGB')
            box=record.get('target_bbox_xyxy')
            if box:
                dr=ImageDraw.Draw(im);dr.rectangle(tuple(box),outline=(240,55,48),width=max(3,min(im.size)//160))
            im.thumbnail((440,460))
            output=BytesIO();im.save(output,format='JPEG',quality=75)
            image_url='data:image/jpeg;base64,'+base64.b64encode(output.getvalue()).decode()
        tags=p.get('canonical_tags') or {}
        imgtag=f'<img alt="target" src="{image_url}">' if image_url else '<p>原图未能加载，禁止评分</p>'
        gtrows.append({'case_id':cid,'task':'tags','image_id':p.get('image_id'),
            'target_instance_id':p.get('target_instance_id'),'gt_category':'','gt_primary_color':'',
            'gt_pattern':'','gt_design_notes':'','review_status':'PENDING','review_note':''})
        opts='<option value="">待审核</option>'+''.join(f'<option value="{c}">{c}</option>' for c in CATS)
        # Static-first. Images and predictions exist in HTML regardless of JS execution.
        imagechunks.append(f'''<article class="card" data-case="{html.escape(str(cid),quote=True)}"><div>{imgtag}
          <p>{html.escape(str(p.get('image_id')))} / {html.escape(str(p.get('target_instance_id')))}</p></div>
          <div><h2>{html.escape(str(cid))}</h2><p><strong>先独立填写人工 GT，不要参考模型输出。</strong></p>
          <details><summary>GT 填写完成后才展开：模型预测（可能引入标注偏倚）</summary><p>{html.escape(json.dumps(tags,ensure_ascii=False))}</p><p>运行状态：{html.escape(str(p.get('status')))}</p></details>
          <label>人工类别 <select name="gt_category">{opts}</select></label>
          <label>人工主色（统一中文色名） <input name="gt_primary_color" placeholder="白色；不确定则留空"></label>
          <label>人工图案（纯色需要确实可见） <input name="gt_pattern" placeholder="纯色/条纹/字母印花；不能确定则留空"></label>
          <label>可见结构备注 <textarea name="gt_design_notes" placeholder="肩带、扣子、领口、下摆等"></textarea></label>
          <label>审核状态 <select name="review_status"><option value="PENDING">待审核</option><option value="VERIFIED">确认完成</option><option value="EXCLUDE">无法可靠审核</option></select></label>
          <label>备注 <textarea name="review_note"></textarea></label></div></article>''')
    datajson=json.dumps(gtrows,ensure_ascii=False).replace('</','<\\/')
    # Only control handler is on buttons; static HTML renders all cards even without scripts.
    doc='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>PRD 3.2 八类属性 GT 审核</title><style>
      body{font:15px system-ui,'Microsoft Yahei',sans-serif;background:#f5f7fb;color:#172a42;max-width:1150px;margin:30px auto;padding:15px}
      header{background:#172640;color:white;padding:26px;border-radius:12px}.card{display:grid;grid-template-columns:370px 1fr;gap:23px;background:white;padding:20px;margin-top:20px;border:1px solid #d7dfe9;border-radius:12px}
      img{max-width:100%;max-height:470px;border-radius:7px}label{display:block;margin:10px 0;font-weight:600}input,select,textarea{width:100%;max-width:520px;box-sizing:border-box;display:block;padding:9px;margin-top:4px;border:1px solid #b8c6d8;border-radius:6px}button{padding:12px 20px;background:#2054ac;border:0;border-radius:8px;color:white;cursor:pointer;margin:14px 0}
      @media(max-width:720px){.card{grid-template-columns:1fr}}
      </style></head><body><header><h1>PRD 3.2 八类服饰属性 · 人工 GT 审核</h1><p>仅开发阶段。请先盲标图片，不参考模型预测；GT 复核后再打开预测。未审核字段不计准确率，不得使用 sealed holdout。</p></header>
      <button onclick="exportCsv()">导出审核 CSV</button><button onclick="exportJson()">导出 JSON 备份</button><main>__CARDS__</main><script>
      function rows(){return [...document.querySelectorAll('.card')].map(c=>{
        const o={case_id:c.dataset.case,task:'tags'};
        for(const n of ['gt_category','gt_primary_color','gt_pattern','gt_design_notes','review_status','review_note'])o[n]=c.querySelector('[name="'+n+'"]').value;
        return o;});}
      function save(name,data,type){const b=new Blob([data],{type});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),1500)}
      function csvEscape(x){const s=String(x??'');return '"'+s.replace(/"/g,'""')+'"'}
      function exportCsv(){const x=rows(),k=['case_id','task','gt_category','gt_primary_color','gt_pattern','gt_design_notes','review_status','review_note'];let all=[k.map(csvEscape).join(',')];for(const r of x)all.push(k.map(q=>csvEscape(r[q])).join(','));save('prd32_8class_human_gt.csv','\\ufeff'+all.join('\\n'),'text/csv;charset=utf-8')}
      function exportJson(){save('prd32_8class_human_gt.json',JSON.stringify({schema:'prd32_8class_gt_dev_v1',results:rows()},null,2),'application/json')}
      </script></body></html>'''.replace('__CARDS__',''.join(imagechunks))
    Path(args.output).write_text(doc,encoding='utf-8')
    print(f'REVIEW_HTML_READY: cards={len(imagechunks)} path={args.output}');return 0


def score(args):
    pred=records(args.predictions)
    gt_raw=records(args.gt)
    if isinstance(gt_raw,dict):gt_raw=gt_raw['results']
    # JSON review export contains nested results; handled by dedicated read.
    if Path(args.gt).suffix.lower()=='.json':
        raw=json.loads(Path(args.gt).read_text(encoding='utf-8-sig'))
        gt_raw=raw['results'] if isinstance(raw,dict) and 'results' in raw else raw
    gts={r['case_id']:r for r in gt_raw if isinstance(r,dict) and r.get('case_id')}
    verified={k:v for k,v in gts.items() if str(v.get('review_status','')).upper()=='VERIFIED'}
    out={'schema':'prd32_8class_scoring_dev_v1','total_predictions':len(pred),'gt_rows':len(gts),
        'gt_verified':len(verified),'dev_only':True,'official_acceptance':False,
        'fields':{},'per_category':{},'missing_gt':[]}
    for field in ('category','primary_color','pattern'):
        a=collections.Counter(); bycat=collections.defaultdict(lambda:collections.Counter())
        for p in pred:
            key=p.get('case_id')
            g=verified.get(key)
            if not g:
                if key not in out['missing_gt']:out['missing_gt'].append(key)
                continue
            truth=g.get('gt_'+field)
            if truth is None or not str(truth).strip():continue  # unknown not automatically scored as correct
            a['eligible']+=1; category=g.get('gt_category') or '_UNKNOWN_'
            predv=(p.get('canonical_tags') or {}).get(field)
            exact=(predv is not None and str(predv).strip()==str(truth).strip())
            a['correct']+=int(exact);a['abstained']+=int(predv is None)
            bycat[category]['eligible']+=1;bycat[category]['correct']+=int(exact)
        out['fields'][field]={'eligible':a['eligible'],'correct':a['correct'],
            'exact_accuracy':round(a['correct']/a['eligible'],4) if a['eligible'] else None,
            'abstained':a['abstained'],'metric_note':'Only independently human-verified nonempty GT fields; strict exact text match'}
        if field=='category':
            out['per_category']={c:{'eligible':bycat[c]['eligible'],'correct':bycat[c]['correct'],
                'accuracy':round(bycat[c]['correct']/bycat[c]['eligible'],4) if bycat[c]['eligible'] else None} for c in CATS}
    out['all_8_gt_categories_covered']=all(out['per_category'][c]['eligible']>0 for c in CATS)
    dump(args.output,out)
    print('SCORE_DEV_ONLY:',json.dumps(out,ensure_ascii=False));return 0


def selftest():
    assert parsebox([0,0,10,10]) and not parsebox([-1,0,2,3]) and not parsebox([0,0,0,1])
    assert not parsebox([float('nan'),0,2,3])
    assert not safe('/workspace/holdout_SEALED_DO_NOT_OPEN.json')
    r={'case_id':'foo','image_id':'im','target_instance_id':'i1','image_path':'/tmp/example.jpg',
       'target_bbox_xyxy':[0,0,11,12],'target_bbox_origin':'prd31_v5_prediction',
       'predicted_category':'top','source_split':'dev_new'}
    obj,why=normalize(r,'test.jsonl',1)
    assert obj['target_bbox_xyxy']==[0,0,11,12] and not why,(obj,why)
    r['source_split']='train';_,why=normalize(r,'test.jsonl',1);assert 'TRAIN_SOURCE_EXCLUDED' in why
    r['source_split']='dev_new';r['predicted_category']='shirt';_,why=normalize(r,'test.jsonl',1);assert 'PREDICTED_CLASS_INVALID' in why
    print('SELF_TEST_PASS: bbox, forbidden files, train leakage, invalid enum, normal data')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare');p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--exclude',action='append');p.add_argument('--skip-image-checks',action='store_true');p.add_argument('--allow-historical-dev',action='store_true')
    p=sub.add_parser('run');p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model-id',default='/workspace/models/Qwen2.5-VL-7B-Instruct');p.add_argument('--max-new-tokens',type=int,default=180)
    p.add_argument('--max-pixels',type=int,default=512*28*28)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--allow-historical-dev',action='store_true')
    p=sub.add_parser('review');p.add_argument('--predictions',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('score');p.add_argument('--predictions',required=True);p.add_argument('--gt',required=True);p.add_argument('--output',required=True)
    sub.add_parser('self-test')
    a=ap.parse_args()
    if a.cmd=='self-test':selftest();return 0
    if a.cmd=='prepare':return prepare(a)
    if a.cmd=='run':return run(a)
    if a.cmd=='review':return review(a)
    if a.cmd=='score':return score(a)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:print(type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(2)
