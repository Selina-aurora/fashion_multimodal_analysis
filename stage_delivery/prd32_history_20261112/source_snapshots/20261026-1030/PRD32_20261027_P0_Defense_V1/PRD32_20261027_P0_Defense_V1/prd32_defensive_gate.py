#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD 3.2 boundary case guard (development). No GPU, no model inference.

Use guard_input_record() at the boundary of a 3.1->3.2 runner, before calling
model inference. It fails closed for uncertain target identity/geometry. Use
sanitize_service_output() after model inference, before releasing a response.

Command line audits/exports guarded dev requests and canonical candidate
service results. Synthetic --self-test uses only temporary generated images.
No freeze metadata or original inputs are modified. Sealed inputs are blocked.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

SCHEMA = 'prd32_engine_neutral_input_v1'
TASKS = {'vqa','tags','copy'}
CLASSES = {'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}
BAD_TEXT = {'', 'none', 'null', 'unknown', 'n/a', 'nan', '无', '无设计', '无明显设计', '无法确定', '不确定'}
BAD_PATTERN = BAD_TEXT | {'无图案','没有图案','蕾丝','lace','无明显图案'}
PATTERNS = {'纯色','条纹','格纹','人字纹','波点','花卉','花卉印花','文字印花','字母印花','印花','几何','动物纹','拼色'}
DEBUG_TERMS = ('红框','红色框','框内','标注框')
MATERIAL_TERMS = ('蕾丝','真丝','纯棉','羊毛成分','涤纶','皮革材质','亚麻材质','丝绸材质')
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 40 * 1000 * 1000


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def restricted(path):
    s = str(path).replace('\\','/').lower()
    return any(part in s for part in ('holdout','sealed_do_not_open','/sealed/'))


def err(code, detail=''):
    return {'code':code,'detail':str(detail)[:200]}


def guard_input_record(rec, *, image_check=True):
    """Return {decision, issues, guarded_record, identity}; no exceptions for bad records.

    BLOCKED cases MUST NOT be sent to image inference. WARN-only cases may be
    inferred with no unverified 3.1 attribute context passed as ground truth.
    """
    issues=[]
    identity={}
    if not isinstance(rec,dict):
        return {'decision':'BLOCKED','issues':[err('RECORD_NOT_OBJECT')], 'guarded_record':None,'identity':{}}
    cid=rec.get('case_id')
    if isinstance(cid,str) and cid.strip(): identity['case_id']=cid.strip()
    else: issues.append(err('CASE_ID_MISSING'))
    for k in ('image_id','target_instance_id'):
        v=rec.get(k)
        if isinstance(v,str) and v.strip(): identity[k]=v.strip()
        else: issues.append(err(k.upper()+'_MISSING'))
    task=rec.get('task')
    identity['task']=task if isinstance(task,str) else None
    if rec.get('schema') != SCHEMA: issues.append(err('INVALID_BRIDGE_SCHEMA'))
    if task not in TASKS: issues.append(err('INVALID_TASK_ENUM',task))
    if rec.get('target_bbox_origin')!='prd31_frozen_prediction':
        issues.append(err('BBOX_ORIGIN_NOT_FROZEN_PREDICTION'))
    if rec.get('split','development') not in ('development','dev'):
        issues.append(err('NON_DEVELOPMENT_SPLIT'))
    query=rec.get('query')
    if not isinstance(query,str) or not query.strip(): issues.append(err('QUERY_MISSING'))
    elif len(query)>3000: issues.append(err('QUERY_TOO_LONG'))

    image_str=rec.get('image_path')
    image_path=None
    if not isinstance(image_str,str) or not image_str.strip(): issues.append(err('IMAGE_PATH_MISSING'))
    elif restricted(image_str): issues.append(err('RESTRICTED_IMAGE_PATH'))
    else:
        image_path=Path(image_str)
        if not image_path.is_absolute() or image_path.suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):
            issues.append(err('BAD_IMAGE_PATH'))
        elif restricted(image_path.resolve()):
            issues.append(err('RESTRICTED_RESOLVED_IMAGE_PATH'))
            image_path=None
    sh=rec.get('image_sha256')
    if not isinstance(sh,str) or not re.fullmatch(r'[0-9a-fA-F]{64}',sh):
        issues.append(err('IMAGE_SHA256_INVALID'))
    size=rec.get('source_size')
    has_size=isinstance(size,(list,tuple)) and len(size)==2 and all(type(x) is int and x>0 for x in size)
    # Real frozen bridge_inputs.jsonl omits source_size. It is optional ONLY
    # when the image is opened and its actual dimensions are checked.
    if size is not None and not has_size: issues.append(err('SOURCE_SIZE_INVALID'))
    if size is None and not image_check: issues.append(err('DIMENSIONS_UNVERIFIED_NO_IMAGE_CHECK'))
    if has_size and size[0]*size[1]>MAX_IMAGE_PIXELS: issues.append(err('SOURCE_IMAGE_TOO_LARGE'))

    bbox=rec.get('target_bbox_xyxy')
    bbox_ok=isinstance(bbox,(list,tuple)) and len(bbox)==4 and all(type(v) in (int,float) and math.isfinite(v) for v in bbox)
    if not bbox_ok: issues.append(err('BBOX_EMPTY_OR_INVALID'))
    else:
        x1,y1,x2,y2 = bbox
        if not (x1 < x2 and y1 < y2): issues.append(err('BBOX_NON_POSITIVE_AREA'))
        if x1<0 or y1<0: issues.append(err('BBOX_OUT_OF_BOUNDS'))
        if has_size and (x2>size[0] or y2>size[1]): issues.append(err('BBOX_OUT_OF_BOUNDS'))
    # Do not silently clamp an uncertain target box, particularly multi-garment images.
    category_hints=[]
    for key in ('target_category','predicted_category','predicted_target_category'):
        if key in rec and rec[key] is not None:
            v=rec[key]
            if not isinstance(v,str) or v not in CLASSES: issues.append(err('INVALID_CATEGORY_ENUM',key))
            else: category_hints.append(v)
    if len(set(category_hints))>1: issues.append(err('CATEGORY_HINT_CONFLICT'))
    attrs=rec.get('predicted_attribute_context')
    if attrs is not None and not isinstance(attrs,list):
        issues.append(err('MALFORMED_PREDICTED_ATTR_CONTEXT'))
    elif isinstance(attrs,list):
        for item in attrs:
            if not isinstance(item,dict) or not isinstance(item.get('name'),str):
                issues.append(err('MALFORMED_PREDICTED_ATTR_ITEM'));break
    # Compare the claimed image size/hash with actual bytes on the real AutoDL data volume.
    if image_check and image_path is not None and not restricted(image_str):
        try:
            if not image_path.is_file(): issues.append(err('IMAGE_NOT_FOUND'))
            elif image_path.stat().st_size > MAX_IMAGE_BYTES: issues.append(err('IMAGE_TOO_LARGE'))
            else:
                from PIL import Image
                with Image.open(image_path) as im:
                    w,h = im.size
                    if w*h>MAX_IMAGE_PIXELS: issues.append(err('IMAGE_PIXELS_EXCEED_LIMIT'))
                    elif has_size and im.size != tuple(size):
                        issues.append(err('ACTUAL_IMAGE_DIMENSION_MISMATCH'))
                    if bbox_ok and (bbox[0]<0 or bbox[1]<0 or bbox[2]>w or bbox[3]>h):
                        issues.append(err('BBOX_OUT_OF_BOUNDS_ACTUAL_IMAGE'))
                    im.verify()
                if isinstance(sh,str) and re.fullmatch(r'[0-9a-fA-F]{64}',sh):
                    if sha256_file(image_path).lower()!=sh.lower(): issues.append(err('IMAGE_IDENTITY_HASH_MISMATCH'))
        except (OSError,ValueError,ImportError) as ex:
            issues.append(err('IMAGE_READ_ERROR',type(ex).__name__))
    # Conservatively treat malformed/unverified 3.1 attributes as unsupported hints.
    warn={'MALFORMED_PREDICTED_ATTR_CONTEXT','MALFORMED_PREDICTED_ATTR_ITEM'}
    blocked=any(x['code'] not in warn for x in issues)
    if blocked:
        return {'decision':'BLOCKED','issues':issues,'guarded_record':None,'identity':identity}
    cleaned=copy.deepcopy(rec)
    # Strong default: do not pass historical predicted attributes to the model.
    cleaned['predicted_attribute_context']=[]
    cleaned['guard_metadata']={'upstream_context_dropped': True,
                               'bbox_policy':'preserved_if_valid_never_silently_clamped',
                               'upstream_fields_are_not_ground_truth':True}
    return {'decision':'PASS_WITH_WARNINGS' if issues else 'PASS',
            'issues':issues,'guarded_record':cleaned,'identity':identity}


def nullable_string(value, *, allowed=None, bad=BAD_TEXT, label='VALUE'):
    if value is None: return None, []
    if not isinstance(value,str): return None,[err(label+'_INVALID_TYPE')]
    s=value.strip()
    if s.lower() in bad: return None,[err(label+'_ABSTAINED')]
    if allowed is not None and s not in allowed: return None,[err(label+'_INVALID_ENUM',s)]
    return s,[]


def sanitize_service_output(service, prediction=None, input_guard=None):
    """Conservative service candidate. Does NOT certify QA/copy or attribute GT."""
    pred=prediction if isinstance(prediction,dict) else {}
    s=service if isinstance(service,dict) else {}
    cid = s.get('case_id') or pred.get('case_id')
    task = s.get('task') or pred.get('task')
    source = input_guard.get('identity',{}) if isinstance(input_guard,dict) else {}
    issues=[]
    if not isinstance(cid,str) or not cid: issues.append(err('SERVICE_CASE_ID_MISSING'))
    if task not in TASKS: issues.append(err('SERVICE_TASK_INVALID'))
    if pred and s.get('case_id')!=pred.get('case_id'):issues.append(err('CASE_PROVENANCE_MISMATCH'))
    if pred and s.get('task') not in (None,pred.get('task')): issues.append(err('TASK_PROVENANCE_MISMATCH'))
    if source.get('case_id') and cid != source['case_id']: issues.append(err('INPUT_SERVICE_IDENTITY_MISMATCH'))
    provenance={k:source.get(k) or pred.get(k) for k in ('image_id','target_instance_id')}
    provenance['case_id']=cid
    provenance['task']=task
    if any(k.endswith('MISMATCH') for k in [x['code'] for x in issues]):
        return {'case_id':cid,'task':task,'status':'BLOCKED_PROVENANCE', 'payload':None,
                'issues':issues,'provenance':provenance,'unverified':True}
    if input_guard is not None and input_guard.get('decision')=='BLOCKED':
        return {'case_id':cid,'task':task,'status':'BLOCKED_INVALID_INPUT','payload':None,
                'issues':input_guard.get('issues',[]),'provenance':provenance,'unverified':True}
    if task == 'tags':
        obj=s.get('canonical_attributes_v2')
        if not isinstance(obj,dict):
            # The bridge adapter payload is narrower; do not invent a class.
            obj=s.get('payload') if isinstance(s.get('payload'),dict) else {}
            issues.append(err('FULL_TAG_SCHEMA_MISSING'))
        cat= obj.get('category')
        if cat is not None and (not isinstance(cat,str) or cat not in CLASSES):
            cat=None;issues.append(err('INVALID_CATEGORY_ENUM'))
        color,e=nullable_string(obj.get('primary_color'), label='COLOR');issues+=e
        pattern,e=nullable_string(obj.get('pattern'), allowed=PATTERNS,bad=BAD_PATTERN,label='PATTERN');issues+=e
        design,e=nullable_string(obj.get('visible_design'), label='DESIGN');issues+=e
        decoration,e=nullable_string(obj.get('visible_decoration'), label='DECORATION');issues+=e
        # 'visible_decoration' from model text has no GT; keep candidate at most.
        payload={'category':cat,'primary_color':color,'pattern':pattern,
                 'visible_design':design,'visible_decoration':decoration}
        payload['unconfirmed']=[k for k,v in payload.items() if v is None]
        return {'case_id':cid,'task':task, 'status':'DEGRADED_REVIEW_REQUIRED' if issues else 'CANDIDATE_UNVERIFIED',
                'payload':payload,'issues':issues,'provenance':provenance,'unverified':True}
    if task in ('vqa','copy'):
        raw=s.get('payload')
        if not isinstance(raw,str) or not raw.strip():
            issues.append(err('EMPTY_TEXT_OUTPUT'));raw=None
        else:
            raw=raw.strip()
            if task=='copy':
                if len(raw)>150:issues.append(err('COPY_TOO_LONG'))
                if any(w in raw for w in DEBUG_TERMS):issues.append(err('DEBUG_WORDING_IN_COPY'))
                if any(w in raw for w in MATERIAL_TERMS):issues.append(err('UNVERIFIED_MATERIAL_CLAIM'))
            if task=='vqa' and len(raw)>1500: issues.append(err('QA_TOO_LONG'))
        # If there are content red flags, hold for review; do not release candidate as verified.
        status='CONTENT_HOLD_FOR_REVIEW' if issues else 'DRAFT_UNVERIFIED'
        return {'case_id':cid,'task':task,'status':status,'payload':None if issues else raw,
                'issues':issues,'provenance':provenance,'unverified':True}
    return {'case_id':cid,'task':task,'status':'BLOCKED_INVALID_SERVICE','payload':None,
            'issues':issues or [err('SERVICE_TASK_MISSING')], 'provenance':provenance,'unverified':True}


def read_jsonl_lenient(path):
    lines=[]
    with open(path, encoding='utf-8') as fp:
        for num,line in enumerate(fp,1):
            if not line.strip():continue
            try: row=json.loads(line)
            except json.JSONDecodeError:
                row={'case_id':f'__invalid_json_line_{num}', '_malformed_line':True}
            lines.append(row)
    return lines


def write_jsonl(path, rows):
    with open(path,'w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def audit(bridge, output, *, service=None, predictions=None, image_check=True):
    if any(restricted(p) for p in (bridge, output, service or '', predictions or '')):
        raise ValueError('Sealed/holdout path forbidden')
    input_records=read_jsonl_lenient(bridge)
    counts=Counter(str(x.get('case_id')) for x in input_records if isinstance(x,dict))
    guarded=[]; decisions=[]; by_id={}
    for i,record in enumerate(input_records,1):
        g=guard_input_record(record, image_check=image_check)
        cid=g['identity'].get('case_id') or f'__invalid_line_{i}'
        if counts[cid]>1:
            g['decision']='BLOCKED';g['guarded_record']=None
            g['issues'].append(err('DUPLICATE_CASE_ID'))
        decisions.append({'line':i,'case_id':cid,'task':g['identity'].get('task'), 'decision':g['decision'],
                          'issues':[x['code'] for x in g['issues']]})
        by_id[cid]=g
        if g['guarded_record'] is not None: guarded.append(g['guarded_record'])
    service_rows=[]
    if service is not None:
        original_service=read_jsonl_lenient(service)
        prediction_map={r['case_id']:r for r in read_jsonl_lenient(predictions) if isinstance(r,dict) and isinstance(r.get('case_id'),str)} if predictions else {}
        service_id_counts=Counter(sr.get('case_id') for sr in original_service if isinstance(sr,dict))
        for sr in original_service:
            cid=sr.get('case_id') if isinstance(sr,dict) else None
            if service_id_counts[cid]>1:
                service_rows.append({'case_id':cid,'task':sr.get('task'),'status':'BLOCKED_DUPLICATE_SERVICE_ID','payload':None,
                                     'issues':[err('DUPLICATE_SERVICE_CASE_ID')],'unverified':True})
                continue
            if cid not in by_id:
                service_rows.append({'case_id':cid,'task':sr.get('task'),'status':'BLOCKED_ORPHAN_SERVICE_ID','payload':None,
                                     'issues':[err('SERVICE_ID_NOT_IN_BRIDGE')],'unverified':True})
                continue
            service_rows.append(sanitize_service_output(sr,prediction_map.get(cid),by_id.get(cid)))
        # Emit controlled fallbacks for input requests that cannot run at all.
    fallback=[]
    for i,g in enumerate(decisions):
        if g['decision']=='BLOCKED':
            fallback.append({'case_id':g['case_id'],'task':g['task'],'status':'BLOCKED_INVALID_INPUT',
                             'payload':None,'unconfirmed':True,'issues':g['issues'],
                             'note':'No model inference; no invented bbox/category/attribute.'})
    out=Path(output)
    if out.exists():raise FileExistsError('Output already exists: '+str(out))
    out.mkdir(parents=True)
    write_jsonl(out/'accepted_bridge_candidates.jsonl',guarded)
    write_jsonl(out/'fallback_blocked_cases.jsonl',fallback)
    write_jsonl(out/'input_audit.jsonl',decisions)
    if service is not None:write_jsonl(out/'service_candidates_guarded.jsonl',service_rows)
    summ={'status':'DEFENSIVE_BOUNDARY_AUDIT_COMPLETED_NOT_PRD_ACCEPTED',
          'input_cases':len(decisions),'input_pass':len(guarded),'input_blocked':len(fallback),
          'input_issue_counts':dict(Counter(issue for x in decisions for issue in x['issues'])),
          'service_checked':len(service_rows),
          'service_status_counts':dict(Counter(r.get('status') for r in service_rows)),
          'freeze_not_modified':True,'model_inference_performed':False,'sealed_holdout_used':False,
          'note':'Deploy guard_input_record at live per-case ingress to isolate bad inputs. This CLI is only an offline audit.'}
    (out/'summary.json').write_text(json.dumps(summ,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summ,ensure_ascii=False,indent=2))
    return summ


def synthetic_tests():
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix='prd32_p0_') as tmp:
        p=Path(tmp)/'example.jpg'
        Image.new('RGB',(80,100),'white').save(p)
        base={'schema':SCHEMA,'case_id':'good','task':'tags','split':'development',
              'image_id':'demo_img','target_instance_id':'pred_0001','image_path':str(p),
              'image_sha256':sha256_file(p),'source_size':[80,100],
              'target_bbox_origin':'prd31_frozen_prediction','target_bbox_xyxy':[5,6,70,90],
              'query':'What is the color?', 'predicted_attribute_context':[]}
        assert guard_input_record(base)['decision']=='PASS'
        without_size=dict(base);without_size.pop('source_size')
        assert guard_input_record(without_size,image_check=True)['decision']=='PASS'
        assert guard_input_record(without_size,image_check=False)['decision']=='BLOCKED'
        tests={'empty_bbox':{'target_bbox_xyxy':None},
               'zero_area':{'target_bbox_xyxy':[0,2,0,5]},
               'out_of_bounds':{'target_bbox_xyxy':[0,0,81,60]},
               'negative':{'target_bbox_xyxy':[-1,0,40,60]},
               'nan':{'target_bbox_xyxy':[float('nan'),0,40,60]},
               'missing_instance':{'target_instance_id':None},
               'invalid_class':{'target_category':'shirt'},
               'class_conflict':{'target_category':'top','predicted_category':'outerwear'},
               'hash_mismatch':{'image_sha256':'0'*64},
               'invalid_task':{'task':'hack'},
               'missing_image':{'image_path':str(Path(tmp)/'missing.jpg')}}
        results={}
        for label,patch in tests.items():
            row=dict(base,**patch);row['case_id']=label
            result=guard_input_record(row)
            assert result['decision']=='BLOCKED',(label,result)
            results[label]='BLOCKED'
        record=dict(base,case_id='soft',predicted_attribute_context='bad')
        soft=guard_input_record(record)
        assert soft['decision']=='PASS_WITH_WARNINGS' and soft['guarded_record']['predicted_attribute_context']==[]
        s={'case_id':'good','task':'tags','canonical_attributes_v2':{'category':'shirt','primary_color':'黑色',
                                                                  'pattern':'无','visible_design':'无设计'}}
        safe=sanitize_service_output(s)
        assert safe['payload']['category'] is None and safe['payload']['pattern'] is None and safe['payload']['visible_design'] is None
        s2={'case_id':'good','task':'copy','payload':'红框内是一件蕾丝衣服'}
        assert sanitize_service_output(s2)['payload'] is None
        # Per-record JSONL input isolation: one invalid line must not abort a whole batch.
        source=Path(tmp)/'sample.jsonl'
        source.write_text(json.dumps(base)+'\nnot-json\n'+json.dumps(dict(base,case_id='bad',target_bbox_xyxy=None))+'\n',encoding='utf-8')
        audit_result=audit(source,Path(tmp)/'audit',image_check=True)
        assert audit_result['input_pass']==1 and audit_result['input_blocked']==2
        print('SELF_TEST_PASS:',json.dumps({'valid':1,'malformed_json_blocked':1,'bad_bbox_blocked':1,
                                             'synthetic_bad_inputs_blocked':len(tests),
                                             'soft_context_abstained':1,'invalid_tag_values_abstained':3,
                                             'risky_copy_held':1},ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bridge',help='Frozen bridge_inputs.jsonl or a development JSONL')
    ap.add_argument('--service',help='Existing service_outputs.jsonl (optional)')
    ap.add_argument('--predictions',help='Original predictions.jsonl (optional)')
    ap.add_argument('--output',help='New output directory (never overwrites)')
    ap.add_argument('--no-image-check',action='store_true',help='Diagnostic only; cannot certify image identity')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:synthetic_tests();return
    if not args.bridge or not args.output:ap.error('--bridge and --output are required unless --self-test')
    audit(args.bridge,args.output,service=args.service,predictions=args.predictions,image_check=not args.no_image_check)

if __name__=='__main__':
    main()
