#!/usr/bin/env python3
"""PRD 3.2 engineering hardening of existing frozen PRD 3.1 bridge replay.

No GPU, no image reads, no pretrained weights. No changes to original model outputs.
Uses only service outputs + earlier read-only, provenance-enriched envelopes.
Never grants publication, GT verification, PRD acceptance, or 3.3 readiness.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import html
import json
import re
from pathlib import Path

TASKS={'vqa','tags','copy'}
CATS={'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}
TAG_FIELDS={'primary_color','pattern','visible_decoration','unconfirmed'}
CANON={'category','primary_color','pattern','visible_design'}
BAD_PATTERNS={'无','没有','无图案','无明显图案','none','n/a','unknown','不确定','无法确定','null','未识别'}
REFUSALS=('无法确定','不能确定','无法判断','看不清','不清楚','无法辨认','不能判断','不确定','未知','无法识别')
DEBUG_BBOX=('红框','红色框','框内','标注框','边框所指','红色边框')
CLAIM_TERMS=('纯棉','真丝','羊毛','涤纶','聚酯','蚕丝','防水','抗菌','透气性','100%','百分之百','品牌','产地')
COLOR_TOKENS=('黑色','白色','浅灰色','灰色','蓝色','红色','绿色','黄色','紫色','粉色','橙色','棕色','米色')


def filehash(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def loadjsonl(p):
    rows=[]
    for i,line in enumerate(p.read_text(encoding='utf-8-sig').splitlines(),1):
        if line.strip():
            try:r=json.loads(line)
            except json.JSONDecodeError as e:raise ValueError(f'{p.name}:{i}: invalid JSON: {e}')
            if not isinstance(r,dict):raise ValueError(f'{p.name}:{i}: row is not JSON object')
            rows.append(r)
    if not rows:raise ValueError(f'{p}: empty input')
    ids=[r.get('case_id') for r in rows]
    if any(not isinstance(v,str) or not v for v in ids) or len(set(ids))!=len(ids):
        raise ValueError(f'{p.name}: missing/duplicate case_id')
    return rows

def save_json(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def target_key(e):return (e['image_id'],e['target_instance_id'])

def text_or_none(v):return v.strip() if isinstance(v,str) and v.strip() else None

def valid_envelope(s,e):
    err=[]
    if e.get('schema')!='prd32_service_dev_envelope_v1':err.append('unexpected_provenance_envelope_schema')
    if e.get('case_id')!=s.get('case_id') or e.get('task')!=s.get('task'):err.append('identity_task_mismatch')
    if e.get('task') not in TASKS:err.append('unsupported_task')
    for key in ('image_id','target_instance_id'):
        if not isinstance(e.get(key),str) or not e.get(key):err.append(f'missing_{key}')
    if e.get('target_bbox_origin')!='prd31_frozen_prediction':err.append('bbox_is_not_prd31_frozen_prediction')
    if e.get('human_verified') is not False or e.get('ready_for_automatic_publishing') is not False:err.append('unsafe_preexisting_publication_or_verified_claim')
    if e.get('source_model_status')!=s.get('status'):err.append('source_status_disagreement')
    if e.get('payload')!=s.get('payload'):err.append('source_payload_disagreement')
    if e.get('canonical_attributes_v2')!=s.get('canonical_attributes_v2'):err.append('source_canonical_tags_disagreement')
    if not isinstance(e.get('structural_errors'),list) or e['structural_errors']:err.append('prior_structural_errors_present')
    return err

def color_disagreement(text, expected):
    if not expected or not isinstance(text,str):return False
    # Deliberately conservative: look for unequivocal main garment color phrasing only.
    colors='|'.join(sorted(map(re.escape,COLOR_TOKENS),key=len,reverse=True))
    matches=re.finditer(r'(?:一件|这件|服饰为|主体为|整体为|主色为)\s*(?:.{0,3}?)(?P<color>'+colors+r')',text)
    return any(m.group('color')!=expected for m in matches)

def adapt(s,e,tag_by_target):
    errors=valid_envelope(s,e)
    flags=set(e.get('review_flags') or [])
    flags.add('not_human_verified')
    task=e.get('task'); raw=s.get('payload'); canonical=s.get('canonical_attributes_v2')
    candidate=None; disposition='REVIEW_REQUIRED'; publisher_blocked=True
    if task=='vqa':
        if s.get('status')!='DRAFT_UNVERIFIED':errors.append('invalid_vqa_status')
        value=text_or_none(raw)
        if value is None:errors.append('empty_vqa_answer')
        elif any(term in value for term in REFUSALS):
            disposition='ABSTAINED_UNVERIFIED';flags.add('vqa_abstention_not_answer_correct')
        else:candidate={'answer':value}
    elif task=='copy':
        if s.get('status')!='DRAFT_UNVERIFIED':errors.append('invalid_copy_status')
        value=text_or_none(raw)
        if value is None:errors.append('empty_copy_draft')
        else:
            if any(x in value for x in DEBUG_BBOX):flags.add('copy_contains_debug_target_marker')
            if any(x in value for x in CLAIM_TERMS):flags.add('copy_unverified_material_function_brand')
            if len(value)>80:flags.add('copy_exceeds_dev_default_80_chars')
            pair=tag_by_target.get(target_key(e))
            if pair and color_disagreement(value,pair.get('primary_color')):
                flags.add('copy_conflicts_with_same_target_tags_main_color')
            if 'copy_contains_debug_target_marker' not in flags and not errors:
                candidate={'copy_draft':value}  # review only, NEVER public API approved
            else: disposition='BLOCKED_FOR_DISPLAY_REWRITE'
    elif task=='tags':
        if s.get('status')!='CANONICALIZED_NOT_HUMAN_VERIFIED':errors.append('invalid_tags_status')
        if not isinstance(raw,dict) or set(raw)!=TAG_FIELDS:errors.append('tags_shape_invalid')
        if not isinstance(canonical,dict) or set(canonical)!=CANON:errors.append('canonical_shape_invalid')
        if isinstance(raw,dict) and set(raw)==TAG_FIELDS:
            if not isinstance(raw['unconfirmed'],list) or any(not isinstance(x,str) for x in raw['unconfirmed']):
                errors.append('invalid_unconfirmed_type')
            elif sorted(raw['unconfirmed'])!=sorted(k for k in TAG_FIELDS-{'unconfirmed'} if raw[k] is None):
                errors.append('unconfirmed_null_mismatch')
            for k in TAG_FIELDS-{'unconfirmed'}:
                if raw[k] is not None and not isinstance(raw[k],str):errors.append('invalid_field_type_'+k)
        if isinstance(canonical,dict) and set(canonical)==CANON:
            if canonical['category'] is not None and canonical['category'] not in CATS:errors.append('invalid_canonical_category')
            for k in CANON-{'category'}:
                if canonical[k] is not None and not isinstance(canonical[k],str):errors.append('invalid_canonical_value_'+k)
        if not errors:
            candidate={'category':canonical['category'],'primary_color':raw['primary_color'],
                       'pattern':raw['pattern'],'visible_decoration':raw['visible_decoration'],
                       'visible_design':canonical['visible_design']}
            if isinstance(raw['pattern'],str) and raw['pattern'].strip().lower() in BAD_PATTERNS:
                candidate['pattern']=None
                flags.add('pattern_ambiguous_abstained_for_client_candidate')
            if raw['visible_decoration'] is None:flags.add('decoration_unconfirmed_not_absent')
            candidate['unconfirmed']=sorted(k for k in ('category','primary_color','pattern','visible_decoration','visible_design') if candidate[k] is None)
    else:errors.append('unsupported_task')
    if errors:
        disposition='STRUCTURAL_BLOCKED';candidate=None
    result={
      'schema':'prd32_development_service_contract_v2','case_id':e.get('case_id'),
      'task':task,'provenance':{'image_id':e.get('image_id'),'target_instance_id':e.get('target_instance_id'),
                              'target_bbox_origin':e.get('target_bbox_origin'),
                              'historical_frozen_bridge_replay':True},
      'source_status':s.get('status'),'disposition':disposition,
      'review_candidate':candidate,'original_model_or_adapter_payload':raw,
      'original_canonical_attributes_v2':canonical,
      'checks':{'structural_valid':not errors,'errors':sorted(set(errors)),'review_flags':sorted(flags)},
      'human_verified':False,'model_output_is_gt':False,'ready_for_automatic_publishing':False,
      'ready_for_prd_acceptance':False,'downstream_contract_signed_off':False,
    }
    return result

def render_review(path,records):
    trs=[]
    for r in records:
        flags=', '.join(r['checks']['review_flags'])
        raw=r['original_model_or_adapter_payload']
        show=json.dumps(raw,ensure_ascii=False) if isinstance(raw,dict) else str(raw)
        trs.append('<tr><td>'+html.escape(str(r['case_id']))+'</td><td>'+html.escape(str(r['task']))+'</td>'
                   '<td>'+html.escape(str(r['provenance']['image_id']))+' / '+html.escape(str(r['provenance']['target_instance_id']))+'</td>'
                   '<td>'+html.escape(str(r['disposition']))+'</td><td>'+html.escape(show)+'</td><td>'+html.escape(flags)+'</td></tr>')
    txt='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PRD 3.2 frozen-bridge engineering gate</title>
<style>body{background:#edf2f7;color:#263446;font:14px/1.5 system-ui;max-width:1300px;margin:auto;padding:25px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border:1px solid #cbd5e1;vertical-align:top;word-break:break-word}th{background:#e4ecf8}td:nth-child(5){max-width:350px}.notice{background:white;padding:18px;border-radius:10px}</style>
<h1>PRD 3.2 · 冻结 3.1 预测框回放 / 接口安全检查</h1><div class="notice">仅使用旧 12 条服务输出与来源关联数据。未运行 V5 或 Qwen；未打开图片；未证明类别准确率。此页不可作为人工视觉 GT 审核或正式 PRD 验收。<br>所有内容未人工核验，永不自动发布；异常文案需人工修改后复核。</div>
<table><thead><tr><th>case_id</th><th>task</th><th>预测目标来源</th><th>服务处置</th><th>原始值</th><th>复核标志</th></tr></thead><tbody>'''+''.join(trs)+'</tbody></table></html>'
    path.write_text(txt,encoding='utf-8')

def verify_frozen_bridge_metadata(rows, envelopes, *, expected_hash=None, bridge_path=None):
    if bridge_path is None:
        return {'checked':False, 'status':'NOT_REVERIFIED_AGAINST_FROZEN_BRIDGE',
                'warning':'Historical read-only envelope alone is not cryptographic provenance authentication'}
    digest=filehash(bridge_path)
    if expected_hash is not None and digest.lower()!=expected_hash.lower():
        raise ValueError('Frozen bridge SHA256 mismatch')
    bridge_rows=loadjsonl(bridge_path)
    bm={r['case_id']:r for r in bridge_rows}
    unknown=[e['case_id'] for e in envelopes if e['case_id'] not in bm]
    if unknown:raise ValueError('Case IDs absent from frozen bridge: '+','.join(unknown))
    warnings=[]
    for e in envelopes:
        cid=e['case_id'];b=bm[cid]
        for k in ('case_id','task','image_id','target_instance_id','target_bbox_origin'):
            if b.get(k)!=e.get(k if k!='target_bbox_origin' else 'target_bbox_origin'):
                raise ValueError(f'{cid}: frozen bridge metadata mismatch at {k}')
        if b.get('schema')!='prd32_engine_neutral_input_v1':raise ValueError(f'{cid}: unexpected bridge schema')
        if b.get('split') not in (None,'development','dev'):
            raise ValueError(f'{cid}: bridge split is not development')
        path=str(b.get('image_path','')).lower()
        if any(part in path for part in ('holdout','sealed_do_not_open','/sealed/')):
            raise ValueError(f'{cid}: a sealed or holdout path is referenced by bridge')
        if '/test/' in path or '/val/' in path or 'fullval' in path:
            warnings.append(cid+': path has test/val segment; development-source permission must be established separately')
        coords=b.get('target_bbox_xyxy')
        if not isinstance(coords,list) or len(coords)!=4 or any(not isinstance(v,(int,float)) for v in coords):
            raise ValueError(f'{cid}: bad frozen predicted bbox metadata')
        if not (0<=coords[0]<coords[2] and 0<=coords[1]<coords[3]):
            raise ValueError(f'{cid}: bad frozen predicted bbox geometry')
    return {'checked':True,'status':'CASE_METADATA_MATCHED_FROZEN_BRIDGE_NOT_LIVE_INFERENCE',
            'bridge_sha256':digest,'checked_cases':len(envelopes),'source_path_warnings':warnings,
            'images_or_checkpoint_opened':False}

def run(sfile,efile,out,bridge=None,expected_bridge_hash=None):
    if out.exists():raise ValueError('Refuse to overwrite existing output: '+str(out))
    services=loadjsonl(sfile);envs=loadjsonl(efile)
    se={r['case_id']:r for r in services};en={r['case_id']:r for r in envs}
    if se.keys()!=en.keys():raise ValueError('service/envelope IDs mismatch, abort')
    # Verify every source row identity before producing any candidate service output.
    errs=[(cid,valid_envelope(se[cid],en[cid])) for cid in se]
    if any(items for _,items in errs):raise ValueError('Provenance source mismatch: '+str([(c,e) for c,e in errs if e]))
    bridge_check=verify_frozen_bridge_metadata(services,envs,expected_hash=expected_bridge_hash,bridge_path=bridge)
    tag_by_target={}
    for e in envs:
        if e['task']=='tags':
            key=target_key(e)
            if key in tag_by_target:raise ValueError('Conflicting tags on same target identity')
            tag_by_target[key]=se[e['case_id']].get('canonical_attributes_v2') or {}
    results=[adapt(s,en[s['case_id']],tag_by_target) for s in services]
    out.mkdir(parents=True,exist_ok=False)
    with (out/'dev_contract_v2.jsonl').open('w',encoding='utf-8') as f:
        for r in results:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    with (out/'engineering_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['case_id','task','image_id','target_instance_id','disposition','structural_valid','review_flags','structural_errors','human_verified','can_publish']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in results:
            w.writerow(dict(case_id=r['case_id'],task=r['task'],image_id=r['provenance']['image_id'],
                target_instance_id=r['provenance']['target_instance_id'],disposition=r['disposition'],
                structural_valid=r['checks']['structural_valid'],review_flags=';'.join(r['checks']['review_flags']),
                structural_errors=';'.join(r['checks']['errors']),human_verified=False,can_publish=False))
    c=collections.Counter(r['disposition'] for r in results)
    task=collections.Counter(r['task'] for r in results)
    flag=collections.Counter(x for r in results for x in r['checks']['review_flags'])
    structurals=sum(not r['checks']['structural_valid'] for r in results)
    summary={
      'schema':'prd32_development_service_hardening_gate_v2',
      'status':'DEV_ENVELOPES_GENERATED_REVIEW_REQUIRED' if not structurals else 'STRUCTURAL_REVIEW_BLOCKED',
      'total':len(results),'tasks':dict(task),'dispositions':dict(c),'structural_errors':structurals,
      'review_flags':dict(flag),'vqa_abstentions':sum(r['disposition']=='ABSTAINED_UNVERIFIED' for r in results),
      'copies_blocked_by_debug_or_structure':sum(r['task']=='copy' and r['disposition']!='REVIEW_REQUIRED' for r in results),
      'tag_pattern_ambiguous_abstentions':sum('pattern_ambiguous_abstained_for_client_candidate' in r['checks']['review_flags'] for r in results),
      'source_files_sha256':{'service_outputs':filehash(sfile),'read_only_envelopes':filehash(efile)},
      'frozen_bridge_metadata_check':bridge_check,
      'review_only':True,'all_publishing_disabled':True,'model_calls':0,'image_reads':0,'holdout_opened':False,
      'freeze_stage':'historical_prd31_frozen_prediction_bridge_replay_not_new_v5_inference',
      'acceptance':False,
      'caution':'Static provenance and text/schema checks. Not an API server, not human QA/copy quality evaluation, not PRD acceptance.'
    }
    save_json(out/'engineering_summary.json',summary)
    render_review(out/'Review_Engineering_Cases.html',results)
    print('DEV_SERVICE_GATE_COMPLETE',json.dumps({'total':len(results),'task_counts':dict(task),'dispositions':dict(c),
                   'structural_errors':structurals,'output':str(out)},ensure_ascii=False))
    return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--service',type=Path,required=True)
    ap.add_argument('--envelopes',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--bridge',type=Path,help='Optional frozen bridge_inputs.jsonl metadata join; no images read')
    ap.add_argument('--expected-bridge-sha256',help='Optional expected SHA256 for bridge metadata')
    a=ap.parse_args()
    if not a.service.is_file() or not a.envelopes.is_file():ap.error('Missing input files')
    if a.expected_bridge_sha256 and not a.bridge:ap.error('--expected-bridge-sha256 requires --bridge')
    if a.bridge and not a.bridge.is_file():ap.error('Frozen bridge file is missing')
    try:run(a.service,a.envelopes,a.output,a.bridge,a.expected_bridge_sha256)
    except (ValueError,KeyError) as e:ap.error(str(e))

if __name__=='__main__':main()
