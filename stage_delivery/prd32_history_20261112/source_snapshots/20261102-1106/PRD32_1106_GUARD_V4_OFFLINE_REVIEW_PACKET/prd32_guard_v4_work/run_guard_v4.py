"""Build V4 offline risk/audit/review package from immutable Guard V3 result zip.
No network, GPU, image decoding, model inference, or mainline modification.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

DATA_CSV = 'dev60_171_identity_class_routing.csv'
MASK_CSV = 'mask_geometry_probe/mask_geometry_171.csv'
SUM_JSON = 'risk_guard_v3_summary.json'
MASK_SUM_JSON = 'mask_geometry_probe/mask_geometry_summary.json'
CSV_FIELDS = ('case_id','instance_id','image_sha256','v5_category_unverified',
              'qwen_category_unverified','category_agreement','user_chat_attested_30',
              'shadow_alias_candidate','shadow_alias_destination','routing_action')


def csv_bytes(content):
    return list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def read_source(src):
    with ZipFile(src) as z:
        names = set(z.namelist())
        required = {DATA_CSV, MASK_CSV, SUM_JSON, MASK_SUM_JSON}
        if not required.issubset(names):
            raise ValueError(f'Archive missing files: {sorted(required - names)}')
        return (csv_bytes(z.read(DATA_CSV)), csv_bytes(z.read(MASK_CSV)),
                json.loads(z.read(SUM_JSON)), json.loads(z.read(MASK_SUM_JSON)))


def build(src: Path, out: Path):
    rows, geom, summary, geom_summary = read_source(src)
    out.mkdir(parents=True, exist_ok=True)
    issues=[]
    def require(cond, issue):
        if not cond: issues.append(issue)
    ids = [(r.get('case_id',''), r.get('instance_id','')) for r in rows]
    gm_ids = [(r.get('case_id',''),r.get('instance_id','')) for r in geom]
    require(len(set(ids))==len(ids), 'DUPLICATE_SOURCE_INSTANCE_ID')
    require(len(set(gm_ids))==len(gm_ids), 'DUPLICATE_GEOM_INSTANCE_ID')
    require(set(ids)==set(gm_ids), 'MASK_CSV_INSTANCE_SET_MISMATCH')
    require(len(rows)==summary.get('source_v5_instances'), 'SOURCE_V5_INSTANCE_TOTAL_MISMATCH')
    require(len(set(x[0] for x in ids))==summary.get('source_images'), 'SOURCE_IMAGE_TOTAL_MISMATCH')
    require(all(all(k in r for k in CSV_FIELDS) for r in rows), 'CSV_REQUIRED_COLUMNS_MISSING')
    require(summary.get('mode')=='OFFLINE_SIDECAR_NO_CHANGE', 'SOURCE_NOT_OFFLINE_SIDECAR')
    require(summary.get('independent_holdout_opened') is False, 'HOLDOUT_OPENED')
    require(summary.get('mainline_deployment_approved') is False, 'MAINLINE_APPROVED_UNEXPECTEDLY')
    require(summary.get('raw_records_changed')==0, 'RAW_SOURCE_RECORDS_CHANGED')
    require(summary.get('qwen_answers_moved')==0, 'QWEN_ANSWERS_MOVED')
    require(summary.get('published_categories')==0, 'CATEGORY_PUBLICATIONS_PRESENT')
    require(geom_summary.get('holdout_opened') is False, 'MASK_PROBE_HOLDOUT_OPENED')
    require(geom_summary.get('raw_masks_modified')==0, 'RAW_MASKS_MODIFIED')
    require(all(g.get('status') == 'GEOMETRY_COMPATIBLE' for g in geom), 'GEOMETRY_STATUS_NOT_ALL_COMPATIBLE')
    bykey = {(r.get('case_id'), r.get('instance_id')): r for r in rows}
    for r in rows:
        key=(r.get('case_id'),r.get('instance_id'))
        c,i=key
        require(bool(re.fullmatch(r'[0-9a-f]{64}',r.get('image_sha256',''))), f'BAD_SHA256:{c}:{i}')
        require((r.get('category_agreement')=='YES')==(r.get('v5_category_unverified')==r.get('qwen_category_unverified')), f'BAD_AGREEMENT:{c}:{i}')
        for field in ('qwen_tag_category_publish_allowed','v5_record_modified','qwen_record_modified','category_auto_overwrite'):
            require(r.get(field)=='NO',f'UNSAFE_FLAG_{field}:{c}:{i}')
        if r.get('shadow_alias_candidate')=='YES':
            dest=(c,r.get('shadow_alias_destination'))
            require(dest in bykey,f'ALIAS_DESTINATION_NOT_IN_IMAGE:{c}:{i}')
            require(dest!=key,f'ALIAS_SELF_REFERENCE:{c}:{i}')
        else:
            require(not r.get('shadow_alias_destination'),f'UNFLAGGED_ALIAS:{c}:{i}')
    sha_by_case=defaultdict(set)
    for r in rows: sha_by_case[r.get('case_id')].add(r.get('image_sha256'))
    for case,sha in sha_by_case.items(): require(len(sha)==1, f'INCONSISTENT_IMAGE_SHA:{case}')
    for r in rows:
        if r.get('user_chat_attested_30')=='YES':
            require(r.get('review_provenance')=='AI_PREFILL_USER_CHAT_ACCEPTED_NO_EXPORTED_CSV',
                    'REVIEW_PROVENANCE_MISMATCH:'+r.get('case_id','')+':'+r.get('instance_id',''))
    if issues:
        (out/'BLOCKED_INTEGRITY_REPORT.json').write_text(json.dumps({'status':'BLOCKED', 'issues':issues},ensure_ascii=False,indent=2),encoding='utf-8')
        raise ValueError('Integrity gate failed: '+ '; '.join(issues[:8]))

    routing=[]; alias=[]; queue=[]
    for r in rows:
        alias_flag=r['shadow_alias_candidate']=='YES'
        conflict=r['category_agreement']=='NO'
        reviewed=r['user_chat_attested_30']=='YES'
        historical_bbox=r.get('v5_bbox_historical','')
        blockers=[]
        if alias_flag: blockers.append('SHADOW_ALIAS_REQUIRES_VISUAL_IDENTITY_CHECK')
        if conflict: blockers.append('CLASS_CONFLICT_REQUIRES_GT')
        if not reviewed and conflict: blockers.append('UNREVIEWED_CONFLICT')
        if reviewed: blockers.append('CHAT_ATTESTED_AI_PREFILL_WITHOUT_EXPORTED_GT_CSV')
        if historical_bbox not in ('CORRECT', ''): blockers.append('HISTORICAL_BBOX_ISSUE_NOT_GT')
        # Agreement is never treated as ground truth.
        if not blockers: blockers.append('CATEGORY_AGREEMENT_NOT_GT')
        policy='BLOCK_SHADOW_ALIAS' if alias_flag else ('BLOCK_CLASS_CONFLICT' if conflict else 'SHADOW_AGREEMENT_ONLY_NO_GT')
        score=(100 if alias_flag else 0)+(35 if conflict and not reviewed else 0)+\
              (20 if historical_bbox not in ('CORRECT','') else 0)+\
              (10 if r.get('qwen_category_unverified')=='top' else 0)+\
              (8 if r.get('v5_class_historical')=='INCORRECT' else 0)
        rec={**r,
            'v4_policy':policy,
            'v4_reasons':';'.join(blockers),
            'priority_score_audit_only':score,
            'visual_gt_verified':'NO',
            'instance_identity_verified':'NO',
            'mainline_approved':'NO'}
        routing.append(rec)
        if alias_flag:
            alias.append({**rec,'alias_target_exists_same_image':'YES',
                          'alias_authorized':'NO'})
        if conflict and not reviewed:
            queue.append(rec)
    require(len(queue)==summary.get('tags_v5_class_conflicts',-1)-summary.get('reviewed_30_user_chat_attested_AI_prefill',-1), 'UNREVIEWED_CONFLICT_COUNT_MISMATCH')
    if issues:
        raise ValueError('Counts inconsistent: '+str(issues))
    def review_fields(r):
        return {**r, 'human_visual_identity_result':'', 'human_visual_category_gt':'',
                'human_image_evidence_reference':'', 'human_review_notes':'',
                'reviewer':'', 'reviewed_at':''}
    review_field_names=list(review_fields(queue[0]).keys()) if queue else []
    routing_fields=list(routing[0].keys())
    write_csv(out/'v4_policy_decisions_171.csv',routing,routing_fields)
    write_csv(out/'v4_shadow_alias_quarantine_22.csv',alias,list(alias[0].keys()) if alias else routing_fields)
    queue=sorted(queue,key=lambda r:(-r['priority_score_audit_only'],r['case_id'],r['instance_id']))
    write_csv(out/'v4_unreviewed_conflict_queue_63.csv',[review_fields(r) for r in queue],review_field_names)

    # 12-item visual review sampler: all 3 agreement+alias, 5 diverse alias+unreviewed,
    # 4 diverse non-alias+unreviewed. These are only suggested priorities, NOT evaluated GT.
    selected=[]; chosen=set()
    def add_one(candidates, limit):
        selected_group=[]; seen_cats=set()
        candidates=sorted(candidates,key=lambda r:(-r['priority_score_audit_only'],r['case_id'],r['instance_id']))
        for r in candidates:
            if len(selected_group)>=limit: break
            key=(r['case_id'],r['instance_id'])
            if key in chosen or r['v5_category_unverified'] in seen_cats: continue
            selected_group.append(r);chosen.add(key);seen_cats.add(r['v5_category_unverified'])
        for r in candidates:
            if len(selected_group)>=limit: break
            key=(r['case_id'],r['instance_id'])
            if key in chosen: continue
            selected_group.append(r);chosen.add(key)
        selected.extend(selected_group)
    add_one([r for r in alias if r['category_agreement']=='YES'],3)
    add_one([r for r in queue if r['shadow_alias_candidate']=='YES'],5)
    add_one([r for r in queue if r['shadow_alias_candidate']!='YES'],4)
    write_csv(out/'v4_priority_visual_review_12.csv',[review_fields(r) for r in selected],review_field_names)

    # Cross model labels are hypotheses only, not confusion vs actual GT.
    mismatch=Counter((r['v5_category_unverified'],r['qwen_category_unverified']) for r in rows if r['category_agreement']=='NO')
    count=Counter(r['v4_policy'] for r in routing)
    result={
      'schema':'prd32_guard_v4_offline_contract_audit_v1',
      'status':'OFFLINE_CONTRACT_AUDIT_PASS_NOT_MODEL_EVALUATION',
      'scope':'FROZEN_DEV60_ONLY',
      'source_zip_filename':src.name,
      'source_zip_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
      'source_instances':len(rows),
      'source_images':len(sha_by_case),
      'mask_bbox_geometry_compatible':sum(g['status']=='GEOMETRY_COMPATIBLE' for g in geom),
      'category_conflicts':sum(r['category_agreement']=='NO' for r in rows),
      'category_agreement_not_gt':sum(r['category_agreement']=='YES' for r in rows),
      'chat_attested_ai_prefill_without_exported_gt':sum(r['user_chat_attested_30']=='YES' for r in rows),
      'unreviewed_conflicts':len(queue),
      'shadow_aliases_quarantined':len(alias),
      'alias_with_category_agreement':sum(r['category_agreement']=='YES' for r in alias),
      'alias_with_class_conflict':sum(r['category_agreement']=='NO' for r in alias),
      'v4_policy_distribution':dict(count),
      'qwen_category_on_conflicts':dict(Counter(r['qwen_category_unverified'] for r in rows if r['category_agreement']=='NO')),
      'top_mismatch_pairs':[{'v5_unverified':x,'qwen_unverified':y,'count':n} for (x,y),n in mismatch.most_common(12)],
      'visual_review_priority_12':[{'case_id':r['case_id'],'instance_id':r['instance_id'],'reason':r['v4_reasons']} for r in selected],
      'source_integrity_gate':'PASS',
      'mask_bbox_geo_not_segmentation_metric':True,
      'independent_holdout_opened':False,
      'raw_source_modified':False,
      'qwen_answers_modified':False,
      'published_categories':0,
      'mainline_deployment_approved':False,
      'warning':'Metadata checks cannot prove that Qwen sees correct garment or that classes are GT-accurate; all code is offline-only.'
    }
    (out/'v4_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).with_name('PRD32_1106_GUARD_V3_RESULTS_INPUT.zip'))
    parser.add_argument('--out',type=Path,default=Path(__file__).with_name('results'))
    args=parser.parse_args()
    res=build(args.input,args.out)
    print(json.dumps(res,ensure_ascii=False,indent=2))
