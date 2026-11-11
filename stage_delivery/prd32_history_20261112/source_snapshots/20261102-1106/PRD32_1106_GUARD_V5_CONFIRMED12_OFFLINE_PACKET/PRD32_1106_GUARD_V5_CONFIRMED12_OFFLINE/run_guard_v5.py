"""Reconcile user-chat-confirmed DEV12 review with immutable V4 results.

Standard library only. Never opens independent holdout, images, model outputs,
GPU checkpoints, or a production repo. Does not publish a category.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

EXPECTED_SCOPE = 171
# Frozen, immutable source snapshots from the verified V4 and 12-case packet.
EXPECTED_SHA256 = {
    'v4_policy_decisions_171.csv': '9fb4b876a98b2165fb018d2f836de229ac09d1ae331b7f78588d99e462f30eb1',
    'PRD32_V4_PRIORITY12_AI_DRAFT_NOT_GT.csv': '7b62f5cc248840f8bd80e3072599da9c28734ff76b522572911250d9c0bdf008',
    'priority12_qwen_three_task_comparison.csv': '4b45df465c9db9ac5cac34706e43c18cc3191c3671e455ca0b254ca63a7e2ddd',
    'v4_summary.json': '08089fb5a7124472857d8f23cf3a33521b48e431beb4cade2e04add5fb35f8eb',
}
VALID_CLASSES = {'top','pants','skirt','outerwear','dress','shoe','bag','accessory'}
UNRESOLVED = '无法判断/不适用'
CONFIRMATION = 'USER_CONFIRMED_IN_CHAT'
DATE = '2026-11-06'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows: raise ValueError('Cannot write empty table')
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),extrasaction='raise')
        w.writeheader(); w.writerows(rows)


def key(r): return (r['case_id'],r['instance_id'])


def checked_map(rows, expected=None):
    out={}
    for r in rows:
        k=key(r)
        if k in out: raise ValueError(f'Duplicate case/instance identity: {k}')
        out[k]=r
    if expected is not None and set(out)!=set(expected):
        raise ValueError(f'Instance set mismatch: missing={sorted(set(expected)-set(out))}, excess={sorted(set(out)-set(expected))}')
    return out


def validated_inputs(root: Path):
    inp=root/'inputs'
    paths={k:inp/v for k,v in {
        'routes':'v4_policy_decisions_171.csv',
        'reviews':'PRD32_V4_PRIORITY12_AI_DRAFT_NOT_GT.csv',
        'qwen':'priority12_qwen_three_task_comparison.csv',
        'v4summary':'v4_summary.json',
    }.items()}
    for p in paths.values():
        if not p.is_file():raise FileNotFoundError(p)
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != EXPECTED_SHA256[p.name]:
            raise ValueError(f'Frozen input SHA256 changed: {p.name}')
    routing=read_csv(paths['routes'])
    reviews=read_csv(paths['reviews'])
    qwen=read_csv(paths['qwen'])
    v4sum=json.loads(paths['v4summary'].read_text(encoding='utf-8'))
    vr=checked_map(routing); rv=checked_map(reviews); qw=checked_map(qwen,rv)
    if len(vr)!=EXPECTED_SCOPE or len(rv)!=12 or v4sum.get('source_instances')!=EXPECTED_SCOPE:
        raise ValueError('Frozen DEV scope/count mismatch')
    if v4sum.get('independent_holdout_opened') is not False or v4sum.get('mainline_deployment_approved') is not False:
        raise ValueError('Source holdout/deployment gate failed')
    if not set(rv).issubset(vr):raise ValueError('Confirmed review not part of frozen V4 DEV scope')
    if sum(x.get('shadow_alias_candidate')=='YES' for x in routing)!=22:
        raise ValueError('Shadow alias source count unexpectedly changed')
    if sum(x.get('category_agreement')=='NO' for x in routing)!=93:
        raise ValueError('Category conflict count unexpectedly changed')
    for k,review in rv.items():
        original=vr[k]
        cmp=qw[k]
        if review.get('review_status') != 'PENDING':
            raise ValueError(f'Input expected immutable AI prefill pending status: {k}')
        if review.get('category_gt') not in VALID_CLASSES|{UNRESOLVED}:
            raise ValueError(f'Invalid category: {k}')
        if not review.get('notes','').startswith('【AI预填'):
            raise ValueError(f'Input not original AI prefill: {k}')
        if (original.get('image_sha256') and cmp.get('qwen_focus_v1_tags') != original.get('qwen_category_unverified')):
            raise ValueError(f'Qwen Focus vs frozen V4 mismatch: {k}')
        if cmp.get('marked_sha_equal_between_variants')!='YES':
            raise ValueError(f'Marked inputs are not known identical: {k}')
    return paths, routing, reviews, qwen, vr, rv, qw, v4sum


def build(root: Path, out: Path):
    paths,routes,reviews,qwen,vr,rv,qw,v4sum=validated_inputs(root)
    out.mkdir(parents=True,exist_ok=True)
    confirmed=[]
    for review in reviews:
        r=vr[key(review)]
        known=review['category_gt'] in VALID_CLASSES
        # Preserve AI-authored analysis separately; do NOT convert unknown to a class.
        confirmed.append({
            'case_id':review['case_id'], 'instance_id':review['instance_id'],
            'image_sha256':r['image_sha256'],
            'target_box':review['target_box'], 'category_gt':review['category_gt'],
            'category_gt_is_determinate':'YES' if known else 'NO',
            'instance_relation':review['instance_relation'],
            'qwen_semantic':review['qwen_semantic'],
            'review_status':CONFIRMATION,
            'user_confirmation_date':DATE,
            'review_provenance':'USER_CHAT_CONFIRMED_AI_PREFILL_NO_SEPARATE_EXPORTED_REVIEW_CSV',
            'ai_origin_notes':review['notes'],
            'human_review_scope':'User confirmed previously displayed AI-prefilled 12-case judgments in chat; unresolved category remains unresolved',
            'alias_partner_human_verified':'NO',
            'qwen_instance_runtime_bound_verified':'NO',
            'mainline_approved':'NO',
        })
    cm=checked_map(confirmed)
    enriched=[]; evaluated=[]
    for r in routes:
        k=key(r); evidence=cm.get(k); compare=qw.get(k)
        reasons=[]
        if r['shadow_alias_candidate']=='YES':reasons.append('SHADOW_ALIAS_QUARANTINE')
        if r['category_agreement']=='NO':reasons.append('V5_QWEN_FOCUS_CLASS_CONFLICT')
        if evidence:
            relation=evidence['instance_relation']
            if relation in ('同物重复','整体与局部'):
                reasons.append('CONFIRMED_DUPLICATE_OR_PART_RELATION_NO_EXPLICIT_FUSION')
            if relation=='不同对象混淆':reasons.append('CONFIRMED_MULTI_OBJECT_AMBIGUITY')
            if evidence['target_box']!='正确对象':reasons.append('CONFIRMED_PARTIAL_OR_MIXED_BOX')
            if evidence['qwen_semantic']=='混合/无法判断':reasons.append('CONFIRMED_CROSS_TASK_SEMANTIC_DISAGREEMENT')
            if evidence['category_gt_is_determinate']=='NO':
                reasons.append('CONFIRMED_CATEGORY_UNRESOLVED')
            else:
                if r['v5_category_unverified']!=evidence['category_gt']:
                    reasons.append('V5_CLASS_DIFFERS_FROM_CONFIRMED_GT')
                if r['qwen_category_unverified']!=evidence['category_gt']:
                    reasons.append('QWEN_FOCUS_CLASS_DIFFERS_FROM_CONFIRMED_GT')
        else:
            reasons.append('NO_USER_CONFIRMED_VISUAL_GT_FOR_THIS_INSTANCE')
        if r['category_agreement']=='YES' and evidence and evidence['category_gt_is_determinate']=='YES' and r['v5_category_unverified']!=evidence['category_gt']:
            reasons.append('AGREE_BUT_BOTH_WRONG_ON_REVIEW')
        decision=('BLOCK_CONFIRMED_REVIEW_RISK' if evidence and (
            evidence['instance_relation']!='独立对象' or evidence['qwen_semantic']=='混合/无法判断'
            or evidence['target_box']!='正确对象' or evidence['category_gt_is_determinate']=='NO'
            or r['v5_category_unverified']!=evidence['category_gt']
            or r['qwen_category_unverified']!=evidence['category_gt'])
            else ('BLOCK_SHADOW_ALIAS' if r['shadow_alias_candidate']=='YES'
                  else ('BLOCK_UNVERIFIED_CLASS_CONFLICT' if r['category_agreement']=='NO'
                        else 'SHADOW_AGREEMENT_ONLY_NOT_GT')))
        if not evidence and r['v4_policy']=='BLOCK_SHADOW_ALIAS' and decision!='BLOCK_SHADOW_ALIAS':
            raise ValueError('Alias quarantine unexpectedly released')
        row={**r,
          'v5_review_status':evidence['review_status'] if evidence else 'NOT_REVIEWED_IN_THIS_BATCH',
          'v5_confirmed_category_gt':evidence['category_gt'] if evidence else '',
          'v5_gt_determinate':evidence['category_gt_is_determinate'] if evidence else '',
          'v5_instance_relation':evidence['instance_relation'] if evidence else '',
          'v5_target_box':evidence['target_box'] if evidence else '',
          'v5_qwen_semantic':evidence['qwen_semantic'] if evidence else '',
          'v5_review_risks':';'.join(reasons),
          'v5_policy':decision,
          'v5_category_auto_override':'NO',
          'v5_duplicate_auto_merge':'NO',
          'v5_mainline_approved':'NO',
          'v5_holdout_opened':'NO',
        }
        enriched.append(row)
        if evidence:
            evaluated.append({
                'case_id':r['case_id'],'instance_id':r['instance_id'],'image_sha256':r['image_sha256'],
                'human_confirmed_gt_or_unresolved':evidence['category_gt'],
                'gt_determinate':evidence['category_gt_is_determinate'],
                'target_box':evidence['target_box'],
                'instance_relation':evidence['instance_relation'],
                'qwen_cross_task_semantic_review':evidence['qwen_semantic'],
                'v5_category':r['v5_category_unverified'],
                'qwen_baseline_category':compare['qwen_baseline_tags'],
                'qwen_focus_category':compare['qwen_focus_v1_tags'],
                'v5_matches_review':('YES' if r['v5_category_unverified']==evidence['category_gt'] else 'NO') if evidence['category_gt_is_determinate']=='YES' else 'N/A',
                'baseline_matches_review':('YES' if compare['qwen_baseline_tags']==evidence['category_gt'] else 'NO') if evidence['category_gt_is_determinate']=='YES' else 'N/A',
                'focus_matches_review':('YES' if compare['qwen_focus_v1_tags']==evidence['category_gt'] else 'NO') if evidence['category_gt_is_determinate']=='YES' else 'N/A',
                'v5_policy':decision,'risk_codes':';'.join(reasons),
                'category_published':'NO','instance_merged':'NO',
            })
    if len(enriched)!=171 or len(evaluated)!=12: raise ValueError('Output coverage changed')
    write_csv(out/'v5_user_confirmed_review_12.csv',confirmed)
    write_csv(out/'v5_dev60_policy_171.csv',enriched)
    write_csv(out/'v5_reviewed12_comparison.csv',evaluated)
    unresolved=[r for r in evaluated if r['gt_determinate']=='NO']
    write_csv(out/'v5_unresolved_category_3.csv',unresolved)
    # No pseudo-GT is manufactured for the remaining 159, and V4 historical
    # chat-attested 30 are not reclassified as independently exported GT.
    known=[r for r in evaluated if r['gt_determinate']=='YES']
    summary={
       'schema':'prd32_guard_v5_confirmed12_offline_audit_v1',
       'status':'OFFLINE_REVIEW_RECONCILIATION_PASS_NOT_PRODUCTION',
       'scope':'FROZEN_DEV60_171_INSTANCES',
       'confirmation_provenance':'USER_CHAT_CONFIRMATION_OF_12_AI_PREFILLS_2026-11-06',
       'confirmed_reviewed_instances':len(evaluated),
       'confirmed_determinate_class_labels':len(known),
       'confirmed_unresolved_categories':len(unresolved),
       'confirmed_duplicate_or_part_relations':sum(r['instance_relation'] in ('同物重复','整体与局部') for r in evaluated),
       'confirmed_mixed_objects':sum(r['instance_relation']=='不同对象混淆' for r in evaluated),
       'confirmed_independent_object':sum(r['instance_relation']=='独立对象' for r in evaluated),
       'confirmed_cross_task_semantic_mixed':sum(r['qwen_cross_task_semantic_review']=='混合/无法判断' for r in evaluated),
       'v5_match_of_confirmed9':sum(r['v5_matches_review']=='YES' for r in known),
       'qwen_baseline_match_of_confirmed9':sum(r['baseline_matches_review']=='YES' for r in known),
       'qwen_focus_match_of_confirmed9':sum(r['focus_matches_review']=='YES' for r in known),
       'focus_changed_vs_baseline_on_confirmed12':sum(r['qwen_baseline_category']!=r['qwen_focus_category'] for r in evaluated),
       'full_dev60_instances':len(enriched),
       'full_dev60_v4_alias_quarantine_preserved':sum(r['shadow_alias_candidate']=='YES' for r in enriched),
       'full_dev60_category_conflicts_preserved':sum(r['category_agreement']=='NO' for r in enriched),
       'v5_policy_distribution':dict(Counter(r['v5_policy'] for r in enriched)),
       'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
       'independent_holdout_opened':False,
       'raw_v5_records_modified':False,'raw_qwen_responses_modified':False,
       'category_auto_overwrites':0,'alias_auto_merges':0,
       'mainline_deployment_approved':False,
       'reviewed12_sample_is_targeted_not_random':True,
       'warning':'Comparisons on targeted 9 determinate labels are diagnostic only. Chat-confirmed labels are not an independent blinded test. No verified alias destination pairs, no runtime image binding authorization, no deployment.'
    }
    (out/'v5_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--out',type=Path,default=None)
    args=parser.parse_args()
    root=args.root.resolve()
    summary=build(root, args.out.resolve() if args.out else root/'results')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
