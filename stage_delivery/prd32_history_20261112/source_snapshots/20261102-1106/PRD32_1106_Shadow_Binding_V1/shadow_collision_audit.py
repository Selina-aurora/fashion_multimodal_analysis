#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD3.2 DEV60: non-mutating, review-bounded instance conflict shadow audit.

NEVER edits model predictions, masks, Qwen outputs, main pipeline, or held-out data.
Only draws candidates from 16 pairwise user-chat-accepted DEV60 reviews.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_IMAGES = 60
EXPECTED_INSTANCES = 171
EXPECTED_PAIRS = 43
REVIEWED_PAIRS = 16
MASK_THRESHOLD = 0.90


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda: f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def key(row):
    return row['case_id'], *sorted((row['instance_a'], row['instance_b']))


def load_and_validate(raw_file, review_file, pair_file):
    records = [json.loads(s) for s in raw_file.read_text(encoding='utf-8').splitlines() if s.strip()]
    assert len(records)==EXPECTED_IMAGES, f'dev image count mismatch {len(records)}'
    ids={}
    for im in records:
        c=im['case_id']
        assert c not in ids, f'duplicate image key {c}'
        assert c.startswith(('fresh_dev_','dev_')), f'unexpected DEV image ID: {c}'
        assert 'HOLDOUT' not in str(im).upper() and 'TEST_SOURCE' not in str(im).upper(), f'forbidden sealed data: {c}'
        assert str(im.get('scope','')).upper().find('DEV')>=0, f'wrong scope: {c}'
        pred={p['instance_id']:p for p in im['instances']}
        assert len(pred)==len(im['instances']), f'nonunique predictions {c}'
        ids[c]=pred
    n=sum(map(len,ids.values()))
    assert n==EXPECTED_INSTANCES, f'prediction count mismatch {n}'
    pairs=read_csv(pair_file)
    reviews=read_csv(review_file)
    assert len(pairs)==EXPECTED_PAIRS, f'43 pair scope mismatch {len(pairs)}'
    assert len(reviews)==REVIEWED_PAIRS, f'16 pair scope mismatch {len(reviews)}'
    all_pair_keys=set()
    for r in pairs:
        k=key(r)
        assert k not in all_pair_keys, f'pair repeated {k}'
        all_pair_keys.add(k)
        c,a,b=k
        assert a in ids[c] and b in ids[c], f'unknown prediction in masks {k}'
        assert 0<=float(r['mask_iou'])<=1, f'invalid mask IoU {k}'
    seen=set()
    for r in reviews:
        k=key(r)
        assert k not in seen, f'duplicate accepted pair: {k}'
        seen.add(k)
        assert k in all_pair_keys, f'accepted review not present in 43 pairs: {k}'
        assert float(r['mask_iou'])>=MASK_THRESHOLD, f'accepted review below frozen scope: {k}'
        assert r['human_verified']=='USER_CHAT_ACCEPTED_2026-11-06', f'chat acceptance not recorded: {k}'
        assert r['pair_relationship'] in {'likely_same_garment_duplicate','same_garment_category_uncertain'}, f'unknown review tag: {k}'
        assert r['instance_a'] in ids[r['case_id']] and r['instance_b'] in ids[r['case_id']], f'unknown reviewed prediction {k}'
    assert len(seen)==REVIEWED_PAIRS
    return records,ids,pairs,reviews


class UnionFind:
    def __init__(self): self.parent={}
    def find(self,x):
        if x not in self.parent: self.parent[x]=x
        if self.parent[x]!=x: self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra!=rb: self.parent[rb]=ra


def analyze(records, ids, pairs, reviews):
    """Returns review-grouped findings, proposed alias rows, and 171 unchanged registry IDs."""
    pair_by_key={key(r):r for r in pairs}
    by_case=defaultdict(list)
    for r in reviews: by_case[r['case_id']].append(r)
    groups=[]; alias={}; blocked_member={}
    for case in sorted(by_case):
        uf=UnionFind()
        for r in by_case[case]: uf.union(r['instance_a'],r['instance_b'])
        clusters=defaultdict(list)
        for r in by_case[case]: clusters[uf.find(r['instance_a'])].append(r)
        for group_review in clusters.values():
            members=sorted(set(x for r in group_review for x in (r['instance_a'],r['instance_b'])))
            categories=sorted({ids[case][id_]['predicted_category'] for id_ in members})
            canonical_set={r['canonical_target_if_confirmed'].strip() for r in group_review if r['canonical_target_if_confirmed'].strip()}
            is_ambiguous=any(r['pair_relationship']!='likely_same_garment_duplicate' for r in group_review)
            record_evidence=[]
            for inst_id in members:
                statuses=[]
                for raw_pair in pairs:
                    if raw_pair['case_id']!=case: continue
                    if inst_id==raw_pair['instance_a']: statuses.append((raw_pair['human_class_a'],raw_pair['bbox_review_a']))
                    if inst_id==raw_pair['instance_b']: statuses.append((raw_pair['human_class_b'],raw_pair['bbox_review_b']))
                record_evidence.append((inst_id, statuses))
            if is_ambiguous:
                status='BLOCKED_CATEGORY_AMBIGUOUS'; canonical=''; reason='人工认可该组为类别不确定；没有足够类别依据选择主实例'
            elif len(canonical_set)!=1:
                status='BLOCKED_INCONSISTENT_CANONICAL';canonical='';reason='同组的保留实例建议不一致或未给出'
            else:
                canonical=next(iter(canonical_set))
                if canonical not in members:
                    status='BLOCKED_CANONICAL_OUTSIDE_HIGH_IOU_GROUP';reason='拟保留实例不属于这组高IoU连通分量，需要扩展人工审核，不允许跨组删除'
                elif not any(s==('CORRECT','CORRECT') for rid, statuses in record_evidence if rid==canonical for s in statuses):
                    status='BLOCKED_CANONICAL_NOT_VERIFIED'; reason='历史类别和框审核未同时标记为正确，不允许进入候选去重'
                else:
                    status='SHADOW_ALIAS_CANDIDATE';reason='仅作开发集非破坏性身份别名建议，用户确认的组内实例不作为新的模型质量证据'
            if status!='SHADOW_ALIAS_CANDIDATE':
                for inst_id in members: blocked_member[(case,inst_id)]=status
            else:
                for inst_id in members:
                    if inst_id!=canonical:
                        assert (case,inst_id) not in alias
                        alias[(case,inst_id)]=canonical
            groups.append({
                'case_id':case,'member_ids':'|'.join(members), 'member_categories':'|'.join(categories),
                'num_members':len(members), 'pair_records':len(group_review),
                'min_mask_iou':f"{min(float(r['mask_iou']) for r in group_review):.5f}",
                'max_mask_iou':f"{max(float(r['mask_iou']) for r in group_review):.5f}",
                'review_canonical_ids':'|'.join(sorted(canonical_set)),
                'shadow_canonical_id':canonical if status=='SHADOW_ALIAS_CANDIDATE' else '',
                'shadow_group_status':status, 'alias_candidate_count':len(members)-1 if status=='SHADOW_ALIAS_CANDIDATE' else 0,
                'reason':reason,
                'review_provenance':'AI_ASSISTED_PREFILL_ACCEPTED_USER_CHAT_2026-11-06_DEV60_ONLY'
            })
    reg=[]
    for img in records:
        c=img['case_id']
        for p in img['instances']:
            k=(c,p['instance_id'])
            if k in alias:
                target=alias[k]; status='SHADOW_SUPPRESS_SUGGESTED'; note='Do not reuse dropped prediction Qwen output as canonical answer'
            elif k in blocked_member:
                target=p['instance_id'];status=blocked_member[k];note='Manual review required, unchanged'
            elif any(x['case_id']==c and x['shadow_canonical_id']==p['instance_id'] for x in groups):
                target=p['instance_id'];status='REVIEWED_SHADOW_CANONICAL';note='One of the reviewed potential canonical predictions'
            else:
                target=p['instance_id'];status='UNCHANGED';note='Not user-reviewed for shadow alias; no new inference'
            reg.append({'case_id':c,'instance_id':p['instance_id'],
                        'v5_predicted_category':p['predicted_category'],
                        'v5_score':f"{float(p['score']):.6f}",
                        'original_instance_id':p['instance_id'],
                        'shadow_alias_target_id':target,
                        'shadow_status':status,
                        'original_prediction_preserved':'YES',
                        'qwen_output_reassigned':'NO',
                        'review_note':note})
    assert len(reg)==EXPECTED_INSTANCES
    assert len(set((x['case_id'],x['instance_id']) for x in reg))==EXPECTED_INSTANCES
    for (case,_),target in alias.items(): assert target in ids[case]
    for g in groups:
        assert (g['shadow_group_status']=='SHADOW_ALIAS_CANDIDATE') == (g['alias_candidate_count']>0)
    unreviewed=[x for x in pairs if key(x) not in {key(r) for r in reviews}]
    assert len(unreviewed)==EXPECTED_PAIRS-REVIEWED_PAIRS
    return groups,alias,reg,unreviewed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default=str(ROOT/'output'),help='Output only; never modifies inputs')
    args=parser.parse_args(); o=Path(args.output_dir).resolve()
    source=ROOT/'input/v5_raw_predictions.jsonl'
    review=ROOT/'input/16_pairs_chat_accepted.csv'
    pair_file=ROOT/'input/43_pairs_mask_audit.csv'
    for f in [source,review,pair_file]: assert f.exists(), f'missing input {f}'
    assert o!=ROOT and o not in [x.resolve() for x in (ROOT/'input',)], 'refuse overwrite of input path'
    records, ids, pairs, reviews=load_and_validate(source, review, pair_file)
    groups, alias, reg, unreviewed=analyze(records,ids,pairs,reviews)
    write_csv(o/'conflict_groups_14.csv',groups,list(groups[0]))
    write_csv(o/'instance_shadow_registry_171.csv',reg,list(reg[0]))
    alias_rows=[{'case_id':case,'suppressed_instance_id':iid,'suggested_canonical_instance_id':tid,
                 'mode':'SHADOW_ONLY_NO_MAINLINE_CHANGE','qwen_output_migrated':'NO'} for (case,iid),tid in sorted(alias.items())]
    write_csv(o/'shadow_alias_candidates.csv',alias_rows, list(alias_rows[0]))
    other=[]
    for p in sorted(unreviewed,key=lambda x: -float(x['mask_iou'])):
        x={k:p[k] for k in ['case_id','instance_a','category_a','instance_b','category_b','bbox_iou','mask_iou','human_class_a','human_class_b','bbox_review_a','bbox_review_b']}
        x.update({'pair_relationship':'PENDING_HUMAN_REVIEW','canonical_target_if_confirmed':'',
                  'human_verified':'NO','action':'NO_CHANGE','risk_hint':'bbox-overlap candidate; MaskIoU alone is not duplicate proof'})
        other.append(x)
    write_csv(o/'unreviewed_27_for_manual_review.csv',other,list(other[0]))
    gc=defaultdict(int)
    for g in groups:gc[g['shadow_group_status']]+=1
    stats={
        'scope':'DEV60_TRAIN_SOURCE_DIAGNOSTIC_ONLY_NOT_HOLDOUT',
        'gate':'SHADOW_ONLY_NO_MAINLINE_UPDATE_NO_QWEN_REASSIGNMENT',
        'source_images':len(records),'source_predictions':len(reg),
        'original_overlap_pairs':len(pairs),'user_chat_accepted_review_pairs':len(reviews),
        'unique_high_iou_review_groups':len(groups),
        'group_status_counts':dict(sorted(gc.items())),
        'shadow_alias_candidate_predictions':len(alias),
        'original_prediction_count_not_modified':len(reg),
        'hypothetical_display_item_count_if_all_proposals_accepted':len(reg)-len(alias),
        'uncertain_or_blocked_groups':len(groups)-gc['SHADOW_ALIAS_CANDIDATE'],
        'remaining_pair_reviews':len(unreviewed),
        'holdout_accessed':False,'model_outputs_recomputed':False,'accuracy_claimed':False,
        'inputs_sha256':{f.name:digest(f) for f in [source,review,pair_file]},
        'limitations':['Review acceptance was expressed in chat; not independent blind adjudication.',
                       'No actual suppression applied to model outputs or downstream predictions.',
                       'Only 16 high-IoU pairs reviewed, 27 other bbox-overlapping pairs pending.',
                       'Theoretical reduced count is not recall/precision nor a validated gain.',
                       'Review annotation is not a new ground-truth test set.']
    }
    (o/'shadow_report.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in stats.items() if k not in ['limitations','inputs_sha256']},ensure_ascii=False,indent=2))
    print('OUTPUT:',str(o))


if __name__=='__main__':main()
