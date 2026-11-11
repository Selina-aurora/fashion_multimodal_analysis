#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""DEV60 full 43-pair conservative shadow identity audit. Never alters model/pipeline/holdout.

This script is an OFFLINE PLANNER. It never changes/deletes predictions and never migrates Qwen outputs.
Accepted labels = prior AI prefill + user chat confirmation; no independent adjudication.
"""
import csv, json, hashlib, argparse
from pathlib import Path
from collections import Counter, defaultdict

ROOT=Path(__file__).resolve().parent
DUP={'SAME_GARMENT_DUPLICATE','likely_same_garment_duplicate'}
MIXED={'MIXED_OR_PARTIAL_TARGET'}
UNCERTAIN={'UNCERTAIN','same_garment_category_uncertain'}
EXPECTED_PAIRS=43


def csvread(path):
    with open(path,encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def csvwrite(path, rows, keys):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pair_key(r):return(r['case_id'],*sorted([r['instance_a'],r['instance_b']]))

class DSU:
    def __init__(self):self.p={}
    def find(self,a):
        self.p.setdefault(a,a)
        if self.p[a]!=a:self.p[a]=self.find(self.p[a])
        return self.p[a]
    def union(self,a,b):
        self.p[self.find(b)]=self.find(a)


def analyze_pairs(rows, instance_ids, historical):
    """Graph merging only based on explicitly same-garment pairs; mixed/uncertain edges never merged."""
    seen=set();edges_by_case=defaultdict(list);DS={}
    for r in rows:
        k=pair_key(r);c,a,b=k
        assert k not in seen,('duplicate reviewed pair',k)
        seen.add(k)
        assert c in instance_ids and a in instance_ids[c] and b in instance_ids[c], ('unknown id',k)
        assert r['pair_relationship'] in DUP|MIXED|UNCERTAIN, ('unknown relation', k)
        assert 0<=float(r['mask_iou'])<=1
        edges_by_case[c].append(r)
        if r['pair_relationship'] in DUP:
            DS.setdefault(c,DSU()).union(a,b)
    groups=[]; candidate=[]; blocked=[]
    for c,d in sorted(DS.items()):
        components=defaultdict(set)
        for r in edges_by_case[c]:
            if r['pair_relationship'] in DUP:
                for i in (r['instance_a'],r['instance_b']):components[d.find(i)].add(i)
        for root,members in components.items():
            members=sorted(members)
            duplication_edges=[r for r in edges_by_case[c] if r['pair_relationship'] in DUP and r['instance_a'] in members and r['instance_b'] in members]
            conflicts=[r for r in edges_by_case[c] if r['pair_relationship'] not in DUP and (r['instance_a'] in members or r['instance_b'] in members)]
            candidate_canonical=set(r['canonical_target_if_confirmed'] for r in duplication_edges if r['canonical_target_if_confirmed'] not in ('','OTHER_NEEDS_GROUP_REVIEW'))
            unresolved=any(r['canonical_target_if_confirmed'] in ('','OTHER_NEEDS_GROUP_REVIEW') for r in duplication_edges)
            highrisk='|'.join(sorted(set(r['pair_relationship']+':'+r['instance_a']+'/'+r['instance_b'] for r in conflicts)))
            canonical=next(iter(candidate_canonical)) if len(candidate_canonical)==1 else ''
            quality=historical.get((c,canonical),{})
            if conflicts:
                status='BLOCK_MIXED_OR_UNCERTAIN_ADJACENCY'
                reason='图内同组成员还参与混合/不确定关系；不得仅靠二元重复边自动合并'
            elif unresolved or len(candidate_canonical)!=1:
                status='BLOCK_CANONICAL_UNRESOLVED'
                reason='没有明确唯一保留实例，或多条边的保留建议不一致'
            elif canonical not in members:
                status='BLOCK_CANONICAL_OUTSIDE_COMPONENT'
                reason='拟保留实例不属于重复候选连通组'
            elif quality.get('bbox_quality')!='CORRECT' or quality.get('v5_category_quality')!='CORRECT':
                status='BLOCK_CANONICAL_HISTORICAL_QUALITY'
                reason='历史开发集审核未同时确认拟保留实例的框和类别正确'
            else:
                status='SHADOW_PROPOSAL_ONLY'
                reason='仅生成开发集假设性去重建议，禁止将被抑制实例的Qwen答案移到保留实例'
            g={
                'case_id':c, 'member_ids':'|'.join(members),'member_count':len(members),
                'duplicate_pair_count':len(duplication_edges), 'cross_relation_count':len(conflicts),
                'cross_relation_edges':highrisk,'preferred_canonicals':'|'.join(sorted(candidate_canonical)),
                'proposed_canonical_id':canonical if status=='SHADOW_PROPOSAL_ONLY' else '',
                'historical_canonical_bbox':quality.get('bbox_quality',''),
                'historical_canonical_category':quality.get('v5_category_quality',''),
                'min_duplicate_mask_iou':f"{min(float(r['mask_iou']) for r in duplication_edges):.5f}",
                'max_duplicate_mask_iou':f"{max(float(r['mask_iou']) for r in duplication_edges):.5f}",
                'status':status,'decision_reason':reason,
                'provenance':'USER_CHAT_ACCEPTED_AI_PREFILL_DEV_ONLY_NOT_BLIND'
            }
            groups.append(g)
            if status=='SHADOW_PROPOSAL_ONLY':
                for mid in members:
                    if mid!=canonical:
                        candidate.append({'case_id':c,'suppressed_instance_id':mid,'proposed_canonical_instance_id':canonical,
                            'mode':'SHADOW_ONLY','original_prediction_modified':'NO','qwen_output_migrated':'NO',
                            'eligible_for_mainline':'NO','group_status':status})
            else:blocked.append(g)
    return groups,candidate,blocked


def run(root=ROOT, output=None):
    src=root/'input'; out=(output or root/'output').resolve();out.mkdir(parents=True,exist_ok=True)
    if out==src.resolve() or src.resolve() in out.parents or out==root.resolve():raise ValueError('Refuse writing into root/input')
    source_files=[src/'v5_raw_predictions.jsonl',src/'nov2_v5_attribution_171.csv',src/'43_pairs_review_prefill_chat_confirmed.csv',src/'focus_v1_qwen_results_513.jsonl']
    assert all(p.is_file() for p in source_files),'missing packaged, fixed DEV60 inputs'
    cases=[json.loads(ln) for ln in source_files[0].read_text(encoding='utf-8').splitlines() if ln.strip()]
    assert len(cases)==60, 'DEV60 only'
    inst={}
    for c in cases:
        case=c['case_id']; assert 'DEV' in c['scope'].upper() and case.startswith(('fresh_dev_','dev_')),'unexpected dataset scope'
        assert case not in inst
        inst[case]={r['instance_id']:r for r in c['instances']}
        assert len(inst[case])==len(c['instances'])
    assert sum(map(len,inst.values()))==171, '171 predictions expected'
    review=csvread(source_files[2]);assert len(review)==EXPECTED_PAIRS
    hist={(r['case_id'],r['instance_id']):r for r in csvread(source_files[1])}
    assert len(hist)==171
    qwen=[json.loads(ln) for ln in source_files[3].read_text(encoding='utf-8').splitlines() if ln.strip()]
    assert len(qwen)==513,'Focus V1 513 unchanged records expected'
    original_ids=set((c,i) for c,im in inst.items() for i in im)
    qwen_key=Counter((r['case_id'],r['target_instance_id'],r['task']) for r in qwen)
    assert all(qwen_key[(c,i,t)]==1 for c,i in original_ids for t in ('tags','vqa','copy')), 'missing/duplicated downstream output'
    groups, candidate, blocked=analyze_pairs(review,inst,hist)
    alias={(r['case_id'],r['suppressed_instance_id']):r['proposed_canonical_instance_id'] for r in candidate}
    assert len(alias)==len(candidate)
    assert all((c,t) in original_ids and (c,i) in original_ids for (c,i),t in alias.items())
    assert all((c,t) not in alias for (c,i),t in alias.items()), 'candidate transitivity not resolved'
    # Preserve the original source review: user-confirmed in chat, not a verified exported CSV.
    accepted=[]
    for r in review:
        x=dict(r)
        x['confirmation_basis']='USER_SAID_REVIEWED_IN_CHAT_2026-11-06;ASSUMES_PREFILL_UNCHANGED_NO_EXPORTED_EDITS'
        x['independent_blind_review']='NO'
        x['safe_to_mainline']='NO'
        accepted.append(x)
    csvwrite(out/'43_pair_chat_attestation_with_original_columns.csv',accepted,list(accepted[0]))
    csvwrite(out/'conflict_groups_full43.csv',groups,list(groups[0]))
    csvwrite(out/'shadow_alias_proposals_strict.csv',candidate,['case_id','suppressed_instance_id','proposed_canonical_instance_id','mode','original_prediction_modified','qwen_output_migrated','eligible_for_mainline','group_status'])
    csvwrite(out/'blocked_conflict_groups.csv',blocked,list(groups[0]))
    other=[dict(r,processing='NO_ALIAS_NO_MERGE') for r in review if r['pair_relationship'] not in DUP]
    csvwrite(out/'mixed_uncertain_no_merge.csv',other,list(other[0]))
    registry=[]
    for c in cases:
        cid=c['case_id']
        for pred in c['instances']:
            pid=pred['instance_id']; keep=alias.get((cid,pid),'')
            registry.append({'case_id':cid,'instance_id':pid,'v5_category':pred['predicted_category'],
                'hypothetical_suppress':'YES' if keep else 'NO', 'hypothetical_alias_to':keep,
                'actual_prediction_preserved':'YES','qwen_text_reassigned':'NO',
                'ready_for_mainline':'NO', 'original_output_count_per_task':'1'})
    assert len(registry)==171
    csvwrite(out/'instance_registry_171_no_mutation.csv',registry,list(registry[0]))
    actual_ids=set((cid,x['instance_id']) for cid,ls in inst.items() for x in ls.values())
    assert actual_ids==original_ids
    rel_count=Counter(r['pair_relationship'] for r in review)
    group_count=Counter(g['status'] for g in groups)
    summary={
        'run_mode':'OFFLINE_SHADOW_ONLY','scope':'DEV60_TRAIN_SOURCE_ONLY_NOT_BLIND_HOLDOUT',
        'review_provenance':'USER_CHAT_CONFIRMED_PRIOR_AI_PREFILL;EXPORTED_EDITED_CSV_NOT_PROVIDED',
        'source_images':60,'source_instances':171,'source_qwen_outputs_untouched':513,
        'all_bbox_screened_pairs':43,'same_garment_pair_edges':sum(rel_count[x] for x in DUP),
        'mixed_pair_edges':sum(rel_count[x] for x in MIXED),
        'uncertain_pair_edges':sum(rel_count[x] for x in UNCERTAIN),
        'duplicate_graph_components':len(groups),'group_status_counts':dict(sorted(group_count.items())),
        'strict_alias_candidate_count':len(candidate),
        'hypothetical_display_count_if_proposals_approved':171-len(candidate),
        'original_instances_changed':0,'qwen_answers_migrated':0,'holdout_opened':False,
        'holdout_decision':'BLOCKED_MENTOR_APPROVAL_REQUIRED',
        'next_actions':['Inspect blocked multi-instance groups; confirm whether mixed edges involve separate garments.',
                        'Code-level instance binding interface validation (case_id, instance_id, image_sha256, bbox, mask).',
                        'Freeze ONE candidate and acceptance metrics before mentor-authorized blind test.'],
        'limitations':['Not an accuracy or recall gain; all counts based on fixed DEV60 predicted instances.',
                       'Chat confirmation accepts earlier AI prefill; downloaded HTML edits were not uploaded.',
                       'Do not merge mixed/uncertain edges into duplicate graph.',
                       'No reassignment of previously generated Qwen Tags/VQA/Copy.',
                       'Shadow proposals remain conditional and cannot enter mainline before independent blind validation.'],
        'input_sha256':{f.name:sha(f) for f in source_files}}
    (out/'shadow_full43_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--output-dir',type=Path,default=ROOT/'output')
    args=a.parse_args()
    print(json.dumps(run(output=args.output_dir),ensure_ascii=False,indent=2))
