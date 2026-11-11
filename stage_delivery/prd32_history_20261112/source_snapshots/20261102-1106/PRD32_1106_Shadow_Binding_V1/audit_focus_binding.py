#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check Focus V1's 513 technical target-binding metadata on DEV60, without any inference.

Structural metadata alignment IS NOT evidence that Qwen semantically answered the right garment.
Tags category vs V5 predicted category disagreement IS NOT a GT error rate.
"""
import csv,json,collections
from pathlib import Path
from shadow_collision_audit import load_and_validate,write_csv,ROOT


def audit():
    raw, ids, _, _=load_and_validate(ROOT/'input/v5_raw_predictions.jsonl',
                                     ROOT/'input/16_pairs_chat_accepted.csv',
                                     ROOT/'input/43_pairs_mask_audit.csv')
    images={r['case_id']:r for r in raw}
    orig_review={ (r['case_id'],r['instance_id']):r for r in
        __import__('shadow_collision_audit').read_csv(ROOT/'input/nov2_v5_attribution_171.csv')}
    records=[json.loads(s) for s in (ROOT/'input/focus_v1_qwen_results_513.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    assert len(records)==513, f'FocusV1 expected 513; got {len(records)}'
    per={}
    for r in records:
        case, pid, task=r['case_id'],r['target_instance_id'],r['task']
        assert case in ids and pid in ids[case], f'missing V5 instance {case}/{pid}'
        assert task in ('tags','vqa','copy'), f'unknown task {task}'
        k=(case,pid,task)
        assert k not in per, f'duplicate task key {k}'
        per[k]=r
    all_rows=[]; mismatch_count=collections.Counter();task_images=collections.Counter()
    for img in raw:
        case=img['case_id']; image_sha=img['image_sha256']
        for pred in img['instances']:
            pid=pred['instance_id']; obj=(case,pid)
            taskrows=[per[(case,pid,t)] for t in ('tags','vqa','copy')]
            issues=[]
            same_image_hash=set();same_marked_hash=set();
            for r in taskrows:
                task=r['task']
                expected_req=f'{case}__{pid}__{task}'
                if r.get('request_id')!=expected_req:issues.append('WRONG_REQUEST_ID_'+task)
                if r.get('image_sha256')!=image_sha:issues.append('IMAGE_HASH_MISMATCH_'+task)
                if r.get('v5_category_unverified')!=pred['predicted_category']:issues.append('V5_CATEGORY_METADATA_MISMATCH_'+task)
                if len(r.get('v5_predicted_bbox_xyxy') or [])!=4 or any(abs(float(a)-float(b))>.002 for a,b in zip(r['v5_predicted_bbox_xyxy'],pred['bbox_xyxy'])):
                    issues.append('BBOX_METADATA_MISMATCH_'+task)
                if r.get('model_variant')!='focus_v1':issues.append('WRONG_VARIANT_'+task)
                if r.get('human_label_or_gt_bbox_used') not in (False,None):issues.append('GT_LABEL_LEAK_FLAG_'+task)
                same_image_hash.add(r.get('image_sha256'))
                same_marked_hash.add(r.get('marked_image_sha256'))
            if len(same_image_hash)!=1:issues.append('THREE_TASKS_DIFFERENT_IMAGE_HASH')
            if len(same_marked_hash)!=1:issues.append('THREE_TASKS_DIFFERENT_MARKED_IMAGE_HASH')
            for i in issues:mismatch_count[i]+=1
            tag=taskrows[0]
            parsed=(tag.get('tag_parse') or {}).get('raw_parsed') or {}
            tag_cat=parsed.get('category')
            classstatus=(orig_review.get(obj) or {}).get('v5_category_quality','NO_REVIEW')
            all_rows.append({
                'case_id':case,'instance_id':pid,'v5_predicted_category':pred['predicted_category'],
                'v5_category_historical_audit':classstatus,
                'qwen_tags_parsed_category':tag_cat or '',
                'tag_category_vs_v5':'SAME' if tag_cat==pred['predicted_category'] else 'DISAGREE_NOT_GT_ERROR',
                'focus_three_tasks_present':'YES','structural_metadata_consistent':'YES' if not issues else 'NO',
                'metadata_issue_types':'|'.join(issues),
                'semantic_target_drift_evaluated':'NO',
                'limitation':'A technically matched red box can still make Qwen describe the wrong garment.'})
    assert len(all_rows)==171
    summary={
        'scope':'DEV60_FOCUS_V1_ORIGINAL_QWEN_LOG_ONLY',
        'total_task_outputs':len(records),
        'unique_instances':len(all_rows),
        'metadata_issue_count_instances':sum(bool(r['metadata_issue_types']) for r in all_rows),
        'metadata_issue_types':dict(mismatch_count),
        'three_tasks_complete':sum(r['focus_three_tasks_present']=='YES' for r in all_rows),
        'tags_v5_category_disagreement_count':sum(r['tag_category_vs_v5']!='SAME' for r in all_rows),
        'tags_v5_category_agreement_count':sum(r['tag_category_vs_v5']=='SAME' for r in all_rows),
        'audit_v5_class_correct_count':sum(r['v5_category_historical_audit']=='CORRECT' for r in all_rows),
        'tags_v5_disagree_when_v5_class_review_correct':sum(r['v5_category_historical_audit']=='CORRECT' and r['tag_category_vs_v5']!='SAME' for r in all_rows),
        'audit_v5_class_incorrect_count':sum(r['v5_category_historical_audit']=='INCORRECT' for r in all_rows),
        'downstream_content_verification_done':False,
        'independent_holdout_evaluated':False,
        'interpretation': 'Metadata alignment can rule out selected ID/bbox/hash propagation faults in saved Focus V1 logs, but cannot rule out pixel-target drift, categorization error or prompt-induced semantic interference. Category disagreement is not a GT error rate.'
    }
    out=ROOT/'output';out.mkdir(exist_ok=True)
    write_csv(out/'focus_v1_binding_metadata_audit_171.csv',all_rows,list(all_rows[0]))
    (out/'focus_v1_binding_metadata_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

if __name__=='__main__':audit()
