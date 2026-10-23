#!/usr/bin/env python3
"""Recompute existing evidence without relabeling any original human decision."""
from collections import Counter
from common import BASE, REGIONS, ATTRS, rows, load, dump, csvout, sha

def main():
    data=rows(BASE/'reference/stage15_human_ORIGINAL.csv');cohort=load(BASE/'reference/cohort.json')
    ids={c['case_id']:c for c in cohort}; seen=set();tasks=[];follow=[]
    for r in data:
        key=(r['case'],r['region'])
        if key in seen:raise ValueError('Duplicate original task')
        seen.add(key)
        if r['source_sha256']!=ids[r['case']]['source_sha256']:raise ValueError('Wrong image join')
        presence=r['human_region_presence_yes_no_uncertain']; grade=r['human_localization_correct_coarse_wrong_missed_uncertain_not_applicable']
        if not presence or not grade or not r['human_review_notes']:raise ValueError('Unexpected missing prior human answer')
        reasons=[]
        if not r['human_reviewer']:reasons.append('reviewer_missing')
        if presence=='yes' and not r['human_bbox_xyxy_if_drawn']:reasons.append('human_bbox_missing')
        if r['case']=='case_10':reasons.append('foreground_vs_background_target_identity')
        if r['case']=='case_11' and r['region']=='pocket':reasons.append('absence_vs_hidden_pocket_conflict')
        if presence=='no' and r['prediction_status']=='detected':reasons.append('potential_absent_part_false_positive')
        if reasons:follow.append({'case_id':r['case'],'region':r['region'],'issues':';'.join(reasons),'original_note':r['human_review_notes'],'action':'human_confirm_append_only'})
    for c in cohort:
        tasks.append({'case_id':c['case_id'],'source_sha256':c['source_sha256'],'task_type':'garment','task':'target','available':'predicted_crop_reference','pending':'human_target_box_and_category'})
        for name in REGIONS:
            tasks.append({'case_id':c['case_id'],'source_sha256':c['source_sha256'],'task_type':'region','task':name,
                'available':'original_qualitative_review' if (c['case_id'],name) in seen else 'no_prior_review',
                'pending':'human_bbox_review_and_same_target_check'})
        for name in ATTRS:
            tasks.append({'case_id':c['case_id'],'source_sha256':c['source_sha256'],'task_type':'attribute','task':name,'available':'annotation_vocabulary','pending':'human_label_and_same_image_prediction'})
    out=BASE/'results';csvout(out/'same12_task_register.csv',tasks);csvout(out/'review_followup.csv',follow)
    summary={'original_human_rows':len(data),'unique_case_region_keys':len(seen),'unique_images':len(ids),
      'source_hash_joins_verified':True,'reviewer_missing':sum(not r['human_reviewer'] for r in data),
      'presence_counts':dict(Counter(r['human_region_presence_yes_no_uncertain'] for r in data)),
      'grade_counts':dict(Counter(r['human_localization_correct_coarse_wrong_missed_uncertain_not_applicable'] for r in data)),
      'task_register_count':len(tasks),'new_human_annotations_completed':0,'new_gpu_inference_completed':False,
      'original_review_sha256':sha(BASE/'reference/stage15_human_ORIGINAL.csv'),'original_review_modified':False}
    dump(out/'offline_audit_summary.json',summary);print(summary)

if __name__=='__main__':main()
