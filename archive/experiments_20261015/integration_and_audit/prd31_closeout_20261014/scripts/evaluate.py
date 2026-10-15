#!/usr/bin/env python3
"""Pair human-confirmed annotations and same-image predictions; retain every task.

Results are development diagnostics with separate garment/region/attribute
denominators. 'Observed failing stage' is not a counterfactual causal proof.
"""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
from common import BASE, ATTRS, REGIONS, load, dump, csvout, sha, pixels, iou, validate_review

def evaluate(review, pred, cohort, taxonomy, threshold=.5):
    pending=validate_review(review,cohort,taxonomy)
    if pred.get('schema_version')!='prd31_same12_predictions_v1' or pred.get('manifest_sha256')!=review['manifest_sha256']:
        raise ValueError('Prediction schema/cohort mismatch')
    expected={c['case_id']:c for c in cohort}
    if set(pred.get('cases',{}))-set(expected):raise ValueError('Unexpected prediction case IDs')
    records=[]
    for cid,c in expected.items():
        human=review['cases'][cid]; p=pred.get('cases',{}).get(cid)
        if p and p.get('source_sha256')!=c['source_sha256']:raise ValueError('Prediction source mismatch: '+cid)
        base={'case_id':cid,'source_sha256':c['source_sha256']}
        ready=cid not in pending
        garment_status='human_review_pending'; match=None; match_iou=None
        if ready:
            if not human.get('target_box') or human['category']=='uncertain' or human['target_status']=='uncertain':garment_status='human_target_unresolved'
            elif not p:garment_status='prediction_not_run'
            else:
                box=pixels(human['target_box'],p['image_size'])
                candidates=p.get('instances',[])
                match=max(candidates,key=lambda x:iou(box,x.get('bbox')),default=None)
                match_iou=iou(box,match.get('bbox')) if match else 0
                if p.get('pipeline_status')!='completed':garment_status='upstream_execution_failed'
                elif not match or match_iou<threshold:garment_status='upstream_no_bbox_match';match=None
                elif match.get('category')!=human['category']:garment_status='upstream_wrong_class'
                else:garment_status='bbox_and_class_match'
        upstream_ok=garment_status=='bbox_and_class_match'
        records.append({**base,'task_type':'garment','task':'target','status':garment_status,'value':match_iou,
                        'human_label':human.get('category',''),'prediction':match.get('category','') if match else '',
                        'detail':'bbox matching only; no human garment mask IoU'})
        region_status={}
        for region in REGIONS:
            h=human.get('regions',{}).get(region,{}); a=(match or {}).get('regions',{}).get(region,{})
            v=None
            if not ready:s='human_review_pending'
            elif not h.get('presence'):s='human_presence_missing'
            elif h['presence']=='uncertain':s='human_presence_uncertain'
            elif garment_status in ['prediction_not_run','human_target_unresolved']:s=garment_status
            elif not upstream_ok:s='blocked_by_upstream'
            elif h['presence']=='no':s='absent_false_positive' if a.get('status')=='detected' else 'absent_no_detection'
            elif h.get('boxes_complete') is not True or not h.get('boxes'):s='human_part_boxes_incomplete'
            elif not a:s='missing_region_output'
            elif a.get('status')!='detected':s='region_not_detected'
            else:
                v=max(iou(pixels(b,p['image_size']),a.get('bbox')) for b in h['boxes'])
                s='region_bbox_pass' if v>=threshold else 'region_bbox_fail'
            region_status[region]=s
            records.append({**base,'task_type':'region','task':region,'status':s,'value':v,
                'human_label':h.get('presence',''),'prediction':a.get('status',''),
                'detail':'any-visible-instance query; complete visible-box set required; not mask accuracy'})
        for attr in ATTRS:
            label=human.get('labels',{}).get(attr,'');a=(match or {}).get('attributes',{}).get(attr,{})
            if not ready:s='human_review_pending'
            elif not label:s='human_label_missing'
            elif label in ['uncertain','not_applicable']:s='human_'+label
            elif garment_status in ['prediction_not_run','human_target_unresolved']:s=garment_status
            elif not upstream_ok:s='blocked_by_upstream'
            elif a.get('status') in ['localization_missing','localization_failed']:s='localization_dependency_missing'
            elif a.get('status')!='predicted' or not a.get('label'):s='attribute_no_prediction'
            elif a.get('label') not in taxonomy[attr]:s='taxonomy_mismatch'
            elif a.get('label')==label:s='attribute_correct'
            elif attr=='neckline' and region_status.get('collar') in ['region_bbox_fail','region_not_detected','missing_region_output']:
                s='attribute_wrong_with_localization_error'
            else:s='attribute_wrong_cause_unresolved'
            records.append({**base,'task_type':'attribute','task':attr,'status':s,'value':None,
                'human_label':label,'prediction':a.get('label',''),'detail':'label comparison; joint causes remain possible'})
    counts={kind:dict(Counter(r['status'] for r in records if r['task_type']==kind)) for kind in ['garment','region','attribute']}
    metrics=[]
    for attr in ATTRS:
        items=[x for x in records if x['task_type']=='attribute' and x['task']==attr]
        measurable={'attribute_correct','attribute_wrong_cause_unresolved','attribute_wrong_with_localization_error',
                    'attribute_no_prediction','localization_dependency_missing','blocked_by_upstream'}
        denom=sum(x['status'] in measurable for x in items)
        correct=sum(x['status']=='attribute_correct' for x in items)
        metrics.append({'attribute':attr,'all_tasks':len(items),'eligible_run_tasks':denom,'correct':correct,
                        'accuracy':correct/denom if denom else None,'excluded_or_pending':len(items)-denom,
                        'scope':'development_selected_targets; upstream/output failures included'})
    result={'status':'PARTIAL_REVIEW' if pending else 'REVIEW_ENTERED',
        'human_pending_cases':pending,'task_counts':{'garment':12,'region':60,'attribute':72},
        'status_counts':counts,'attribute_metrics':metrics,'bbox_iou_diagnostic_threshold':threshold,
        'formal_prd_acceptance':False,'causal_three_way_error_percentage':None,
        'limitations':['Different task units are not pooled into one accuracy or error fraction.',
        'BBox threshold is a diagnostic protocol, not an approved PRD region metric.',
        'Uncertain, unannotated, unrun and taxonomy-mismatch rows are retained and reported.',
        'No human mask GT; current cohort only 12 development images.',
        'Even correct garment bbox/class does not prove mask quality; attribution is observational.']}
    return records,result

def main():
    p=argparse.ArgumentParser();p.add_argument('--review',required=True,type=Path)
    p.add_argument('--predictions',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    cohort=load(BASE/'reference/cohort.json');tax=load(BASE/'reference/taxonomy.json')
    records,summary=evaluate(load(a.review),load(a.predictions),cohort,tax)
    csvout(a.out/'unified_task_results.csv',records);dump(a.out/'evaluation_summary.json',summary)
    dump(a.out/'input_hashes.json',{'review':sha(a.review),'predictions':sha(a.predictions),'cohort':sha(BASE/'reference/cohort.json')})
    print(summary['status']);print(a.out.resolve())

if __name__=='__main__':main()
