#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce 2026-11-06 Dev60 root-cause diagnostic. Standard library only.
Reads DEV data and user-attested review records. NEVER loads a holdout set.
The review annotations are diagnostic labels, NOT inference inputs or independent GT.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv, json, math, hashlib, sys
BASE=Path(__file__).resolve().parent
I=BASE/'input'; O=BASE/'output'
def rows(path):
    with open(path,encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write_rows(path,fields,data):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data)
def intersection(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def area(b): return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def iou(a,b):
    x=intersection(a,b);u=area(a)+area(b)-x
    return x/u if u>0 else 0

def main():
    O.mkdir(parents=True,exist_ok=True)
    subtype=rows(I/'nov4_focus_crop_confirmed_171.csv')
    v5_audit=rows(I/'nov2_v5_attribution_171.csv')
    nov5=rows(I/'nov5_copy_revised_171.csv')
    v5_pred=[json.loads(x) for x in (I/'v5_raw_predictions.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    assert len(subtype)==171 and len(v5_audit)==171 and len(v5_pred)==60 and len(nov5)==171
    audits={(r['case_id'],r['instance_id']):r for r in v5_audit}
    subtypes={(r['case_id'],r['instance_id'].split('/')[-1]):r for r in subtype}
    nov5score={(r['case_id'],r['instance_id']):r for r in nov5}
    allrows=[]; manifest=[]; flags=Counter(); pair_geom=Counter(); categories=Counter()
    seen=set()
    for p in v5_pred:
        c=p['case_id'];h=p['image_sha256'];w,hpix=p['image_size'];instances=p['instances']
        manifest.append({'case_id':c,'image_sha256':h,'count_v5_predictions':len(instances)})
        for idx,a in enumerate(instances):
            pred=a['instance_id'];key=(c,pred)
            assert key not in seen and key in audits and key in subtypes and key in nov5score,key
            seen.add(key)
            av=audits[key];sv=subtypes[key];cv=nov5score[key]
            assert sv['image_sha256']==h and cv['image_sha256']==h,key
            bbox=a['bbox_xyxy']; assert len(bbox)==4 and area(bbox)>0
            assert (-0.001<=bbox[0]<bbox[2]<=w+.001 and -0.001<=bbox[1]<bbox[3]<=hpix+.001),key
            partners=[(b['instance_id'],b['predicted_category'],iou(bbox,b['bbox_xyxy'])) for j,b in enumerate(instances) if j!=idx]
            pmax=max(partners,key=lambda t:t[2]) if partners else ('','',0.)
            overlap50=[q for q in partners if q[2]>=.5]
            box_ok=av['bbox_quality']=='CORRECT'
            cls_ok=av['v5_category_quality']=='CORRECT'
            focus=sv['focus_subtype']
            if focus=='SPATIAL_DRIFT':priority='P0_SPATIAL_TARGET'
            elif focus=='SEMANTIC_CATEGORY_INTERFERENCE':priority='P0_SEMANTIC_CONFIRMED'
            elif focus=='SEMANTIC_CANDIDATE':priority='P1_SEMANTIC_CANDIDATE'
            elif focus=='SPATIAL_OR_SEMANTIC_UNRESOLVED':priority='P1_BINDING_UNRESOLVED'
            elif focus=='UPSTREAM_CONFOUNDED':priority='P1_UPSTREAM_CONFOUNDED'
            elif not box_ok or not cls_ok:priority='P2_UPSTREAM_OR_UNCERTAIN'
            else:priority='P2_OTHER_OR_CLEAR'
            category=(a['predicted_category'] or '').lower()
            categories[category]+=1
            flags[focus]+=1
            if pmax[2]>=.5: pair_geom['instances_max_iou_ge_0.5']+=1
            if pmax[2]>=.7: pair_geom['instances_max_iou_ge_0.7']+=1
            if any(x[2]>=.5 and x[1]!=category for x in partners):pair_geom['instances_cross_category_overlap_ge_0.5']+=1
            if box_ok and cls_ok:pair_geom['v5_bbox_and_class_both_reviewed_correct']+=1
            row={
                'case_id':c,'instance_id':pred,'target_uid':c+'__'+pred,'image_sha256':h,
                'v5_category':category,'v5_score':a['score'],'bbox_xyxy':json.dumps(bbox,separators=(',',':')),
                'bbox_area_ratio':round(area(bbox)/(w*hpix),4),'instances_in_image':len(instances),
                'max_bbox_iou_other_prediction':round(pmax[2],4),'max_iou_partner':pmax[0],
                'partner_v5_category':pmax[1],'overlap_ge_0.5_count':len(overlap50),
                'has_cross_category_overlap_ge_0.5':any(x[2]>=.5 and x[1]!=category for x in partners),
                'v5_bbox_review':av['bbox_quality'],'v5_class_review':av['v5_category_quality'],
                'nov2_responsibility':av['responsibility'],'nov2_qwen_focus_binding':av['qwen_binding_focus'],
                'focus_error_subtype':focus,'crop_error_subtype':sv['crop_subtype'],
                'nov4_review_status':sv['review_status'],
                'focus_copy_revised_1105':cv['focus_revised_20261105'],
                'cv1_copy_revised_1105':cv['cv1_revised_20261105'],
                'rootcause_priority':priority,
                'evidence_grade':'USER_ACCEPTED_AI_REVIEW_DEV_NOT_BLIND_CAUSAL',
            }
            allrows.append(row)
    assert len(seen)==171 and len(set((r['case_id'],r['image_sha256']) for r in manifest))==60
    assert len(set(r['image_sha256'] for r in manifest))==60
    assert len(set((r['case_id'],r['instance_id']) for r in allrows))==171
    write_rows(O/'DEV60_IMAGE_FROZEN_MANIFEST.csv',list(manifest[0]),sorted(manifest,key=lambda x:x['case_id']))
    allrows.sort(key=lambda r:(r['rootcause_priority'],r['case_id'],r['instance_id']))
    write_rows(O/'171_INSTANCE_ROOTCAUSE_FEATURE_MATRIX.csv',list(allrows[0]),allrows)
    focus_counts=Counter(r['focus_error_subtype'] for r in allrows)
    bypriority=Counter(r['rootcause_priority'] for r in allrows)
    both=Counter((r['v5_bbox_review'],r['v5_class_review']) for r in allrows)
    spatial=[r for r in allrows if r['focus_error_subtype']=='SPATIAL_DRIFT']
    semantic=[r for r in allrows if r['focus_error_subtype']=='SEMANTIC_CATEGORY_INTERFERENCE']
    candidates=[r for r in allrows if r['focus_error_subtype']=='SEMANTIC_CANDIDATE']
    low_conf=[r for r in allrows if r['focus_error_subtype'] in ('SPATIAL_OR_SEMANTIC_UNRESOLVED','UNCERTAIN')]
    assert focus_counts['SPATIAL_DRIFT']==5 and focus_counts['SEMANTIC_CATEGORY_INTERFERENCE']==1 and focus_counts['SEMANTIC_CANDIDATE']==5
    nclean=sum(r['v5_bbox_review']=='CORRECT' and r['v5_class_review']=='CORRECT' for r in allrows)
    official={'schema':'PRD32_DEV60_ROOTCAUSE_DIAGNOSTIC_20261106_V1',
       'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY_NOT_OFFICIAL_HOLDOUT',
       'source_provenance':'V5 frozen 2026-10-30; 11-02 V5 audit; 11-04 accepted AI-assisted error subtypes; 11-05 revised Copy scores',
       'unique_images':60,'v5_predictions':171,'v5_category_counts':dict(categories),
       'nov4_focus_subtypes':dict(focus_counts),'priority_queue_counts':dict(bypriority),
       'bbox_and_class_both_reviewed_correct':nclean,
       'v5_bbox_review':dict(Counter(r['v5_bbox_review'] for r in allrows)),
       'v5_category_review':dict(Counter(r['v5_class_review'] for r in allrows)),
       'bbox_geometry_flags':dict(pair_geom),
       'note_bbox_iou':'Bounding-box overlap risk signal only; not duplicate GT proof or mask IoU.',
       'key_case_ids':{'focus_spatial':[r['target_uid'] for r in spatial],
                       'focus_semantic_confirmed':[r['target_uid'] for r in semantic],
                       'focus_semantic_candidates':[r['target_uid'] for r in candidates]},
       'n_ambiguous_spatial_semantic_or_unclassified':len(low_conf),
       'official_holdout_used':False,'w7_holdout_assumed_official':False,
       'no_additional_gpu_inference':True,
       'mechanism_not_proven_causal':True,'source_chats_global_attestation_not_independent_blind':True,
    }
    (O/'ROOTCAUSE_DEV60_SUMMARY.json').write_text(json.dumps(official,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in official.items() if k in ('unique_images','v5_predictions','v5_bbox_review','v5_category_review','nov4_focus_subtypes','bbox_and_class_both_reviewed_correct','bbox_geometry_flags','priority_queue_counts')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
