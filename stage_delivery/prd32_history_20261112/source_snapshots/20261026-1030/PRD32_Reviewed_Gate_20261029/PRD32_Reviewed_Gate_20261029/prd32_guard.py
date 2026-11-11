#!/usr/bin/env python3
"""PRD 3.2 GT-free static service gate + separate human review backtest.

Input: frozen V5 predicted-instance Qwen results (tags, vqa, copy).
Runtime gate NEVER reads human review unless --review is explicitly supplied;
even then review data is used only by an OFFLINE evaluation after decisions.
Does not alter model text, masks, boxes, labels, source files, or publish anything.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

TASKS = ("tags", "vqa", "copy")
EXPECTED_REVIEW_IDS = {
    ("fresh_dev_001", "pred_0000"), ("fresh_dev_001", "pred_0001"),
    ("fresh_dev_004", "pred_0000"), ("fresh_dev_004", "pred_0001"),
    ("fresh_dev_004", "pred_0002"),
}
ENUMS = {
    "instance_binding": {"CORRECT", "WRONG_INSTANCE", "UNCERTAIN", "UNREVIEWED"},
    "duplicate_assessment": {"DUPLICATE", "DISTINCT", "UNCERTAIN", "UNREVIEWED"},
    "tags_quality": {"CORRECT", "INCORRECT", "UNCERTAIN", "UNREVIEWED"},
    "vqa_quality": {"CORRECT", "INCORRECT", "ABSTENTION_REASONABLE", "UNCERTAIN", "UNREVIEWED"},
    "copy_quality": {"CORRECT", "WRONG_INSTANCE", "UNSUPPORTED_CLAIM", "DEBUG_TEXT", "OTHER_ERROR", "UNCERTAIN", "UNREVIEWED"},
}
DEBUG_RE = re.compile(r"红色(?:边框|方框|矩形框|框)|红框|标注框|框选|bounding\s*box|\bbbox\b", re.I)
ABSTAIN_RE = re.compile(r"无法确定|不能确定|无法判断|看不清|无法辨认|不确定|不知道")
COLORS = {
    "white": ("白色", "米白", "乳白"), "black": ("黑色",),
    "blue": ("蓝色", "牛仔蓝", "天蓝"), "yellow": ("黄色", "明黄", "姜黄"),
    "orange": ("橙色", "橘色", "橘黄"), "green": ("绿色",),
    "red": ("红色", "酒红", "玫红"), "pink": ("粉色", "粉红"),
    "gray": ("灰色",), "brown": ("棕色", "褐色", "咖啡色"),
    "purple": ("紫色",),
}
CATEGORY_TEXT = (
    ("dress", ("连衣裙", "连身裙")),
    ("pants", ("短裤", "长裤", "牛仔裤", "裤装", "裤子", "阔腿裤")),
    ("skirt", ("半身裙", "百褶裙", "蓬蓬裙", "短裙", "裙子")),
    ("outerwear", ("外套", "大衣", "风衣", "夹克")),
    ("top", ("上衣", "背心", "T恤", "衬衫", "卫衣", "针织衫")),
    ("shoe", ("鞋子", "高跟鞋", "运动鞋")),
    ("bag", ("背包", "手提包", "挎包")),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda: f.read(1024*1024), b''): h.update(c)
    return h.hexdigest()


def save_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')


def valid_box(b: Any) -> bool:
    if not isinstance(b, list) or len(b) != 4: return False
    if not all(type(v) in (int,float) and math.isfinite(v) for v in b): return False
    return b[0] >= 0 and b[1] >= 0 and b[2] > b[0] and b[3] > b[1]


def iou(a: list[float], b: list[float]) -> float:
    if not valid_box(a) or not valid_box(b): return 0.0
    intersect = max(0, min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area1=(a[2]-a[0])*(a[3]-a[1]);area2=(b[2]-b[0])*(b[3]-b[1])
    return intersect/(area1+area2-intersect) if area1+area2>intersect else 0.0


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def colors(text: str) -> set[str]:
    return {c for c, words in COLORS.items() if any(w in text for w in words)}


def copy_class(text: str) -> str | None:
    if '裙裤' in text or '裤裙' in text: return None
    candidates=[cat for cat, words in CATEGORY_TEXT if any(w in text for w in words)]
    return candidates[0] if len(candidates)==1 else None


def parse_tags(row: dict) -> dict | None:
    p=row.get('tag_parse') or {}
    for key in ('canonical_service_json','raw_parsed'):
        if isinstance(p.get(key),dict): return p[key]
    return None


def validate_runtime(rows: list[dict]) -> dict:
    """Fail closed for unsafe provenance / inconsistent task identities."""
    if not isinstance(rows, list) or not rows:
        raise ValueError('EMPTY_OR_INVALID_REQUESTS')
    by_key: dict[tuple[str,str],dict[str,dict]] = {}
    for row in rows:
        if not isinstance(row,dict): raise ValueError('NON_OBJECT_REQUEST')
        key=(row.get('case_id'), row.get('target_instance_id'))
        if not all(isinstance(x,str) and x for x in key): raise ValueError('MISSING_INSTANCE_ID')
        task=row.get('task')
        if task not in TASKS: raise ValueError('UNEXPECTED_TASK')
        if row.get('human_label_or_gt_bbox_used') is not False: raise ValueError('GT_CONTAMINATION')
        if row.get('publish_allowed') is not False: raise ValueError('UNSAFE_PUBLISH_PERMISSION')
        if row.get('target_bbox_origin')!='prd31_v5_live_prediction': raise ValueError('GT_OR_UNKNOWN_BBOX_ORIGIN')
        if not valid_box(row.get('v5_predicted_bbox_xyxy')): raise ValueError('INVALID_V5_BOX')
        if not isinstance(row.get('v5_category_unverified'),str): raise ValueError('MISSING_V5_CATEGORY')
        record=by_key.setdefault(key,{})
        if task in record: raise ValueError('DUPLICATE_TASK_REQUEST')
        record[task]=row
    for key, group in by_key.items():
        if set(group)!=set(TASKS): raise ValueError(f'INCOMPLETE_INSTANCE_TASKS:{key}')
        classes={r['v5_category_unverified'] for r in group.values()}
        boxes={tuple(r['v5_predicted_bbox_xyxy']) for r in group.values()}
        if len(classes)!=1 or len(boxes)!=1: raise ValueError(f'INCONSISTENT_INSTANCE_PROVENANCE:{key}')
        if any(r.get('status')!='GENERATED_REVIEW_REQUIRED' for r in group.values()):
            raise ValueError(f'UNEXPECTED_GENERATION_STATUS:{key}')
    return by_key


def runtime_gate(rows: list[dict], threshold: float=.80) -> dict:
    """Generic gate: no human label or review accessed and NO automatic dedup."""
    if not 0 < threshold <=1: raise ValueError('INVALID_IOU_THRESHOLD')
    by_key=validate_runtime(rows)
    candidates=[]
    for case in sorted({k[0] for k in by_key}):
        keys=sorted((k for k in by_key if k[0]==case),key=lambda k:k[1])
        for ia,a in enumerate(keys):
            for b in keys[ia+1:]:
                ba=by_key[a]['tags']['v5_predicted_bbox_xyxy']
                bb=by_key[b]['tags']['v5_predicted_bbox_xyxy']
                overlap=iou(ba,bb)
                if overlap >= threshold:
                    candidates.append({"case_id":case,"instance_a":a[1],"instance_b":b[1],"box_iou":round(overlap,6),
                                       "status":"POSSIBLE_DUPLICATE_NOT_AUTOMATIC_MERGE",
                                       "note":"bbox IoU only; review masks / garment layering before dedup"})
    involved={(p['case_id'], i) for p in candidates for i in (p['instance_a'],p['instance_b'])}
    results=[]
    for key, grp in sorted(by_key.items()):
        t=parse_tags(grp['tags']);tags=t or {}
        v5=grp['tags']['v5_category_unverified']
        copy=grp['copy'].get('raw_text','')
        vqa=grp['vqa'].get('raw_text','')
        if not isinstance(copy,str) or not isinstance(vqa,str): raise ValueError('INVALID_MODEL_TEXT')
        flags=[];hard=[]
        if t is None: flags.append('TAGS_UNPARSEABLE');hard.append('TAGS_UNPARSEABLE')
        if t is not None and tags.get('category')!=v5:
            flags.append('V5_QWEN_CATEGORY_DISAGREEMENT')
        inferred=copy_class(copy)
        if inferred and t is not None and isinstance(tags.get('category'),str) and inferred!=tags['category']:
            hard.append('COPY_CATEGORY_CONFLICT_WITH_TAGS')
        cleaned=DEBUG_RE.sub('',copy)
        tcolor=colors(str(tags.get('primary_color','')))
        ccolor=colors(cleaned)
        if len(tcolor)==1 and len(ccolor)==1 and tcolor.isdisjoint(ccolor):
            hard.append('COPY_COLOR_CONFLICT_WITH_TAGS')
        if DEBUG_RE.search(copy): hard.append('COPY_CONTAINS_DEBUG_MARKER')
        if ABSTAIN_RE.search(vqa): flags.append('VQA_ABSTENTION_REQUIRES_VALIDITY_CHECK')
        if tags.get('pattern') in ('纯色','无','无图案') and ('字样' in copy or 'logo' in copy.lower() or '印花' in copy):
            flags.append('SOLID_PATTERN_WITH_VISIBLE_TEXT_CLAIM_REVIEW')
        if key in involved: flags.append('HIGH_BOX_IOU_DUPLICATE_CANDIDATE')
        flags=sorted(set(flags));hard=sorted(set(hard))
        copy_status='BLOCKED' if hard else 'REVIEW_REQUIRED'
        row={"case_id":key[0],"instance_id":key[1],"v5_category":v5,
             "qwen_tag_category":tags.get('category'),"v5_score":grp['tags'].get('v5_predicted_score_uncalibrated'),
             "bbox":grp['tags']['v5_predicted_bbox_xyxy'],
             "task_request_ids":{task:grp[task].get('request_id') for task in TASKS},
             "duplicate_candidate":key in involved,"flags":flags,"copy_block_reasons":hard,
             "copy_disposition":copy_status,
             "vqa_disposition":'ABSTAINED_UNVERIFIED' if ABSTAIN_RE.search(vqa) else 'REVIEW_REQUIRED',
             "tags_disposition":'REVIEW_REQUIRED',
             "service_disposition":'BLOCKED_AND_REVIEW' if hard or key in involved else 'REVIEW_REQUIRED',
             "auto_merge_performed":False,"publish_allowed":False}
        results.append(row)
    return {"schema":"prd32_gt_free_runtime_consistency_gate_v1", "mode":"RUNTIME_ONLY_NO_HUMAN_GT",
      "scope":"TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY", "images":len({r['case_id'] for r in results}),
      "instances":len(results),"requests":len(rows),"duplicate_iou_threshold":threshold,
      "duplicate_candidate_pairs":candidates,"rows":results,
      "counts":{"copy_blocked":sum(r['copy_disposition']=='BLOCKED' for r in results),
                "duplicate_candidate_instances":sum(r['duplicate_candidate'] for r in results),
                "vqa_abstentions":sum(r['vqa_disposition']=='ABSTAINED_UNVERIFIED' for r in results),
                "v5_qwen_class_disagreements":sum('V5_QWEN_CATEGORY_DISAGREEMENT' in r['flags'] for r in results)},
      "publish_allowed":False,"auto_merge_performed":False,"gt_used_to_choose_instance":False,
      "model_calls":0,"note":"Keyword checks are conservative heuristics, not human semantic ground truth. Original outputs unchanged."}


def validate_review(doc:dict, expected_keys:set[tuple[str,str]]) -> dict:
    errors=[]
    if doc.get('schema')!='prd32_v5_qwen_human_review_v1': errors.append('INVALID_REVIEW_SCHEMA')
    if doc.get('scope')!='TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY': errors.append('INVALID_REVIEW_SCOPE')
    if doc.get('reviewed_by_human') is not True: errors.append('NOT_HUMAN_ATTESTED')
    r=doc.get('rows')
    if not isinstance(r,list): errors.append('INVALID_ROWS');r=[]
    byid={}
    for x in r:
        if not isinstance(x,dict):errors.append('NON_OBJECT_REVIEW_ROW');continue
        k=(x.get('case_id'),x.get('instance_id'))
        if k in byid: errors.append(f'DUPLICATE_REVIEW_ID:{k}')
        byid[k]=x
        for fld, enum in ENUMS.items():
            if x.get(fld) not in enum or x.get(fld)=='UNREVIEWED':errors.append(f'INVALID_REVIEW_{fld}:{k}')
        if not isinstance(x.get('reviewer_note'),str) or not x['reviewer_note'].strip():errors.append(f'MISSING_REVIEW_NOTE:{k}')
    if set(byid)!=expected_keys:errors.append('REVIEW_IDENTITY_SET_MISMATCH')
    for case,a,b in [('fresh_dev_001','pred_0000','pred_0001'),('fresh_dev_004','pred_0000','pred_0002')]:
        if (case,a) in byid and (case,b) in byid and byid[(case,a)]['duplicate_assessment']!=byid[(case,b)]['duplicate_assessment']:
            errors.append(f'ASYMMETRIC_PAIR:{case}')
    return {"valid":not errors,"errors":errors,"reviewed_by_human":doc.get('reviewed_by_human') is True,
            "reviewed_instances":len(byid),"counts":{f:dict(collections.Counter(x.get(f) for x in byid.values())) for f in ENUMS}}


def offline_backtest(runtime:dict, review:dict) -> dict:
    """Review only evaluates frozen outcomes, NEVER modifies runtime decisions."""
    expected={(r['case_id'],r['instance_id']) for r in runtime['rows']}
    check=validate_review(review, expected)
    if not check['valid']: raise ValueError('HUMAN_REVIEW_INVALID: '+';'.join(check['errors']))
    byid={(x['case_id'],x['instance_id']):x for x in review['rows']}
    pairs=[]
    for p in runtime['duplicate_candidate_pairs']:
        a=(p['case_id'],p['instance_a']);b=(p['case_id'],p['instance_b'])
        pairs.append({"case_id":p['case_id'],"instance_a":a[1],"instance_b":b[1],
                      "human_assessment_a":byid[a]['duplicate_assessment'],
                      "human_assessment_b":byid[b]['duplicate_assessment'],
                      "confirmation":"BOTH_DUPLICATE" if byid[a]['duplicate_assessment']==byid[b]['duplicate_assessment']=='DUPLICATE' else 'NOT_BOTH_CONFIRMED'})
    copies_wrong={(k) for k,v in byid.items() if v['copy_quality'] not in ('CORRECT','UNCERTAIN')}
    copies_blocked={(r['case_id'],r['instance_id']) for r in runtime['rows'] if r['copy_disposition']=='BLOCKED'}
    review_fail_vqa={(k) for k,v in byid.items() if v['vqa_quality']=='INCORRECT'}
    vqa_flagged={(r['case_id'],r['instance_id']) for r in runtime['rows'] if r['vqa_disposition']=='ABSTAINED_UNVERIFIED'}
    return {"schema":"prd32_review_backtest_v1","separate_from_runtime":True,"review_validation":check,
            "human_confirmed_duplicate_pairs":pairs,
            "copy_human_errors_n":len(copies_wrong),"copy_heuristic_blocked_n":len(copies_blocked),
            "copy_human_errors_caught_n":len(copies_wrong & copies_blocked),
            "copy_false_positive_in_reviewed_5_n":len(copies_blocked-copies_wrong),
            "vqa_human_errors_n":len(review_fail_vqa),"vqa_abstention_flags_n":len(vqa_flagged),
            "vqa_human_errors_flagged_n":len(review_fail_vqa & vqa_flagged),
            "in_sample_only":True,"not_independent_validation":True,"no_runtime_selection_from_human_review":True,
            "publish_allowed":False}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qwen', required=True, type=Path)
    p.add_argument('--review',type=Path,default=None,help='Optional offline backtest only; never read by runtime logic')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--duplicate-iou-threshold',type=float,default=.80)
    args=p.parse_args()
    if args.output.exists():raise SystemExit('Refusing to overwrite existing output: '+str(args.output))
    raw=read_jsonl(args.qwen)
    runtime=runtime_gate(raw,args.duplicate_iou_threshold) # review has NOT been opened yet
    args.output.mkdir(parents=True,exist_ok=False)
    save_json(args.output/'runtime_gate.json',runtime)
    save_json(args.output/'duplicate_candidates.json',runtime['duplicate_candidate_pairs'])
    with (args.output/'runtime_gate_rows.csv').open('w',newline='',encoding='utf-8-sig') as out:
        fields=['case_id','instance_id','v5_category','qwen_tag_category','duplicate_candidate','copy_disposition','copy_block_reasons','vqa_disposition','service_disposition','flags','publish_allowed']
        w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
        for r in runtime['rows']:
            v={k:r.get(k) for k in fields};v['copy_block_reasons']='|'.join(v['copy_block_reasons']);v['flags']='|'.join(v['flags']);w.writerow(v)
    provenance={"qwen_results_sha256":sha(args.qwen),"human_review_sha256":None,"runtime_gate_computed_without_review":True,"original_files_untouched":True}
    if args.review:
        human=json.loads(args.review.read_text(encoding='utf-8'))
        result=offline_backtest(runtime,human)
        provenance['human_review_sha256']=sha(args.review)
        save_json(args.output/'human_review_backtest.json',result)
    save_json(args.output/'artifact_provenance.json',provenance)
    print('PRD32_RUNTIME_GATE_OK',json.dumps(runtime['counts'],ensure_ascii=False),flush=True)
    if args.review: print('HUMAN_REVIEW_CONFIRMED_OFFLINE_BACKTEST_OK',flush=True)

if __name__=='__main__':main()
