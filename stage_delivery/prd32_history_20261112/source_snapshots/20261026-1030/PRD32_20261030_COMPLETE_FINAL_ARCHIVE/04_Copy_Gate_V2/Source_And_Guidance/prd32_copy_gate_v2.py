#!/usr/bin/env python3
"""PRD 3.2 additive review-only Copy safety gate (2026-10-30).

Runtime never uses human labels, modifies model outputs, deduplicates, or grants publish.
Human review is accepted ONLY by the offline `regression` command.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

TASKS = ('tags', 'vqa', 'copy')
SCHEMA = 'prd32_copy_gate_v2_review_only_20261030'
# Narrowly match model descriptions of the annotator's box. Plain garments with
# red/rectangular fabric patterns must not be blocked based on shape/color alone.
DEBUG_PATTERNS = [
    ("ANNOTATION_FRAME", re.compile(r'(?:红色\s*)?(?:边框|方框|矩形框?|框)(?:之?内|中|所指|指向|框选)|红框(?:之?内|中|所指|指向)|红色矩形(?:指向|所指|框选)', re.I)),
    ("ANNOTATION_INSTRUCTION", re.compile(r'(?:标注框|标注区域|标记框|框选区域|图中框选|框内目标|框中的目标)', re.I)),
    ("BBOX_LITERAL", re.compile(r'\b(?:bounding\s*box|bbox)\b', re.I)),
]
REVIEW_PATTERNS = [
    ("UNVERIFIED_STRUCTURE_CLAIM", re.compile(r'(?:无|没有|未见|看不到)裤腿分叉|裤腿(?:分开|分叉)|独立半身裙')),
    ("FINE_DETAIL_CLAIM", re.compile(r'裤脚(?:略?显|有|处)(?:明显)?磨损|口袋(?:设计)?清晰可见|脚踝处(?:清晰)?可见')),
    ("SUBJECTIVE_OR_NONVISUAL_CLAIM", re.compile(r'柔软舒适|触感|手感|高品质|纯棉|真丝|保暖舒适|显瘦|适合(?:春|夏|秋|冬)')),
    ("MULTI_GARMENT_SCOPE_CLAIM", re.compile(r'(?:上衣|T恤|衬衫).{0,9}(?:搭配|配)(?:一条|一件)?(?:裙|裤|外套)|(?:裙|裤).{0,9}(?:搭配|配)(?:上衣|T恤|衬衫)')),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    result = []
    with path.open('r', encoding='utf-8') as stream:
        for line_num, line in enumerate(stream, 1):
            if line.strip():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'BAD_JSONL_LINE_{line_num}') from exc
                if not isinstance(entry, dict):
                    raise ValueError(f'NON_OBJECT_LINE_{line_num}')
                result.append(entry)
    return result


def validate_triplets(rows: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    if not rows:
        raise ValueError('EMPTY_PREDICTIONS')
    by_instance: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        c, i, t = (r.get(k) for k in ('case_id','target_instance_id','task'))
        if not isinstance(c, str) or not c or not isinstance(i, str) or not i or t not in TASKS:
            raise ValueError('INVALID_REQUEST_IDENTITY')
        if r.get('request_id') != f'{c}__{i}__{t}':
            raise ValueError(f'REQUEST_ID_MISMATCH:{c}:{i}:{t}')
        if r.get('publish_allowed') is not False or r.get('human_label_or_gt_bbox_used') is not False:
            raise ValueError('UNSAFE_PERMISSION_OR_GT_CONTAMINATION')
        if r.get('target_bbox_origin') != 'prd31_v5_live_prediction':
            raise ValueError('UNTRUSTED_TARGET_BBOX')
        if r.get('status') != 'GENERATED_REVIEW_REQUIRED':
            raise ValueError('UNSUCCESSFUL_GENERATION')
        if t in by_instance[(c,i)]:
            raise ValueError(f'DUPLICATE_TASK:{c}:{i}:{t}')
        if not isinstance(r.get('raw_text'), str):
            raise ValueError('INVALID_RAW_TEXT')
        by_instance[(c,i)][t] = r
    for key, tr in by_instance.items():
        if set(tr) != set(TASKS):
            raise ValueError(f'INCOMPLETE_TASKS:{key}')
        # Identical target evidence for all 3 tasks; avoid cross-instance mixing.
        for field in ('image_sha256','marked_image_sha256','v5_category_unverified'):
            values = [r.get(field) for r in tr.values()]
            if not isinstance(values[0], str) or not values[0] or len(set(values)) != 1:
                raise ValueError(f'INCONSISTENT_{field}:{key}')
        boxes_raw = [r.get('v5_predicted_bbox_xyxy') for r in tr.values()]
        for box in boxes_raw:
            if not isinstance(box, list) or len(box) != 4 or not all(type(v) in (int,float) and math.isfinite(v) for v in box) or not (box[0]>=0 and box[1]>=0 and box[2]>box[0] and box[3]>box[1]):
                raise ValueError(f'INVALID_BBOX:{key}')
        boxes = [json.dumps(box) for box in boxes_raw]
        if len(set(boxes)) != 1:
            raise ValueError(f'INCONSISTENT_BBOX:{key}')
        if not isinstance(tr['vqa'].get('question_text'), str) or not tr['vqa']['question_text']:
            raise ValueError(f'MISSING_VQA_QUESTION:{key}')
        q = tr['vqa']['question_text'].encode('utf-8')
        if tr['vqa'].get('question_sha256') != hashlib.sha256(q).hexdigest():
            raise ValueError(f'VQA_QUESTION_HASH_MISMATCH:{key}')
    return dict(by_instance)


def detect_copy(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError('INVALID_COPY')
    hard = sorted({name for name, regex in DEBUG_PATTERNS if regex.search(text)})
    if not text.strip():
        hard = sorted(set(hard) | {'EMPTY_COPY_TEXT'})
    manual = sorted({name for name, regex in REVIEW_PATTERNS if regex.search(text)})
    # REVIEW_REQUIRED is not approved/released. No writes to the model output.
    return {
        'hard_block_reasons': hard,
        'manual_review_flags': manual,
        'copy_disposition': 'BLOCKED_DEBUG_TEXT' if hard else 'REVIEW_REQUIRED',
        'publish_allowed': False,
        'original_text_preserved': True,
    }


def run_gate(rows: list[dict]) -> dict:
    by_instance = validate_triplets(rows)
    output = []
    for (case, instance), grp in sorted(by_instance.items()):
        result = detect_copy(grp['copy']['raw_text'])
        tp = grp['tags'].get('tag_parse')
        canonical = tp.get('canonical_service_json') if isinstance(tp, dict) else None
        canonical = canonical if isinstance(canonical, dict) else {}
        output.append({
            'case_id': case, 'instance_id': instance,
            'copy_request_id': grp['copy']['request_id'],
            'copy_sha256': hashlib.sha256(grp['copy']['raw_text'].encode('utf-8')).hexdigest(),
            'copy_text': grp['copy']['raw_text'],
            'marked_image_sha256': grp['copy']['marked_image_sha256'],
            'v5_category_unverified': grp['copy']['v5_category_unverified'],
            'qwen_tag_category_unverified': canonical.get('category'),
            **result,
        })
    return {
        'schema': SCHEMA, 'mode':'RUNTIME_NO_HUMAN_GROUND_TRUTH',
        'instances': len(output), 'requests': len(rows),
        'counts': {
            'debug_blocked': sum(bool(x['hard_block_reasons']) for x in output),
            'manual_risk_flagged': sum(bool(x['manual_review_flags']) for x in output),
            'review_required_without_regex_flag': sum(not x['hard_block_reasons'] and not x['manual_review_flags'] for x in output),
        },
        'rows': output,
        'publish_allowed': False, 'automatic_rewrite': False,
        'automatic_dedup': False, 'human_labels_used': False,
    }


def save_json(path: Path, value: Any):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def save_csv(path: Path, rows: list[dict], names: list[str]):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        w = csv.DictWriter(stream, fieldnames=names, extrasaction='ignore')
        w.writeheader()
        for entry in rows:
            row = dict(entry)
            for k in ('hard_block_reasons','manual_review_flags'):
                if k in row: row[k] = '|'.join(row[k])
            w.writerow(row)


def evaluate(baseline_path: Path, focus_path: Path, review_path: Path) -> tuple[dict,list[dict],list[dict]]:
    # Labels consulted only here, never in run_gate(), model inference, or rule selection.
    baseline = run_gate(read_jsonl(baseline_path))
    focus = run_gate(read_jsonl(focus_path))
    human = json.loads(review_path.read_text(encoding='utf-8'))
    if human.get('review_status') != 'HUMAN_REVIEW_COMPLETED_DIAGNOSTIC_ONLY' or human.get('reviewed_by_human') is not True:
        raise ValueError('REVIEW_NOT_CONFIRMED')
    for key,path in [('baseline_qwen_sha256',baseline_path),('focus_qwen_sha256',focus_path)]:
        if human.get(key) != sha256_file(path):
            raise ValueError(f'HUMAN_REVIEW_SOURCE_SHA_MISMATCH:{key}')
    b={(r['case_id'],r['instance_id']):r for r in baseline['rows']}
    f={(r['case_id'],r['instance_id']):r for r in focus['rows']}
    if set(b)!=set(f): raise ValueError('BASELINE_FOCUS_INSTANCE_MISMATCH')
    hr={}
    for r in human.get('rows',[]):
        k=r.get('case_id'),r.get('instance_id')
        if k in hr: raise ValueError(f'DUPLICATE_HUMAN_ROW:{k}')
        if r.get('instance_review_confirmed') is not True: raise ValueError(f'UNCONFIRMED_ROW:{k}')
        hr[k]=r
    if set(hr)!=set(b): raise ValueError('HUMAN_INSTANCE_SET_MISMATCH')
    detailed=[]
    for k in sorted(b):
        x,y,z=b[k],f[k],hr[k]
        if x['marked_image_sha256'] != y['marked_image_sha256'] or x['marked_image_sha256'] != z.get('marked_image_sha256'):
            raise ValueError(f'MARKED_IMAGE_MISMATCH:{k}')
        detailed.append({
            'case_id':k[0],'instance_id':k[1],
            'instance_binding':z.get('instance_binding'), 'duplicate_assessment':z.get('duplicate_assessment'),
            'baseline_copy_human_label': z['baseline_quality']['copy'],
            'focus_copy_human_label': z['focus_quality']['copy'],
            'baseline_debug_block': bool(x['hard_block_reasons']),
            'focus_debug_block': bool(y['hard_block_reasons']),
            'baseline_manual_risk_flags':'|'.join(x['manual_review_flags']),
            'focus_manual_risk_flags':'|'.join(y['manual_review_flags']),
            'focus_copy_correct':z['focus_quality']['copy']=='CORRECT',
            'focus_tags_quality':z['focus_quality']['tags'],
            'focus_vqa_quality':z['focus_quality']['vqa'],
            'offline_priority': 'P0_BINDING_OR_TAGS' if z.get('instance_binding') != 'CORRECT' or z['focus_quality']['tags']=='INCORRECT' else (
                'P1_COPY_ISSUE' if z['focus_quality']['copy'] not in ('CORRECT','UNCERTAIN') else 'P2_MONITOR'),
        })
    summary={
        'schema':'prd32_20261030_copy_guard_v2_offline_regression',
        'scope':'TRAIN_SOURCE_DEV_DIAGNOSTIC_ONLY',
        'validation_type':'SAME_DEVELOPMENT_SET_RULE_CALIBRATION_NOT_HELD_OUT_VALIDATION',
        'images':len({x[0] for x in b}),'instances':len(b),'requests_per_variant':len(baseline['rows'])*3,
        'source_sha256':{'baseline':sha256_file(baseline_path),'focus':sha256_file(focus_path),'human':sha256_file(review_path)},
        'baseline_rule_counts':baseline['counts'],'focus_rule_counts':focus['counts'],
        'human_primary_labels':{
            variant: dict(Counter(r[f'{variant}_quality']['copy'] for r in hr.values())) for variant in ('baseline','focus')},
        'debug_primary_label_coverage': {
            variant: {
                'human_primary_label_DEBUG_TEXT':sum(r[f'{variant}_quality']['copy']=='DEBUG_TEXT' for r in hr.values()),
                'human_debug_labeled_and_rule_blocked':sum(r[f'{variant}_quality']['copy']=='DEBUG_TEXT' and g[k]['hard_block_reasons'] != [] for k,r in hr.items()),
                'rule_flags_total_including_other_primary_errors':sum(bool(g[k]['hard_block_reasons']) for k in hr),
            } for variant,g in [('baseline',b),('focus',f)]},
        'priority_counts': dict(Counter(r['offline_priority'] for r in detailed)),
        'official_prd_acceptance':False,'publish_allowed':False,'main_pipeline_changed':False,
        'note':'Copy risk rules triage for review, not ground-truth correctness. Reviewer labels are AI-prefilled and user-confirmed, not blinded. No claims of external generalization.'
    }
    return summary, detailed, [baseline, focus]


def cli(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    gate=sub.add_parser('gate',help='runtime output-only gate: never reads human review')
    gate.add_argument('--qwen', type=Path,required=True)
    gate.add_argument('--output',type=Path,required=True)
    check=sub.add_parser('regression', help='offline comparison against user-confirmed review')
    check.add_argument('--baseline',type=Path,required=True)
    check.add_argument('--focus',type=Path,required=True)
    check.add_argument('--human',type=Path,required=True)
    check.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('OUTPUT_EXISTS_AND_NOT_EMPTY_NO_OVERWRITE')
    if args.command=='gate':
        result=run_gate(read_jsonl(args.qwen))
        args.output.mkdir(parents=True,exist_ok=True)
        save_json(args.output/'copy_gate_v2.json',result)
        save_csv(args.output/'copy_gate_v2.csv',result['rows'],
                 ['case_id','instance_id','copy_disposition','hard_block_reasons','manual_review_flags','copy_sha256','copy_text','publish_allowed'])
        print('COPY_GATE_V2_REVIEW_ONLY',json.dumps(result['counts'],ensure_ascii=False))
    else:
        summary,detail,gates=evaluate(args.baseline,args.focus,args.human)
        args.output.mkdir(parents=True,exist_ok=True)
        save_json(args.output/'offline_regression_summary.json',summary)
        save_csv(args.output/'paired_copy_risks.csv',detail,list(detail[0]))
        save_json(args.output/'baseline_gate.json',gates[0])
        save_json(args.output/'focus_gate.json',gates[1])
        print('OFFLINE_COPY_GUARD_V2_OK',json.dumps({
            'instances':summary['instances'], 'baseline_debug_block':summary['baseline_rule_counts']['debug_blocked'],
            'focus_debug_block':summary['focus_rule_counts']['debug_blocked'],
            'priority_counts':summary['priority_counts']},ensure_ascii=False))
        print('NO_PUBLISH_NO_MAIN_STRATEGY_CHANGE')

if __name__=='__main__':
    cli()
