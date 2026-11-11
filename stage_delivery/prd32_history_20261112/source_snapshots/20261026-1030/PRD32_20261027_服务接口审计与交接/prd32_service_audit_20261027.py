#!/usr/bin/env python3
"""Read-only PRD 3.2 service-output audit for 2026-10-27 offline dev replay.

No model execution, no automatic modification of emitted attributes, no GT inference.
Produces:
  audit_summary.json       -- structural and semantic-review findings
  audit_per_case.csv       -- individual request findings
  handoff_envelopes.jsonl  -- provenance-enriched copies; still UNVERIFIED
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

VALID_TASKS = {'vqa', 'tags', 'copy'}
TAGS_KEYS = {'primary_color', 'pattern', 'visible_decoration', 'unconfirmed'}
CANONICAL_KEYS = {'category', 'primary_color', 'pattern', 'visible_design'}
TAG_TYPES = {'primary_color': (type(None), str),
             'pattern': (type(None), str),
             'visible_decoration': (type(None), str)}
BAD_PATTERN_SENTINELS = {'无', '没有', '无图案', '无明显图案', 'none', 'unknown', '无法确定', '不确定'}


def read_jsonl(path):
    rows = []
    for index, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f'{path} line {index} not a JSON object')
        rows.append(record)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if any(not p.is_file() for p in [args.service, args.predictions]):
        parser.error('service and predictions files must both exist')
    if args.output.exists():
        parser.error('output directory already exists; originals are never overwritten')
    sr = read_jsonl(args.service)
    pr = read_jsonl(args.predictions)
    ids = [r.get('case_id') for r in sr]
    pred_ids = [r.get('case_id') for r in pr]
    if any(not isinstance(i, str) or not i for i in ids + pred_ids):
        raise ValueError('Missing case_id in records')
    if len(set(ids)) != len(ids) or len(set(pred_ids)) != len(pred_ids):
        raise ValueError('Duplicate case_id')
    pm = {r['case_id']: r for r in pr}
    if set(ids) != set(pm):
        raise ValueError('Service and predictions case IDs are not identical')

    scored, enriched = [], []
    structural_errors = 0
    warnings = Counter()
    tag_by_target = {}
    for s in sr:
        cid = s['case_id']
        pred = pm[cid]
        task = s.get('task')
        errs, flags = [], []
        if task not in VALID_TASKS or task != pred.get('task'):
            errs.append('task_mismatch_or_invalid')
        if task == 'tags':
            payload = s.get('payload')
            canon = s.get('canonical_attributes_v2')
            if not isinstance(payload, dict) or set(payload) != TAGS_KEYS:
                errs.append('tags_payload_keys_or_type_invalid')
            else:
                for k, allowed in TAG_TYPES.items():
                    if not isinstance(payload[k], allowed):
                        errs.append(f'tags_field_wrong_type:{k}')
                uc = payload.get('unconfirmed')
                if not isinstance(uc, list) or any(not isinstance(x, str) for x in uc):
                    errs.append('unconfirmed_type_invalid')
                elif sorted(uc) != sorted([k for k in TAG_TYPES if payload.get(k) is None]):
                    errs.append('unconfirmed_not_matching_nulls')
                p = payload.get('pattern')
                if isinstance(p, str) and p.strip().lower() in BAD_PATTERN_SENTINELS:
                    flags.append('pattern_noncanonical_or_ambiguous')
                if payload.get('visible_decoration') is None:
                    flags.append('decoration_abstained_not_absent')
            if not isinstance(canon, dict) or set(canon) != CANONICAL_KEYS:
                errs.append('canonical_attributes_shape_invalid')
            if s.get('status') != 'CANONICALIZED_NOT_HUMAN_VERIFIED':
                errs.append('tags_status_invalid')
            target = (pred.get('image_id'), pred.get('target_instance_id'))
            if isinstance(canon, dict):
                tag_by_target[target] = canon
        else:
            if not isinstance(s.get('payload'), str) or not s['payload'].strip():
                errs.append('empty_or_nontext_payload')
            if s.get('status') != 'DRAFT_UNVERIFIED':
                errs.append('text_status_invalid')
            if task == 'copy' and isinstance(s.get('payload'), str):
                t = s['payload']
                if '红框' in t or '红色框' in t:
                    flags.append('copy_mentions_debug_bbox')
                if '蕾丝' in t or any(w in t for w in ('真丝', '纯棉', '羊毛成分')):
                    flags.append('copy_unverified_material_or_fabric_claim')
                if len(t) > 80:
                    flags.append('copy_exceeds_default_80chars')
        if errs:
            structural_errors += 1
        flags.append('content_not_human_verified')
        for f in flags:
            warnings[f] += 1
        item = {
            'case_id': cid, 'task': task, 'image_id': pred.get('image_id'),
            'target_instance_id': pred.get('target_instance_id'),
            'target_bbox_origin': pred.get('bbox_origin'),
            'structural_valid': not bool(errs),
            'structural_errors': ';'.join(errs), 'review_flags': ';'.join(flags),
            'model_or_adapter_status': s.get('status'),
        }
        scored.append(item)
        enriched.append({
            'schema': 'prd32_service_dev_envelope_v1', 'case_id': cid,
            'task': task, 'image_id': pred.get('image_id'),
            'target_instance_id': pred.get('target_instance_id'),
            'target_bbox_origin': pred.get('bbox_origin'),
            'source_model_status': s.get('status'),
            'payload': s.get('payload'),
            'canonical_attributes_v2': s.get('canonical_attributes_v2'),
            'human_verified': False,
            'review_required': True,
            'ready_for_automatic_publishing': False,
            'review_flags': flags,
            'structural_errors': errs,
        })
    # Read-only cross-task checks, matched by image and target instance.
    for rec in scored:
        if rec['task'] != 'copy':
            continue
        source = next(s for s in sr if s['case_id'] == rec['case_id'])
        text = source.get('payload', '')
        canon = tag_by_target.get((rec['image_id'], rec['target_instance_id']))
        # Concrete, conservative check for the observed white vs light-gray case;
        # do not treat a mention of other colored decorations as main-color conflict.
        if canon and canon.get('primary_color') == '白色' and re.search(r'一件浅灰色|服饰为浅灰色', text):
            rec['review_flags'] += ';copy_tag_primary_color_disagreement'
            warnings['copy_tag_primary_color_disagreement'] += 1
            e = next(x for x in enriched if x['case_id'] == rec['case_id'])
            e['review_flags'].append('copy_tag_primary_color_disagreement')

    output = args.output
    output.mkdir(parents=True)
    with (output/'audit_per_case.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    with (output/'handoff_envelopes.jsonl').open('w', encoding='utf-8') as f:
        for row in enriched:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    summary = {
        'status': 'OFFLINE_SERVICE_SHAPE_CHECK_PASS' if not structural_errors else 'SERVICE_STRUCTURAL_REVIEW_REQUIRED',
        'scope': '2026-10-27 frozen 3.1 bridge replay, not live 3.1, not PRD acceptance',
        'service_records': len(sr),
        'matched_prediction_records': len(pr),
        'task_counts': dict(Counter(r.get('task') for r in sr)),
        'structural_errors': structural_errors,
        'review_flag_counts': dict(warnings),
        'accepted_for_dev_replay_only': not bool(structural_errors),
        'ready_for_automatic_publishing': False,
        'needs_downstream_schema_agreement': True,
        'tag_payload_does_not_include_category': True,
        'qa_copy_are_unverified_drafts': True,
        'unverified_3_1_attributes_not_truth': True,
        'note': 'Envelopes enrich provenance only. They do not correct predictions or constitute human review.'
    }
    (output/'audit_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
