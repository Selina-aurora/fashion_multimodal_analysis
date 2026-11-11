#!/usr/bin/env python3
"""PRD 3.2 V5 offline semantic audit / fail-closed service fallback.

Reads the *existing* consistency_case_XX.json. No GPU, downloads, or
modifications to the 3.1 frozen inputs or previous model results.
The fallback is only a constrained serialization of unverified model labels.
It must not be counted as factual accuracy or free-form generative VQA.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CATS = {'top':'上衣','pants':'裤子','skirt':'半身裙','outerwear':'外套',
        'dress':'连衣裙','shoe':'鞋','bag':'包','accessory':'配饰'}
FIELDS = ('primary_color','pattern','visible_design')
NON_DESIGNS = {
    '无','没有','无明显设计','无设计','没有设计','未见设计','无特殊设计',
    '无图案','无装饰','纯色','素色','solid','plain','none','unknown','n/a',
    '不确定','未知','无法确定','看不清','不可见','无明显特征',
}


def _is_non_design(value):
    if not isinstance(value, str):
        return False
    normalized = ''.join(value.strip().casefold().split()).strip('。！!，,；;')
    return normalized in NON_DESIGNS


def audit(record, category):
    parsed = record.get('parsed')
    errors = list(record.get('errors') or [])
    if not isinstance(parsed, dict):
        return {'model_errors': errors + ['unparseable_or_non_object'],
                'strictly_usable': False, 'candidate_fallback': None,
                'fallback_edits': [], 'unverified_model_copy': None}

    missing = [f for f in FIELDS if f not in parsed]
    if missing:
        return {'model_errors': errors + ['missing_keys:' + ','.join(missing)],
                'strictly_usable': False, 'candidate_fallback': None,
                'fallback_edits': [], 'unverified_model_copy': parsed.get('marketing_copy')}
    tags = {'category': category}
    edits = []
    for name in FIELDS:
        value = parsed.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append('invalid_type_or_empty:' + name)
            value = None
            edits.append({'field': name, 'operation':'invalidate_to_null',
                          'reason':'invalid_type_or_empty'})
        elif isinstance(value, str):
            value = value.strip()
        if name == 'visible_design' and _is_non_design(value):
            errors.append('non_structural_visible_design:' + value)
            edits.append({'field': name, 'operation': 'invalidate_to_null',
                          'reason': 'not_an_observable_structure', 'original': value})
            value = None
        tags[name] = value

    tags['uncertain_fields'] = [key for key in FIELDS if tags[key] is None]
    qa = tags['primary_color'] if tags['primary_color'] is not None else '无法确定'
    cat = CATS.get(category, '服饰')
    copy = (tags['primary_color'] or '') + cat
    # Include only non-null, appropriately typed visible attributes; never render “无”。
    if isinstance(tags['pattern'], str) and tags['pattern']:
        if tags['pattern'] in ('纯色', '素色'):
            copy += '，纯色设计'
        else:
            copy += '，' + tags['pattern'] + '图案'
    if isinstance(tags['visible_design'], str) and tags['visible_design']:
        copy += '，' + tags['visible_design']
    copy += '。'
    fallback = {'qa_color': qa, 'tags': tags, 'marketing_copy': copy}
    # Do not count postprocessed candidates as model compliance; retains errors even if safe fallback exists.
    return {'model_errors': sorted(set(errors)),
            'strictly_usable': not errors and record.get('raw_native_json') is True,
            'candidate_fallback': fallback, 'fallback_edits': edits,
            'unverified_model_copy': parsed.get('marketing_copy'),
            'qa_is_derived_from_unverified_tags': True,
            'copy_is_template_not_human_rated': True,
            'auto_acceptance': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Output already exists; select another file to preserve prior audit')
    src = json.loads(args.input.read_text(encoding='utf-8'))
    if src.get('schema') != 'prd32_consistency_dev_v5':
        ap.error('Input is not PRD32 V5 consistency report')
    cat = src.get('target_category')
    if cat not in CATS:
        ap.error('Missing or invalid target category')
    measurements = src.get('measurements', [])
    audits = []
    for idx, rec in enumerate(measurements):
        result = audit(rec,cat)
        audits.append({'measurement_index': idx, 'variant':rec.get('variant'),
                       'mode': rec.get('mode'), 'end_to_end_ms':rec.get('end_to_end_ms'),
                       'raw_native_json':rec.get('raw_native_json'),
                       'previous_attributes_valid':rec.get('attributes_valid'),
                       **result})
    summary = {
        'total_measurements':len(audits),
        'parseable_count':sum(bool(x.get('parsed') is not None) for x in measurements),
        'model_raw_native_json_count':sum(r.get('raw_native_json') is True for r in measurements),
        'previous_validator_accepted_count':sum(r.get('attributes_valid') is True for r in measurements),
        'audit_detected_placeholder_design_count':sum(any(e['field']=='visible_design' for e in a['fallback_edits']) for a in audits),
        'model_strict_usable_count':sum(a['strictly_usable'] for a in audits),
        'service_fallback_candidates_not_auto_accepted':sum(a['candidate_fallback'] is not None for a in audits),
        'errors':dict(Counter(err for a in audits for err in a['model_errors'])),
    }
    result={'schema':'prd32_v5_semantic_audit_v1',
            'status':'AUDITED_NO_PRD_ACCEPTANCE',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'input_file':str(args.input),
            'bridge_case_id':src.get('bridge_case_id'),
            'target_category':cat,
            'target_category_source':'manually_supplied_diagnostic_not_model_output',
            'summary':summary, 'audits':audits,
            'limitations':[
                'No image-grounded GT; even a plausible color cannot be counted as validated factual accuracy.',
                'Postprocessed fallback normalizes invalid placeholders to null but is not a corrected model prediction.',
                'The color QA is synthesized from model tags and is not general multi-modal QA.',
                'Template copy has no merchant satisfaction review.',
                'Original inputs, weights, outputs and sealed holdout are not modified.',
            ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],**summary},ensure_ascii=False,indent=2))
    print('Output:',args.output)
if __name__ == '__main__':
    main()
