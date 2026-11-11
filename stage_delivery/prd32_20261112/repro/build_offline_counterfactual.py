#!/usr/bin/env python3
"""Read only verified DEV60 Stage2 171 records, estimate *scope-specific* timings.
Does NOT claim this is a measured A/B or official PRD acceptance.
"""
import argparse,csv,io,json,zipfile,collections
from pathlib import Path
from arch_ab_common import make_copy,stats,DEBUG_TERMS

def jsonl(z,name):
    return [json.loads(line) for line in z.read(name).decode('utf-8').splitlines() if line.strip()]

def source_records(path):
    out={}; tasks={}; chunk_count=0
    with zipfile.ZipFile(path) as outer:
        for zn in sorted(n for n in outer.namelist() if n.startswith('CHUNK_ZIPS/') and n.endswith('.zip')):
            chunk_count+=1
            with zipfile.ZipFile(io.BytesIO(outer.read(zn))) as inner:
                ev=next(n for n in inner.namelist() if n.endswith('/per_target_events.jsonl'))
                ca=next(n for n in inner.namelist() if n.endswith('/qwen_calls.jsonl'))
                for r in jsonl(inner,ev):
                    if r['cohort']=='warmup_excluded':continue
                    key=(r['case_id'],r['instance_id'])
                    if key in out:raise ValueError('duplicate '+str(key))
                    out[key]=r
                for c in jsonl(inner,ca):
                    if c['cohort']=='warmup_excluded':continue
                    key=(c['case_id'],c['instance_id'],c['task'])
                    if key in tasks:raise ValueError('duplicate task '+str(key))
                    tasks[key]=c
    if chunk_count!=8 or len(out)!=171 or len(tasks)!=513:raise ValueError(f'incomplete source: chunks={chunk_count}, instances={len(out)}, tasks={len(tasks)}')
    return out,tasks

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,default=Path('.'))
    args=ap.parse_args();o=args.output;o.mkdir(parents=True,exist_ok=True)
    rr,qq=source_records(args.source)
    rows=[]
    for (case,inst),r in sorted(rr.items()):
        bt=float(r['total_host_wall_ms']);tag=float(r['tags_task_host_wall_ms']);vqa=float(r['vqa_task_host_wall_ms']);copy=float(r['copy_task_host_wall_ms'])
        ov=bt-(tag+vqa+copy)
        # same fixed V5 cost + unmodified task timing, NOT rerun measurements
        vqa_route=ov+vqa; merchant=ov+tag+copy;merch_template=ov+tag
        tags=qq[(case,inst,'tags')]['response'];vqa_raw=qq[(case,inst,'vqa')]['response'];copy_raw=qq[(case,inst,'copy')]['response']
        draft=make_copy(tags)
        rows.append({'case_id':case,'instance_id':inst,
          'baseline_measured_ms':round(bt,3),'v5_measured_ms':r['v5_host_wall_ms'],
          'tags_measured_ms':round(tag,3),'vqa_measured_ms':round(vqa,3),'copy_measured_ms':round(copy,3),
          'other_measured_ms':round(ov,3),
          'vqa_only_offline_estimate_ms':round(vqa_route,3),
          'merchant_tags_copy_offline_estimate_ms':round(merchant,3),
          'merchant_tags_template_offline_lower_bound_estimate_ms':round(merch_template,3),
          'tags_tokens':qq[(case,inst,'tags')]['model_runner_timing']['generated_tokens'],
          'vqa_tokens':qq[(case,inst,'vqa')]['model_runner_timing']['generated_tokens'],
          'copy_tokens':qq[(case,inst,'copy')]['model_runner_timing']['generated_tokens'],
          'tags_response':tags,'vqa_response':vqa_raw,'baseline_copy':copy_raw,
          'template_copy':draft['text'],'template_status':draft['status'],
          'template_copy_chars':len(draft['text']),
          'template_had_debug_phrase':any(k in draft['text'] for k in DEBUG_TERMS),
          'baseline_had_debug_phrase':any(k in copy_raw for k in DEBUG_TERMS),
          'baseline_guard_status':r['guard_status']})
    with (o/'OFFLINE_171_ARCHITECTURE_COUNTERFACTUAL_NOT_LIVE.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    route_metrics={'baseline_three_tasks_measured':stats([r['baseline_measured_ms'] for r in rows]),
        'vqa_only_offline_estimate':stats([r['vqa_only_offline_estimate_ms'] for r in rows]),
        'merchant_tags_copy_offline_estimate':stats([r['merchant_tags_copy_offline_estimate_ms'] for r in rows]),
        'merchant_tags_template_offline_lower_bound':stats([r['merchant_tags_template_offline_lower_bound_estimate_ms'] for r in rows])}
    m={'status':'OFFLINE_ROUTE_COUNTERFACTUAL_NOT_GPU_AB', 'scope':'171 developer-predicted targets, previously measured frozen Stage2 DEV60',
       'important':'Subtracting unused task wall time is additive approximation; different scopes/quality, cache and scheduling effects unmeasured. Template estimate omits template CPU construction/guard/service overhead.',
       'route_metrics':route_metrics,'template_status_counts':dict(collections.Counter(r['template_status'] for r in rows)),
       'template_length_20_to_80_count':sum(20<=r['template_copy_chars']<=80 and r['template_status']=='TEMPLATE_DRAFT_NEEDS_HUMAN_REVIEW' for r in rows),
       'baseline_debug_phrase_count':sum(r['baseline_had_debug_phrase'] for r in rows),
       'template_debug_phrase_count':sum(r['template_had_debug_phrase'] for r in rows),
       'quality_warning':'The template inherits any Tags misclassification and cannot be called accurate until human review. Copy quality comparison needs user review, not automatic success.',
       'not_formal_prd_acceptance':True,'no_gpu_calls':True,'holdout_used':False,'publication_or_release_allowed':False}
    (o/'OFFLINE_SUMMARY.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(m,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
