#!/usr/bin/env python3
"""Read-only audit of pinned DEV12 V7 timings or an explicit cold-staged GPU timing run.

Not a PRD warm end-to-end latency benchmark. No holdout access, no changes to models.
"""
import argparse
import csv
import datetime as dt
import json
import math
import re
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

TASKS = ("tags", "vqa", "copy")


def nearest_p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs)-1, math.ceil(0.95 * len(xs))-1)] if xs else None


def metrics(vals):
    return {"n": len(vals), "mean_ms": round(sum(vals)/len(vals), 3) if vals else None,
            "p95_ms": round(nearest_p95(vals), 3) if vals else None,
            "max_ms": round(max(vals), 3) if vals else None}


def validate_and_report(producer_events, qwen_results, guard):
    if len({(e['case_id'], e['instance_id']) for e in producer_events}) != len(producer_events):
        raise ValueError('duplicate producer identity')
    if guard.get('production_approved') is not False or guard.get('holdout_accessed') is not False:
        raise ValueError('Guard safety flags not verified')
    by_task = defaultdict(list)
    per_instance = defaultdict(dict)
    seen = set()
    for q in qwen_results:
        key = (q.get('case_id'), q.get('target_instance_id'))
        ident = (*key, q.get('task'))
        if q.get('task') not in TASKS or ident in seen:
            raise ValueError('invalid or duplicate task identity')
        seen.add(ident)
        if q.get('status') != 'GENERATED_REVIEW_REQUIRED' or q.get('publish_allowed') is not False:
            raise ValueError('unexpected response release/status')
        t = q['timing']
        elapsed = float(t['end_to_end_ms'])
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid task latency')
        if abs(elapsed - (float(t['preprocess_ms']) + float(t['generation_ms']))) > 1:
            raise ValueError('task time breakdown inconsistent')
        by_task[q['task']].append(elapsed)
        per_instance[key][q['task']] = elapsed
    if not seen or any(set(tasks) != set(TASKS) for tasks in per_instance.values()):
        raise ValueError('missing tasks')
    # For these previously recorded sequential task timings, the sum is a task-time sum,
    # NOT a measured per-request end-to-end wallclock.
    sums = [sum(tasks.values()) for tasks in per_instance.values()]
    forward = {}
    for e in producer_events:
        case = e['case_id']; ms = float(e['producer_inference_ms'])
        if case in forward and abs(forward[case]-ms)>1e-4:
            raise ValueError('inconsistent V5 forward duration for a shared image')
        forward[case] = ms
    return {
        'status': 'OBSERVED_TASK_TIMINGS_ONLY_NOT_PRD_ACCEPTANCE',
        'source': 'V7 staged DEV-only historical live GPU run',
        'v5_unique_image_forwards': len(forward),
        'v5_forward_only_ms': metrics(list(forward.values())),
        'qwen_task_e2e_ms_by_task': {t: metrics(by_task[t]) for t in TASKS},
        'sequential_qwen_triplet_sum_ms_NOT_WALLCLOCK': metrics(sums),
        'qwen_target_count': len(per_instance),
        'qwen_task_count': len(seen),
        'guard_state': guard.get('guard_service_status'),
        'full_chain_warm_latency_ms': None,
        'prd_latency_pass': 'NOT_ASSESSED',
        'developer_selected_high_risk': True,
        'independent_holdout_used': False,
        'release_allowed': False,
    }


def read_v7_zip(path):
    with zipfile.ZipFile(path) as z:
        by_suffix = {}
        for suffix in ('/producer_events.json', '/qwen_shadow_results.jsonl', '/guard_shadow_report.json'):
            candidates = [n for n in z.namelist() if n.endswith(suffix)]
            if len(candidates) != 1:
                raise ValueError('missing or ambiguous '+suffix)
            by_suffix[suffix] = z.read(candidates[0]).decode('utf8')
        pe = json.loads(by_suffix['/producer_events.json'])
        q = [json.loads(line) for line in by_suffix['/qwen_shadow_results.jsonl'].splitlines() if line.strip()]
        g = json.loads(by_suffix['/guard_shadow_report.json'])
        return validate_and_report(pe, q, g)


def read_v7_dirs(producer, qwen):
    pe = json.loads((producer/'producer_events.json').read_text(encoding='utf8'))
    q = [json.loads(line) for line in (qwen/'qwen_shadow_results.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    g = json.loads((qwen/'guard_shadow_report.json').read_text(encoding='utf8'))
    return validate_and_report(pe, q, g)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def make_report(data):
    rows = ['# PRD 3.2 latency evidence — scope controlled', '',
        '**These are V7 DEV-only small-sample observations, not PRD acceptance results.**',
        'Qwen E2E is one Qwen task (preprocessing + generation), not V5→Qwen→Guard wallclock.',
        'V5 timings exclude image disk I/O and model load. Historical task sums exclude Guard, queue and serialization.',
        'True warm per-user-request wallclock and full DEV60 representative timing remain unmeasured.', '',
        '| Item | n | Mean ms | P95 ms |', '|---|---:|---:|---:|']
    for label, m in [('V5 forward (image)',data['v5_forward_only_ms'])] + [
        (f'Qwen {t} task E2E', data['qwen_task_e2e_ms_by_task'][t]) for t in TASKS
    ]+[('Three Qwen task durations summed (not actual pipeline wallclock)',data['sequential_qwen_triplet_sum_ms_NOT_WALLCLOCK'])]:
        rows.append(f'| {label} | {m["n"]} | {m["mean_ms"]} | {m["p95_ms"]} |')
    if 'cold_staged_total_ms' in data:
        rows += ['',f'Cold-staged serial wallclock including process/model loading: **{data["cold_staged_total_ms"]} ms**.',
                 'This is a real elapsed wallclock measurement for a staged multi-process batch, but NOT warm per-request E2E.']
    rows += ['', '## Next real acceptance instrumentation',
             'Instrument a persistent-model request handler from input image arrival → V5 + marked frame → all requested Qwen outputs → Guard decision; '
             'record start/end monotonic timestamps per image/target, mean/P50/P95 across a fixed DEV cohort and full DEV60 where feasible.',
             'Report success/failure/blocked coverage alongside latency, and label cold-start/model-load latency separately.',
             'Do not open holdout, change prompts, or alter model thresholds to force speed.']
    return '\n'.join(rows)+'\n'


def run_stage(args, name, command, out):
    start = time.perf_counter_ns()
    res = subprocess.run(command, cwd=args.v7_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = (time.perf_counter_ns()-start)/1e6
    (out/(name+'_stdout.txt')).write_text(res.stdout,encoding='utf8')
    print(name, 'exit',res.returncode,'wall_ms',round(elapsed,1),flush=True)
    if res.returncode != 0:
        raise RuntimeError(f'{name} failed exit={res.returncode}: '+res.stdout[-1400:])
    return {'elapsed_wall_ms': round(elapsed,3),'exit_code':res.returncode}


def run_gpu(args):
    if not args.execute_models:
        raise SystemExit('GPU model run requires explicit --execute-models')
    if not args.v7_dir.is_dir():
        raise SystemExit('V7 directory not found; original V7 packet must be installed')
    if not (args.v7_dir/'run_v5_runtime_producer_v7.py').is_file():
        raise SystemExit('V7 source files missing')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out = args.workspace/('PRD32_1109_TIMED_V7_'+stamp)
    out.mkdir(exist_ok=False,parents=True)
    components={}
    producer = out/'V5_PRODUCER'; qwen=out/'QWEN'
    start=time.perf_counter_ns()
    try:
        commands=[
            ('v5_preflight', [sys.executable,'run_v5_runtime_producer_v7.py','--mode','preflight','--out',str(out/'V5_PREFLIGHT')]),
            ('v5_live',[sys.executable,'run_v5_runtime_producer_v7.py','--mode','live','--out',str(producer)]),
            ('qwen_preflight',[sys.executable,'run_qwen_from_v5_producer_v7.py','--mode','preflight','--producer-dir',str(producer),'--out',str(out/'QWEN_PREFLIGHT')]),
            ('qwen_live',[sys.executable,'run_qwen_from_v5_producer_v7.py','--mode','live','--producer-dir',str(producer),'--out',str(qwen)]),
        ]
        for name, command in commands:
            components[name]=run_stage(args,name,command,out)
        observed=read_v7_dirs(producer,qwen)
        observed['cold_staged_total_ms']=round((time.perf_counter_ns()-start)/1e6,3)
        observed['cold_staged_component_wall_ms']=components
        observed['cold_staged_is_warm_e2e']=False
        write_json(out/'latency_evidence.json',observed)
        (out/'LATENCY_README.md').write_text(make_report(observed),encoding='utf8')
        print('TIMED_V7_STAGED_COMPLETE_NOT_PRD_E2E',out,flush=True)
    except Exception as exc:
        write_json(out/'failure.json',{'status':'INCOMPLETE_NO_RELEASE', 'error_type':type(exc).__name__,'message':str(exc), 'completed_stages':components})
        raise
    finally:
        # Results may include user-specific model responses. Review before sharing.
        archive=args.workspace/(out.name+'.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob('*')):
                if f.is_file():
                    z.write(f,f.relative_to(args.workspace))
        print('UPLOAD_THIS:',archive,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['audit-existing','timed-staged'],default='audit-existing')
    p.add_argument('--existing-zip',type=Path)
    p.add_argument('--v7-dir',type=Path,default=Path('/workspace/PRD32_1106_DEV12_RUNTIME_V7'))
    p.add_argument('--workspace',type=Path,default=Path('/workspace'))
    p.add_argument('--execute-models',action='store_true')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.mode=='audit-existing':
        if a.existing_zip is None or a.out is None:
            p.error('audit-existing requires --existing-zip and --out')
        observed=read_v7_zip(a.existing_zip)
        a.out.mkdir(parents=True,exist_ok=False)
        write_json(a.out/'latency_evidence.json',observed)
        (a.out/'LATENCY_README.md').write_text(make_report(observed),encoding='utf8')
        print('EXISTING_V7_TIMINGS_AUDITED',a.out)
    else:
        run_gpu(a)

if __name__=='__main__':main()
