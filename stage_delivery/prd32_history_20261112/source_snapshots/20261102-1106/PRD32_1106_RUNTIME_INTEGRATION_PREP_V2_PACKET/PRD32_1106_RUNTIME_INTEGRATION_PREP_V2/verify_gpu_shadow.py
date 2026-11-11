"""Read-only consistency audit of user-uploaded GPU shadow outputs."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def audit(directory):
    p=Path(directory)
    s=json.loads((p/'shadow_summary.json').read_text(encoding='utf-8'))
    rows=load_csv(p/'shadow_decisions_171.csv')
    queue=load_csv(p/'shadow_risk_queue.csv')
    keys=lambda rs: [(r['case_id'],r['instance_id'],r['image_sha256']) for r in rs]
    violations=[]
    if s.get('frozen_dev60_instances')!=len(rows):violations.append('instance_count_differs_from_summary')
    if len(set(keys(rows)))!=len(rows):violations.append('duplicate_instance_identity')
    if len(set(keys(queue)))!=len(queue):violations.append('duplicate_risk_queue_identity')
    if set(keys(queue))!=set(keys(r for r in rows if r.get('risk_flags'))):
        violations.append('risk_queue_not_equal_to_flagged_rows')
    if sum(int(r['qwen_task_records']) for r in rows)!=s.get('qwen_records_seen'):
        violations.append('task_record_count_mismatch')
    if any(r['qwen_task_records']!=r['expected_task_records'] for r in rows):
        violations.append('missing_qwen_tasks')
    if any(r['publish_allowed']!='NO' or r['auto_alias_merge']!='NO' or r['auto_category_override']!='NO' for r in rows):
        violations.append('unsafe_publication_or_override')
    if any(r['replay_status']!='SHADOW_ONLY_RUNTIME_BINDING_UNVERIFIED' for r in rows):
        violations.append('unexpected_runtime_verification_claim')
    if s.get('metadata_violation_counts') or not s.get('marked_image_file_bytes_verified'):
        violations.append('metadata_or_image_verification_incomplete')
    if s.get('holdout_opened') or s.get('model_called') or s.get('source_predictions_modified'):
        violations.append('unexpected_access_or_prediction_mutation')
    flags=Counter(z for r in rows for z in r.get('risk_flags','').split(';') if z)
    policies=Counter(r['frozen_guard_v5_policy'] for r in rows)
    return {'schema':'prd32_shadow_gpu_audit_v2','audit_pass':not violations,
            'violations':violations,'instances':len(rows),
            'unique_images':len(set(r['case_id'] for r in rows)),
            'qwen_task_records':sum(int(r['qwen_task_records']) for r in rows),
            'risk_queue_size':len(queue),'policy_counts':dict(policies),
            'risk_flag_counts':dict(flags),
            'qwen_baseline_focus_class_changes':sum(r['qwen_baseline_tags']!=r['qwen_focus_tags'] for r in rows),
            'runtime_binding_verified':False,'production_approved':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shadow-dir',required=True)
    ap.add_argument('--out',required=True);a=ap.parse_args()
    report=audit(a.shadow_dir)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['audit_pass']:raise SystemExit(2)

if __name__=='__main__':main()
