#!/usr/bin/env python3
"""Checks development-only soft labels against frozen image hashes, never authorizes training."""
import argparse,csv,json,collections,hashlib,pathlib,math

def read_csv(path):
 with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 inp=pathlib.Path(a.inputs);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 soft=read_csv(inp/'soft_targets_4class_DEVELOPMENT_ONLY.csv');cand=read_csv(inp/'future_train_candidates_REQUIRES_ISOLATION_CHECK.csv')
 frozen=json.loads((inp/'sleeve_v3_holdout20_BLIND_MANUAL_20261013_last.json').read_text(encoding='utf-8'))
 holdout={r['sha256'] for r in frozen['records']}
 core={line.strip().lower() for line in (inp/'core400_source_sha256_20261012.txt').read_text().splitlines() if len(line.strip())==64}
 all_hashes=[r['source_image_sha256'] for r in soft]; c_hashes=[r['source_image_sha256'] for r in cand]
 sums=[];incorrect=[];bad_ratios=[]
 for r in soft:
  probabilities=[float(r['p_'+x]) for x in ('sleeveless','cap','short','elbow')]
  sums.append(sum(probabilities))
  if abs(sum(probabilities)-1)>1e-6 or any(v<0 or v>1 for v in probabilities):incorrect.append(r['record_id'])
  if not (0<=float(r['ratio'])<=1):bad_ratios.append(r['record_id'])
 report={'n_soft_targets':len(soft),'n_training_candidates':len(cand),'n_unique_candidate_source_hashes':len(set(c_hashes)),
 'candidate_human_class_distribution':dict(collections.Counter(r['human_class'] for r in cand)),
 'candidate_soft_top1_distribution':dict(collections.Counter(r['soft_top1'] for r in cand)),
 'candidate_human_soft_top1_disagreement':[r['record_id'] for r in cand if r['human_class']!=r['soft_top1']],
 'probability_invalid_ids':incorrect,'ratio_invalid_ids':bad_ratios,'candidate_core400_sha_overlap':sorted(set(c_hashes)&core),
 'candidate_v3_holdout20_sha_overlap':sorted(set(c_hashes)&holdout),
 'candidate_internal_duplicate_sha':len(c_hashes)-len(set(c_hashes)),
 'runtime_train_val_image_hash_audit_done':False,'training_authorized':False,'model_trained':False,
 'status':'DEVELOPMENT_LABEL_DATA_QC_ONLY',
 'limitations':['Candidate vs frozen Core400 and V3 holdout image-hash checks do not establish full train/val split isolation.','Four classes only; source ratio is old projected-arm definition, distinct from V3 opening midpoint.','Human-label agreement does not imply predictive accuracy.']}
 assert len(soft)==18 and len(cand)==10
 (out/'softlabel_dev_QC_20261014.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
