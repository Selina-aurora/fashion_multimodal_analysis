#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD32 / frozen DEV60: sidecar identity & category risk routing.

NO model calls. NO output reassignment. NO changes to predictions or holdout.
The reviewed Tags30 labels are AI-prefilled labels accepted verbally by the user,
NOT a separately exported edited CSV and NOT independently adjudicated ground truth.
"""
import argparse, collections, csv, hashlib, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EXPECTED={'images':60,'instances':171,'task_records':513,'reviewed':30,'shadow_aliases':22}
TASKS=('tags','vqa','copy')
CLASS_CODES={'SPATIAL','SEMANTIC','UPSTREAM_MIXED','MIXED_EVIDENCE','UNCERTAIN'}
ACTIONS={
 'SPATIAL':'BLOCK_AND_DIAGNOSE_VISUAL_IDENTITY',
 'SEMANTIC':'BLOCK_AND_REVIEW_CATEGORY_TAXONOMY',
 'UPSTREAM_MIXED':'BLOCK_AND_REVIEW_V5_MASK_AND_BBOX',
 'MIXED_EVIDENCE':'BLOCK_AND_VERIFY_ATTRIBUTES_PER_INSTANCE',
 'UNCERTAIN':'BLOCK_UNTIL_VISUAL_EVIDENCE',
}

def load_csv(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def load_jsonl(p):
 with p.open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]

def write_csv(path,rows,fields):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows)

def extract_tags_category(rec):
 p=rec.get('tag_parse') or {}
 if not isinstance(p,dict):return ''
 x=p.get('canonical_service_json') or p.get('raw_parsed') or {}
 if not isinstance(x,dict):return ''
 return str(x.get('category') or '')

def assert_frozen_metadata(im,pred,rec):
 """Only proves stored metadata consistency, NOT whether model visually attended to the target."""
 failures=[]
 if im['case_id']!=rec.get('case_id'):failures.append('case_id')
 if pred['instance_id']!=rec.get('target_instance_id'):failures.append('instance_id')
 if im['image_sha256']!=rec.get('image_sha256'):failures.append('image_sha256')
 if pred['predicted_category']!=rec.get('v5_category_unverified'):failures.append('v5_category')
 if rec.get('target_bbox_origin')!='prd31_v5_live_prediction':failures.append('target_bbox_origin')
 if rec.get('human_label_or_gt_bbox_used') is not False:failures.append('gt_or_human_bbox_used')
 b1=pred['bbox_xyxy'];b2=rec.get('v5_predicted_bbox_xyxy')
 if not isinstance(b2,list) or len(b2)!=4:failures.append('bbox_shape')
 else:
  try:
   if any(not math.isfinite(float(x)) for x in b1+b2) or any(abs(float(a)-float(b))>1e-3 for a,b in zip(b1,b2)):
    failures.append('bbox_value')
  except (TypeError,ValueError):failures.append('bbox_type')
 if rec.get('task') not in TASKS:failures.append('task')
 return failures

def route_case(root=ROOT, output=None):
 root=Path(root).resolve(); src=root/'input'; out=Path(output or root/'output').resolve()
 if out==root or src==out or src in out.parents:raise ValueError('unsafe output directory')
 out.mkdir(parents=True,exist_ok=True)
 manifests={
  'v5':src/'v5_raw_predictions.jsonl',
  'qwen':src/'focus_v1_qwen_results_513.jsonl',
  'history':src/'nov2_v5_attribution_171.csv',
  'review':src/'tags30_prior_AI_prefill_chat_attested.csv',
  'aliases':src/'shadow_alias_proposals_strict.csv',
 }
 for p in manifests.values():
  if not p.is_file():raise FileNotFoundError(p)
 images=load_jsonl(manifests['v5']);qwen=load_jsonl(manifests['qwen']);historical=load_csv(manifests['history']);review=load_csv(manifests['review']);alias=load_csv(manifests['aliases'])
 if len(images)!=EXPECTED['images'] or len(qwen)!=EXPECTED['task_records'] or len(historical)!=EXPECTED['instances'] or len(review)!=EXPECTED['reviewed'] or len(alias)!=EXPECTED['shadow_aliases']:
  raise ValueError('frozen source cardinality changed; refuse to silently continue')
 inst={}
 for im in images:
  cid=im['case_id']
  if 'DEV' not in im['scope'].upper() or not cid.startswith(('fresh_dev_','dev_')):
   raise ValueError('non-DEV input blocked')
  if cid in inst:raise ValueError('duplicate image case_id')
  for pred in im['instances']:
   k=(cid,pred['instance_id'])
   if k in inst:raise ValueError('duplicate instance key')
   inst[k]=(im,pred)
 if len(inst)!=EXPECTED['instances']:raise ValueError('unexpected instance cardinality')
 qs={}
 for r in qwen:
  key=(r['case_id'],r['target_instance_id'],r['task'])
  if key in qs:raise ValueError('duplicate Qwen task '+str(key))
  qs[key]=r
 if set(qs)!=set((cid,iid,t) for cid,iid in inst for t in TASKS):raise ValueError('Qwen task set changed')
 hist={(r['case_id'],r['instance_id']):r for r in historical}
 if set(hist)!=set(inst):raise ValueError('historical keys changed')
 rev={}
 for r in review:
  key=(r['case_id'],r['instance_id'])
  if key in rev or key not in inst or r['ai_primary_class'] not in CLASS_CODES:
   raise ValueError('review record invalid '+str(key))
  if r.get('human_primary_class') or str(r.get('human_confirmed')).lower()=='true':
   raise ValueError('unexpected edit/checkbox; user chat attestation must not fake HTML export')
  rev[key]=r
 shadow={}
 for r in alias:
  key=(r['case_id'],r['suppressed_instance_id'])
  val=(r['case_id'],r['proposed_canonical_instance_id'])
  if key in shadow or key not in inst or val not in inst or key==val:raise ValueError('invalid alias')
  if r['mode']!='SHADOW_ONLY' or r['eligible_for_mainline']!='NO':raise ValueError('alias not shadow-only')
  shadow[key]=val
 if any(val in shadow for val in shadow.values()):raise ValueError('transitive alias forbidden')
 if rev.keys() & shadow.keys():raise ValueError('unexpected overlapping reviewed/suppression; re-evaluate manually')

 rows=[];failures=[]
 for (cid,iid),(im,pred) in sorted(inst.items()):
  records=[qs[(cid,iid,t)] for t in TASKS]
  issues={r['task']:assert_frozen_metadata(im,pred,r) for r in records}
  for t, errors in issues.items():
   if errors: failures.append({'case_id':cid,'instance_id':iid,'task':t,'issues':'|'.join(errors)})
  tags=qs[(cid,iid,'tags')];v5=pred['predicted_category'];tc=extract_tags_category(tags)
  metadata_pass=not any(issues.values())
  conflict=(tc!=v5 or not tc)
  r=rev.get((cid,iid));typ=r['ai_primary_class'] if r else ''
  if not metadata_pass:action='BLOCK_METADATA_INTEGRITY'
  elif not tc:action='BLOCK_TAG_CATEGORY_MISSING'
  elif r:action=ACTIONS[typ]
  elif conflict:action='BLOCK_UNREVIEWED_CLASS_CONFLICT'
  else:action='AGREEMENT_ONLY_NOT_GT_VERIFIED'
  if r and not conflict:raise ValueError('Tags30 review is no longer a conflict '+str((cid,iid)))
  b=hist[(cid,iid)]
  rows.append({
   'case_id':cid,'instance_id':iid,'image_sha256':im['image_sha256'],
   'v5_category_unverified':v5,'qwen_category_unverified':tc,
   'v5_bbox_historical':b.get('bbox_quality',''),'v5_class_historical':b.get('v5_category_quality',''),
   'metadata_integrity':'PASS' if metadata_pass else 'FAIL',
   'category_agreement':'NO' if conflict else 'YES',
   'user_chat_attested_30':'YES' if r else 'NO',
   'prior_ai_review_root_cause_chat_accepted':typ,
   'review_confidence_prior_ai':r.get('ai_confidence','') if r else '',
   'routing_action':action,'qwen_tag_category_publish_allowed':'NO',
   'shadow_alias_candidate':'YES' if (cid,iid) in shadow else 'NO',
   'shadow_alias_destination':shadow[(cid,iid)][1] if (cid,iid) in shadow else '',
   'v5_record_modified':'NO','qwen_record_modified':'NO','category_auto_overwrite':'NO',
   'review_provenance':'AI_PREFILL_USER_CHAT_ACCEPTED_NO_EXPORTED_CSV' if r else 'NOT_IN_30_REVIEW',
  })
 if failures:raise ValueError('metadata integrity failures: '+str(failures[:3]))
 assert len(rows)==171
 by_action=collections.Counter(r['routing_action'] for r in rows)
 labels=collections.Counter(r['prior_ai_review_root_cause_chat_accepted'] for r in rows if r['user_chat_attested_30']=='YES')
 mismatch=sum(r['category_agreement']=='NO' for r in rows)
 if mismatch!=93 or len(rev)!=30 or set(labels)!=CLASS_CODES:
  raise ValueError('DEV60 regression signature mismatch; inspect source before proceeding')
 summary={
  'mode':'OFFLINE_SIDECAR_NO_CHANGE','scope':'FROZEN_DEV60_TRAIN_SOURCE_ONLY',
  'source_images':60,'source_v5_instances':171,'source_focus_v1_qwen_tasks':513,
  'tags_v5_class_conflicts':mismatch,'tags_v5_class_agreement_not_GT':171-mismatch,
  'reviewed_30_user_chat_attested_AI_prefill':len(rev),
  'prior_ai_root_cause_distribution':dict(sorted(labels.items())),
  'routing_distribution':dict(sorted(by_action.items())),
  'shadow_alias_proposals_untouched':len(alias),
  'reviewed_30_intersection_shadow_alias':0,
  'raw_records_changed':0,'qwen_answers_moved':0,'published_categories':0,
  'independent_holdout_opened':False,'mainline_deployment_approved':False,
  'caution':['This routes risks; it does NOT fix model perception or demonstrate accuracy improvements.',
             'User confirmed prior AI prefill in chat; no exported edited CSV exists in this archive.',
             'A V5/Qwen class agreement cannot establish true category accuracy.',
             'Mainline must remain unchanged until candidate freeze and authorized independent blind test.'],
  'sha256_of_immutable_inputs':{key:hashlib.sha256(p.read_bytes()).hexdigest() for key,p in manifests.items()},
 }
 write_csv(out/'dev60_171_identity_class_routing.csv',rows,list(rows[0]))
 write_csv(out/'confirmed30_targeted_fix_worklist.csv',[r for r in rows if r['user_chat_attested_30']=='YES'],list(rows[0]))
 (out/'risk_guard_v3_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output-dir',type=Path)
 args=p.parse_args()
 print(json.dumps(route_case(output=args.output_dir),indent=2,ensure_ascii=False))
