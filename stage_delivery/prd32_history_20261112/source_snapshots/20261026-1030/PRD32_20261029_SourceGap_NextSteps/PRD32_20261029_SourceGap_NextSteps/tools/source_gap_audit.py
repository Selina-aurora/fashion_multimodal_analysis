#!/usr/bin/env python3
"""Read source metadata; open ONLY explicitly permitted TRAIN image bytes for SHA matching.

This does not approve a new dataset, train a model, or open val/test/fullval/holdout.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,re
from pathlib import Path
from collections import Counter,defaultdict

ROOT=Path(__file__).resolve().parent.parent
CATS=('top','pants','skirt','outerwear','dress','shoe','bag','accessory')
RESTRICTED={'holdout','sealed','fullval','val','valid','validation','test','testing','testset','eval','evaluation'}
IMAGE_EXT={'.jpg','.jpeg','.png','.webp','.bmp'}
REHEX=re.compile(r'^[0-9a-f]{64}$')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def read_csv(path):
 with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write_csv(path,rows,fields):
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def old_hashes(path):
 rows=read_csv(path)
 result={r['image_sha256'].strip().lower() for r in rows}
 if len(rows)!=40 or len(result)!=40 or not all(REHEX.fullmatch(x) for x in result):raise ValueError('Prior 40 image registry is invalid')
 return result

def new_hashes(path):
 data=json.loads(Path(path).read_text(encoding='utf-8'))
 rows=data.get('rows',[])
 result={r['image_sha256'].strip().lower() for r in rows}
 if len(rows)!=12 or len(result)!=12 or not all(REHEX.fullmatch(x) for x in result):raise ValueError('Prior 12 image registry is invalid')
 return result

def validate_bbox(r):
 try:b=[float(r['bbox_'+k]) for k in ('x1','y1','x2','y2')]
 except (KeyError,ValueError,TypeError):return False
 return all(math.isfinite(v) for v in b) and b[0]>=0 and b[1]>=0 and b[2]>b[0] and b[3]>b[1]

def permitted_path(source,project,data):
 if not source:return None,'EMPTY_SOURCE'
 p=(project/source).resolve(strict=False)
 if not p.is_relative_to(data):return None,'OUTSIDE_ALLOWED_DATA_ROOT'
 parts=[x.lower() for x in p.parts]
 for s in parts:
  if s in RESTRICTED or any(s.startswith(t+'_') or s.endswith('_'+t) for t in RESTRICTED):return None,'RESTRICTED_SPLIT'
 if 'train' not in parts:return None,'NOT_EXPLICIT_TRAIN'
 if p.suffix.lower() not in IMAGE_EXT:return None,'NOT_IMAGE'
 if not p.is_file():return None,'MISSING_ALLOWED_TRAIN_FILE'
 return p,None

def audit(project,data,inventory,old,new,out):
 if out.exists():raise FileExistsError(f'Output already exists: {out}; choose _run02')
 rows=read_csv(inventory)
 expected={'source_image','garment_category','bbox_x1','bbox_y1','bbox_x2','bbox_y2'}
 if not rows or not expected.issubset(rows[0]):raise ValueError(f'Inventory missing: {expected-set(rows[0] if rows else [])}')
 old_set=old_hashes(old);new_set=new_hashes(new)
 if old_set & new_set:raise ValueError('Old40 and new12 SHA registry overlap: recheck data')
 log=[];raw_eligible=[];counts=defaultdict(Counter)
 permitted_hash_reads=0
 for i,r in enumerate(rows,1):
  cat=(r.get('garment_category') or '').strip().lower()
  shown=cat if cat in CATS else 'UNKNOWN'
  counts[shown]['inventory_rows']+=1
  source=(r.get('source_image') or '').strip()
  p,reason=permitted_path(source,project,data)
  if cat not in CATS:reason=reason or 'BAD_CLASS_HINT'
  if not validate_bbox(r):reason=reason or 'BAD_BBOX'
  digest=''
  if not reason:
   try:
    digest=sha(p);permitted_hash_reads+=1
   except (OSError,PermissionError):reason='TRAIN_SHA_READ_ERROR'
  if not reason:
   if digest in old_set:reason='SAME_SHA_AS_PRIOR40'
   elif digest in new_set:reason='SAME_SHA_AS_PRIOR12'
  if reason:
   counts[shown][reason]+=1
  else:
   counts[shown]['post_exclusion_rows']+=1
   raw_eligible.append({'row_number':i,'source_hint_NOT_GT':cat,'sha256':digest,
                        'record_id':r.get('record_id',''),'garment_id':r.get('garment_id',''),
                        'bbox_x1':r['bbox_x1'],'bbox_y1':r['bbox_y1'],'bbox_x2':r['bbox_x2'],'bbox_y2':r['bbox_y2']})
  log.append({'row_number':i,'source_hint_NOT_GT':shown,'reason':reason or 'TRAIN_DEV_CANDIDATE',
              'image_sha_prefix':digest[:12] if digest else ''})
 # An image can have several item boxes or hints. Count unique image identity, not rows.
 grouped=defaultdict(list)
 for r in raw_eligible:grouped[r['sha256']].append(r)
 for digest,rr in grouped.items():
  if len(rr)>1:
   for r in rr:counts[r['source_hint_NOT_GT']]['multi_instance_rows']+=1
 available={c:len({r['sha256'] for r in raw_eligible if r['source_hint_NOT_GT']==c}) for c in CATS}
 # A single image is not automatically used for two independent class samples.
 globally_unique=len(grouped)
 selected=sorted(raw_eligible,key=lambda r:(CATS.index(r['source_hint_NOT_GT']),r['sha256'],r['row_number']))
 used=set();dedup=[]
 for r in selected:
  if r['sha256'] in used:continue
  used.add(r['sha256']);dedup.append(r)
 out.mkdir(parents=True)
 write_csv(out/'exclusion_reason_audit.csv',log,['row_number','source_hint_NOT_GT','reason','image_sha_prefix'])
 write_csv(out/'potential_train_dev_candidates.csv',dedup,['row_number','source_hint_NOT_GT','sha256','record_id','garment_id','bbox_x1','bbox_y1','bbox_x2','bbox_y2'])
 summary={'schema':'prd32_source_gap_after_old40_new12_v1','status':'METADATA_AND_ALLOWED_TRAIN_SHA_AUDIT_COMPLETE',
  'scope':'TRAIN_SOURCE_DEV_ONLY_NOT_INDEPENDENT','inventory_rows':len(rows),
  'prior40_hashes':40,'prior12_hashes':12,'allowed_train_hash_reads':permitted_hash_reads,
  'quarantined_image_reads':0,'model_calls':0,'holdout_opened':False,
  'by_category_source_hint_NOT_GT':{c:dict(counts[c]) for c in CATS},
  'new_distinct_image_hashes_by_hint':available,'post_exclusion_distinct_image_hashes_all_categories':globally_unique,
  'globally_deduplicated_sample_candidate_rows':len(dedup),
  'skirt_gap':available['skirt']==0,'outerwear_gap':available['outerwear']==0,
  'candidate_labels_are_not_human_gt':True,'provenance_independence_not_proven':True,
  'priority_action':'ACQUIRE_APPROVED_NEW_SKIRT_OUTERWEAR_IMAGES' if available['skirt']==0 or available['outerwear']==0 else 'BLIND_REVIEW_AVAILABLE_TRAIN_DEV'}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 lines=['# PRD 3.2 来源缺口审计（10.29）','',f'共审计 {len(rows)} 行元数据，只对符合明确 TRAIN 路径规则的文件计算 SHA256，共读取 {permitted_hash_reads} 次。','',
        '| 来源类别提示（非GT） | 原始条目 | 与旧40同图 | 与新12同图 | 去重后剩余图像 |','|---|---:|---:|---:|---:|']
 for c in CATS:
  x=counts[c];lines.append(f'| {c} | {x["inventory_rows"]} | {x["SAME_SHA_AS_PRIOR40"]} | {x["SAME_SHA_AS_PRIOR12"]} | {available[c]} |')
 lines+=['','**所有来源类别均仅为候选线索；不是人工类别。**',
 '如果 skirt/outerwear 剩余为 0，现有明确训练来源清单不足以继续补这两类的未用开发样本，不应改从 fullval/matched/holdout 取图。',
 '后续需要另行获得可使用的新增图片及来源审批；之后先做盲审和类别/目标框确认，再运行固定 V2/V3 对照。',
 '该审计不保证与未提供的训练清单无重叠，也不保证图像近重复已经排除。']
 (out/'SOURCE_GAP_REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('SOURCE_GAP_AUDIT_OK',json.dumps({'rows':len(rows),'new_unique_images':globally_unique,'by_hint':available,'result':str(out)},ensure_ascii=False))
 return summary

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--project-root',default='/workspace/fashion_multimodal_analysis')
 ap.add_argument('--data-root',default='/workspace/fashion_data')
 ap.add_argument('--inventory')
 ap.add_argument('--old40',default=str(ROOT/'evidence/HumanEval40_Comparison.csv'))
 ap.add_argument('--new12',default=str(ROOT/'evidence/new12_blind_review.json'))
 ap.add_argument('--output',required=True)
 a=ap.parse_args();project=Path(a.project_root).resolve();data=Path(a.data_root).resolve()
 inv=Path(a.inventory).resolve() if a.inventory else project/'configs/prd_8class_unified_manifest_v2.csv'
 audit(project,data,inv,Path(a.old40),Path(a.new12),Path(a.output).resolve())
if __name__=='__main__':main()
