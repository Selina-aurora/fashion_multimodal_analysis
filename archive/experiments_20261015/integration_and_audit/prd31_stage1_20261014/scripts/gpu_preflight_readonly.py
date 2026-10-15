#!/usr/bin/env python3
"""Read-only inventory: do not evaluate, train, change manifests, or trust stale checkpoints."""
import argparse, csv, json, pathlib, hashlib, sys, os, datetime, zipfile
CORE_SHA='c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89'
CONFIGS=['configs/prd_8class_train_v5_balanced150_20261008.csv',
         'configs/prd_8class_train_v5_full_targets.csv',
         'configs/prd_8class_val_v2_full_targets.csv',
         'benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()

def inspect_csv(p):
 with open(p,'r',encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f);head=reader.fieldnames or [];r=list(reader)
  return {'rows':len(r),'columns':head,'sample_first_row':r[0] if r else None}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',default='/workspace/fashion_multimodal_analysis');parser.add_argument('--out',default='/workspace/prd31_preflight_20261014');a=parser.parse_args()
 root=pathlib.Path(a.root).resolve();out=pathlib.Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
 rep={'date':'2026-10-14','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'root':str(root),'root_exists':root.is_dir(),'read_only_project':True,'configs':{},'checkpoint_candidates':[],
 'core400_expected_sha256':CORE_SHA,'pass_overall':False,'warnings':[],
 'limitations':['This script only inventories inputs/checkpoints; no model forward pass, accuracy/latency measurement or image-level cross-split hash audit.']}
 for rel in CONFIGS:
  p=root/rel
  if p.is_file():
   d={'exists':True,'sha256':sha(p),'bytes':p.stat().st_size}
   try:d.update(inspect_csv(p))
   except Exception as exc:d['csv_error']=str(exc)
   if rel.endswith('segmentation_test_v1.csv'):
    d['sha_match_expected']=d['sha256']==CORE_SHA
    if not d['sha_match_expected']:rep['warnings'].append('CORE400_MANIFEST_HASH_MISMATCH_STOP')
  else:
   d={'exists':False};rep['warnings'].append('MISSING '+rel)
  rep['configs'][rel]=d
 outputs=root/'outputs'
 if outputs.is_dir():
  ckpts=sorted((p for p in outputs.rglob('*.pth') if p.is_file()),key=lambda p:p.stat().st_mtime,reverse=True)
  rep['checkpoint_candidates']=[{'relpath':str(p.relative_to(root)),'bytes':p.stat().st_size,'mtime':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat()} for p in ckpts[:100]]
  if len(ckpts)>100:rep['warnings'].append('CHECKPOINT_LIST_TRUNCATED')
 if not rep['checkpoint_candidates']:rep['warnings'].append('NO_PTH_CHECKPOINTS_FOUND')
 rep['pass_overall']=bool(root.is_dir() and rep['configs'][CONFIGS[2]]['exists'] and rep['configs'][CONFIGS[3]]['exists'] and rep['configs'][CONFIGS[3]].get('sha_match_expected') and rep['checkpoint_candidates'])
 (out/'gpu_preflight_20261014.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf-8')
 summary=['# GPU 只读回归前检查','',f'项目路径存在：{rep["root_exists"]}',f'Core400 冻结清单哈希匹配：{rep["configs"][CONFIGS[3]].get("sha_match_expected",False)}',f'发现 checkpoint：{len(rep["checkpoint_candidates"])} 个（最多列出100个）',f'基础资料齐备（非模型可运行证明）：{rep["pass_overall"]}','','## 警告']+[f'- {v}' for v in rep['warnings']]+['','**本检查不启动模型、不计算精度、不改变文件。**','把 JSON 发回后确认具体 checkpoint、代码入口、依赖版本，才能安排正式回归。']
 (out/'README_GPU检查结果.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
 zip_path=out.parent/(out.name+'_UPLOAD_THIS.zip')
 with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
  for p in out.iterdir():
   if p.is_file():z.write(p,p.name)
 print('PREFLIGHT_DONE_NO_TRAINING',zip_path);print('FOUND_CHECKPOINTS',len(rep['checkpoint_candidates']));print('CORE400_HASH_MATCH',rep['configs'][CONFIGS[3]].get('sha_match_expected',False))

if __name__=='__main__':main()
