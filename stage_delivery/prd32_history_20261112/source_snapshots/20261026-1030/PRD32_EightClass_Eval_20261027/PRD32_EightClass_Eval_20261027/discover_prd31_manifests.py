#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only discovery of potential existing 3.1 detection/instance manifests; no image data opened."""
import argparse,csv,os,sys
from pathlib import Path
SAFE_TOKENS=('holdout','sealed','private','do_not_open','train','__pycache__')
INTERESTING=('pred','instance','manifest','detect','bbox','infer','core400','export')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',default='/workspace/fashion_multimodal_analysis')
 p.add_argument('--output',default='/workspace/PRD32_EIGHTCLASS_SOURCE_DISCOVERY_20261027.csv')
 p.add_argument('--max-results',type=int,default=100)
 a=p.parse_args()
 root=Path(a.root)
 if not root.is_dir():raise NotADirectoryError(root)
 results=[]
 for dirname,dirs,files in os.walk(root):
  dirs[:]=[d for d in dirs if not any(t in d.lower() for t in SAFE_TOKENS) and not d.startswith('.')]
  for name in files:
   if Path(name).suffix.lower() not in ('.csv','.jsonl','.json'):continue
   low=name.lower()
   if any(t in low for t in SAFE_TOKENS):continue
   if not any(t in low for t in INTERESTING):continue
   full=Path(dirname)/name
   if not full.is_file() or not full.stat().st_size:continue
   results.append((str(full),full.stat().st_size))
   if len(results)>=a.max_results:break
  if len(results)>=a.max_results:break
 out=Path(a.output)
 if any(t in str(out).lower() for t in ('holdout','sealed')):raise ValueError('No writes into restricted paths')
 if out.exists():raise FileExistsError(out)
 with out.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['path','size_bytes']);w.writerows(results)
 print(f'DISCOVERY_DONE candidates={len(results)} output={out} (filenames only, not confirmed schema)')
 for path,n in results[:30]:print(n,path)
if __name__=='__main__':
 try:main()
 except Exception as e:print(f'{type(e).__name__}: {e}',file=sys.stderr);sys.exit(2)
