#!/usr/bin/env python3
"""Create deterministic, leakage-screened *image-only* holdout manifest.

No predictions and no GT are inspected. Fail closed if the training or validation
manifests cannot be interpreted. Does not modify the project/dataset.
"""
import argparse,csv,hashlib,json,sys
from pathlib import Path
from PIL import Image

HERE=Path(__file__).resolve().parent
TRAIN='configs/prd_8class_train_v5_balanced150_20261008.csv'
VAL='configs/prd_8class_val_v2_full_targets.csv'
CORE='benchmark/prd_3_1_v1/manifests/segmentation_test_v1.csv'
EXPECTED={TRAIN:'e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f',VAL:'39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93',CORE:'c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89'}
EXT={'.jpg','.jpeg','.png','.webp'}
PATH_FIELDS={'image_path','source_image_path','image_file','image_file_path','img_path','file_path','filename','file_name','image_name','path','image','image_relpath','source_path','source_image_file','source_filename','source_image','image_abs_path','absolute_image_path'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
def save_json(path,obj):
 Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def exclusion_from_csv(csvpath, project_root=None):
 with Path(csvpath).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f); columns=rd.fieldnames or []
  col=[x for x in columns if x.strip().lower() in PATH_FIELDS]
  if not col:
   raise ValueError(f'Cannot identify image columns in {csvpath}. Columns: {columns}. STOP, do not bypass exclusions.')
  names=set(); paths=set(); hashes=set(); rows=0; used=0; resolved_files=0
  for r in rd:
   rows+=1
   for c in col:
    val=(r.get(c) or '').strip().strip('"').replace('\\','/')
    if not val:continue
    p=Path(val)
    if p.suffix.lower() in EXT:
     names.add(p.name.lower()); paths.add(val.lower());used+=1
     # If referenced source bytes are mounted, also exclude their SHA (even if renamed).
     probes=[p] if p.is_absolute() else ([Path(project_root)/p] if project_root else [])
     for source in probes:
      if source.is_file():
       hashes.add(sha(source));resolved_files+=1;break
  if not names:
   raise ValueError(f'CSV image fields {col} have no recognizable file names: {csvpath}; stop.')
 return {'basename_set':names,'path_set':paths,'sha256_set':hashes,'columns':col,'rows':rows,'image_refs':used,'resolved_files':resolved_files}

def main():
 pa=argparse.ArgumentParser()
 pa.add_argument('--root',default='/workspace/fashion_multimodal_analysis')
 pa.add_argument('--candidate-dir',default='/workspace/fashion_data/raw/fashionpedia/images/test')
 pa.add_argument('--n',type=int,default=20)
 pa.add_argument('--seed',default='prd31-v3-holdout-20261013-v1')
 pa.add_argument('--out-dir',default='/workspace/sleeve_v3_holdout20_20261013')
 a=pa.parse_args();root=Path(a.root).resolve();src=Path(a.candidate_dir).resolve();out=Path(a.out_dir).resolve()
 if not 1<=a.n<=40:raise ValueError('n must be between 1 and 40')
 if out.exists():raise FileExistsError(f'Will not overwrite: {out}')
 for rel,want in EXPECTED.items():
  path=root/rel
  if not path.is_file():raise FileNotFoundError(f'Missing frozen CSV: {path}')
  got=sha(path)
  if got!=want:raise ValueError(f'Frozen manifest SHA changed: {rel}: {got}; STOP')
 exclusions={k:exclusion_from_csv(root/k,root) for k in (TRAIN,VAL)}
 excluded_names=set().union(*(r['basename_set'] for r in exclusions.values()))
 excluded_hashes=set().union(*(r['sha256_set'] for r in exclusions.values()))
 core_hashes=set((HERE/'inputs/core400_sha256.txt').read_text().split())
 if len(core_hashes)!=400 or any(len(h)!=64 for h in core_hashes):raise ValueError('Core400 fingerprint list invalid')
 old=json.loads((HERE/'inputs/dev8_images.json').read_text(encoding='utf-8'))
 old_hashes={x['sha256'] for x in old};old_names={Path(x['path']).name.lower() for x in old}
 excluded_names|=old_names
 history=json.loads((HERE/'inputs/previous_reviewed_or_used_images.json').read_text(encoding='utf-8'))
 excluded_names.update(history['excluded_filenames'])
 excluded_hashes.update(history['excluded_sha256'])
 if not src.is_dir():raise FileNotFoundError(f'Candidate folder missing: {src}')
 all_images=[p for p in src.rglob('*') if p.is_file() and p.suffix.lower() in EXT]
 if len(all_images)<a.n:raise ValueError('Insufficient raw images')
 # The cohort is selected by a deterministic hash of seed and RELATIVE PATH alone.
 # No human labels, model outputs, class GT, or image appearance is used to rank.
 all_images.sort(key=lambda p:hashlib.sha256((a.seed+'\0'+p.relative_to(src).as_posix()).encode()).digest())
 selected=[];counts={'blocked_manifest_or_dev_filename':0,'blocked_train_val_core_or_dev_sha256':0,'duplicate_sha256':0,'broken_image':0}
 seen=set()
 for p in all_images:
  if p.name.lower() in excluded_names:
   counts['blocked_manifest_or_dev_filename']+=1;continue
  digest=sha(p)
  if digest in core_hashes or digest in old_hashes or digest in excluded_hashes:
   counts['blocked_train_val_core_or_dev_sha256']+=1;continue
  if digest in seen:
   counts['duplicate_sha256']+=1;continue
  try:
   with Image.open(p) as im:
    im.verify()
   with Image.open(p) as im:
    w,h=im.size
  except Exception:
   counts['broken_image']+=1;continue
  if min(w,h)<300:
   counts['broken_image']+=1;continue
  seen.add(digest)
  selected.append({'image_id':f'holdout_{len(selected)+1:03d}','path':str(p),'sha256':digest,'size':[w,h],'source_name':p.name})
  if len(selected)>=a.n:break
 if len(selected)<a.n:raise ValueError(f'Only {len(selected)} eligible images; STOP')
 out.mkdir(parents=True,exist_ok=False)
 save_json(out/'holdout_images_FROZEN.json',selected)
 excludes_info={k:{'rows':x['rows'],'columns':x['columns'],'distinct_images':len(x['basename_set']),'resolved_for_SHA':x['resolved_files']} for k,x in exclusions.items()}
 provenance={'purpose':'fresh cohort for V3 dev_v1 fixed-rule diagnostic; independent of *geometry rule development* only',
  'sampling_protocol':'deterministic seed+relative path; no GT, model prediction, or visual curation',
  'seed':a.seed,'selected_n':len(selected),'candidate_dir':str(src),'candidate_files_count':len(all_images),
  'checked_frozen_manifest_sha256':EXPECTED,'train_val_image_exclusions':excludes_info,
  'core400_n_sha256':len(core_hashes),'dev8_n_sha256':len(old_hashes),'historical_candidates_excluded_filenames':len(history['excluded_filenames']),'historical_candidates_excluded_sha256':len(history['excluded_sha256']),
  'excluded_counts_during_selection':counts,'sampling_inference_rule_sha256':sha(HERE/'geometry_v3_proposals.py'),
  'selected_file_manifest_sha256':sha(out/'holdout_images_FROZEN.json'),
  'limitations':['Filename+byte-SHA exclusions cannot rule out visually near-identical recompressed duplicates',
                 'Selection not stratified by sleeve class; no target garment in some images is possible',
                 'This cohort cannot become a blind PRD test after examining its results']}
 save_json(out/'cohort_provenance.json',provenance)
 print('HOLDOUT_PREPARED',len(selected))
 print('TRAIN_VAL_EXCLUSION_COUNTS',excludes_info)
 print('SELECTED_MANIFEST_SHA256',provenance['selected_file_manifest_sha256'])
 print('OUTPUT_DIR',out)
 print('NEXT: run_frozen_inference.py --out-dir',out)
if __name__=='__main__': main()
