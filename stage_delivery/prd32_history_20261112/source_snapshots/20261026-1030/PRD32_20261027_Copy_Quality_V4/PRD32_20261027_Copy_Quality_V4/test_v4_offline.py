#!/usr/bin/env python3
import base64, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image
from importlib.util import spec_from_file_location,module_from_spec
SRC=Path(__file__).parent
spec=spec_from_file_location('copy_v4',SRC/'prd32_copy_quality_v4.py');m=module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory(prefix='prd32_v4_synth_') as tmp:
 root=Path(tmp)/'frozen'; root.mkdir()
 image=root/'safe_dev_test.jpg';Image.new('RGB',(80,100),(20,40,80)).save(image)
 hash_=hashlib.sha256(image.read_bytes()).hexdigest()
 (root/'summary.json').write_text(json.dumps({'status':'PRD31_STAGE_FREEZE_READY','phase_freeze_ready':True,'candidate_id':'SYNTHETIC_DEV_TEST'}))
 (root/'freeze_record.json').write_text(json.dumps({'frozen':True,'candidate_id':'SYNTHETIC_DEV_TEST','checkpoint_sha256':'0'*64}))
 records=[]
 for i,t in enumerate(['vqa','tags','copy','copy','copy']):
  records.append({'schema':'prd32_engine_neutral_input_v1','case_id':f'synth_{i}','task':t,'split':'development','target_bbox_origin':'prd31_frozen_prediction','image_path':str(image),'image_sha256':hash_,'image_id':'synth_img','target_instance_id':'synth_target','target_bbox_xyxy':[10,10,70,80],'query':'请描述该图片的目标服饰','predicted_attribute_context':[]})
 (root/'bridge_inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
 cmd=[sys.executable,str(SRC/'prd32_copy_quality_v4.py'),'--dry-run','--freeze-dir',str(root)]
 proc=subprocess.run(cmd,capture_output=True,text=True)
 assert proc.returncode==0,(proc.stdout,proc.stderr)
 assert 'copy_cases=3 eligible=3 blocked=0' in proc.stdout, proc.stdout
 # Wrong image bytes after hash -> all copy blocked; no GPU attempt.
 Image.new('RGB',(80,100),(220,40,40)).save(image)
 proc=subprocess.run(cmd,capture_output=True,text=True)
 assert proc.returncode==0,(proc.stdout,proc.stderr)
 assert 'copy_cases=3 eligible=0 blocked=3' in proc.stdout,proc.stdout
 # Static-first HTML must include static cards even when script disabled.
 testimg=root/'marked.jpg';Image.new('RGB',(30,30),'black').save(testimg)
 view=root/'html';view.mkdir()
 sample=[{'case_id':'copy_01','image_id':'x','target_instance_id':'t','draft':'黑色背带裙，带有背带设计。','facts':{},'status':'DRAFT_REVIEW_REQUIRED','issues':[],'marked_image':str(testimg)}]
 m.html_review(view,sample,{'copy_01':'黑色无袖连衣裙。'})
 h=(view/'COPY_V4_compare_review.html').read_text()
 assert '<section class="card">' in h and 'data:image/jpeg;base64,' in h
 js=h.split('<script>')[1].split('</script>')[0]
 (view/'test.js').write_text(js)
 subprocess.run(['node','--check',str(view/'test.js')],check=True)
 print('V4_OFFLINE_FULL_PASS: guarded preflight, mismatched SHA quarantine, static image HTML, JS syntax')
