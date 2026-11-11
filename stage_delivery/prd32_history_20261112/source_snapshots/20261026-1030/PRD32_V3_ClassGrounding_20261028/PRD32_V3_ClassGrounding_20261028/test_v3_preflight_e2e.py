#!/usr/bin/env python3
"""Synthetic end-to-end no-GPU preflight validation; no real training data used."""
from __future__ import annotations
import csv,json,subprocess,sys,tempfile,hashlib
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);imroot=root/'fashion_data'/'raw'/'train'/'train'/'image'; imroot.mkdir(parents=True)
    selected=root/'train_dev_selected.csv';base=root/'predictions.jsonl'
    fields=['resolved_image_path','inventory_category_NOT_GT','decision','sha256','record_id','garment_id','bbox_x1','bbox_y1','bbox_x2','bbox_y2']
    with selected.open('w',newline='',encoding='utf-8') as f1,base.open('w',encoding='utf-8') as f2:
        writer=csv.DictWriter(f1,fieldnames=fields);writer.writeheader()
        for i in range(1,41):
            p=imroot/f'{i:03d}.jpg';Image.new('RGB',(160,160),(i*5%256,80,100)).save(p)
            d=sha(p);cid=f'train_dev_{i:03d}'
            writer.writerow({'resolved_image_path':str(p),'inventory_category_NOT_GT':'pants' if i in range(9,15) else 'top',
              'decision':'TRAIN_DEV_ONLY','sha256':d,'record_id':f'r{i}','garment_id':f'item{i}',
              'bbox_x1':12,'bbox_y1':15,'bbox_x2':130,'bbox_y2':150})
            f2.write(json.dumps({'case_id':cid,'task':'tags','image_id':p.stem,'input_image_sha256':d,
              'bbox_origin':'historical_inventory_gt_pilot_NOT_V5_prediction','status':'GENERATED_UNVERIFIED',
              'canonical_tags':{'category':'top'}},ensure_ascii=False)+'\n')
    for subset,n in [('reviewed15',15),('all40',40)]:
        dest=root/f'preflight_{subset}'
        cmd=[sys.executable,str(HERE/'prd32_v3_class_grounding.py'),'preflight','--selected',str(selected),
             '--baseline',str(base),'--allowed-root',str(root/'fashion_data'),'--subset',subset,
             '--output',str(dest)]
        subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        info=json.loads((dest/'preflight_summary.json').read_text())
        assert info['selected_cases']==n and info['model_not_loaded'] and not info['human_labels_sent_to_model']
    # Deliberately relabel an image path to a forbidden source; must stop before GPU.
    rows=list(csv.DictReader(selected.open()))
    rows[0]['resolved_image_path']=str(root/'fashion_data'/'raw'/'test'/'001.jpg')
    with selected.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    r=subprocess.run([sys.executable,str(HERE/'prd32_v3_class_grounding.py'),'preflight',
      '--selected',str(selected),'--baseline',str(base),'--allowed-root',str(root/'fashion_data'),
      '--subset','all40','--output',str(root/'blocked')],capture_output=True,text=True)
    assert r.returncode!=0,'Forbidden test path not blocked'
print('V3_40_CASE_SYNTHETIC_PREFLIGHT_E2E_PASS')
