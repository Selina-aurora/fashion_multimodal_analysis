#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv,hashlib,importlib.util,json,tempfile
from pathlib import Path
from PIL import Image,ImageDraw
spec=importlib.util.spec_from_file_location('d',Path(__file__).parent/'build_target_binding_review.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as t:
    root=Path(t); images=root/'fashion_data/raw/train/train/image';images.mkdir(parents=True)
    selected=root/'selected.csv';preds=root/'preds.jsonl';cases=[]
    for i,ref,model in [(1,'pants','top'),(2,'top','top')]:
        f=images/f'{i}.jpg';im=Image.new('RGB',(200,240),'white');ImageDraw.Draw(im).rectangle((20,70,130,210),fill='black');im.save(f)
        mp=root/f'{i}_marked.jpg';im.save(mp)
        digest=hashlib.sha256(f.read_bytes()).hexdigest()
        cases.append({'decision':'TRAIN_DEV_ONLY','inventory_category_NOT_GT':ref,'resolved_image_path':str(f),'sha256':digest,'bbox_x1':20,'bbox_y1':70,'bbox_x2':130,'bbox_y2':210})
        with preds.open('a',encoding='utf-8') as out:
            out.write(json.dumps({'case_id':f'train_dev_{i:03d}','source_split':'train','bbox_origin':'historical_inventory_gt_pilot_NOT_V5_prediction','source_hint_NOT_GT':ref,'input_image_sha256':digest,'image_id':str(i),'target_instance_id':f'instance_{i}','canonical_tags':{'category':model,'primary_color':'黑色','pattern':None,'visible_design':'长裤'},'issues':[],'timing':{'end_to_end_ms':1000},'marked_image':str(mp)},ensure_ascii=False)+'\n')
    with selected.open('w',newline='',encoding='utf-8') as out:
        w=csv.DictWriter(out,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    out=root/'report';m.build(type('A',(),{'predictions':str(preds),'selected':str(selected),'output':str(out)}))
    page=(out/'TARGET_BINDING_REVIEW_STATIC.html').read_text(encoding='utf-8')
    assert page.count('<section class="card"')==1 and 'data:image/jpeg;base64,' in page
    assert (out/'mismatch19_review_queue.csv').exists() and (out/'analysis_summary.json').exists()
    assert json.loads((out/'analysis_summary.json').read_text())['hint_mismatch_count']==1
    print('STATIC_DIAGNOSTIC_TEST_PASS')
