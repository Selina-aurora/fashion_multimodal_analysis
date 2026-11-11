#!/usr/bin/env python3
"""No-GPU synthetic test: exact 40-case cohort, 19-case selection, image hashes and forbidden split."""
import csv,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def invoke(command, must_succeed=True):
    r=subprocess.run(command,capture_output=True,text=True)
    if must_succeed and r.returncode!=0:
        raise AssertionError(r.stdout+'\n'+r.stderr)
    if not must_succeed and r.returncode==0:
        raise AssertionError('Expected blocked preflight')
    return r

with tempfile.TemporaryDirectory() as t:
    root=Path(t); imroot=root/'fashion_data'/'raw'/'train'/'train'/'image';imroot.mkdir(parents=True)
    selected=root/'train_dev_selected.csv'; base=root/'v2.jsonl';v3=root/'v3.jsonl'
    fields=['resolved_image_path','inventory_category_NOT_GT','decision','sha256','record_id','garment_id','bbox_x1','bbox_y1','bbox_x2','bbox_y2']
    with selected.open('w',newline='',encoding='utf-8') as f1,base.open('w',encoding='utf-8') as f2,v3.open('w',encoding='utf-8') as f3:
        writer=csv.DictWriter(f1,fieldnames=fields); writer.writeheader()
        for i in range(1,41):
            p=imroot/f'{i:03d}.jpg';Image.new('RGB',(160,160),(i*5%256,80,100)).save(p)
            d=sha(p);cid=f'train_dev_{i:03d}'
            writer.writerow({'resolved_image_path':str(p),'inventory_category_NOT_GT':'pants' if i in range(9,15) else 'top',
              'decision':'TRAIN_DEV_ONLY','sha256':d,'record_id':f'r{i}','garment_id':f'item{i}',
              'bbox_x1':12,'bbox_y1':15,'bbox_x2':130,'bbox_y2':150})
            f2.write(json.dumps({'case_id':cid,'task':'tags','image_id':p.stem,'input_image_sha256':d,
              'bbox_origin':'historical_inventory_gt_pilot_NOT_V5_prediction','status':'GENERATED_UNVERIFIED',
              'canonical_tags':{'category':'top'}},ensure_ascii=False)+'\n')
            f3.write(json.dumps({'case_id':cid,'status':'GENERATED_UNVERIFIED','category':'top',
              'input_image_sha256':d},ensure_ascii=False)+'\n')
    script=str(HERE/'prd32_v4_target_focus.py')
    for subset,count in [('reviewed19',19),('all40',40)]:
        dest=root/f'out_{subset}'
        invoke([sys.executable,script,'preflight','--selected',str(selected),'--baseline',str(base),
            '--v3-full-tags',str(v3),'--allowed-root',str(root/'fashion_data'),
            '--subset',subset,'--output',str(dest)])
        summ=json.loads((dest/'preflight_summary.json').read_text())
        plans=[json.loads(line) for line in (dest/'case_plan.jsonl').read_text().splitlines()]
        assert summ['selected_cases']==count and len(plans)==count
        assert summ['human_labels_sent_to_model'] is False and summ['model_not_loaded']
        if subset=='reviewed19':
            assert {'train_dev_001','train_dev_003','train_dev_005','train_dev_008'}<=set(x['case_id'] for x in plans)
    # Reject forbidden split even when source CSV claims TRAIN_DEV_ONLY.
    rows=list(csv.DictReader(selected.open()))
    rows[0]['resolved_image_path']=str(root/'fashion_data'/'raw'/'holdout'/'001.jpg')
    with selected.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)
    invoke([sys.executable,script,'preflight','--selected',str(selected),'--baseline',str(base),
        '--v3-full-tags',str(v3),'--allowed-root',str(root/'fashion_data'),
        '--subset','all40','--output',str(root/'reject')],False)
print('V4_SYNTHETIC_END_TO_END_PASS')
