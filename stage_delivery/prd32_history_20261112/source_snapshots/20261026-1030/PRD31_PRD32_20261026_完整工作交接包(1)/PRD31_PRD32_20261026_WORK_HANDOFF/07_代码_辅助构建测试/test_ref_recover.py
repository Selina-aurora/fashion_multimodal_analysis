import importlib.util, json, hashlib, tempfile, csv, sys
from pathlib import Path
from PIL import Image
src='/mnt/data/prd32_w7_recover_refs_v3_20261026.py'
spec=importlib.util.spec_from_file_location('recover',src)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
with tempfile.TemporaryDirectory(prefix='prd32_recovery_test_') as d:
    d=Path(d)
    records=[];review=[]
    for i, ref in enumerate(r.REFERENCE):
        im=d/(ref['image_id']+'.jpg');Image.new('RGB',(1100,1200),'white').save(im)
        h=hashlib.sha256(im.read_bytes()).hexdigest()
        ref['source_image']=str(im);ref['source_sha256']=h
        records.append({'image_id':ref['image_id'],'source_image':str(im),'source_sha256':h,'width':1100,'height':1200})
        rv={'image_id':ref['image_id'],'source_image':str(im),'source_sha256':h,'target_category':ref['target_category']}
        if ref['image_id']=='dev_11': rv['source_sha256']='a'*64
        review.append(rv)
    selected=d/'selected.json'; selected.write_text(json.dumps(records))
    csvp=d/'review.csv'
    with csvp.open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(review[0]));w.writeheader();w.writerows(review)
    r.EXPECTED_SELECTED_SHA256=hashlib.sha256(selected.read_bytes()).hexdigest()
    sys.argv=['', '--selected',str(selected),'--review',str(csvp),'--output',str(d/'output')]
    r.main()
    summary=json.loads((d/'output'/'summary.json').read_text())
    assert summary['statuses']=={'REFERENCE_CANDIDATE_VISUAL_REVIEW_REQUIRED':10, 'REFERENCE_PENDING_PRIOR_REVIEW':1,'QUARANTINED':1},summary
    print('TEST_SUCCESS')
