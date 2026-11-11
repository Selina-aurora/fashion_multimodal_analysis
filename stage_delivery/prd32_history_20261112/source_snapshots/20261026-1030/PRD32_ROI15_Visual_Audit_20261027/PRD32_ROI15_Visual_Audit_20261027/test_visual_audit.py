#!/usr/bin/env python3
"""One-case browser regression: statically visible cards and usable exports."""
import importlib.util, json, tempfile
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('roi15_audit',here/'build_roi_visual_audit.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
with tempfile.TemporaryDirectory() as d:
    d=Path(d);src=d/'fashion_data/raw/train/train/image';src.mkdir(parents=True)
    run=d/'run';(run/'roi_marked_inputs').mkdir(parents=True)
    img=src/'a.jpg';Image.new('RGB',(180,160),'white').save(img)
    roi=run/'roi_marked_inputs'/'a_roi.jpg';Image.new('RGB',(120,100),'white').save(roi)
    plan={'case_id':'train_dev_014','cause':'MODEL_CLASSIFICATION','human_category_FOR_EVAL_ONLY':'pants','baseline_category':'skirt','image_path':str(img),'image_sha256':mod.file_sha256(img),'bbox':[20,20,140,135]}
    pred={'case_id':'train_dev_014','cause':'MODEL_CLASSIFICATION','baseline_category':'skirt','roi_category':'pants','roi_input':str(roi),'schema_valid':True,'native_json':False,'issues':[],'timing':{'end_to_end_ms':1000}}
    (run/'case_plan.jsonl').write_text(json.dumps(plan)+'\n');(run/'roi_predictions.jsonl').write_text(json.dumps(pred)+'\n')
    (run/'summary.json').write_text(json.dumps({'historical_GT_pilot_bbox_not_frozen_V5':True,'holdout_used':False}))
    mod.build(run,d/'out',d/'fashion_data',expected=1)
    htmlfile=(d/'out'/'ROI15_VISUAL_REVIEW_STATIC.html')
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(htmlfile.read_text(),'html.parser')
    js=soup.find('script').text
    jsfile=d/'test.js';jsfile.write_text(js)
    import subprocess
    subprocess.run(['node','--check',str(jsfile)],check=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(accept_downloads=True)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.set_content(htmlfile.read_text(),wait_until='load')
        assert page.locator('.card').count()==1
        assert page.locator('.images img').count()==2
        assert page.locator('.images img').evaluate_all('(nodes)=>nodes.every(n=>n.complete && n.naturalWidth>0)')
        page.locator('[data-key="target_visible"]').select_option('YES')
        with page.expect_download() as dw:page.get_by_text('导出人工审核 JSON').click()
        down=dw.value;dest=d/'download.json';down.save_as(dest)
        data=json.loads(dest.read_text())
        assert data['rows'][0]['target_visible']=='YES'
        with page.expect_download() as dw2:page.get_by_text('导出 CSV').click()
        assert dw2.value.suggested_filename.endswith('.csv')
        assert not errors, errors
        browser.close()
    print('BROWSER_STATIC_IMAGES_AND_EXPORT_PASS')
