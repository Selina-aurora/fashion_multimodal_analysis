#!/usr/bin/env python3
"""Read-only Stage 11: isolate runtime production DINO/geometry-vs-batch drift.

This diagnoses Stage10 mismatch; no training, no original repository edits.
Only a small number of current diagnostic inputs are read. Does not claim accuracy.
"""
from __future__ import annotations
import argparse,hashlib,inspect,json,math,os,statistics,sys,time,traceback,zipfile
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import stage6_helpers as h
import stage9_dino_impl as dino
ROOT=Path('/workspace/fashion_multimodal_analysis')
OUTROOT=Path('/workspace/prd31_stage11_runs_20261014')

def compare_det(a,b):
    an,bn=len(a),len(b)
    n=min(an,bn)
    diffs=[max(abs(float(x)-float(y)) for x,y in zip(a[i]['box'],b[i]['box'])) for i in range(n)]
    sdiff=[abs(float(a[i]['score'])-float(b[i]['score'])) for i in range(n)]
    label_eq=all(a[i]['label']==b[i]['label'] for i in range(n))
    topa=a[0] if a else None;topb=b[0] if b else None
    return {'candidate_count_a':an,'candidate_count_b':bn,
            'max_box_abs_px':max(diffs,default=0),'max_score_abs':max(sdiff,default=0),
            'labels_equal':label_eq,
            'candidate_parity':an==bn and label_eq and max(diffs,default=0)<=0.5 and max(sdiff,default=0)<=0.005,
            'top_a':topa,'top_b':topb}

def source_audit(backend,out):
    from fashion_multimodal_analysis.grounding.refinement.run_shoulder_waist_pattern_groundingdino_refinement_v1 import run_grounding_dino
    data={}
    for k,obj in [('runtime_ModelBackends_ground',backend.ground),('runtime_run_grounding_dino',run_grounding_dino),
                  ('experimental_predict_dino',dino.predict_dino)]:
        try:
            src=inspect.getsource(obj)
            text=(out/(k+'.txt'))
            text.write_text(src,encoding='utf-8')
            data[k]={'source_file':str(inspect.getfile(obj)),
                      'sha256':hashlib.sha256(src.encode()).hexdigest(),
                      'source_lines':len(src.splitlines()),
                      'tokens':{token:(token in src) for token in ['rejected_region_geometry','semantic','max_local_region_crop_area_ratio','neckline_context','crop_bounds','run_grounding_dino','run_sam']}}
        except Exception as e:data[k]={'source_error':str(e)}
    h.write_json(out/'runtime_source_audit.json',data)
    return data

def run(backend,selection,out,case_limit,query_count):
    from PIL import Image
    from fashion_multimodal_analysis.integration.pipeline import roi_images
    from fashion_multimodal_analysis.benchmarking.prepare_prd31_recovery import PROMPTS
    from fashion_multimodal_analysis.grounding.refinement.run_shoulder_waist_pattern_groundingdino_refinement_v1 import run_grounding_dino
    rows=[]; selection_out=[]
    for j,record in enumerate(selection,1):
        if len(selection_out)>=case_limit:break
        with Image.open(record['source_image']) as f:im=f.convert('RGB')
        rois,_=backend.segment(im)
        if not rois:continue
        roi=sorted(rois,key=h.preferred)[0]
        regions=h.allowed_regions(roi.category)[:query_count]
        if len(regions)<2:continue
        crop,mask,origin=roi_images(im,roi)
        isolated=Image.composite(crop,Image.new('RGB',crop.size,'white'),mask)
        texts=[PROMPTS[x] for x in regions]
        batch=dino.predict_dino(backend,isolated,texts,batch=True)
        serial=dino.predict_dino(backend,isolated,texts,batch=False)
        selection_out.append({'case':f'case_{j:02d}','roi_class':roi.category,
                              'crop_size':list(crop.size),'regions':regions})
        for idx,region in enumerate(regions):
            native,_=run_grounding_dino(isolated,PROMPTS[region],
                 backend.config['grounding_box_threshold'],backend.config['grounding_text_threshold'],
                 backend.ground_processor,backend.ground_model,backend.device)
            baseline,_=backend.ground(im,roi,region)
            ab=compare_det(native,serial[idx]);bc=compare_det(serial[idx],batch[idx]);ac=compare_det(native,batch[idx])
            rows.append({'case':f'case_{j:02d}','region':region,'production_status':baseline.get('status'),
                         'production_dino_phrase':str(baseline.get('model_phrase','')),
                         'production_dino_score':baseline.get('score'),
                         'production_proposal_bbox':json.dumps(baseline.get('proposal_bbox'),ensure_ascii=False),
                         'native_candidate_count':len(native),'helper_serial_candidate_count':len(serial[idx]),
                         'helper_batch_candidate_count':len(batch[idx]),
                         'native_serial_parity':ab['candidate_parity'],
                         'serial_batch_parity':bc['candidate_parity'],
                         'native_batch_parity':ac['candidate_parity'],
                         'native_batch_max_box_px':ac['max_box_abs_px'],
                         'native_batch_max_score_abs':ac['max_score_abs'],
                         'native_batch_labels_equal':ac['labels_equal'],
                         'native_top_json':json.dumps(ac['top_a'],ensure_ascii=False),
                         'batch_top_json':json.dumps(ac['top_b'],ensure_ascii=False)})
            print(f"{rows[-1]['case']} {region}: prod={baseline.get('status')} native/batch={ac['candidate_parity']}",flush=True)
    h.write_csv(out/'dino_three_path_comparison.csv',rows)
    h.write_json(out/'case_selection.json',selection_out)
    if not rows:raise RuntimeError('No eligible queries checked')
    summary={'status':'STAGE11_ROOTCAUSE_DIAGNOSTIC_COMPLETED','cases':len(selection_out),'queries':len(rows),
      'native_vs_helper_serial_pass':sum(r['native_serial_parity'] for r in rows),
      'helper_serial_vs_batch_pass':sum(r['serial_batch_parity'] for r in rows),
      'native_vs_helper_batch_pass':sum(r['native_batch_parity'] for r in rows),
      'production_status_counts':{s:sum(r['production_status']==s for r in rows) for s in sorted({str(r['production_status']) for r in rows})},
      'scope':'read-only same-image same-prompt candidate parity; no localization GT or speed acceptance',
      'next_step':'Inspect runtime ground source and proposal differences; do not integrate Stage10 experimental path'}
    h.write_json(out/'rootcause_summary.json',summary)
    return summary

def pack(out):
    dest=out.parent/f'prd31_stage11_{out.name}_UPLOAD_THIS.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(out))
    with zipfile.ZipFile(dest) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'Corrupt ZIP entry: {bad}')
    return dest

def main():
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-cases',type=int,default=4);p.add_argument('--query-count',type=int,default=4)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out-root',type=Path,default=OUTROOT)
    a=p.parse_args()
    if not(1<=a.max_cases<=8 and 2<=a.query_count<=8):p.error('Invalid cases/query count')
    root=a.root.resolve();outroot=a.out_root.resolve()
    if root==outroot or root in outroot.parents or outroot==root.parent:p.error('Output must stay outside repo')
    out=h.new_out(outroot);audit={'status':'STARTED','project_files_modified':False}
    code=0
    try:
        os.environ.setdefault('HF_HOME','/workspace/hf_cache')
        os.environ.setdefault('HF_HUB_CACHE','/workspace/hf_cache/hub')
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,str(root/'src'))
        cfg,sel=h.check_inputs(root,out)
        if a.dry_run:audit['status']='DRY_RUN_PASS'
        else:
            audit['checkpoint']=h.inspect_checkpoint(root)
            from fashion_multimodal_analysis.integration.model_backends import ModelBackends
            backend=ModelBackends(cfg,device='cuda')
            audit['runtime_source_audit']=source_audit(backend,out)
            audit['result']=run(backend,sel,out,a.max_cases,a.query_count)
            audit['status']=audit['result']['status']
    except Exception as e:
        audit['status']='STAGE11_FAILED';audit['error']=f'{type(e).__name__}: {e}'
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('ERROR:',audit['error'],flush=True);code=2
    finally:
        h.write_json(out/'run_audit.json',audit)
        try:
            zip_path=pack(out)
            print('STATUS:',audit['status'])
            if a.dry_run: print('DRY_RUN_PASS' if audit['status']=='DRY_RUN_PASS' else 'DRY_RUN_FAILED')
            print('UPLOAD_THIS_ZIP:',zip_path)
            print('PROJECT_FILES_MODIFIED: False')
        except Exception as e:
            print('PACK_FAILED:',str(e));code=3
    return code
if __name__=='__main__':sys.exit(main())
