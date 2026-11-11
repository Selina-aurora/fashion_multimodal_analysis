#!/usr/bin/env python3
"""Offline full-runner fault injection; never touches GPU/holdout/real freezes."""
from __future__ import annotations
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from PIL import Image

HERE=Path(__file__).resolve().parent


def base_record(image, *, cid, image_id, target_id, task):
    return {
        'schema':'prd32_engine_neutral_input_v1','case_id':cid,'task':task,
        'split':'development','image_id':image_id,'target_instance_id':target_id,
        'image_path':str(image),'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
        'source_size':[80,100], 'target_bbox_origin':'prd31_frozen_prediction',
        'target_bbox_xyxy':[5,6,70,90],'query':'测试问题','predicted_attribute_context':[]
    }


def run_fixture():
    with tempfile.TemporaryDirectory(prefix='prd32_v2_fault_') as tmp:
        root=Path(tmp);freeze=root/'freeze';freeze.mkdir()
        pic=root/'test.png';Image.new('RGB',(80,100),(220,220,220)).save(pic)
        v1=base_record(pic,cid='pass_vqa',image_id='good',target_id='pred01',task='vqa')
        t1=base_record(pic,cid='pass_tags',image_id='good',target_id='pred01',task='tags')
        t1['predicted_attribute_context']='not_a_list'  # warning only; strip the hint
        c1=base_record(pic,cid='pass_copy',image_id='good',target_id='pred01',task='copy')
        bad_bbox=base_record(pic,cid='bad_bbox',image_id='bad',target_id='pred02',task='vqa')
        bad_bbox['target_bbox_xyxy']=[0,0,85,50]
        bad_enum=base_record(pic,cid='bad_enum',image_id='enum',target_id='pred03',task='tags')
        bad_enum['target_category']='shirt'
        bad_task=base_record(pic,cid='bad_task_type',image_id='badtype',target_id='pred04',task='vqa')
        bad_task['task']=['tags']
        dupe=base_record(pic,cid='duplicated_id',image_id='dupe',target_id='pred05',task='vqa')
        conflict1=base_record(pic,cid='conflict_a',image_id='conflict',target_id='pred06',task='vqa')
        conflict2=base_record(pic,cid='conflict_b',image_id='conflict',target_id='pred06',task='tags')
        conflict2['target_bbox_xyxy']=[6,6,71,90]
        catcross1=base_record(pic,cid='crosscat_1',image_id='catcross',target_id='pred08',task='vqa')
        catcross1['target_category']='top'
        catcross2=base_record(pic,cid='crosscat_2',image_id='catcross',target_id='pred08',task='copy')
        catcross2['target_category']='outerwear'
        hash_bad=base_record(pic,cid='bad_hash',image_id='hashbad',target_id='pred07',task='vqa')
        hash_bad['image_sha256']='0'*64
        rows=[v1,t1,c1,bad_bbox,bad_enum,bad_task,dupe,dupe,conflict1,conflict2,catcross1,catcross2,hash_bad]
        (freeze/'summary.json').write_text(json.dumps({'status':'PRD31_STAGE_FREEZE_READY',
          'phase_freeze_ready':True,'candidate_id':'SYNTHETIC_TEST_ONLY'}),encoding='utf-8')
        (freeze/'freeze_record.json').write_text(json.dumps({'frozen':True,
          'candidate_id':'SYNTHETIC_TEST_ONLY'}),encoding='utf-8')
        bridge=freeze/'bridge_inputs.jsonl'
        with bridge.open('w',encoding='utf-8') as f:
            for record in rows:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            f.write('{not json}\n')
            f.write(json.dumps(dict(v1,case_id='nonfinite')) .replace('"target_bbox_xyxy": [5, 6, 70, 90]', '"target_bbox_xyxy": [5, NaN, 70, 90]')+'\n')
        tool=HERE/'prd32_resilient_runner.py'
        cmd=[sys.executable,str(tool),'--bridge',str(bridge),'--freeze-dir',str(freeze),
             '--v1-script',str(HERE/'prd32_integrated_pipeline_20261027.py')]
        report_dir=root/'preflight'
        pre=subprocess.run(cmd+['--dry-run','--preflight-output',str(report_dir)],
            capture_output=True,text=True,timeout=35)
        assert pre.returncode==0,pre.stderr+'\n'+pre.stdout
        report=json.loads((report_dir/'summary.json').read_text())
        assert report['source_requests']==15,report
        assert report['eligible_for_inference']==3,report
        assert report['blocked_input_cases']==12,report
        assert report['error_counts']['TARGET_IDENTITY_GEOMETRY_CONFLICT']==2,report
        assert report['error_counts']['DUPLICATE_CASE_ID']==2,report
        assert report['error_counts']['CROSS_REQUEST_CATEGORY_CONFLICT']==2,report
        assert report['error_counts']['INVALID_TASK_ENUM']>=1,report
        assert report['error_counts']['IMAGE_IDENTITY_HASH_MISMATCH']==1,report
        assert report['error_counts']['MALFORMED_PREDICTED_ATTR_CONTEXT']==1,report
        out=root/'simulate'
        proc=subprocess.run(cmd+['--simulate','--output',str(out)],
            capture_output=True,text=True,timeout=45)
        assert proc.returncode==0,proc.stderr+'\n'+proc.stdout
        summary=json.loads((out/'summary.json').read_text())
        services=[json.loads(t) for t in (out/'service_outputs_guarded.jsonl').read_text().splitlines()]
        preds=[json.loads(t) for t in (out/'predictions.jsonl').read_text().splitlines()]
        assert len(services)==len(preds)==15
        assert all(s['schema']=='prd32_guarded_service_contract_dev_v2' and
                   isinstance(s['issues'],list) and isinstance(s['provenance'],dict) and
                   isinstance(s['unconfirmed'],list) for s in services),services
        assert summary['simulated_backend'] is True
        assert summary['status']=='SIMULATED_FAULT_INJECTION_NOT_REAL_MODEL'
        assert summary['inference_completed']==3 and summary['inference_failed']==0
        assert sum(s['status']=='BLOCKED_INVALID_INPUT' for s in services)==12
        tag=next(s for s in services if s['case_id']=='pass_tags')
        assert tag['status']=='DEGRADED_REVIEW_REQUIRED',tag
        assert tag['payload']['pattern'] is None and tag['payload']['visible_design'] is None,tag
        assert next(s for s in services if s['case_id']=='pass_copy')['status']=='DRAFT_UNVERIFIED'
        assert all(r.get('inference_performed') is False for r in services if r['status']=='BLOCKED_INVALID_INPUT')
        assert all(r['simulated_backend'] is True for r in services)
        assert not any('ACCEPTED' in x['status'] for x in services)
        print('FULL_RUNNER_FAULT_INJECTION_PASS')
        print(json.dumps({'source_requests':report['source_requests'],
            'eligible':report['eligible_for_inference'],'blocked':report['blocked_input_cases'],
            'simulated_inference_completed':summary['inference_completed'],
            'service_status_counts':summary['output_status_counts'],
            'no_real_gpu_used':True,'holdout_used':False},ensure_ascii=False,indent=2))
        return summary

if __name__=='__main__':run_fixture()
