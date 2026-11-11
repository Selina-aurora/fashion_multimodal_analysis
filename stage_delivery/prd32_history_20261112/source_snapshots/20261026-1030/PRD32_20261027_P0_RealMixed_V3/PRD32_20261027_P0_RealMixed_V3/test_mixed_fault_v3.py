#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic local self-test. Does NOT invoke Qwen or real PRD frozen data."""
from __future__ import annotations
import hashlib, json, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image

HERE=Path(__file__).resolve().parent

def invoke(args):
    p=subprocess.run([sys.executable,*map(str,args)],capture_output=True,text=True,timeout=90)
    if p.returncode!=0:
        raise AssertionError(f'Command failed {p.returncode}\nSTDOUT {p.stdout}\nSTDERR {p.stderr}')
    return p

def test():
    with tempfile.TemporaryDirectory(prefix='prd32_fault_v3_') as tmp:
        root=Path(tmp);fr=root/'real_freeze_mock';fr.mkdir()
        photo=root/'image.png'
        Image.new('RGB',(120,140),(60,85,105)).save(photo)
        h=hashlib.sha256(photo.read_bytes()).hexdigest()
        (fr/'summary.json').write_text(json.dumps({'status':'PRD31_STAGE_FREEZE_READY','phase_freeze_ready':True,'candidate_id':'MOCK_ONLY_FOR_UNIT_TEST'}),encoding='utf-8')
        (fr/'freeze_record.json').write_text(json.dumps({'frozen':True,'candidate_id':'MOCK_ONLY_FOR_UNIT_TEST'}),encoding='utf-8')
        allrec=[]
        for task,cid in [('vqa','vqa_color_01'),('tags','tags_01'),('copy','copy_01')]:
            allrec.append(dict(schema='prd32_engine_neutral_input_v1',case_id=cid,task=task,split='development',
                image_id='fixture_photo',target_instance_id='pred_0001',image_path=str(photo),image_sha256=h,
                target_bbox_origin='prd31_frozen_prediction',target_bbox_xyxy=[10,10,100,120],
                query='请描述红框所指目标服饰的颜色',predicted_attribute_context=[]))
        bridge=fr/'bridge_inputs.jsonl'
        bridge.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in allrec),encoding='utf-8')
        fixture=root/'mixed'
        invoke([HERE/'create_mixed_fault_fixture.py','--freeze-dir',fr,'--output',fixture])
        manifest=json.loads((fixture/'manifest.json').read_text())
        assert manifest['expected_valid']==3 and manifest['expected_blocked']==12
        runner=HERE/'prd32_resilient_runner_v3.py'
        preflight=root/'preflight'
        base=[runner,'--freeze-dir',fr,'--v1-script',HERE/'prd32_integrated_pipeline_20261027.py',
              '--fault-bridge',fixture/'fault_injection_only.jsonl','--fault-manifest',fixture/'manifest.json']
        invoke(base+['--dry-run','--preflight-output',preflight])
        audit=json.loads((preflight/'summary.json').read_text())
        assert audit['eligible_for_inference']==3,audit
        assert audit['blocked_input_cases']==12,audit
        decisions=[json.loads(x) for x in (preflight/'input_decisions.jsonl').read_text().splitlines()]
        assert len(decisions)==15,decisions
        assert sum(x['decision']=='BLOCKED' for x in decisions)==12,decisions
        sim=root/'simulation'
        invoke(base+['--simulate','--output',sim])
        report=json.loads((sim/'summary.json').read_text())
        service=[json.loads(x) for x in (sim/'service_outputs_guarded.jsonl').read_text().splitlines()]
        assert report['inference_completed']==3,report
        assert report['inference_failed']==0,report
        assert report['test_mode']=='isolated_fault_injection_only'
        assert report['simulated_backend'] is True
        assert len(service)==15
        assert sum(x['status']=='BLOCKED_INVALID_INPUT' for x in service)==12
        assert all(x['payload'] is None and x['inference_performed'] is False for x in service if x['status']=='BLOCKED_INVALID_INPUT')
        assert sum(x['inference_performed'] is True for x in service)==3
        assert all(x['simulated_backend'] is True for x in service)
        assert all(x['unverified'] is True for x in service)
        assert manifest['injected_bridge_sha256']==report['injected_fixture_sha256']
        # Tampering with synthetic JSONL must fail closed, NOT run any inference.
        (fixture/'fault_injection_only.jsonl').write_text((fixture/'fault_injection_only.jsonl').read_text()+'{}\n')
        tampered=subprocess.run([sys.executable,*map(str,base+['--dry-run'])],capture_output=True,text=True,timeout=45)
        assert tampered.returncode!=0 and 'edited after creation' in tampered.stderr, tampered.stderr
        print('MIXED_FAULT_V3_LOCAL_TEST_PASS')
        print(json.dumps({'source_requests':15,'eligible':3,'blocked':12,'fake_inference_completed':3,
                          'real_gpu_used':False,'original_bridge_changed':False,
                          'tampering_blocked':True},ensure_ascii=False,indent=2))

if __name__=='__main__':test()
