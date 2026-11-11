#!/usr/bin/env python3
"""Dry-run fault injection tests; zero GPU, zero access to sealed inputs."""
import contextlib
import hashlib
import io
import json
import tempfile
from pathlib import Path
from PIL import Image
from prd32_defensive_gate import guard_input_record, sanitize_service_output, synthetic_tests
from prd32_resilient_runner import load_v1, select_with_guard


def run():
    with contextlib.redirect_stdout(io.StringIO()): synthetic_tests()
    with tempfile.TemporaryDirectory(prefix='prd32_runner_fi_') as d:
        root=Path(d);freeze=root/'freeze';freeze.mkdir()
        im=root/'img.png';Image.new('RGB',(70,60),'white').save(im)
        h=hashlib.sha256(im.read_bytes()).hexdigest()
        good={'schema':'prd32_engine_neutral_input_v1','case_id':'valid','image_id':'img01',
              'target_instance_id':'pred_01','task':'tags','split':'development',
              'target_bbox_origin':'prd31_frozen_prediction','target_bbox_xyxy':[5,5,50,50],
              'image_sha256':h,'image_path':str(im),'query':'颜色是什么？',
              'predicted_attribute_context':[]}
        bbox=dict(good,case_id='bad_box',target_bbox_xyxy=[0,0,71,50])
        cat=dict(good,case_id='bad_class',target_category='shirt')
        records=[good,bbox,cat]
        # Critical: real bridge omits source_size; image dimensions are read from image bytes.
        status=[g['decision'] for g in select_with_guard(records)]
        assert status==['PASS','BLOCKED','BLOCKED'],status
        # Distinct case with a conflicting class hint must be blocked.
        conflict=dict(good,case_id='conflict',target_category='top',predicted_category='outerwear')
        assert guard_input_record(conflict)['decision']=='BLOCKED'
        svc={'case_id':'valid','task':'tags','canonical_attributes_v2':{'category':'shirt',
             'primary_color':'白色','pattern':'无','visible_design':'无设计'}}
        resp=sanitize_service_output(svc)
        assert resp['payload']['category'] is None and resp['payload']['pattern'] is None
        assert resp['payload']['visible_design'] is None
        # A bad service string must not masquerade as a valid released product copy.
        copy=sanitize_service_output({'case_id':'good','task':'copy','payload':'红框中是纯棉蕾丝外套'})
        assert copy['payload'] is None and copy['status']=='CONTENT_HOLD_FOR_REVIEW'
        # Full runner validates freeze identity and performs isolated per-case preflight.
        (freeze/'summary.json').write_text(json.dumps({'status':'PRD31_STAGE_FREEZE_READY',
            'phase_freeze_ready':True,'candidate_id':'fixture-only'}),encoding='utf-8')
        (freeze/'freeze_record.json').write_text(json.dumps({'frozen':True,
            'candidate_id':'fixture-only'}),encoding='utf-8')
        (freeze/'bridge_inputs.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records),encoding='utf-8')
        mod=load_v1(Path(__file__).parent.parent/'PRD32_20261027_pipeline'/'prd32_integrated_pipeline_20261027.py') if (Path(__file__).parent.parent/'PRD32_20261027_pipeline'/'prd32_integrated_pipeline_20261027.py').exists() else None
        if mod:
            freeze_info=mod.validate_freeze(freeze,freeze/'bridge_inputs.jsonl')
            assert freeze_info['candidate_id']=='fixture-only'
        print('FAULT_INJECTION_PASS: 1 valid continues; out-of-bounds, invalid enum and category conflicts blocked; text/pattern fallbacks verified')

if __name__=='__main__':run()
