#!/usr/bin/env python3
"""No-GPU regression tests for tag degradation/provenance in V3."""
from __future__ import annotations
import hashlib, tempfile
from pathlib import Path
from PIL import Image
import prd32_resilient_runner_v3 as runner

def main():
    with tempfile.TemporaryDirectory(prefix='prd32_tags_v3_') as tmp:
        img=Path(tmp)/'white.png';Image.new('RGB',(100,120),'white').save(img)
        sha=hashlib.sha256(img.read_bytes()).hexdigest()
        rec={'schema':'prd32_engine_neutral_input_v1','case_id':'lace_case','task':'tags',
             'split':'development','image_id':'fixture','target_instance_id':'p001',
             'image_path':str(img),'image_sha256':sha,'target_bbox_origin':'prd31_frozen_prediction',
             'target_bbox_xyxy':[3,5,90,99],'query':'属性', 'predicted_attribute_context':[]}
        v1=runner.load_v1(Path(__file__).parent/'prd32_integrated_pipeline_20261027.py')
        v1.run_model_case=lambda *args, **kwargs:('''```json
{"category":"dress","primary_color":"白色","pattern":"蕾丝","visible_design":"交叉肩带"}
```''',{'preprocess_ms':0.0,'generation_ms':0.0,'end_to_end_ms':0.0,'generated_tokens':0})
        marked_dir=Path(tmp)/'marked';marked_dir.mkdir()
        pred,s=runner.infer_case(v1,None,None,rec,marked_dir,'marked_full_image',0.05)
        assert s['payload']['pattern'] is None,s
        assert s['status']=='DEGRADED_REVIEW_REQUIRED',s
        assert any(x['code'].startswith('UPSTREAM_TAG_PARSE_PATTERN_SEMANTICS') for x in s['issues']),s
        from prd32_defensive_gate import guard_input_record
        g=guard_input_record(rec)
        standardized=runner.standardize_service(s,g,inference_performed=True,simulated_backend=True)
        assert 'pattern' in standardized['unconfirmed'],standardized
        assert standardized['provenance']['case_id']==rec['case_id']
        print('TAG_ABSTENTION_AND_UNCONFIRMED_PROPAGATION_PASS')
if __name__=='__main__':main()
