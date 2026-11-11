#!/usr/bin/env python3
"""Read-only local regression tests. No GPU/model access and no development images."""
from pathlib import Path
import tempfile
import json
import copy
from bridge_engineering_gate import run, loadjsonl, adapt, valid_envelope, color_disagreement, filehash

ROOT=Path(__file__).resolve().parent
SF=ROOT/'evidence/ServiceOutputs_12_20261027.jsonl'
EF=ROOT/'evidence/ReadOnlyEnvelopes_12_20261027.jsonl'

def test():
    assert SF.is_file() and EF.is_file()
    srv=loadjsonl(SF);env=loadjsonl(EF)
    assert len(srv)==len(env)==12
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)/'audit'
        s=run(SF,EF,out)
        assert s['total']==12 and s['structural_errors']==0
        assert s['tasks']=={'vqa':6,'tags':3,'copy':3}
        assert s['vqa_abstentions']==3
        assert s['copies_blocked_by_debug_or_structure']==2
        assert s['tag_pattern_ambiguous_abstentions']==1
        records={r['case_id']:r for r in loadjsonl(out/'dev_contract_v2.jsonl')}
        assert all(r['ready_for_automatic_publishing'] is False for r in records.values())
        assert records['tags_01']['review_candidate']['pattern'] is None
        assert records['tags_01']['original_model_or_adapter_payload']['pattern']=='无'
        assert records['copy_02']['review_candidate'] is None
        assert records['copy_03']['review_candidate'] is None
        assert records['copy_01']['review_candidate'] is not None
        assert records['vqa_neck_02']['disposition']=='ABSTAINED_UNVERIFIED'
        assert color_disagreement('红色框内展示了一件浅灰色蕾丝吊带连衣裙','白色')
        assert not color_disagreement('服饰为白色，带有黑色波点图案和纽扣设计','白色')
        try:run(SF,EF,out);raise AssertionError('Output overwrite allowed')
        except ValueError as e:assert 'overwrite' in str(e)
        # Poison identity, status, and publisher flags -- all must fail provenance validation.
        e0=copy.deepcopy(env[0]);s0=srv[0]
        assert valid_envelope(s0,e0)==[]
        e0['ready_for_automatic_publishing']=True
        assert 'unsafe_preexisting_publication_or_verified_claim' in valid_envelope(s0,e0)
        e0=copy.deepcopy(env[0]);e0['image_id']='swapped_image'
        # For changed ID alone, existing old envelope is insufficient evidence to prevent forged provenance;
        # a signed bridge would be required. Test that the program keeps provenance unverified, never publishes.
        safe=adapt(s0,e0,{})
        assert safe['ready_for_automatic_publishing'] is False
        assert safe['human_verified'] is False
        broken=copy.deepcopy(s0);broken['payload']='伪造数据'
        assert 'source_payload_disagreement' in valid_envelope(broken,env[0])
        # Duplicate service IDs fail before output directory is created.
        duplicate=Path(td)/'duplicate.jsonl'
        duplicate.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in srv+[srv[0]])+'\n',encoding='utf-8')
        try:run(duplicate,EF,Path(td)/'dupout');raise AssertionError('Duplicate ID accepted')
        except ValueError as e:assert 'duplicate' in str(e)
        assert not (Path(td)/'dupout').exists()
        # Stronger optional join: bridge metadata must match image and predicted instance IDs.
        br=Path(td)/'bridge_inputs.jsonl'
        bridge_rows=[dict(schema='prd32_engine_neutral_input_v1',case_id=e['case_id'],task=e['task'],
                          split='development',image_id=e['image_id'],target_instance_id=e['target_instance_id'],
                          target_bbox_origin='prd31_frozen_prediction',target_bbox_xyxy=[1,1,30,40],
                          image_path='/workspace/fashion_data/raw/train/demo.jpg') for e in env]
        br.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in bridge_rows)+'\n',encoding='utf-8')
        bridged=run(SF,EF,Path(td)/'bridge_joined',bridge=br,expected_bridge_hash=filehash(br))
        assert bridged['frozen_bridge_metadata_check']['checked'] is True
        bridge_rows[0]['target_instance_id']='a_forged_instance'
        br.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in bridge_rows)+'\n',encoding='utf-8')
        try:run(SF,EF,Path(td)/'invalid_bridge',bridge=br);raise AssertionError('Forged bridge identity accepted')
        except ValueError as e:assert 'metadata mismatch' in str(e)
        assert not (Path(td)/'invalid_bridge').exists()
    print('ENGINEERING_GATE_SYNTHETIC_AND_HISTORICAL_E2E_PASS')

if __name__=='__main__':test()
