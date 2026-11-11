from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json,hashlib,importlib.util
spec=importlib.util.spec_from_file_location('gate',Path(__file__).with_name('v5_live_preflight.py'))
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
def writej(p,obj):p.write_text(json.dumps(obj),encoding='utf8')
def writel(p,rows):p.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
with TemporaryDirectory() as d:
    base=Path(d);project=base/'project';c=project/'outputs/recovery_20261008/v5_balanced150/checkpoint_last.pth'
    c.parent.mkdir(parents=True);c.write_bytes(b'fake checkpoint test only');h=gate.sha(c)
    froze=base/'prd31f';froze.mkdir()
    writej(froze/'summary.json',{'status':'PRD31_STAGE_FREEZE_READY','phase_freeze_ready':True,'candidate_id':'testv5',
          'checkpoint_check':{'passed':True,'path':str(c),'expected_sha256':h,'actual_sha256':h}})
    writej(froze/'freeze_record.json',{'frozen':True,'candidate_id':'testv5','checkpoint_sha256':h})
    data=base/'data';(data/'train').mkdir(parents=True);(data/'test').mkdir(parents=True)
    restricted=data/'test'/'restricted.jpg';restricted.write_bytes(b'never-read-this')
    writel(froze/'bridge_inputs.jsonl',[{'case_id':'oldtest','task':'tags','image_path':str(restricted)}])
    fresh=base/'new';fresh.mkdir();rows=[]
    for i in range(12):
        img=data/'train'/f'{i}.jpg';img.write_bytes(f'approved {i}'.encode())
        rows.append({'case_id':f'fresh_dev_{i+1:03d}','image_path':str(img),'image_sha256':gate.sha(img),
                     'bbox_origin':'historical_inventory_gt_pilot_NOT_V5_prediction','bbox':[0,0,1,1],
                     'category':'SHOULD_NOT_COPY'})
    writel(fresh/'frozen_inference_plan.jsonl',rows)
    writel(fresh/'frozen_human_reference.jsonl',[{'case_id':r['case_id'],'category':'SHOULD_NOT_COPY'} for r in rows])
    writej(fresh/'freeze.json',{'status':'FROZEN_DEV_NOT_INDEPENDENT','total':12,
                 'plan_sha256':gate.sha(fresh/'frozen_inference_plan.jsonl'),
                 'human_reference_sha256':gate.sha(fresh/'frozen_human_reference.jsonl')})
    args=SimpleNamespace(project=project,freeze_dir=froze,newdev_frozen=fresh,data_root=data,
          output=base/'out',skip_checkpoint_hash=False)
    gate.run(args)
    out=gate.jlread(args.output/'v5_image_only_inputs.jsonl')
    assert len(out)==12
    assert all('bbox' not in v and 'category' not in v for v in out)
    s=gate.jread(args.output/'v5_live_preflight_summary.json')
    assert s['ready_for_live_v5_invocation'] and s['image_reads_from_restricted_splits']==0
    assert s['bridge_source_warnings'][0]['path_policy']=='RESTRICTED_SPLIT'
    print('SYNTHETIC_FULL_GATE_PASSED')
