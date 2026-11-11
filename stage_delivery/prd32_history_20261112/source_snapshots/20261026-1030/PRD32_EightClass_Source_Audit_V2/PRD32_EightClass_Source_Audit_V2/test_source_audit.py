from pathlib import Path
from tempfile import TemporaryDirectory
import csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_prd31_sources import inspect_source,main

def write(path,headers,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(rows)

def test():
 with TemporaryDirectory() as t:
  r=Path(t)/'repo';(r/'configs').mkdir(parents=True)
  m=r/'configs/prd_8class_unified_manifest_v2.csv'
  write(m,['image_id','predicted_category','source_split','target_bbox_xyxy','target_instance_id','image_path'],[
    {'image_id':'a1','predicted_category':'top','source_split':'dev_new','target_bbox_xyxy':'[0,0,20,20]','target_instance_id':'p1','image_path':'/tmp/a.jpg'},
    {'image_id':'a2','predicted_category':'shoe','source_split':'train','target_bbox_xyxy':'[1,2,30,30]','target_instance_id':'p2','image_path':'/tmp/b.jpg'},
    {'image_id':'a1','predicted_category':'top','source_split':'dev_new','target_bbox_xyxy':'[0,0,20,20]','target_instance_id':'p3','image_path':'/tmp/a.jpg'},
  ])
  s1,k1,kind=inspect_source(m)
  assert s1['record_count']==3 and s1['identity_duplicate_rows']==1
  assert s1['category_counts_by_column']['predicted_category']['top']==2
  assert s1['has_train_split_rows'] and s1['eligibility'].startswith('NOT_APPROVED')
  assert kind=='image_id' and len(k1)==2
  full=r/'configs/prd_3_1_predicted_roi_fullval_v1.csv'
  write(full,['image_id','category'],[{'image_id':'b','category':'dress'}])
  s2,_,_=inspect_source(full)
  assert s2['source_role'].startswith('REFERENCE_ONLY')
  matched=r/'configs/prd_3_1_predicted_roi_matched_v2.csv'
  write(matched,['image_id','bbox'],[{'image_id':'a1','bbox':'[0,0,20,30]'}])
  assert main(['--project-root',str(r),'--output-dir',str(Path(t)/'result')])==0
  assert main(['--project-root',str(r),'--output-dir',str(Path(t)/'result')])==0
  out=json.loads((Path(t)/'result'/'source_schema_audit.json').read_text())
  assert len(out['sources'])==3 and out['images_opened']==0
  assert any(v['exact_overlap_distinct_keys']==1 for v in out['identity_overlap'] if v['comparable'])
  assert (Path(t)/'result_run02'/'source_schema_audit.json').exists()
  restricted=r/'sealed_holdout.csv';restricted.write_text('a,b\n',encoding='utf-8')
  info,_,_=inspect_source(restricted)
  assert info['read_mode']=='REFUSED_RESTRICTED_PATH'
 print('AUDIT_TEST_PASS')
if __name__=='__main__':test()
