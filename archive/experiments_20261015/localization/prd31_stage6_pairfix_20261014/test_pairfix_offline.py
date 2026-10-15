import importlib.util
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).with_name('run_stage6.py')
spec=importlib.util.spec_from_file_location('s6fix',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for pref,mask in [('bbox_','mask_path'),('gt_bbox_','gt_mask_path')]:
  r={'source_image':'foo.jpg','garment_category':'top','garment_id':'a',mask:'mask.png'}
  for k,v in zip(('x1','y1','x2','y2'),(1,2,10,20)):r[pref+k]=str(v)
  schema=m.resolve_val_schema([r]); assert schema['bbox_keys']==[pref+k for k in ('x1','y1','x2','y2')]
  assert schema['mask_key']==mask
  assert m.read_gt_box(r,schema)==[1.0,2.0,10.0,20.0]
assert m.summary_ms([1,2,3],2)['exceed_target_count']==1
try:
  m.resolve_val_schema([{'source_image':'test.jpg'}]);raise AssertionError('Missing GT was not rejected')
except ValueError as e:assert 'Unsupported validation schema' in str(e)
try:
  r={'source_image':'x','garment_category':'top','garment_id':'x','mask_path':'y','bbox_x1':'2','bbox_y1':'2','bbox_x2':'1','bbox_y2':'3'}
  m.read_gt_box(r,m.resolve_val_schema([r])); raise AssertionError('Invalid bbox not rejected')
except ValueError:pass
print('PAIRFIX_OFFLINE_TESTS_PASS')
