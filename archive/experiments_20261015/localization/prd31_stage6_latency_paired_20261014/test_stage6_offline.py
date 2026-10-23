import importlib.util
from pathlib import Path
p=Path(__file__).with_name('run_stage6.py')
spec=importlib.util.spec_from_file_location('stage6',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.CLASSES['accessory']==8
assert m.allowed_regions('top')==m.ALL_REGIONS
assert m.allowed_regions('shoe')==['pattern','decoration']
assert m.allowed_regions('pants')==['hem','pocket','waist','pattern','decoration']
assert m.summary_ms([])['n']==0
assert m.summary_ms([5,7,9],7)['exceed_target_count']==1
assert m.summary_ms([5,7,9],7)['median_ms']==7
assert len(m.EXPECTED_SHA)==3
for path in ['selected_images.json','run_config.json','priority_review_queue.csv']:
    assert (p.parent/path).is_file(),path
print('OFFLINE_SMOKE_TESTS_PASSED: 9')
