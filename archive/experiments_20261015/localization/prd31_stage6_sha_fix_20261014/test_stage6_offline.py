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
assert m.EXPECTED_SHA[m.VAL]=='39141f545ea5512d16ed074788b6f98474b33baff111d3b663ec0ffcc4c40b93'
assert m.EXPECTED_SHA[m.TRAIN]=='e8c277055efe4d042b4353bf2d4553bddddc7706eead9398f07f5efcd90bb38f'
assert m.EXPECTED_SHA[m.CORE]=='c83204aca87c9c4dee304b774fd459bf5a18a2d1d65f0a3a9a79b92721224b89'
for path in ['selected_images.json','run_config.json','priority_review_queue.csv']:
    assert (p.parent/path).is_file(),path
print('OFFLINE_SMOKE_TESTS_PASSED: 12')
