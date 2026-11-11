import csv
import json
from pathlib import Path
import tempfile
from audit_provenance_v3 import audit, norm_path, bbox_check

def write(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

base_cols=['source_dataset','source_image','garment_id','garment_category','bbox_x1','bbox_y1','bbox_x2','bbox_y2']
with tempfile.TemporaryDirectory() as t:
    root=Path(t)/'project'; (root/'configs').mkdir(parents=True)
    inv=base_cols+['record_id','source_manifest']
    write(root/'configs/prd_8class_unified_manifest_v2.csv',inv,[
        dict(zip(base_cols,['A','dir/one.jpg','g1','top','1','2','40','50']),record_id='a',source_manifest='no_split'),
        dict(zip(base_cols,['A','train/two.jpg','g2','pants','1','1','10','10']),record_id='b',source_manifest='train/source.csv'),
        dict(zip(base_cols,['A','new/three.jpg','g3','dress','5','1','5','10']),record_id='c',source_manifest='no_split'),
    ])
    write(root/'configs/prd_3_1_predicted_roi_fullval_v1.csv',base_cols,[
        dict(zip(base_cols,['A','dir\\one.jpg','old1','top','1','2','40','50']))
    ])
    write(root/'configs/prd_3_1_predicted_roi_matched_v2.csv',base_cols,[
        dict(zip(base_cols,['A','other/two.jpg','old2','pants','1','1','10','10']))
    ])
    s=audit(root,Path(t)/'out')
    assert s['sources']['inventory']['row_count']==3
    assert s['sources']['inventory']['unique_image_keys']==3
    assert s['inventory_rows_approved_for_accuracy']==0
    assert s['inventory_rows_quarantined']==3
    assert s['pairwise_overlap'][0]['exact_dataset_and_normalized_path_overlap_image_keys']==1
    assert s['pairwise_overlap'][1]['weak_dataset_and_basename_overlap_image_keys']==1
    assert s['sources']['inventory']['bbox_geometry_metadata_only']['ZERO_OR_NEGATIVE_BBOX_AREA']==1
    assert s['sources']['fullval']['eligible_for_3_2_accuracy']==False
    assert len(list(csv.DictReader(open(Path(t)/'out/inventory_quarantined_review.csv',encoding='utf-8-sig'))))==3
    assert norm_path('dir\\one.jpg')=='dir/one.jpg'
    assert bbox_check(dict(zip(base_cols,['A','x','g','top','NaN','0','1','2'])))=='MISSING_OR_NONFINITE_BBOX'
    try:
        audit(root,Path(t)/'out')
        raise AssertionError('should refuse overwrite')
    except FileExistsError:
        pass
print('V3_PROVENANCE_SYNTHETIC_TEST_PASS')
