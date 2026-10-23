"""CPU-only behavior checks, no GPU models or original project required."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from run_stage10 import parity_row


def test_status_pair():
    a={'status':'missed','bbox':None,'mask':None}
    b={'status':'missed','bbox':None,'mask':None}
    assert parity_row('c','hem',a,b)['parity_pass']
    b['status']='empty_mask'
    assert not parity_row('c','hem',a,b)['parity_pass']


def test_masks_and_scores():
    m=np.zeros((4,6),np.uint8);m[1:3,2:5]=1
    a={'status':'detected','mask':m,'bbox':[2,1,5,3],
       'proposal_bbox':[1,0,5,4],'score':.55,'sam_score':.98,'model_phrase':'hem'}
    b=dict(a,mask=m.copy(),score=.5501,sam_score=.98001)
    r=parity_row('c','hem',a,b)
    assert r['parity_pass'] and r['mask_pixel_equal'] and r['mask_iou']==1.0
    b['mask']=np.zeros_like(m)
    assert not parity_row('c','hem',a,b)['parity_pass']
    b['mask']=m.copy();b['model_phrase']='pocket'
    assert not parity_row('c','hem',a,b)['parity_pass']


if __name__=='__main__':
    test_status_pair();test_masks_and_scores();print('OFFLINE_TESTS_PASSED: 2')
