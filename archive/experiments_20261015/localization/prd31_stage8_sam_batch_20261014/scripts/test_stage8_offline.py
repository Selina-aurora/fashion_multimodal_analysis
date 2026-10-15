from pathlib import Path
import ast
import sys
base=Path(__file__).parent
for p in [base/'run_stage8.py',base/'stage6_helpers.py']:
    ast.parse(p.read_text(encoding='utf-8'))
sys.path.insert(0,str(base))
from run_stage8 import safe_mask_iou
import numpy as np
assert safe_mask_iou(np.zeros((3,3)),np.zeros((3,3)))==1.0
assert safe_mask_iou(np.ones((3,3)),np.zeros((3,3)))==0.0
assert safe_mask_iou(np.array([[1,0]]),np.array([[1,1]]))==0.5
assert safe_mask_iou(np.ones((2,3)),np.ones((3,2))) is None
print('OFFLINE_TESTS_PASS')
