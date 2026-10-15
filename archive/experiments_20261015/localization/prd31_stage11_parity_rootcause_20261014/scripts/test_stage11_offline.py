import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_stage11 import compare_det

def t():
    a=[{'box':[0.,0.,10.,10.],'score':.9,'label':'collar'}]
    assert compare_det(a,a)['candidate_parity']
    b=[{'box':[1,0,10,10],'score':.9,'label':'collar'}]
    assert not compare_det(a,b)['candidate_parity']
    assert not compare_det(a,[])['candidate_parity']
    assert compare_det([],[])['candidate_parity']
    c=[{'box':[0,0,10,10],'score':.903,'label':'collar'}]
    assert compare_det(a,c)['candidate_parity']
    c[0]['label']='hem'
    assert not compare_det(a,c)['candidate_parity']
if __name__=='__main__':t();print('OFFLINE_TESTS_PASSED')
