from pathlib import Path
import tempfile
from PIL import Image
from source_gap_audit import sha
from intake_fresh8 import scan

def main():
 with tempfile.TemporaryDirectory() as tmp:
  d=Path(tmp)/'PRD32_NEW_DEV_IMAGES';p=d/'skirt';p.mkdir(parents=True)
  im=p/'new_skirt.jpg';Image.new('RGB',(160,160),'white').save(im)
  digest=sha(im)
  rs=scan(d,set(),prior_hashes={digest})
  assert len(rs)==1 and 'SHA_MATCH_PRIOR52_DO_NOT_USE' in rs[0]['flag_codes'],rs
  print('PRIOR52_CONTENT_HASH_GATE_PASS (renamed duplicate blocked)')
if __name__=='__main__':main()
