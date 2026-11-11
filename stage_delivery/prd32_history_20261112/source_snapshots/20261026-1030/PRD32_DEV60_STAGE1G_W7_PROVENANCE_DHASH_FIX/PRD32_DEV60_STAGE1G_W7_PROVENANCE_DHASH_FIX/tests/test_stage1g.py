import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageOps

SCRIPT = Path(__file__).resolve().parents[1]/'check_w7_provenance_corrected_dhash.py'
spec = importlib.util.spec_from_file_location('stage1g', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_good_original_algorithm_signature(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'select_week7.py'
            p.write_text("g=ImageOps.exif_transpose(image).convert('L').resize((9,8),Image.Resampling.LANCZOS)\n"+
                         "bits=(bits<<1)|int(pix[y*9+x]>pix[y*9+x+1])",encoding='utf8')
            self.assertTrue(m.verify_w7_algorithm_source(p))
    def test_bad_reversed_algorithm_signature(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'select_week7.py'
            p.write_text("g=ImageOps.exif_transpose(image).convert('L').resize((9,8),Image.Resampling.LANCZOS)\n"+
                         "bits=(bits<<1)|int(pix[y*9+x+1]>pix[y*9+x])",encoding='utf8')
            self.assertFalse(m.verify_w7_algorithm_source(p))
    def test_exif_required(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'s.py'
            p.write_text("g=image.convert('L').resize((9,8),Image.Resampling.LANCZOS)\n"+
                         "bits=(bits<<1)|int(pix[y*9+x]>pix[y*9+x+1])",encoding='utf8')
            self.assertFalse(m.verify_w7_algorithm_source(p))
    def test_dhash_white_left_black_right(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.png'
            im=Image.new('L',(90,80))
            for y in range(80):
                for x in range(90): im.putpixel((x,y),255 if x < 45 else 0)
            im.save(p)
            self.assertNotEqual(m.original_w7_dhash(p),0)
    def test_dhash_black_left_white_right(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.png'
            im=Image.new('L',(90,80))
            for y in range(80):
                for x in range(90): im.putpixel((x,y),0 if x < 45 else 255)
            im.save(p)
            self.assertNotEqual(m.original_w7_dhash(p),0xffffffffffffffff)
    def test_original_polarity_distinct_from_stage1f(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.png'
            im=Image.new('L',(90,80))
            for y in range(80):
                for x in range(90): im.putpixel((x,y),max(0,255-2*x))
            im.save(p)
            old=m.original_w7_dhash(p)
            with Image.open(p) as image:
                z=image.convert('L').resize((9,8),resample=Image.Resampling.LANCZOS)
                pix=z.load(); bits=0
                for y in range(8):
                    for x in range(8): bits=(bits<<1)|int(pix[x+1,y]>pix[x,y])
            self.assertNotEqual(old,bits)
    def test_original_dhash_range(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.png'; Image.new('RGB',(18,16),(50,50,50)).save(p)
            self.assertTrue(0 <= m.original_w7_dhash(p) < 2**64)
    def test_sha_file(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x'; p.write_bytes(b'abc')
            self.assertEqual(m.sha_file(p),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
    def test_sealed_rejects_wrong_sha_before_read(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'sealed.json';p.write_text('[]')
            with self.assertRaisesRegex(ValueError,'W7_SEALED_FILE_SHA_NOT_ORIGINAL'):m.valid_sealed_8(p)
    def test_m3_requires_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'W7_M3_ORIGINAL_EVIDENCE_FILES_MISSING'):
                m.verify_w7_original_split(Path(td),Path(td)/'sealed.json',['0'*64]*8)
    def test_dev60_count_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'manifest').write_text('{}\n')
            (root/'ref').write_text(json.dumps({'images':[]}))
            with self.assertRaisesRegex(ValueError,'DEV60_COUNT_INVALID'):
                m.verify_dev60(root/'manifest',root/'ref',root)
    def test_no_access_without_explicit_flag(self):
        import subprocess,sys
        p=subprocess.run([sys.executable,str(SCRIPT)],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('allow-sealed-metadata-read',p.stderr)

if __name__=='__main__': unittest.main()
