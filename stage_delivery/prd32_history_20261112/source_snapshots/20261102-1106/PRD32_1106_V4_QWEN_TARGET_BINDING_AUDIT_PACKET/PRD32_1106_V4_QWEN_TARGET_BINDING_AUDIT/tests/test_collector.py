from pathlib import Path
from tempfile import TemporaryDirectory
import unittest,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collect_marked12 import sha256_file,strict_source,try_copy_verified,FORBIDDEN
import hashlib

class TestEvidenceCollector(unittest.TestCase):
    def test_expected_bytes_verified(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)/'marked_inputs';p.mkdir()
            src=p/'abc_marked_full_image.jpg';src.write_bytes(b'test data')
            digest=hashlib.sha256(b'test data').hexdigest()
            self.assertEqual(sha256_file(src),digest)
            dst=Path(temp)/'out'/'x.jpg'
            self.assertEqual(try_copy_verified(src,dst,digest),digest)
            self.assertEqual(dst.read_bytes(),src.read_bytes())
    def test_digest_mismatch_blocked(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)/'a.jpg';p.write_bytes(b'test data')
            with self.assertRaises(ValueError): try_copy_verified(p,Path(temp)/'b.jpg','0'*64)
    def test_outside_approved_directory_blocked(self):
        with TemporaryDirectory() as temp:
            d=Path(temp)/'marked_inputs';d.mkdir()
            src=Path(temp)/'bad.jpg';src.write_bytes(b'x')
            with self.assertRaises(ValueError):strict_source(src,d)
    def test_allow_valid_inside_directory(self):
        with TemporaryDirectory() as temp:
            d=Path(temp)/'marked_inputs';d.mkdir()
            src=d/'valid.jpg';src.write_bytes(b'abc')
            self.assertEqual(strict_source(str(src),d),src.resolve())
    def test_block_symlink_escape(self):
        with TemporaryDirectory() as temp:
            d=Path(temp)/'marked_inputs';d.mkdir()
            outside=Path(temp)/'b.jpg';outside.write_bytes(b'abc')
            link=d/'escape.jpg';link.symlink_to(outside)
            with self.assertRaises(ValueError):strict_source(str(link),d)
    def test_policy_forbidden(self):
        self.assertIn('holdout',FORBIDDEN)
        self.assertIn('sealed',FORBIDDEN)
        self.assertIn('blind',FORBIDDEN)

if __name__=='__main__':unittest.main()
