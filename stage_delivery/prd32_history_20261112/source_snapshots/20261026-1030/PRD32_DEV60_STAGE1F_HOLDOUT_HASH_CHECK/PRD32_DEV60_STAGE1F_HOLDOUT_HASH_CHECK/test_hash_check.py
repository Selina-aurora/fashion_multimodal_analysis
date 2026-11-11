import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import check_holdout_isolation as m

class SafeDigestTests(unittest.TestCase):
    def test_accept_hex16(self):
        self.assertEqual(m.parse_dhash('000000000000000f'), 15)
    def test_accept_0x(self):
        self.assertEqual(m.parse_dhash('0xff'), 255)
    def test_reject_invalid_hash(self):
        with self.assertRaises(ValueError): m.parse_dhash('xyz')
    def test_reject_boolean_as_dhash(self):
        with self.assertRaises(ValueError): m.parse_dhash(True)
    def test_allow_int(self):
        self.assertEqual(m.parse_dhash(5), 5)
    def test_reject_overflow(self):
        with self.assertRaises(ValueError): m.parse_dhash(1 << 64)
    def test_sha_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_bytes(b'ab')
            self.assertEqual(m.sha_file(p), hashlib.sha256(b'ab').hexdigest())
    def test_sealed_eight_sha(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.json'
            p.write_text(json.dumps([{'source_sha256': format(i, '064x'), 'dhash64':format(i,'016x')} for i in range(1,9)]))
            sha, dh = m.parse_sealed_metadata(p)
            self.assertEqual(len(sha),8)
            self.assertEqual(len(dh),8)
    def test_sealed_duplicate_sha_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.json'
            p.write_text(json.dumps([{'source_sha256':'1'*64,'dhash64':'2'*16} for _ in range(8)]))
            with self.assertRaises(ValueError): m.parse_sealed_metadata(p)
    def test_sealed_wrong_size(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.json';p.write_text('[]')
            with self.assertRaises(ValueError): m.parse_sealed_metadata(p)
    def test_sealed_wrong_digest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.json'
            p.write_text(json.dumps([{'source_sha256':'INVALID','dhash64':'2'*16} for _ in range(8)]))
            with self.assertRaises(ValueError): m.parse_sealed_metadata(p)
    def test_no_sealed_metadata_in_output(self):
        # Local report must contain only aggregates, not sealed identifiers/paths/sha/dHash.
        with patch.object(m,'read_cohort',return_value=([{'image_sha256':'f'*64}]*60, [Path('/tmp/fake')]*60)), patch.object(m,'parse_sealed_metadata',return_value=({'a'*64},[7]*8)):
            r=m.audit(Path('none'),Path('none'),Path('/private/sealed.json'),Path('/tmp'),skip_dhash=True)
            serialized=json.dumps(r)
            self.assertNotIn('a'*64,serialized)
            self.assertNotIn('/private/sealed.json',serialized)
            self.assertFalse(r['official_freeze_approved'])

if __name__=='__main__': unittest.main()
