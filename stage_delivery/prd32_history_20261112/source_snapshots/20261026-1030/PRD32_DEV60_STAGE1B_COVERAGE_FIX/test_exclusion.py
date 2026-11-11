import unittest,json
from check_holdout_digest_registry import compare_cohort
class DigestRegistryTests(unittest.TestCase):
    def test_safe_registry_no_overlap(self):
        lines=[json.dumps({'case_id':f'f_{i}','image_sha256':f'{i:064x}'}) for i in range(60)]
        self.assertTrue(compare_cohort(lines,[f'{i:064x}' for i in range(100,108)])['no_exact_sha_overlap'])
    def test_overlap_blocks(self):
        lines=[json.dumps({'case_id':f'f_{i}','image_sha256':f'{i:064x}'}) for i in range(60)]
        a=compare_cohort(lines,[f'{i:064x}' for i in range(5,13)])
        self.assertFalse(a['no_exact_sha_overlap'])
        self.assertEqual(a['overlap_images'],8)
    def test_too_few_hashes_blocks(self):
        lines=[json.dumps({'case_id':f'f_{i}','image_sha256':f'{i:064x}'}) for i in range(60)]
        with self.assertRaises(ValueError):compare_cohort(lines,[f'{i:064x}' for i in range(100,102)])
if __name__=='__main__':unittest.main()
