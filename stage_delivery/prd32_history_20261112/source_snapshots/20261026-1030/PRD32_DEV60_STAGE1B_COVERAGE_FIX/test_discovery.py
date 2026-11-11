import unittest
from pathlib import Path
from inspect_training_manifests import safe_train_image_path, safe_metadata_path, parse_classes
class SafeDiscoveryTests(unittest.TestCase):
    def test_holdout_rejected(self):
        self.assertFalse(safe_metadata_path(Path('/workspace/holdout_SEALED_DO_NOT_OPEN.jsonl')))
    def test_sealed_nested_rejected(self):
        self.assertFalse(safe_metadata_path(Path('/workspace/fashion_data/sealed/train/meta.csv')))
    def test_val_train_image_rejected(self):
        self.assertFalse(safe_train_image_path(Path('/workspace/fashion_data/train/val/a.jpg')))
    def test_train_image_allowed(self):
        self.assertTrue(safe_train_image_path(Path('/workspace/fashion_data/raw/train/train/image/a.jpg')))
    def test_no_trivial_class_alias(self):
        self.assertEqual(parse_classes('coat'),[])
    def test_multi_class(self):
        self.assertEqual(parse_classes('top;bag'),['top','bag'])
    def test_testset_rejected(self):
        self.assertFalse(safe_train_image_path(Path('/workspace/testset/train/x.png')))
if __name__=='__main__':unittest.main()
