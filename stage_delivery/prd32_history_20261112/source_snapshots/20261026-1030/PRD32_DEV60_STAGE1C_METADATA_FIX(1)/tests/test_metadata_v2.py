import importlib.util
import tempfile
import unittest
from pathlib import Path

p=Path(__file__).resolve().parents[1]/'discover_train_metadata_v2.py'
spec=importlib.util.spec_from_file_location('metadata_v2',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_source_image_is_known(self):self.assertIn('source_image',m.PATH_KEYS)
    def test_exact_category(self):self.assertEqual(m.category_tokens('skirt'),['skirt'])
    def test_do_not_map_shorts_to_pants(self):self.assertEqual(m.category_tokens('shorts'),[])
    def test_multiclass(self):self.assertEqual(m.category_tokens('bag;shoe'),['shoe','bag'])
    def test_holdout_restricted(self):self.assertFalse(m.safe_parts(Path('/workspace/holdout/img.jpg')))
    def test_val_restricted(self):self.assertFalse(m.safe_parts(Path('/workspace/fashion_data/val/img.jpg')))
    def test_train_allowed(self):self.assertTrue(m.safe_parts(Path('/workspace/fashion_data/raw/train/img.jpg')))
    def test_safe_real_source(self):
        with tempfile.TemporaryDirectory() as root:
            rt=Path(root); data=rt/'fashion_data';project=rt/'repo';
            pic=data/'raw'/'train'/'train'/'image'/'01.jpg';pic.parent.mkdir(parents=True);pic.write_bytes(b'test-image')
            meta=project/'configs'/'unified.csv';meta.parent.mkdir(parents=True)
            meta.write_text('source_image,garment_category\n'+str(pic)+',top\n',encoding='utf-8')
            records,counts,_=m.analyze([meta],data.resolve(),project.resolve(),100)
            self.assertEqual(len(records),1)
            self.assertEqual(records[0]['category_hint_NOT_GT'],'top')
            self.assertEqual(counts['rows_missing_known_path_field'],0)
    def test_no_test_image_leak(self):
        with tempfile.TemporaryDirectory() as root:
            rt=Path(root);data=rt/'fashion_data';project=rt/'repo'
            pic=data/'raw'/'test'/'image'/'01.jpg';pic.parent.mkdir(parents=True);pic.write_bytes(b'test')
            img,reason=m.resolve_train_image(str(pic),project/'configs'/'a.csv',data.resolve(),project.resolve())
            self.assertIsNone(img)
    def test_missing_field_flagged(self):
        with tempfile.TemporaryDirectory() as root:
            rt=Path(root);meta=rt/'one.csv';meta.write_text('garment_category,wrong_path\ntop,something\n')
            rows,counts,_=m.analyze([meta],(rt/'data').resolve(),(rt/'repo').resolve(),100)
            self.assertEqual(rows,[])
            self.assertEqual(counts['rows_missing_known_path_field'],1)
if __name__=='__main__':unittest.main()
