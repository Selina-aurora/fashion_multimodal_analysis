import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from collect_qwen_live_entrypoints import collect

class CollectorTests(unittest.TestCase):
    def test_reads_only_explicit_dev_qwen_and_scripts(self):
        with TemporaryDirectory() as t:
            w=Path(t)/'workspace';w.mkdir()
            allowed=w/'PRD32_DEV60_QWEN_BASELINE_20261030_run01';allowed.mkdir()
            (allowed/'run_qwen.py').write_text('import transformers\n# qwen\nx = model.generate(x)\n')
            excluded=allowed/'holdout_data';excluded.mkdir()
            (excluded/'qwen.py').write_text('import transformers\n# qwen\n')
            unrelated=w/'fashion_data';unrelated.mkdir()
            (unrelated/'train_qwen.py').write_text('import transformers\n# qwen\n')
            out=w/'out';out.mkdir()
            m=collect(w,out)
            self.assertEqual(len(m['selected']),1)
            self.assertEqual(m['selected'][0]['source_path'],str(allowed/'run_qwen.py'))
            self.assertTrue(m['no_holdout_scan'])
            self.assertFalse(any('holdout' in str(p) for p in out.rglob('*')))
    def test_secret_literal_is_not_exported(self):
        with TemporaryDirectory() as t:
            w=Path(t)/'workspace';w.mkdir()
            allowed=w/'PRD32_DEV60_QWEN_BASELINE_20261030_run01';allowed.mkdir()
            (allowed/'runner.py').write_text('import transformers\n# qwen\napi_key = "secret1234567890123"\n')
            out=w/'out';out.mkdir()
            m=collect(w,out)
            self.assertEqual(len(m['selected']),1)
            self.assertTrue(m['selected'][0]['secret_pattern_skipped'])
            self.assertFalse(m['selected'][0]['source_exported'])

if __name__=='__main__':unittest.main()
