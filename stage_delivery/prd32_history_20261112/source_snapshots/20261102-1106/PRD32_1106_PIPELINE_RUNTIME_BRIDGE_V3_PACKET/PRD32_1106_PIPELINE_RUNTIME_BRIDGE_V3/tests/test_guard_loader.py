import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from verified_guard_loader import load_guard

BASE=Path(__file__).resolve().parents[1]

class GuardLoaderTest(unittest.TestCase):
    def test_verified_source_loads(self):
        guard=load_guard(BASE/'reference_guard/prd32_output_guard.py')
        self.assertTrue(callable(guard))
    def test_changed_source_is_rejected(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'guard.py'
            p.write_text('def guard_outputs(x): return {}')
            with self.assertRaisesRegex(ValueError,'SHA changed'):
                load_guard(p)

if __name__=='__main__':unittest.main()
