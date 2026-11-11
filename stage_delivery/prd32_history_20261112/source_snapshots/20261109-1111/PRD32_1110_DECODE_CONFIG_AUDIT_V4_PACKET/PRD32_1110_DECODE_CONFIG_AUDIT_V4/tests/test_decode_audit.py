import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import decode_config_audit_v4 as m

class Tests(unittest.TestCase):
    def test_parses_generate_kw(self):
        x=m.static_calls('def run_model_case():\n    return model.generate(**inputs, use_cache=True, max_new_tokens=32)', 'r.py')
        self.assertEqual(x['call_sites'][0]['selected_explicit_kwargs']['use_cache'], True)
        self.assertTrue(x['call_sites'][0]['has_asterisk_kwargs'])
    def test_dynamic_kw_not_invented(self):
        x=m.static_calls('def run_model_case():\n    return model.generate(**args, use_cache=maybe)', 'r.py')
        self.assertEqual(x['call_sites'][0]['selected_explicit_kwargs']['use_cache'],'DYNAMIC_EXPR')
    def test_no_generate_calls(self):
        x=m.static_calls('def hello():\n    return 2','r.py')
        self.assertEqual(x['call_sites'],[])
    def test_refuse_holdout(self):
        with self.assertRaises(ValueError):m.refuse_restricted(Path('/workspace/holdout/private.json'))
    def test_read_config_only_allowlist(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'config.json';p.write_text(json.dumps({'use_cache':True,'api_key':'DO_NOT_SHOW','text_config':{'use_cache':False}}))
            r=m.sanitized_config(p)
            self.assertNotIn('api_key',str(r))
            self.assertEqual(r['safe_selected_fields']['use_cache'],True)
    def test_missing_generation_config(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(m.sanitized_config(Path(td)/'generation_config.json')['status'],'NOT_PRESENT')
    def test_digest(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a';p.write_bytes(b'abc')
            self.assertEqual(m.digest(p),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

if __name__=='__main__':unittest.main()
