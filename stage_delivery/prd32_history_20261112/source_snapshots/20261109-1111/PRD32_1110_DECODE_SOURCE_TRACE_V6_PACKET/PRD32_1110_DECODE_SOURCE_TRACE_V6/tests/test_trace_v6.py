import importlib.util, pathlib, unittest
P=pathlib.Path(__file__).resolve().parents[1]/'trace_source_helper_v6.py'
spec=importlib.util.spec_from_file_location('v6',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TestTrace(unittest.TestCase):
 def test_ast_find(self):
  x=m.analyze('def run_model_case(model, p, limit=170):\n return model.generate(ids,max_new_tokens=limit,use_cache=True,do_sample=False)\n','x.py')
  self.assertEqual(x['definitions'][0]['name'],'run_model_case')
  self.assertEqual(x['interesting_calls'][0]['literal_decode_kwargs']['use_cache'],True)
  self.assertEqual(x['interesting_calls'][0]['literal_decode_kwargs']['max_new_tokens'],'DYNAMIC_NOT_RESOLVED')
 def test_no_string_leak(self):
  self.assertEqual(m.literal_value(__import__('ast').parse("'a'*100",mode='eval').body),'DYNAMIC_NOT_RESOLVED')
 def test_sensitive_redacted(self):
  import ast
  self.assertEqual(m.literal_value(ast.Constant(value='my-secret-token')),'COMPLEX_OR_REDACTED')
 def test_unsafe_paths(self):
  with self.assertRaises(ValueError):m.guarded_path('../bad.py',root=pathlib.Path('/tmp'))
 def test_imported_source_no_execution(self):
  a=m.analyze('raise RuntimeError("do not import")\ndef load_helper():\n pass\n','example.py')
  self.assertEqual(a['definitions'][0]['name'],'load_helper')
 def test_parse_dynamic_spread(self):
  a=m.analyze('def run_model_case(model):\n return model.generate(toks,**kwargs)\n','example.py')
  self.assertTrue(a['interesting_calls'][0]['unpacked_kwargs_present'])
if __name__=='__main__':unittest.main()
