"""Final GPU rerun of installed 3.1 chain, with geometry and output-contract checks.
No training, no data/weight/source modification. Uses existing offline models.
"""
import json,os,shutil,time,zipfile
from pathlib import Path
from collections import Counter
from PIL import Image
from smoke_export import save_prediction,sha,dump
from verify_chain_contracts import verify
from check_geometry import run_checks


def main():
 from fashion_multimodal_analysis.common.paths import project_root,resolve_path
 from fashion_multimodal_analysis.integration.pipeline import FashionPipeline
 import fashion_multimodal_analysis.integration.pipeline as pipeline_module
 import fashion_multimodal_analysis.integration.semantic_consistency as consistency_module
 import fashion_multimodal_analysis.integration.model_backends as backend_module
 root=project_root()
 baseline=root/'reports/reruns/recovery_20261009/prd31_neckline_context_20261009_012052'
 config=json.loads((baseline/'run_config.json').read_text())
 selected=json.loads((baseline/'cases.json').read_text())
 config['semantic_consistency_review']=True
 config['regression_patch_version']='final_installed_chain_rerun_20261009'
 if 'semantic_consistency_review' not in Path(pipeline_module.__file__).read_text():
  raise RuntimeError('Installed pipeline lacks previously supplied consistency patch; do not silently test an older version')
 geometry=run_checks()
 print('GEOMETRY_CHECKS_PASSED:',len(geometry['tests']),flush=True)
 for key,expected in json.loads((baseline/'summary.json').read_text())['input_sha256'].items():
  if sha(resolve_path(config[key]))!=expected:raise ValueError('Frozen input changed: '+key)
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 out=root/'reports/reruns/recovery_20261009'/('prd31_final_chain_'+time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000:06d}')
 out.mkdir(parents=True);(out/'source_images').mkdir()
 dump(out/'run_config.json',config);dump(out/'geometry_tests.json',geometry)
 code={str(Path(m.__file__).resolve()):sha(Path(m.__file__)) for m in [pipeline_module,consistency_module,backend_module]}
 dump(out/'installed_source_hashes.json',code)
 from fashion_multimodal_analysis.integration.model_backends import ModelBackends
 pipeline=FashionPipeline(ModelBackends(config,device='cuda'))
 cases=[]
 for n,selected_case in enumerate(selected,1):
  source=resolve_path(selected_case['source_image'])
  expected=selected_case.get('source_sha256')
  if expected and sha(source)!=expected:raise ValueError('Source image changed')
  case=f'case_{n:02d}';shutil.copyfile(source,out/'source_images'/(case+'.jpg'))
  print(f'Running full chain {n}/{len(selected)}',flush=True)
  with Image.open(source) as raw:image=raw.convert('RGB')
  started=time.perf_counter();result=pipeline.run(image,config['required_regions'],config['required_attributes'])
  payload=save_prediction(image,result,out/case)
  cases.append({'case_dir':case,'source_image':str(source),'source_sha256':sha(source),
                'selection_category':selected_case['selection_category'],'pipeline_status':result['status'],
                'instance_count':len(result['instances']),'failures':result['failures'],
                'interface_issues':payload['interface_issues'],'chain_wall_ms':(time.perf_counter()-started)*1000})
  dump(out/'cases.json',cases)
 check=verify(out);check['model_inference_run']=True;dump(out/'contract_check.json',check)
 summary={'status':'FULL_CHAIN_INTERFACE_PASS' if check['status']=='PASS' else 'FULL_CHAIN_INTERFACE_FAIL',
          'images':len(cases),'predicted_instances':sum(c['instance_count'] for c in cases),
          'geometry_tests_passed':len(geometry['tests']),'contract_checks':check['counts']['checks'],
          'interface_issues':len(check['issues']),'failed_downstream_calls':sum(len(c['failures']) for c in cases),
          'model_inference_run':True,'semantic_consistency_review_enabled':True,
          'accuracy_evaluated':False,'formal_prd_acceptance':False,
          'scope':'Installed-source GPU rerun on 8 existing diagnostic images; interface completion only.',
          'limitations':['Not a blind accuracy test or latency benchmark.',
                         'Requires-review statuses preserve uncertainty; they do not fix semantic predictions.',
                         'Semantic errors documented by previous human reviews remain unresolved.']}
 dump(out/'summary.json',summary)
 with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(out))
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
 print('UPLOAD_THIS_ZIP:',out.with_suffix('.zip'),flush=True)
 if check['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
