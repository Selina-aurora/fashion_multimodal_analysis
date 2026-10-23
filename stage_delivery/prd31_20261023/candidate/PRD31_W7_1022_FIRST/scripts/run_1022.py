"""Bounded 10/22 workflow: delivery CLI, added samples, compiler screen."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import traceback
import zipfile
from collect_new_regression import collect
from run_attributes import HERE, read, write, verify


def child(command, folder, timeout):
    """Run GPU stages in fresh processes; kill compiler workers on timeout."""
    log=folder.with_suffix('.log')
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace',bufsize=1,start_new_session=True)
        def pump():
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line,end='',flush=True)
        reader=threading.Thread(target=pump,daemon=True)
        reader.start()
        print('RUNNING: '+' '.join(command),flush=True)
        print('LOG: '+str(log),flush=True)
        timed_out=False
        interrupted=False
        try:
            code=process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            interrupted=isinstance(exc,KeyboardInterrupt)
            timed_out=True
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait(timeout=10)
            code=124
        reader.join(timeout=10)
        if not reader.is_alive():process.stdout.close()
        if interrupted:raise KeyboardInterrupt
        result=read(folder/'summary.json') if (folder/'summary.json').is_file() else {'complete':False,'status':'NO_CHILD_SUMMARY'}
        if timed_out:
            result={'status':'STAGE_TIMEOUT_OR_INTERRUPTION','complete':False,'previous_partial_summary':result}
        result['process_exit_code']=code
        result['log_file']=str(log)
        return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='/workspace/fashion_multimodal_analysis')
    ap.add_argument('--output-root',default='/workspace/prd31_week7_1022_first')
    ap.add_argument('--w7-identities')
    ap.add_argument('--pool-root',action='append')
    ap.add_argument('--skip-compile',action='store_true')
    ap.add_argument('--compile-timeout-seconds',type=int,default=1200)
    args=ap.parse_args()
    root=Path(args.output_root).resolve()
    out=root/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    stages={}
    code=1
    try:
        verify(HERE,read(HERE/'PACKAGE_SHA256.json'))
        project=Path(args.project).resolve()
        sample_dir=out/'new_samples'
        sample_dir.mkdir()
        print('STAGE 1/4 NEW CANDIDATE IDENTITY SCREEN',flush=True)
        try:
            stages['new_samples']=collect(project,sample_dir,20,args.w7_identities,args.pool_root)
        except Exception:
            stages['new_samples']={'status':'COLLECTION_BLOCKED','selected':0,'error':traceback.format_exc()}
            write(sample_dir/'selection_audit.json',stages['new_samples'])
        write(out/'stage_status.json',stages)
        print('STAGE 2/4 DELIVERY SINGLE-IMAGE CLI CHECK',flush=True)
        source=read(HERE/'inputs/SMOKE_REFERENCE.json')['source']['path']
        stages['delivery_entry']=child([sys.executable,'-u',str(HERE/'scripts/run_candidate.py'),
            '--project',str(project),'--image',source,'--mode','white-sam','--check-smoke-reference',
            '--output-dir',str(out/'delivery_entry')],out/'delivery_entry',600)
        write(out/'stage_status.json',stages)
        entry_ok=stages['delivery_entry'].get('complete',False) and stages['delivery_entry']['process_exit_code']==0
        print('STAGE 3/4 NEW IMAGE PAIRED REGRESSION',flush=True)
        if entry_ok and stages['new_samples'].get('selected')==20:
            stages['new_regression']=child([sys.executable,'-u',str(HERE/'scripts/run_new_regression.py'),
                '--project',str(project),'--manifest',str(sample_dir/'images.json'),
                '--output-dir',str(out/'new_regression')],out/'new_regression',1200)
        else:
            stages['new_regression']={'status':'SKIPPED_ENTRY_OR_SAMPLE_COLLECTION_NOT_READY','complete':False}
        write(out/'stage_status.json',stages)
        print('STAGE 4/4 CORE COMPILATION SCREEN',flush=True)
        if entry_ok and not args.skip_compile:
            stages['compile_screen']=child([sys.executable,'-u',str(HERE/'scripts/core_compile_probe.py'),
                '--project',str(project),'--count','64','--output-dir',str(out/'compile_screen')],
                out/'compile_screen',args.compile_timeout_seconds)
        else:
            stages['compile_screen']={'status':'SKIPPED_BY_OPTION_OR_ENTRY_NOT_READY','complete':False}
        write(out/'stage_status.json',stages)
        ready=entry_ok and stages['new_regression'].get('all_output_checks_pass',False)
        result={'status':'ENTRY_AND_NEW_REGRESSION_CHECKS_COMPLETE' if ready else 'PARTIAL_CHECKS_COMPLETE',
            'delivery_entry_passed':entry_ok,'additional_regression_passed':stages['new_regression'].get('all_output_checks_pass',False),
            'stages':stages,'compiled_model_promoted':False,'production_installed':False,
            'certified_independent_accuracy_evaluation_done':False,
            'sealed_test_images_used':False,'all_PRD_metrics_passed':False}
        write(out/'summary.json',result)
        code=0 if ready else 1
    except BaseException:
        error=traceback.format_exc()
        print(error,flush=True)
        write(out/'summary.json',{'status':'WORKFLOW_INTERRUPTED','stages':stages,'error':error,'production_installed':False})
    finally:
        archive=out.with_name(out.name+'_UPLOAD_THIS.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob('*')):
                if f.is_file() and not f.name.endswith('_UPLOAD_THIS.zip'):
                    z.write(f,f.relative_to(out))
        print('UPLOAD_THIS: '+str(archive),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
