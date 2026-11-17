"""After accounting, continue only a verified, successful, productive batch."""
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'llm'))
from ldc_main_common_v33 import ROOT,WORK,read,save,digest,verify


def run(job):
    verify()
    receipt=read(WORK/'submissions'/(job+'.json'))
    account=read(WORK/'scheduler'/job/'summary.json')
    result=dict(previous_job=job,main_test_opened=False)
    if receipt['phase']!='main' or account['state']!='COMPLETED' or account['exit_code']!='0:0':
        result.update(status='CONTINUATION_STOPPED',reason='Previous GPU batch failed; no automatic retry')
    else:
        review=read(WORK/'main/batches'/job/'review.json')
        if not all(review['checks'].values()):
            result.update(status='CONTINUATION_STOPPED',reason='Batch receipt checks failed')
        elif review['all_done']:
            result.update(status='TRAINING_STAGE_COMPLETE_TEST_PENDING',reason='No more training; final sealed TEST is a separate stage')
        elif review['status']!='LDC_MAIN_BATCH_PARTIAL_PASS' or review['all_attempts_done']<=receipt['done_before_start']:
            result.update(status='CONTINUATION_STOPPED',reason='Batch not productive or not cleanly drained')
        else:
            command=[str(ROOT/'envs/visualization/bin/python'),'-B',str(ROOT/'test/analysis/submit_ldc_main_batch_v33.py')]
            response=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90)
            result.update(status='NEXT_BATCH_SUBMITTED' if response.returncode==0 else 'CONTINUATION_STOPPED',
                          returncode=response.returncode,stdout=response.stdout,stderr=response.stderr,
                          operational_manifest_sha256=digest(WORK/'operational_manifest.json'))
    save(WORK/'scheduler'/job/'continuation.json',result)
    print(result['status'],result.get('stdout',result.get('reason','')),flush=True)
    return result['status'] in ['NEXT_BATCH_SUBMITTED','TRAINING_STAGE_COMPLETE_TEST_PENDING']


if __name__=='__main__':raise SystemExit(0 if run(sys.argv[1]) else 1)
