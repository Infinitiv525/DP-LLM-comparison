"""Account every batch; last callback starts one combined16GPU successor."""
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'llm'))
from ldc_main_common_v33 import ROOT,WORK,read,save,digest
from submit_ldc_parallel_v35 import POOL,verify_parallel,queue_snapshot,main_records


def peer_decision(current,active_callbacks):
    return 'YIELD' if any(int(other)>int(current) for other in active_callbacks) else 'LEADER'


def run(job):
    result=dict(previous_gpu_job=job,main_test_opened=False)
    try:
        verify_parallel()
        account=read(WORK/'scheduler'/job/'summary.json')
        if account['state']!='COMPLETED' or account['exit_code']!='0:0':raise ValueError('GPU batch failed;no automatic retry')
        review=read(WORK/'main/batches'/job/'review.json')
        if not all(review['checks'].values()):raise ValueError('GPU batch receipt checks failed')
        queue=queue_snapshot();active_main={str(r['job']) for r in main_records()}&set(queue)
        if active_main:
            result.update(status='WAIT_FOR_OTHER_GPU_BATCH',active_main_jobs=sorted(active_main))
        else:
            current=os.environ['SLURM_JOB_ID']
            callback_ids={p.stem for p in (POOL/'callbacks').glob('*.json')}
            peers=callback_ids&set(queue)-{current}
            if peer_decision(current,peers)=='YIELD':
                result.update(status='YIELD_TO_LATER_CALLBACK')
            else:
                deadline=time.monotonic()+45
                while peers:
                    if time.monotonic()>deadline:raise ValueError('Peer callback did not finish;node slot needed')
                    time.sleep(1);peers=callback_ids&set(queue_snapshot())-{current}
                state=read(WORK/'main/state.json')
                if all(r['status']=='DONE' for r in state['tasks'].values()):
                    result.update(status='TRAINING_STAGE_COMPLETE_TEST_PENDING')
                    save(POOL/'training_complete.json',dict(main_test_opened=False,state_sha256=digest(WORK/'main/state.json'),
                        full_review_candidates=[str(p) for p in (WORK/'main/batches').glob('*/review.json') if read(p)['all_done']]))
                else:
                    response=subprocess.run([str(ROOT/'envs/visualization/bin/python'),'-B',
                        str(ROOT/'test/analysis/submit_ldc_parallel_v35.py'),'--mode','rollover'],cwd=ROOT,
                        capture_output=True,text=True,timeout=90)
                    if response.returncode:raise RuntimeError(response.stdout+'\n'+response.stderr)
                    result.update(status='NEXT_COMBINED_BATCH_SUBMITTED',stdout=response.stdout)
    except Exception as error:
        result.update(status='PARALLEL_CONTINUATION_REVIEW_REQUIRED',reason=str(error))
        save(POOL/'hold.json',result)
    save(WORK/'scheduler'/job/'parallel_continuation.json',result)
    print(result['status'],result.get('stdout',result.get('reason','')),flush=True)
    return result['status']!='PARALLEL_CONTINUATION_REVIEW_REQUIRED'


if __name__=='__main__':raise SystemExit(0 if run(sys.argv[1]) else 1)
