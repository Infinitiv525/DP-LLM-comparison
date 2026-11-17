"""Two-node NFS locking test in a fixture-only directory; no real trainings."""
import argparse
import os
from pathlib import Path
import socket
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'llm'))
from ldc_main_common_v33 import WORK,read,save,digest,verify
from ldc_main_scheduler_v33 import initialize_state,claim,finish,search_id

DIRECTORY=WORK/'operational_crossnode_v35'


def initialize():
    verify();DIRECTORY.mkdir(exist_ok=False)
    plan=[]
    for step in range(5):
        for chain in range(5):
            method=f'unit_fixture_{chain}'
            plan.append(dict(id=search_id(method,0,step),kind='search',method=method,run=0,candidate=step,
                             seed=0,dependencies=[] if not step else [search_id(method,0,step-1)]))
    initialize_state(DIRECTORY,plan)
    save(DIRECTORY/'fixture_manifest.json',dict(source_sha256=digest(__file__),
        frozen_science_sha256=digest(WORK/'freeze.json'),fixture_only=True,real_training=False))
    print('CROSSNODE_FIXTURE_READY',flush=True)


def worker():
    verify();rank=int(os.environ['SLURM_PROCID']);job=os.environ['SLURM_JOB_ID'];done=[]
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        payload,status=claim(DIRECTORY,rank,job)
        if status=='ALL_DONE':break
        if payload is None:time.sleep(.1);continue
        task=payload['task'];time.sleep(.02)
        # Explicit synthetic unit fixture; never used as experiment feedback.
        outcome=dict(status='COMPLETE',synthetic_unit_fixture_only=True,
                     history=dict(id=task['id'],parameters={},score=float(task['candidate'])))
        finish(DIRECTORY,task['id'],rank,job,outcome);done.append(task['id'])
    else:raise RuntimeError('Cross-node fixture deadline exceeded')
    save(DIRECTORY/f'rank_{rank}.json',dict(rank=rank,host=socket.gethostname(),claimed=done,job=job,fixture_only=True))
    print('CROSSNODE_WORKER_DONE',rank,len(done),flush=True)


def review():
    verify();rows=[read(DIRECTORY/f'rank_{i}.json') for i in range(2)]
    state=read(DIRECTORY/'state.json');claimed=sum((r['claimed'] for r in rows),[])
    checks=dict(two_distinct_nodes=len({r['host'] for r in rows})==2,
                exactly_25_unique_claims=len(claimed)==len(set(claimed))==25,
                all_fixture_tasks_done=all(r['status']=='DONE' for r in state['tasks'].values()))
    checks['dependency_order']=all(state['tasks'][p]['finished_unix']<=r['claimed_unix']
         for r in state['tasks'].values() for p in r['task']['dependencies'])
    result=dict(status='TWO_NODE_QUEUE_LOCK_TEST_PASS' if all(checks.values()) else 'CROSSNODE_TEST_FAILED',
        checks=checks,rows=rows,real_training=False,main_test_opened=False,
        scientific_freeze_sha256=digest(WORK/'freeze.json'))
    save(DIRECTORY/'review.json',result);print(result['status'],checks,flush=True)
    return all(checks.values())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--initialize',action='store_true');p.add_argument('--review',action='store_true');a=p.parse_args()
    if a.initialize:initialize()
    elif a.review:raise SystemExit(0 if review() else 1)
    else:worker()
