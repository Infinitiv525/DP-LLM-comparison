"""24GPU operations only; immutable V33 science and seed plan stay unchanged."""
import os
from pathlib import Path
import re
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'llm'))
from ldc_main_common_v33 import ROOT,WORK,read,save,digest,verify as verify_science
from ldc_main_scheduler_v33 import locked

POOL=WORK/'operational_pool_v36'
TERMINAL={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}


def command(args):
    return subprocess.run(args,cwd=ROOT,capture_output=True,text=True,check=True,timeout=30).stdout.strip()


def queue_snapshot():
    raw=command(['squeue','-h','-u',os.environ.get('USER','lutoke823'),'-o','%i|%D'])
    return {line.split('|')[0].strip():int(line.split('|')[1]) for line in raw.splitlines() if line.strip()}


def main_records():
    return [read(p) for p in (WORK/'submissions').glob('*.json') if read(p)['phase']=='main']


def verify():
    verify_science();manifest=read(WORK/'operational_manifest_v36.json')
    if any(digest(p)!=h for p,h in manifest['files'].items()):raise ValueError('24GPU operational source changed')
    proof=WORK/'operational_crossnode_v35b/review.json'
    if digest(proof)!=manifest['crossnode_review_sha256'] or not all(read(proof)['checks'].values()):
        raise ValueError('Concurrent distributed queue proof required')
    if read(WORK/'integration/review.json')['status']!='LDC_MAIN_INTEGRATION_PASS':
        raise ValueError('Actual production integration required')


def checked_id(text):
    job=text.split(';')[0]
    if not re.fullmatch(r'\d+',job):raise ValueError('Unexpected Slurm ID')
    return job


def after_any(ids):
    if not ids or any(not re.fullmatch(r'\d+',str(j)) for j in ids):raise ValueError('Invalid dependency IDs')
    return '?'.join('afterany:'+str(j) for j in ids)


def submit_collector(group_id,parents,held=False):
    queue=queue_snapshot()
    if sum(queue.values())+1>4:raise ValueError('No reserved-node slot for accounting collector')
    args=['sbatch','--parsable','--dependency='+after_any(parents)]
    if held:args.append('--hold')
    args+=['test/scripts/263_ldc24_collector_v36.sbatch',group_id]
    job=checked_id(command(args))
    save(WORK/'submissions'/(job+'.json'),dict(job=int(job),phase='operational_collector_v36',group=group_id,
        parents=parents,registered_before_dependency_can_complete=True,unix=time.time(),
        operational_manifest_sha256=digest(WORK/'operational_manifest_v36.json')))
    path=POOL/'groups'/(group_id+'.json');group=read(path)
    group['collectors'].append(job);save(path,group)
    return job


def launch_gpu(gpus,group_id,predecessor_cpu=None):
    verify()
    if (POOL/'hold.json').exists():raise ValueError('Operational pool on review hold')
    if gpus not in [8,16,24]:raise ValueError('Unsupported GPU count')
    nodes=gpus//8;queue=queue_snapshot()
    if sum(queue.values())+nodes>4:raise ValueError('Four-node ceiling would be exceeded')
    records=main_records()
    if len(records)>=16:raise ValueError('Maximum registered main batch count reached')
    state=read(WORK/'main/state.json')
    if state['attention_required'] or any(r['status']=='ATTENTION' for r in state['tasks'].values()):
        raise ValueError('Experiment needs review')
    active={str(r['job']) for r in records}&set(queue)
    if sum(r.get('gpus',8) for r in records if str(r['job']) in active)+gpus>24:
        raise ValueError('Maximum24concurrentGPUs would be exceeded')
    owners={str(r['job']) for r in state['tasks'].values() if r['status']=='RUNNING'}
    if not owners<=active:raise ValueError('Stale running training requires review')
    if predecessor_cpu is not None and active:raise ValueError('Successor waits for previous GPU group to finish')
    scripts={8:'261_ldc24_8_v36.sbatch',16:'264_ldc24_16_v36.sbatch',24:'262_ldc24_24_v36.sbatch'}
    args=['sbatch','--hold','--parsable']
    if predecessor_cpu is not None:args.append('--dependency=afterok:'+str(predecessor_cpu))
    args.append('test/scripts/'+scripts[gpus])
    job=checked_id(command(args))
    save(WORK/'submissions'/(job+'.json'),dict(job=int(job),phase='main',gpus=gpus,nodes=nodes,group=group_id,
        accounting_job=None,accounting_mode='Group collector,registered by bootstrap before training',
        predecessor_collector=predecessor_cpu,registered_before_release=True,unix=time.time(),
        done_before_start=sum(r['status']=='DONE' for r in state['tasks'].values()),
        freeze_sha256=digest(WORK/'freeze.json'),operational_manifest_sha256=digest(WORK/'operational_manifest_v36.json')))
    return job


def capture(job,optional=False):
    path=WORK/'scheduler'/str(job)/'summary.json'
    if path.exists():return read(path)
    response=subprocess.run([str(ROOT/'envs/visualization/bin/python'),'-B',
        str(ROOT/'test/analysis/capture_ldc_main_v40.py'),str(job)],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if response.returncode:
        if optional:
            save(POOL/'accounting_gaps'/(str(job)+'.json'),dict(job=str(job),unknown=True,
                 stdout=response.stdout,stderr=response.stderr,note='Controller CPU accounting unavailable;not imputed as zero'))
            return None
        raise RuntimeError(response.stdout+'\n'+response.stderr)
    return read(path)


def bootstrap(gpu_job):
    verify();registration=read(WORK/'submissions'/(gpu_job+'.json'))
    if (POOL/'hold.json').exists():raise ValueError('Accounting pool needs review before training')
    if registration['phase']!='main':raise ValueError('Not a main batch')
    group_id=registration['group']
    with locked(POOL/'control'):
        group=read(POOL/'groups'/(group_id+'.json'))
        if gpu_job not in group['jobs']:raise ValueError('GPU not a group member')
        active_collectors=set(group['collectors'])&set(queue_snapshot())
        collector=sorted(active_collectors,key=int)[-1] if active_collectors else submit_collector(group_id,[gpu_job])
        save(POOL/'bootstrap'/(gpu_job+'.json'),dict(gpu_job=gpu_job,group=group_id,collector_job=collector,
             status='ACCOUNTING_READY_BEFORE_TRAINING',unix=time.time()))
    predecessor=registration.get('predecessor_collector')
    if predecessor is not None:capture(predecessor,optional=True)
    print('ACCOUNTING_READY_BEFORE_TRAINING',gpu_job,collector,flush=True)


def prepare_successor(current_cpu):
    verify()
    with locked(POOL/'control'):
        if (POOL/'hold.json').exists():raise ValueError('Operational pool on review hold')
        state=read(WORK/'main/state.json');queue=queue_snapshot()
        active={str(r['job']) for r in main_records()}&set(queue)
        if active or any(r['status'] in ['RUNNING','ATTENTION'] for r in state['tasks'].values()):
            raise ValueError('Group has not drained cleanly')
        pending=sum(r['status']=='PENDING' for r in state['tasks'].values())
        if not pending:return None
        gpus=8 if pending<=8 else 16 if pending<=16 else 24
        group_id=f'group_{len(list((POOL/"groups").glob("*.json"))):03d}'
        job=launch_gpu(gpus,group_id,current_cpu)
        save(POOL/'groups'/(group_id+'.json'),dict(id=group_id,jobs=[job],collectors=[],
             status='WAITING_FOR_PREDECESSOR_CPU_EXIT',predecessor_cpu=current_cpu))
        save(POOL/'current_group.json',dict(group=group_id,gpu_job=job,gpus=gpus))
        # No CPU collector is reserved here. The GPU job registers it on startup
        # AFTER this predecessor CPU exits, keeping the reserved-node total <=4.
        command(['scontrol','release',job])
        return dict(job=job,gpus=gpus,group=group_id)
