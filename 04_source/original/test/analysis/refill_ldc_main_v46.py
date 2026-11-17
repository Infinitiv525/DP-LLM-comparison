"""Refill a partially drained V36 group from 8 to 24 main GPUs.

Replace only the CPU collector so it caches the complete amended group.
Use the unchanged, audited V36 16-GPU launcher and original V33 workers.
No training is interrupted and no scientific attempt is added or retried.
"""
import argparse
from collections import Counter
from datetime import datetime
import re
from zoneinfo import ZoneInfo

from ldc24_ops_v36 import (ROOT, WORK, POOL, TERMINAL, read, save, digest,
    verify, command, queue_snapshot, submit_collector, launch_gpu, capture, main_records)
from ldc_main_scheduler_v33 import locked

OUT=ROOT/'results/ldc_main_refill_20261106_v46'
OLD='144648'
EXISTING_GPU='144649'
GROUP='group_000'


def now():
    return datetime.now(ZoneInfo('Europe/Bratislava')).isoformat()


def fields(job):
    return dict(re.findall(r'(?:^|\s)(\w+)=([^\s]+)',command(['scontrol','show','job','-o',str(job)])))


def verified_state():
    verify()
    state=read(WORK/'main/state.json')
    assert len(state['tasks'])==395
    assert not state['attention_required'] and not (POOL/'hold.json').exists()
    return state


def prepare():
    state=verified_state()
    assert not OUT.exists(), 'Do not repeat an already registered migration'
    queue=queue_snapshot()
    assert set(queue)=={OLD,EXISTING_GPU},queue
    assert fields(OLD).get('TresPerNode','').find('gpu')<0
    assert not any(r['status']=='RUNNING' and r['job']==OLD for r in state['tasks'].values())
    steps=command(['squeue','--steps','-j',OLD,'-h','-o','%i'])
    assert all(s in [OLD+'.batch',OLD+'.extern'] for s in steps.splitlines())
    ready=[r['task']['id'] for r in state['tasks'].values() if r['status']=='PENDING'
           and all(state['tasks'][d]['status']=='DONE' for d in r['task']['dependencies'])]
    assert len(ready)>=16
    OUT.mkdir(parents=True)
    save(OUT/'intent.json',dict(at=now(),old_cpu_collector=OLD,existing_gpu_job=EXISTING_GPU,
         old_group=read(POOL/'groups'/(GROUP+'.json')),old_overview=read(POOL/'current_group.json'),
         original_queue=queue,ready_tasks=ready,ready_count=len(ready),
         original_counts=dict(Counter(r['status'] for r in state['tasks'].values())),
         original_main_freeze_sha256=digest(WORK/'freeze.json'),source_sha256=digest(__file__),
         target_main_gpus=24,total_attempts=395,new_capacity_runs=0,TEST_opened=False))
    with locked(POOL/'control'):
        cpu=submit_collector(GROUP,[OLD],held=True)
        save(OUT/'progress.json',dict(stage='NEW_CPU_HELD',cpu=cpu,old=OLD,at=now()))
        command(['scancel',OLD])
        save(OUT/'progress.json',dict(stage='OLD_CPU_CANCEL_REQUESTED',cpu=cpu,old=OLD,at=now()))
    print('CPU_REPLACEMENT_REGISTERED_HELD',cpu,'OLD_CPU_CANCEL_REQUESTED',OLD,flush=True)


def register():
    verified_state();p=read(OUT/'progress.json')
    assert p['stage']=='OLD_CPU_CANCEL_REQUESTED'
    if OLD in queue_snapshot() or fields(OLD)['JobState'] not in TERMINAL:
        print('WAIT_OLD_CPU_TERMINATION',flush=True);return
    account=capture(OLD)
    assert account['state']=='CANCELLED' and account['gpu_count']==0
    save(OUT/'old_cpu_accounting.json',account)
    with locked(POOL/'control'):
        group=read(POOL/'groups'/(GROUP+'.json'))
        assert group['jobs']==read(OUT/'intent.json')['old_group']['jobs']
        assert p['cpu'] in group['collectors']
        gpu=launch_gpu(16,GROUP)
        group=read(POOL/'groups'/(GROUP+'.json'))
        group['jobs'].append(gpu)
        group['active_collector']=p['cpu']
        group['status']='ACTIVE_MAIN_GROUP_CPU_ACCOUNTING'
        group['refill_v46']=dict(old_cpu=OLD,cpu=p['cpu'],added_main_job=gpu,added_gpus=16,
             intent_sha256=digest(OUT/'intent.json'))
        save(POOL/'groups'/(GROUP+'.json'),group)
        save(POOL/'current_group.json',dict(group=GROUP,jobs=group['jobs'],collector=p['cpu'],
             target_gpus=24,status='REFILL_REGISTERED_BEFORE_RELEASE',updated_at=now(),
             migration_receipt=str(OUT/'progress.json')))
        p.update(stage='MAIN16_REGISTERED_HELD',gpu=gpu,at=now());save(OUT/'progress.json',p)
        command(['scontrol','release',p['cpu']])
        p.update(stage='CPU_RELEASED_MAIN16_HELD',at=now());save(OUT/'progress.json',p)
    print('CPU_RELEASED',p['cpu'],'MAIN16_REGISTERED_HELD',gpu,flush=True)


def activate():
    verified_state();p=read(OUT/'progress.json')
    assert p['stage']=='CPU_RELEASED_MAIN16_HELD'
    status=POOL/'collector_status'/(p['cpu']+'.json')
    if fields(p['cpu'])['JobState']!='RUNNING' or not status.exists():
        print('WAIT_NEW_CPU_START',p['cpu'],flush=True);return
    collector=read(status)
    assert set(collector['waiting'])=={EXISTING_GPU,p['gpu']}
    assert set(collector['accounted'])=={'143342','143656','143684'}
    command(['scontrol','release',p['gpu']])
    p.update(stage='MAIN16_RELEASED',at=now());save(OUT/'progress.json',p)
    print('MAIN16_RELEASED',p['gpu'],'CPU_COLLECTOR',p['cpu'],flush=True)


def closeout():
    state=verified_state();p=read(OUT/'progress.json')
    assert p['stage']=='MAIN16_RELEASED'
    bootstrap_path=POOL/'bootstrap'/(p['gpu']+'.json')
    if not bootstrap_path.exists():
        print('WAIT_NEW_GPU_BOOTSTRAP',p['gpu'],flush=True);return
    bootstrap=read(bootstrap_path)
    assert bootstrap['status']=='ACCOUNTING_READY_BEFORE_TRAINING' and bootstrap['collector_job']==p['cpu']
    probe=WORK/'main/batches'/p['gpu']/'device_assignment.json'
    if not probe.exists():
        print('WAIT_DEVICE_PROBE',p['gpu'],flush=True);return
    devices=read(probe)['rows'];existing=read(WORK/'main/batches'/EXISTING_GPU/'device_assignment.json')['rows']
    assert len(devices)==16 and len(existing)==8
    assert len({r['uuid'] for r in devices+existing})==24
    assert all(r['offline'] and 'H200' in r['device'] for r in devices+existing)
    owned=[dict(id=k,task=r['task'],worker=r['worker']) for k,r in state['tasks'].items()
           if r['status']=='RUNNING' and r['job']==p['gpu']]
    if len(owned)!=16:
        print('WAIT_16_WORKER_CLAIMS',len(owned),flush=True);return
    assert len({r['worker'] for r in owned})==16
    for r in owned:
        config=read(WORK/'main/executions'/r['id']/p['gpu']/'training/config.json')
        assert config['width']==192 and config['epochs']==400 and config['seed']==r['task']['seed']
    queue=queue_snapshot()
    assert set(queue)=={EXISTING_GPU,p['gpu'],p['cpu']} and sum(queue.values())==4
    allocated=sum(r.get('gpus',8) for r in main_records() if str(r['job']) in queue)
    assert allocated==24
    with locked(POOL/'control'):
        overview=read(POOL/'current_group.json');overview['status']='ACTIVE_CPU_ACCOUNTING'
        overview['verified_at']=now();save(POOL/'current_group.json',overview)
    result=dict(status='PASS',at=now(),added_main_job=p['gpu'],added_gpus=16,
         existing_main_job=EXISTING_GPU,cpu_collector=p['cpu'],replaced_cpu_collector=OLD,
         assigned_distinct_H200=24,reserved_nodes=4,counts=dict(Counter(r['status'] for r in state['tasks'].values())),
         new_owned_tasks=owned,main_395_attempts_unchanged=True,frozen_sources_unchanged=True,
         existing_trainings_interrupted=False,larger_network_training=False,TEST_opened=False,
         source_sha256=digest(__file__))
    save(OUT/'closeout.json',result)
    print('REFILL_PASS', {k:v for k,v in result.items() if k!='new_owned_tasks'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','register','activate','closeout'])
    globals()[parser.parse_args().stage]()
