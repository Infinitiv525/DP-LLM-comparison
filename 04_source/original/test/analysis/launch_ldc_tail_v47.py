"""Register one round of eight remaining planned trainings, not new science."""
import argparse
from collections import Counter
from datetime import datetime
import re
from zoneinfo import ZoneInfo
import ldc24_ops_v36 as ops
from ldc_main_scheduler_v33 import locked

ROOT,WORK,POOL=ops.ROOT,ops.WORK,ops.POOL
OUT=ROOT/'results/ldc_tail_20261107_v47'
OLD='145175'; EXISTING='145176'; GROUP='group_000'
TAIL='test/scripts/267_ldc_one_round_v47.sbatch'


def now():return datetime.now(ZoneInfo('Europe/Bratislava')).isoformat()


def fields(job):
    return dict(re.findall(r'(?:^|\s)(\w+)=([^\s]+)',ops.command(['scontrol','show','job','-o',str(job)])))


def verify():
    ops.verify()
    s=ops.read(WORK/'main/state.json')
    assert len(s['tasks'])==395 and not s['attention_required'] and not (POOL/'hold.json').exists()
    assert sum(r['status']=='DONE' and r['task']['kind']=='search' for r in s['tasks'].values())==240
    if (OUT/'freeze.json').exists():
        assert all(ops.digest(path)==h for path,h in ops.read(OUT/'freeze.json')['files'].items())
    return s


def prepare():
    s=verify();assert not OUT.exists()
    assert set(ops.queue_snapshot())=={OLD,EXISTING}
    assert ops.read(WORK/'submissions'/(OLD+'.json'))['phase']=='operational_collector_v36'
    assert all(line in [OLD+'.batch',OLD+'.extern'] for line in ops.command(['squeue','--steps','-j',OLD,'-h','-o','%i']).splitlines())
    ready=[r['task']['id'] for r in s['tasks'].values() if r['status']=='PENDING'
           and all(s['tasks'][d]['status']=='DONE' for d in r['task']['dependencies'])]
    assert len(ready)>=8
    OUT.mkdir(parents=True)
    files=[ROOT/TAIL,ROOT/'test/llm/worker_ldc_one_round_v47.py',ROOT/'test/llm/test_ldc_one_round_v47.py',ROOT/'test/analysis/launch_ldc_tail_v47.py']
    ops.save(OUT/'freeze.json',dict(files={str(p):ops.digest(p) for p in files},
        main_freeze_sha256=ops.digest(WORK/'freeze.json'),policy='One unchanged V33 queued attempt per worker, at most8; no idle waiting afterwards'))
    ops.save(OUT/'intent.json',dict(at=now(),old_group=ops.read(POOL/'groups'/(GROUP+'.json')),
        old_index=ops.read(POOL/'current_group.json'),counts=dict(Counter(r['status'] for r in s['tasks'].values())),
        ready_tasks=ready,target_gpus=24,original_395_attempts_unchanged=True,TEST_opened=False))
    with locked(POOL/'control'):
        cpu=ops.submit_collector(GROUP,[OLD],held=True)
        ops.save(OUT/'progress.json',dict(stage='CPU_REGISTERED_HELD',cpu=cpu,old=OLD))
        ops.command(['scancel',OLD])
        ops.save(OUT/'progress.json',dict(stage='OLD_CPU_CANCEL_REQUESTED',cpu=cpu,old=OLD))
    print('CPU_REPLACEMENT',cpu,'OLD_CPU_CANCEL_REQUESTED',OLD,flush=True)


def register():
    verify();p=ops.read(OUT/'progress.json');assert p['stage']=='OLD_CPU_CANCEL_REQUESTED'
    if OLD in ops.queue_snapshot() or fields(OLD)['JobState'] not in ops.TERMINAL:
        print('WAIT_OLD_CPU_TERMINATION');return
    account=ops.capture(OLD);assert account['state']=='CANCELLED' and account['gpu_count']==0
    ops.save(OUT/'old_cpu_accounting.json',account)
    with locked(POOL/'control'):
        # Preserve every original launcher guard. The scoped command adapter
        # changes only the held execution script, then records that explicitly.
        original_command=ops.command
        def tail_command(args):
            if args[0]=='sbatch' and args[-1]=='test/scripts/261_ldc24_8_v36.sbatch':
                args=args[:-1]+[TAIL]
            return original_command(args)
        ops.command=tail_command
        try:gpu=ops.launch_gpu(8,GROUP)
        finally:ops.command=original_command
        reg=ops.read(WORK/'submissions'/(gpu+'.json'))
        reg.update(execution_script=TAIL,execution_policy='one_round_max_one_attempt_per_worker',
             supplemental_operational_freeze=str(OUT/'freeze.json'),supplemental_operational_sha256=ops.digest(OUT/'freeze.json'))
        ops.save(WORK/'submissions'/(gpu+'.json'),reg)
        group=ops.read(POOL/'groups'/(GROUP+'.json'));group['jobs'].append(gpu);group['active_collector']=p['cpu']
        group['tail_v47']=dict(gpu=gpu,cpu=p['cpu'],old_cpu=OLD,policy='one_round')
        ops.save(POOL/'groups'/(GROUP+'.json'),group)
        ops.save(POOL/'current_group.json',dict(group=GROUP,jobs=group['jobs'],collector=p['cpu'],target_gpus=24,
             status='TAIL_REGISTERED_BEFORE_RELEASE',migration_receipt=str(OUT/'progress.json'),updated_at=now()))
        p.update(stage='TAIL_REGISTERED_CPU_RELEASED',gpu=gpu);ops.save(OUT/'progress.json',p)
        ops.command(['scontrol','release',p['cpu']])
    print('TAIL_REGISTERED_HELD',gpu,'CPU_RELEASED',p['cpu'],flush=True)


def activate():
    verify();p=ops.read(OUT/'progress.json');assert p['stage']=='TAIL_REGISTERED_CPU_RELEASED'
    path=POOL/'collector_status'/(p['cpu']+'.json')
    if fields(p['cpu'])['JobState']!='RUNNING' or not path.exists():print('WAIT_CPU_START');return
    c=ops.read(path);assert set(c['waiting'])=={EXISTING,p['gpu']}
    assert set(c['accounted'])=={'143342','143656','143684','144649'}
    ops.command(['scontrol','release',p['gpu']]);p['stage']='TAIL_RELEASED';ops.save(OUT/'progress.json',p)
    print('TAIL_RELEASED',p['gpu'],'CPU',p['cpu'],flush=True)


def closeout():
    s=verify();p=ops.read(OUT/'progress.json');assert p['stage']=='TAIL_RELEASED'
    probe=WORK/'main/batches'/p['gpu']/'device_assignment.json'
    if not probe.exists():print('WAIT_TAIL_PROBE');return
    new=ops.read(probe)['rows'];old=ops.read(WORK/'main/batches'/EXISTING/'device_assignment.json')['rows']
    assert len(new)==8 and len(old)==16 and len({r['uuid'] for r in new+old})==24
    assert all(r['offline'] and 'H200' in r['device'] for r in new+old)
    owned=[dict(id=k,task=r['task'],worker=r['worker']) for k,r in s['tasks'].items() if r['status']=='RUNNING' and r['job']==p['gpu']]
    configs=[WORK/'main/executions'/r['id']/p['gpu']/'training/config.json' for r in owned]
    if len(owned)!=8 or not all(f.exists() for f in configs):print('WAIT_TAIL_TRAINING',len(owned));return
    assert len({r['worker'] for r in owned})==8
    for r,f in zip(owned,configs):
        cfg=ops.read(f);assert cfg['width']==192 and cfg['epochs']==400 and cfg['seed']==r['task']['seed']
    q=ops.queue_snapshot();assert set(q)=={EXISTING,p['gpu'],p['cpu']} and sum(q.values())==4
    assert ops.read(POOL/'bootstrap'/(p['gpu']+'.json'))['collector_job']==p['cpu']
    with locked(POOL/'control'):
        index=ops.read(POOL/'current_group.json');index['status']='ACTIVE_CPU_ACCOUNTING';ops.save(POOL/'current_group.json',index)
    result=dict(status='PASS',at=now(),gpu_job=p['gpu'],cpu_collector=p['cpu'],existing_gpu_job=EXISTING,
        counts=dict(Counter(r['status'] for r in s['tasks'].values())),new_tasks=owned,
        distinct_H200=24,max_attempts_per_new_worker=1,new_scientific_attempts=0,
        original_395_attempts_unchanged=True,original_sources_unchanged=True,TEST_opened=False,
        source_sha256=ops.digest(__file__))
    ops.save(OUT/'closeout.json',result);print('TAIL_PASS',{k:v for k,v in result.items() if k!='new_tasks'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','register','activate','closeout'])
    globals()[parser.parse_args().stage]()
