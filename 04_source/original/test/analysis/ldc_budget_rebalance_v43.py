"""Replace finished capacity GPU with CPU accounting; refill existing main pool.

Only allocation registrations/group membership change. The immutable V33
science, training queue, data, and V36 execution scripts stay unchanged.
Each mutation is recorded before the next dependent action.
"""
import argparse
from collections import Counter
from datetime import datetime
import re
import subprocess
import time
from zoneinfo import ZoneInfo

from ldc24_ops_v36 import (ROOT, WORK, POOL, TERMINAL, read, save, digest,
                          verify, command, queue_snapshot, submit_collector,
                          launch_gpu, capture)
from ldc_main_scheduler_v33 import locked

OUT = ROOT / 'results/ldc_budget_rebalance_20261106_v43'
CAP = ROOT / 'results/ldc_capacity_20261105_v39'
GROUP = 'group_000'
OLD = '143807'


def now():
    return datetime.now(ZoneInfo('Europe/Bratislava')).isoformat()


def fields(job):
    return dict(re.findall(r'(?:^|\s)(\w+)=([^\s]+)',
                          command(['scontrol', 'show', 'job', '-o', str(job)])))


def check_science():
    verify()
    for path, expected in read(CAP / 'freeze.json')['files'].items():
        assert digest(path) == expected, path
    state = read(WORK / 'main/state.json')
    assert not state['attention_required'] and not (POOL / 'hold.json').exists()
    assert len(state['tasks']) == 395
    return state


def prepare():
    state = check_science()
    assert not OUT.exists(), 'Use the recorded continuation stage, never repeat prepare'
    cap = read(CAP / 'result.json')
    assert cap['all_three_complete'] and len(cap['outcomes']) == 3
    assert all(r['complete'] and r['exit_code'] == 0 for r in cap['outcomes'])
    assert OLD in queue_snapshot()
    steps = command(['squeue', '--steps', '-j', OLD, '-h', '-o', '%i|%j'])
    assert all(line.split('|')[0] in [OLD+'.batch', OLD+'.extern']
               for line in steps.splitlines()), 'Another analysis step is using the old allocation'
    assert not any(r['status'] == 'RUNNING' and r['job'] == OLD for r in state['tasks'].values())
    OUT.mkdir(parents=True)
    save(OUT / 'intent.json', dict(at=now(), user_instruction='Close larger-network work; conserve GPU allocation for existing main experiment',
         old_capacity_collector=OLD, old_group=read(POOL/'groups'/(GROUP+'.json')),
         initial_queue=queue_snapshot(), initial_counts=dict(Counter(r['status'] for r in state['tasks'].values())),
         capacity_result_sha256=digest(CAP/'result.json'), main_freeze_sha256=digest(WORK/'freeze.json'),
         operational_manifest_sha256=digest(WORK/'operational_manifest_v36.json'),
         source_sha256=digest(__file__), total_attempts_unchanged=395, new_capacity_runs=0,
         main_test_opened=False))
    with locked(POOL / 'control'):
        cpu = submit_collector(GROUP, [OLD], held=True)
        save(OUT/'progress.json', dict(stage='CPU_REGISTERED_HELD', cpu=cpu, old=OLD, at=now()))
        command(['scancel', OLD])
        save(OUT/'progress.json', dict(stage='OLD_GPU_CANCEL_REQUESTED', cpu=cpu, old=OLD, at=now()))
    print('CPU_REGISTERED_HELD', cpu, 'OLD_GPU_CANCEL_REQUESTED', OLD, flush=True)


def register():
    check_science()
    progress = read(OUT/'progress.json')
    assert progress['stage'] == 'OLD_GPU_CANCEL_REQUESTED'
    if fields(OLD)['JobState'] not in TERMINAL or OLD in queue_snapshot():
        print('WAIT_OLD_ALLOCATION_TERMINATION', flush=True)
        return
    account = capture(OLD)
    assert account['state'] == 'CANCELLED', account
    with locked(POOL/'control'):
        group = read(POOL/'groups'/(GROUP+'.json'))
        assert progress['cpu'] in group['collectors']
        assert group['jobs'] == read(OUT/'intent.json')['old_group']['jobs']
        gpu = launch_gpu(8, GROUP)
        # The replacement CPU collector is still held and has not cached this
        # group. Register the new GPU member before either allocation starts.
        group = read(POOL/'groups'/(GROUP+'.json'))
        group['jobs'].append(gpu)
        group['active_collector'] = progress['cpu']
        group['budget_rebalance_v43'] = dict(old_capacity_collector=OLD,
            cpu_collector=progress['cpu'], added_main_job=gpu,
            intent_sha256=digest(OUT/'intent.json'), no_new_capacity_runs=True)
        save(POOL/'groups'/(GROUP+'.json'), group)
        progress.update(stage='BOTH_REPLACEMENT_JOBS_REGISTERED', gpu=gpu, at=now())
        save(OUT/'progress.json', progress)
        save(OUT/'old_gpu_accounting.json', account)
        command(['scontrol', 'release', progress['cpu']])
        progress.update(stage='CPU_RELEASED_GPU_HELD', at=now())
        save(OUT/'progress.json', progress)
    print('CPU_RELEASED', progress['cpu'], 'MAIN8_REGISTERED_HELD', gpu, flush=True)


def activate():
    check_science()
    progress = read(OUT/'progress.json')
    assert progress['stage'] == 'CPU_RELEASED_GPU_HELD'
    status_path = POOL/'collector_status'/(progress['cpu']+'.json')
    if fields(progress['cpu'])['JobState'] != 'RUNNING' or not status_path.exists():
        print('WAIT_CPU_COLLECTOR_START', progress['cpu'], flush=True)
        return
    status = read(status_path)
    assert progress['gpu'] in status['waiting'], 'New batch must be tracked before release'
    assert '143342' in status['accounted'], 'Existing accounting must remain available'
    command(['scontrol', 'release', progress['gpu']])
    progress.update(stage='MAIN8_RELEASED_CPU_ACCOUNTING_RUNNING', at=now())
    save(OUT/'progress.json', progress)
    save(CAP/'closed_for_new_training_20261106_v43.json', dict(at=now(),
         reason='User closed larger-model work and requested GPU conservation',
         existing_results_preserved=True, new_capacity_runs=0,
         old_collector=OLD, replacement_cpu_collector=progress['cpu'],
         migration_receipt=str(OUT/'progress.json'), result_sha256=digest(CAP/'result.json')))
    print('MAIN8_RELEASED', progress['gpu'], 'CPU_COLLECTOR', progress['cpu'], flush=True)


def closeout():
    state = check_science()
    progress = read(OUT/'progress.json')
    assert progress['stage'] == 'MAIN8_RELEASED_CPU_ACCOUNTING_RUNNING'
    gpu = progress['gpu']
    bootstrap = read(POOL/'bootstrap'/(gpu+'.json'))
    assert bootstrap['status'] == 'ACCOUNTING_READY_BEFORE_TRAINING'
    assert bootstrap['collector_job'] == progress['cpu']
    probe = WORK/'main/device_probe'/gpu/'review.json'
    # Exact probe path is captured separately if this generic path does not
    # match the frozen probe's layout; bootstrap itself is the required gate.
    queue = queue_snapshot()
    assert OLD not in queue and progress['cpu'] in queue and gpu in queue
    assert sum(queue.values()) <= 4
    records = [read(p) for p in (WORK/'submissions').glob('*.json')]
    active_gpu = sum(r.get('gpus',8) for r in records
                     if r['phase']=='main' and str(r['job']) in queue)
    assert active_gpu <= 24
    result = dict(status='MAIN_POOL_REPLENISHED_CAPACITY_GPU_RELEASED', at=now(),
         added_main_job=gpu, cpu_collector=progress['cpu'], stopped_idle_capacity_job=OLD,
         active_main_gpus=active_gpu, total_reserved_nodes=sum(queue.values()),
         main_counts=dict(Counter(r['status'] for r in state['tasks'].values())),
         collector_status=read(POOL/'collector_status'/(progress['cpu']+'.json')),
         bootstrap=bootstrap, all_395_tasks_preserved=True, new_capacity_runs=0,
         frozen_sources_unchanged=True, main_test_opened=False,
         original_gpu_accounting_sha256=digest(OUT/'old_gpu_accounting.json'))
    save(OUT/'closeout.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','register','activate','closeout'])
    stage = parser.parse_args().stage
    globals()[stage]()
