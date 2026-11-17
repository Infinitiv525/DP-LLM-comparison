"""Locked persistent dependency queue; one new training per logical attempt."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import time
from ldc_main_common_v33 import read, save

METHODS = ['qwen3_8b', 'ministral3_8b', 'phi4_mini', 'granite33_8b', 'random', 'tpe']


def search_id(method, run, candidate):
    return f'{method}_r{run:02d}_c{candidate:02d}'


def make_plan(phase):
    runs, candidates, repeats = (5, 8, 5) if phase == 'main' else (1, 3, 1)
    tasks = []
    for run in range(runs):
        for candidate in range(candidates):
            for method in METHODS:
                identifier = search_id(method, run, candidate)
                tasks.append(dict(id=identifier, kind='search', method=method, run=run, candidate=candidate,
                    seed=(140001 if phase=='main' else 320001)+run,
                    dependencies=[] if candidate==0 else [search_id(method,run,candidate-1)]))
    for run in range(runs):
        for repeat in range(repeats):
            for method in METHODS:
                tasks.append(dict(id=f'{method}_r{run:02d}_k{repeat:02d}', kind='retrain', method=method,
                    run=run, repeat=repeat, seed=(150001 if phase=='main' else 330001)+repeat,
                    dependencies=[search_id(method,run,c) for c in range(candidates)]))
    for repeat in range(repeats):
        tasks.append(dict(id=f'fixed_k{repeat:02d}', kind='fixed', method='fixed', repeat=repeat,
                          seed=(150001 if phase=='main' else 330001)+repeat, dependencies=[]))
    return tasks


@contextmanager
def locked(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'scheduler.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def initialize_state(directory, plan):
    directory=Path(directory)
    with locked(directory):
        if (directory/'state.json').exists():
            raise ValueError('Existing experiment state')
        save(directory/'state.json', dict(tasks={t['id']:dict(status='PENDING',task=t) for t in plan},
                                         attention_required=False, events=[]))


def choose_selection(state, task):
    candidates = [state['tasks'][i] for i in task['dependencies']]
    successful = [r for r in candidates if r.get('outcome',{}).get('status')=='COMPLETE']
    if not successful:
        return None
    best = min(successful,key=lambda r:(r['outcome']['history']['score'],r['task']['candidate']))
    return dict(source_task=best['task']['id'], parameters=best['outcome']['history']['parameters'],
                score=best['outcome']['history']['score'],
                trial_summary_sha256=best['outcome']['trial_summary_sha256'])


def claim(directory, worker, job):
    directory=Path(directory)
    with locked(directory):
        state=read(directory/'state.json')
        if state['attention_required']:
            return None, 'ATTENTION_REQUIRED'
        changed=False
        for identifier,row in state['tasks'].items():
            task=row['task']
            if row['status']!='PENDING' or any(state['tasks'][x]['status']!='DONE' for x in task['dependencies']):
                continue
            selection=None
            if task['kind']=='retrain':
                selection=choose_selection(state,task)
                if selection is None:
                    row.update(status='DONE',outcome=dict(status='NO_VALID_CONFIGURATION'));changed=True
                    continue
                path=directory/'selections'/f'{task["method"]}_r{task["run"]:02d}.json'
                if path.exists() and read(path)!=selection:
                    raise ValueError('Previously selected configuration changed')
                if not path.exists():
                    save(path,selection)
            row.update(status='RUNNING',worker=worker,job=str(job),claimed_unix=time.time())
            history=[]
            if task['kind']=='search':
                for candidate in range(task['candidate']):
                    previous=state['tasks'][search_id(task['method'],task['run'],candidate)]
                    history.append(previous['outcome']['history'])
            save(directory/'state.json',state)
            return dict(task=task,history=history,selection=selection), 'CLAIMED'
        if changed:save(directory/'state.json',state)
        return None, 'ALL_DONE' if all(r['status']=='DONE' for r in state['tasks'].values()) else 'WAITING'


def finish(directory, identifier, worker, job, outcome, attention=False):
    directory=Path(directory)
    with locked(directory):
        state=read(directory/'state.json');row=state['tasks'][identifier]
        if row['status']!='RUNNING' or row['worker']!=worker or row['job']!=str(job):
            raise ValueError('Task ownership mismatch')
        row.update(status='ATTENTION' if attention else 'DONE',outcome=outcome,finished_unix=time.time())
        if attention:
            state['attention_required']=True
        save(directory/'state.json',state)
