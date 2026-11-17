"""CPU review of the final new training results and one-round GPU release."""
from collections import Counter
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import torch
import audit_ldc_main_v44 as audit
from ldc24_ops_v36 import ROOT, WORK, POOL, read, save, digest, verify, queue_snapshot
from ldc_main_scheduler_v33 import locked

OUT=ROOT/'results/ldc_status_20261108_v48'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);assert not torch.cuda.is_available()
    verify()
    base=ROOT/'results/ldc_full_audit_20261106_v44/final'
    assert read(base/'closeout.json')['status']=='PASS'
    assert digest(audit.__file__)==read(base/'summary.json')['source_sha256']
    previous=[(base/'completed_trials.csv',base/'summary.json'),
        (ROOT/'results/ldc_status_20261106_v45/new_completed_trials.csv',ROOT/'results/ldc_status_20261106_v45/summary.json'),
        (ROOT/'results/ldc_status_20261107_v47/new_completed_trials.csv',ROOT/'results/ldc_status_20261107_v47/summary.json')]
    seen=set()
    for path,proof in previous:
        assert read(proof)['status']=='PASS'
        with path.open() as f:seen.update(r['id'] for r in csv.DictReader(f))
    with locked(WORK/'main'):state=read(WORK/'main/state.json')
    save(OUT/'state_snapshot.json',state)
    protocol=read(WORK/'protocol.json');prepared=read(protocol['prepared_path'])
    split=read(protocol['split_path'])['indices'];norm=read(Path(prepared['cache'])/'normalization.json')
    new=[];initial={};paired={}
    for identifier,row in state['tasks'].items():
        if row['status']!='DONE' or identifier in seen:continue
        new.append(identifier)
        try:audit.audit_completed(identifier,row,state,protocol,split,norm,initial,paired)
        except Exception:
            import traceback
            audit.check(identifier+':review_exception',False,traceback.format_exc())
        if len(new)%10==0:print('NEW_MODELS_REVIEWED',len(new),'PROBLEMS',len(audit.problems),flush=True)
    pairs={}
    for identifier,row in state['tasks'].items():
        if row['status']!='DONE' or row['outcome']['status']!='COMPLETE':continue
        o=row['outcome'];pair=(o['initial_weight_sha256'],o['sample_sha256'])
        audit.check(identifier+':global_seed_pairing',pairs.setdefault(row['task']['seed'],pair)==pair)
    account=read(WORK/'scheduler/145597/summary.json')
    review=read(WORK/'main/batches/145597/review.json')
    audit.check('145597:completed',account['state']=='COMPLETED' and account['exit_code']=='0:0')
    audit.check('145597:batch_checks',all(review['checks'].values()))
    tail=ROOT/'results/ldc_tail_20261107_v47'
    freeze=read(tail/'freeze.json')
    audit.check('tail:supplement_unchanged',all(digest(p)==h for p,h in freeze['files'].items()))
    policy=[]
    for rank in range(8):
        row=read(WORK/f'main/batches/145597/one_round/rank_{rank}.json')
        worker=read(WORK/f'main/batches/145597/worker_{rank}.json')
        audit.check(f'tail:rank{rank}:one_original_attempt',row['rank']==rank
            and worker['completed']==[row['task']] and row['stop_reason']=='ONE_ATTEMPT_ASSIGNED'
            and row['operational_freeze_sha256']==digest(tail/'freeze.json')
            and not row['main_test_opened'])
        task=state['tasks'][row['task']]
        audit.check(f'tail:rank{rank}:completed',task['status']=='DONE' and task['job']=='145597'
            and task['worker']==rank and task['outcome']['status']=='COMPLETE')
        policy.append(row)
    audit.check('tail:exactly8unique_attempts',len({r['task'] for r in policy})==8)
    queue=queue_snapshot()
    audit.check('tail:GPU_allocation_released','145597' not in queue)
    group=read(POOL/'groups/group_000.json');cpu=group['active_collector']
    collector=read(POOL/'collector_status'/(cpu+'.json'))
    audit.check('collector:active',cpu in queue and not collector['main_test_opened'])
    audit.check('collector:all_jobs_tracked',set(group['jobs'])==set(collector['accounted'])|set(collector['waiting']))
    audit.check('collector:tail_accounted','145597' in collector['accounted'])
    live=[]
    for identifier,row in state['tasks'].items():
        if row['status']!='RUNNING':continue
        audit.check(identifier+':owner_active',row['job'] in queue)
        folder=WORK/'main/executions'/identifier/row['job']/'training'
        history=read(folder/'history.json');config=read(folder/'config.json')
        audit.check(identifier+':live_full_config',config['width']==192 and config['epochs']==400 and config['seed']==row['task']['seed'])
        live.append(dict(id=identifier,kind=row['task']['kind'],job=row['job'],seed=row['task']['seed'],
            latest_update=history[-1]['update'],equivalent_epoch=history[-1]['equivalent_epoch']))
    audit.check('state:no_attention_or_hold',not state['attention_required'] and not (POOL/'hold.json').exists())
    audit.check('state:all395started',len(state['tasks'])==395 and not any(r['status']=='PENDING' for r in state['tasks'].values()))
    verify()
    result=dict(status='PASS' if not audit.problems else 'REVIEW_REQUIRED',
        at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat(),counts=dict(Counter(r['status'] for r in state['tasks'].values())),
        by_kind={k:dict(Counter(r['status'] for r in state['tasks'].values() if r['task']['kind']==k)) for k in ['search','retrain','fixed']},
        previous_reviewed=len(seen),new_reviewed=len(new),checks_total=len(audit.checks),
        checks_passed=sum(c['pass_check'] for c in audit.checks),problems=len(audit.problems),
        completed_job=account,completed_review_checks=len(review['checks']),one_round_records=policy,
        live_trainings=live,collector=collector,CPU_only=True,new_training=False,TEST_opened=False,
        source_sha256=digest(__file__))
    save(OUT/'summary.json',result);save(OUT/'checks.json',audit.checks);save(OUT/'problems.json',audit.problems)
    save(OUT/'artifact_hashes.json',audit.manifests)
    if audit.rows:
        with (OUT/'new_completed_trials.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(audit.rows[0]));writer.writeheader();writer.writerows(audit.rows)
    lines=['STAV PERUN — '+result['at'],'',result['status'],'Úloha145597 skončila COMPLETED/0:0.',
       'Overených8pôvodných tréningov, presne jeden na pracovníka. GPU alokácia je uvoľnená.',
       str(result['counts']),f"Nových modelov preverených naCPU: {len(new)}.",
       f"Kontroly {result['checks_passed']}/{result['checks_total']}, nálezy {len(audit.problems)}.",
       'Nič ďalšie nečaká na spustenie. Ostávajú3finálne opakovania a5fixed baseline.',
       'Po ich dobehnutí sa skončí pôvodná tréningová etapa395pokusov.',
       'TEST zatiaľ uzavretý. Žiadna nová GPU úloha, väčšia sieť alebo pilot spotreby.']
    (OUT/'STAV.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REVIEW_RESULT',{k:v for k,v in result.items() if k not in ['one_round_records','live_trainings','collector','completed_job']},flush=True)
    print('LIVE_PROGRESS',live,flush=True)
    return not audit.problems


if __name__=='__main__':raise SystemExit(0 if main() else 1)
