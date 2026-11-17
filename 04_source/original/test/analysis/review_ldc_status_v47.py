"""Review all new immutable main results since the V44/V45 audits, on CPU."""
from collections import Counter
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import torch
import audit_ldc_main_v44 as audit
from ldc24_ops_v36 import ROOT, WORK, POOL, read, save, digest, verify, queue_snapshot
from ldc_main_scheduler_v33 import locked, choose_selection

OUT=ROOT/'results/ldc_status_20261107_v47'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);assert not torch.cuda.is_available()
    verify()
    base=ROOT/'results/ldc_full_audit_20261106_v44/final'
    previous=ROOT/'results/ldc_status_20261106_v45'
    assert read(base/'closeout.json')['status']==read(previous/'summary.json')['status']=='PASS'
    assert digest(audit.__file__)==read(base/'summary.json')['source_sha256']
    seen=set()
    for f in [base/'completed_trials.csv',previous/'new_completed_trials.csv']:
        with f.open() as stream:seen.update(r['id'] for r in csv.DictReader(stream))
    with locked(WORK/'main'):state=read(WORK/'main/state.json')
    save(OUT/'state_snapshot.json',state)
    protocol=read(WORK/'protocol.json');prepared=read(protocol['prepared_path'])
    split=read(protocol['split_path'])['indices'];norm=read(Path(prepared['cache'])/'normalization.json')
    expected={};paired={};new=[]
    for identifier,row in state['tasks'].items():
        if row['status']!='DONE' or identifier in seen:continue
        new.append(identifier)
        try:audit.audit_completed(identifier,row,state,protocol,split,norm,expected,paired)
        except Exception:
            import traceback
            audit.check(identifier+':review_exception',False,traceback.format_exc())
        if len(new)%20==0:print('NEW_MODELS_REVIEWED',len(new),'PROBLEMS',len(audit.problems),flush=True)
    pairs={}
    for identifier,row in state['tasks'].items():
        if row['status']!='DONE' or row['outcome']['status']!='COMPLETE':continue
        o=row['outcome'];pair=(o['initial_weight_sha256'],o['sample_sha256'])
        audit.check(identifier+':global_seed_pairing',pairs.setdefault(row['task']['seed'],pair)==pair)
    search=[r for r in state['tasks'].values() if r['task']['kind']=='search']
    audit.check('search:all240complete',len(search)==240 and all(r['status']=='DONE' and r['outcome']['status']=='COMPLETE' for r in search))
    selections=[]
    for method in protocol['methods']:
        for run in range(5):
            task=state['tasks'][f'{method}_r{run:02d}_k00']['task']
            selected=choose_selection(state,task)
            actual=read(WORK/'main/selections'/f'{method}_r{run:02d}.json')
            audit.check(f'selection:{method}:{run}',selected==actual)
            selections.append(dict(method=method,run=run,**actual))
    account=read(WORK/'scheduler/144649/summary.json');review=read(WORK/'main/batches/144649/review.json')
    audit.check('144649:completed_successfully',account['state']=='COMPLETED' and account['exit_code']=='0:0')
    audit.check('144649:all_review_checks',all(review['checks'].values()))
    group=read(POOL/'groups/group_000.json');cpu=group['active_collector']
    collector=read(POOL/'collector_status'/(cpu+'.json'));queue=queue_snapshot()
    audit.check('collector:live',cpu in queue and not collector['main_test_opened'])
    audit.check('collector:tracks_every_batch',set(group['jobs'])==set(collector['accounted'])|set(collector['waiting']))
    audit.check('state:no_attention',not state['attention_required'] and not (POOL/'hold.json').exists())
    audit.check('state:running_owned_by_active_jobs',all(r['job'] in queue for r in state['tasks'].values() if r['status']=='RUNNING'))
    verify()
    result=dict(status='PASS' if not audit.problems else 'REVIEW_REQUIRED',
        at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat(),counts=dict(Counter(r['status'] for r in state['tasks'].values())),
        by_kind={k:dict(Counter(r['status'] for r in state['tasks'].values() if r['task']['kind']==k)) for k in ['search','retrain','fixed']},
        previous_reviewed=len(seen),new_reviewed=len(new),checks_total=len(audit.checks),
        checks_passed=sum(c['pass_check'] for c in audit.checks),problems=len(audit.problems),
        all_search_complete=True,selected_configurations=len(selections),completed_accounting=account,
        completed_batch_review_checks=len(review['checks']),collector=collector,
        CPU_only=True,TEST_opened=False,new_training=False,source_sha256=digest(__file__))
    save(OUT/'summary.json',result);save(OUT/'checks.json',audit.checks);save(OUT/'problems.json',audit.problems)
    save(OUT/'artifact_hashes.json',audit.manifests);save(OUT/'selected_configurations.json',selections)
    if audit.rows:
        with (OUT/'new_completed_trials.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(audit.rows[0]));writer.writeheader();writer.writerows(audit.rows)
    lines=['STAV A KONTROLA PERUN — '+result['at'],'',result['status'],
        'Úloha144649: COMPLETED, exit0:0, všetkých3263pôvodných kontrol prešlo.',
        str(result['counts']),f"Nových jednotlivo preverených modelov: {len(new)}.",
        f"Kontroly: {result['checks_passed']}/{result['checks_total']}; nálezy: {len(audit.problems)}.",
        'Všetkých240 vyhľadávacích tréningov je hotových. Overených30výberov konfigurácie.',
        'Ostávajú finálne opakovania a fixed baseline. TEST ostáva uzavretý.',
        'Stav koncovej8GPU dávky a aktuálny súbeh sú v samostatnom výsledku ldc_tail_20261107_v47.']
    (OUT/'STAV.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REVIEW_RESULT',{k:v for k,v in result.items() if k not in ['collector','completed_accounting']},flush=True)
    return not audit.problems


if __name__=='__main__':raise SystemExit(0 if main() else 1)
