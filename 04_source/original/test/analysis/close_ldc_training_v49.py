"""CPU-only final training audit and 155-model manifest, before any TEST access."""
from collections import Counter
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import torch
import audit_ldc_main_v44 as audit
from ldc24_ops_v36 import ROOT,WORK,POOL,read,save,digest,verify,capture
from ldc_main_scheduler_v33 import locked,choose_selection

FINAL=ROOT/'results/ldc_final_test_20261108_v49'
OUT=FINAL/'training_closeout'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);assert not torch.cuda.is_available()
    verify()
    base=ROOT/'results/ldc_full_audit_20261106_v44/final'
    records=[(base/'completed_trials.csv',base/'summary.json')]
    for folder in ['ldc_status_20261106_v45','ldc_status_20261107_v47','ldc_status_20261108_v48']:
        p=ROOT/'results'/folder;records.append((p/'new_completed_trials.csv',p/'summary.json'))
    seen=set()
    for path,proof in records:
        assert read(proof)['status']=='PASS'
        with path.open() as f:seen.update(r['id'] for r in csv.DictReader(f))
    with locked(WORK/'main'):state=read(WORK/'main/state.json')
    assert len(state['tasks'])==395 and all(r['status']=='DONE' and r['outcome']['status']=='COMPLETE' for r in state['tasks'].values())
    assert not state['attention_required'] and not (POOL/'hold.json').exists()
    protocol=read(WORK/'protocol.json');prep=read(protocol['prepared_path'])
    split=read(protocol['split_path'])['indices'];normalization=read(Path(prep['cache'])/'normalization.json')
    initial={};paired={};new=[]
    for identifier,row in state['tasks'].items():
        if identifier in seen:continue
        new.append(identifier)
        audit.audit_completed(identifier,row,state,protocol,split,normalization,initial,paired)
    pairs={}
    for identifier,row in state['tasks'].items():
        o=row['outcome'];pair=(o['initial_weight_sha256'],o['sample_sha256'])
        audit.check(identifier+':global_seed_pairing',pairs.setdefault(row['task']['seed'],pair)==pair)
    review=read(WORK/'main/batches/145176/review.json')
    audit.check('training:final_review',review['status']=='LDC_MAIN_TRAINING_COMPLETE' and all(review['checks'].values()))
    completion=read(POOL/'training_complete.json')
    audit.check('training:sealed_completion',completion['status']=='TRAINING_STAGE_COMPLETE_TEST_PENDING'
          and not completion['main_test_opened'] and completion['state_sha256']==digest(WORK/'main/state.json'))
    # Capture the final tiny CPU accounting task too; no additional GPU work.
    final_accounting=str(completion['final_accounting_job'])
    if not (WORK/'scheduler'/final_accounting/'summary.json').exists():capture(final_accounting)
    ledger=[]
    for job in read(POOL/'groups/group_000.json')['jobs']:
        account=read(WORK/'scheduler'/job/'summary.json')
        audit.check('accounting:'+job,account['state']=='COMPLETED' and account['exit_code']=='0:0')
        ledger.append(account)
    models=[]
    for identifier,row in state['tasks'].items():
        task=row['task']
        if task['kind'] not in ['retrain','fixed']:continue
        summary_path=Path(row['outcome']['trial_summary_path']);folder=summary_path.parent
        summary=read(summary_path);checkpoint=folder/'best.pt'
        audit.check(identifier+':final_weight_hash',digest(checkpoint)==summary['checkpoint_sha256'])
        selection=choose_selection(state,task) if task['kind']=='retrain' else None
        if selection is not None:
            audit.check(identifier+':final_selection',selection==read(folder.parent/'selection.json'))
        models.append(dict(id=identifier,kind=task['kind'],method=task['method'],search_run=task.get('run'),
            repeat=task['repeat'],seed=task['seed'],checkpoint=str(checkpoint),checkpoint_sha256=summary['checkpoint_sha256'],
            training_summary=str(summary_path),training_summary_sha256=digest(summary_path),
            config=read(folder/'config.json'),normalization_sha256=digest(Path(prep['cache'])/'normalization.json'),
            selected_by_VAL=selection))
    audit.check('final:155models',len(models)==155 and len({m['id'] for m in models})==155)
    for method in protocol['methods']:
        for run in range(5):
            group=[m for m in models if m['method']==method and m['search_run']==run]
            audit.check(f'final:{method}:{run}',len(group)==5 and sorted(m['seed'] for m in group)==list(range(150001,150006)))
    fixed=[m for m in models if m['kind']=='fixed']
    audit.check('final:5fixed',len(fixed)==5 and sorted(m['seed'] for m in fixed)==list(range(150001,150006)))
    verify()
    save(OUT/'checks.json',audit.checks);save(OUT/'problems.json',audit.problems)
    save(OUT/'artifact_hashes.json',audit.manifests);save(OUT/'main_gpu_accounting.json',ledger)
    if audit.problems:raise RuntimeError('Final training audit requires review: '+str(audit.problems))
    save(FINAL/'models.json',dict(models=models,main_state_sha256=digest(WORK/'main/state.json'),
         main_freeze_sha256=digest(WORK/'freeze.json'),normalization=normalization,TEST_opened=False))
    result=dict(status='TRAINING_AUDITED_MODEL_MANIFEST_READY',at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat(),
        completed_trainings=395,newly_audited=len(new),checks=len(audit.checks),problems=0,
        final_models=155,models_sha256=digest(FINAL/'models.json'),
        main_gpu_allocated_seconds=sum(r['allocated_gpu_seconds'] for r in ledger),
        CPU_only=True,TEST_opened=False,source_sha256=digest(__file__))
    save(OUT/'summary.json',result)
    (OUT/'VYSLEDOK.txt').write_text('Všetkých395tréningov úspešne uzavretých.\n'
        'Posledné modely prešli CPU auditom. Manifest155finálnych sietí je pripravený.\n'
        'TEST ešte neotvorený; evaluator a protokol sa musia samostatne zmraziť.\n',encoding='utf-8')
    print(result,flush=True)


if __name__=='__main__':main()
