"""CPU-only incremental review of results added since the full V44 audit."""
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import audit_ldc_main_v44 as audit
from ldc24_ops_v36 import ROOT, WORK, POOL, read, save, digest, verify, queue_snapshot
from ldc_main_scheduler_v33 import locked

OUT=ROOT/'results/ldc_status_20261106_v45'
BASE=ROOT/'results/ldc_full_audit_20261106_v44/final'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    verify()
    assert read(BASE/'closeout.json')['status']=='PASS'
    assert digest(audit.__file__)==read(BASE/'summary.json')['source_sha256']
    with (BASE/'completed_trials.csv').open() as f:
        reviewed={r['id'] for r in csv.DictReader(f)}
    with locked(WORK/'main'):
        state=read(WORK/'main/state.json')
    save(OUT/'state_snapshot.json',state)
    protocol=read(WORK/'protocol.json');prep=read(protocol['prepared_path'])
    split=read(protocol['split_path'])['indices'];norm=read(Path(prep['cache'])/'normalization.json')
    expected={};paired={};new=[]
    for identifier,row in state['tasks'].items():
        if row['status']!='DONE' or identifier in reviewed:continue
        new.append(identifier)
        try:
            audit.audit_completed(identifier,row,state,protocol,split,norm,expected,paired)
        except Exception:
            import traceback
            audit.check(identifier+':incremental_auditor_exception',False,traceback.format_exc())
        if len(new)%10==0:print('NEW_COMPLETED_REVIEWED',len(new),'PROBLEMS',len(audit.problems),flush=True)
    pairs={}
    for identifier,row in state['tasks'].items():
        if row['status']!='DONE' or row['outcome']['status']!='COMPLETE':continue
        o=row['outcome'];pair=(o['initial_weight_sha256'],o['sample_sha256'])
        audit.check(identifier+':pairing_across_all_completed',pairs.setdefault(row['task']['seed'],pair)==pair)
    accounts=[]
    for job in ['143656','143684']:
        account=read(WORK/'scheduler'/job/'summary.json')
        review=read(WORK/'main/batches'/job/'review.json')
        audit.check(job+':terminal_success',account['state']=='COMPLETED' and account['exit_code']=='0:0')
        audit.check(job+':all_batch_checks',all(review['checks'].values()))
        accounts.append(dict(job=job,accounting=account,review_status=review['status'],
             review_checks=len(review['checks']),review_sha256=digest(WORK/'main/batches'/job/'review.json')))
    queue=queue_snapshot()
    audit.check('live:owners_active',all(r['job'] in queue for r in state['tasks'].values() if r['status']=='RUNNING'))
    collector=read(POOL/'collector_status/144648.json')
    audit.check('collector:correctly_waiting',set(collector['accounted'])=={'143342','143656','143684'}
                and collector['waiting']==['144649'] and '144648' in queue)
    audit.check('live:no_attention_or_hold',not state['attention_required'] and not (POOL/'hold.json').exists())
    audit.check('plan:unchanged',len(state['tasks'])==395)
    verify()
    counts=dict(Counter(r['status'] for r in state['tasks'].values()))
    result=dict(status='PASS' if not audit.problems else 'REVIEW_REQUIRED',
        observed_at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat(),counts=counts,
        by_kind={k:dict(Counter(r['status'] for r in state['tasks'].values() if r['task']['kind']==k)) for k in ['search','retrain','fixed']},
        completed_outcomes=dict(Counter(r['outcome']['status'] for r in state['tasks'].values() if r['status']=='DONE')),
        newly_reviewed=len(new),previously_reviewed=len(reviewed),
        checks_total=len(audit.checks),checks_passed=sum(c['pass_check'] for c in audit.checks),
        problem_count=len(audit.problems),completed_job_accounts=accounts,collector=collector,
        running_owners=dict(Counter(r['job'] for r in state['tasks'].values() if r['status']=='RUNNING')),
        new_training=False,TEST_opened=False,larger_network_training=False,CPU_only=True,source_sha256=digest(__file__))
    save(OUT/'summary.json',result);save(OUT/'checks.json',audit.checks)
    save(OUT/'problems.json',audit.problems);save(OUT/'artifact_hashes.json',audit.manifests)
    if audit.rows:
        with (OUT/'new_completed_trials.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(audit.rows[0]));writer.writeheader();writer.writerows(audit.rows)
    lines=['KONTROLA STAVU PERUN — '+result['observed_at'],'',result['status'],
        'Skončili dávky143656 a143684: COMPLETED, exit0:0, všetky kontroly prešli.',
        f"Hotové: {counts.get('DONE',0)}/395; beží: {counts.get('RUNNING',0)}; čaká: {counts.get('PENDING',0)}.",
        f"Od kompletného auditu pribudlo {len(new)} dokončených behov; všetky jednotlivo preverené na CPU.",
        f"Dodatočné kontroly: {result['checks_passed']}/{result['checks_total']}; kritické nálezy: {len(audit.problems)}.",
        'Overené aj finálne opakovania: výber podľa pôvodnej VAL, správny finálny seed, čerstvé váhy,400epoch.',
        'Váhy, obnovovací stav, raw latencia, LLM návrhy, sampler replay a fyzikálne záznamy boli skontrolované.',
        'GPU dávka144649 pokračuje s8GPU; CPU zberač144648 čaká na jej koniec.',
        'Automatický pokračovací mechanizmus je aktívny. V tejto kontrole sa nespúšťala ďalšia GPU dávka.',
        'TEST ostáva uzavretý; väčšia sieť sa nespúšťa. Váhy a veľké polia zostávajú na Perune.']
    (OUT/'STAV.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ['completed_job_accounts','collector']}),flush=True)
    return not audit.problems


if __name__=='__main__':
    raise SystemExit(0 if main() else 1)
