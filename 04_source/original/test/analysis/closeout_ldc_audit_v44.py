"""Finish the CPU audit, verify live progress, repair only the stale index."""
from collections import Counter
from datetime import datetime
import csv
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

import torch
import audit_ldc_main_v44 as audit
from ldc24_ops_v36 import ROOT, WORK, POOL, read, save, digest, verify, queue_snapshot
from ldc_main_scheduler_v33 import locked, choose_selection
from ldc_main_contract_v33 import training_config

OUT=audit.OUT


def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    assert read(OUT/'summary.json')['status']=='PASS'
    assert not (OUT/'closeout.json').exists()
    verify()
    with (OUT/'completed_trials.csv').open() as f:
        original_rows=list(csv.DictReader(f))
    seen={r['id'] for r in original_rows}
    with locked(WORK/'main'):
        state=read(WORK/'main/state.json')
    protocol=read(WORK/'protocol.json')
    prep=read(protocol['prepared_path']); split=read(protocol['split_path'])['indices']
    norm=read(Path(prep['cache'])/'normalization.json')
    for identifier,row in state['tasks'].items():
        if row['status']=='DONE' and identifier not in seen:
            audit.audit_completed(identifier,row,state,protocol,split,norm,{}, {})
    # Cross-run pairing of any newly completed trials is checked against the
    # original snapshot, not only against one another.
    pairs={}
    for row in state['tasks'].values():
        if row['status']!='DONE' or row['outcome']['status']!='COMPLETE':continue
        s=read(row['outcome']['trial_summary_path'])
        pair=(s['initial_weight_sha256'],s['sample_sha256'])
        audit.check(row['task']['id']+':cross_snapshot_seed_pairing',pairs.setdefault(s['seed'],pair)==pair)
    live=[]
    for identifier,row in state['tasks'].items():
        if row['status']!='RUNNING':continue
        folder=WORK/'main/executions'/identifier/row['job']
        request=read(folder/'request.json'); config=read(folder/'training/config.json')
        audit.check(identifier+':live_config',config==training_config(protocol['base_model'],request['parameters'],'main',row['task']['seed']))
        hp=folder/'training/history.json'
        history=read(hp) if hp.exists() else []
        age=time.time()-hp.stat().st_mtime if hp.exists() else time.time()-row['claimed_unix']
        audit.check(identifier+':live_progress_present',bool(history) and 0<=history[-1]['update']<=64000)
        audit.check(identifier+':progress_not_stale',age<1800,dict(seconds_since_assessment=age))
        log=(folder/'training.log').read_text(errors='replace')
        errors=[s for s in ['Traceback (most recent call last)','OutOfMemoryError','CUDA error:',
                            'Nonfinite loss','Segmentation fault'] if s in log]
        audit.check(identifier+':no_live_runtime_error',not errors,errors)
        if row['task']['kind']=='retrain':
            selection=choose_selection(state,row['task'])
            audit.check(identifier+':live_final_selection',selection==read(folder/'selection.json')
                        and request['parameters']==selection['parameters'])
        live.append(dict(id=identifier,job=row['job'],worker=row['worker'],kind=row['task']['kind'],
              seed=row['task']['seed'],latest_update=history[-1]['update'] if history else None,
              equivalent_epoch=history[-1]['equivalent_epoch'] if history else None,
              assessment_age_seconds=age,physics_weight=config['physics_weight'],bc_weight=config['bc_weight']))
    selections=[]
    for p in (WORK/'main/selections').glob('*.json'):
        m=re.fullmatch(r'(.+)_r(\d+)',p.stem)
        assert m
        task=next(r['task'] for r in state['tasks'].values() if r['task']['kind']=='retrain'
                  and r['task']['method']==m[1] and r['task']['run']==int(m[2]))
        expected=choose_selection(state,task)
        audit.check('selection:'+p.stem,expected==read(p))
        selections.append(dict(method=m[1],run=int(m[2]),**expected))
    unit_log=ROOT/'results/ldc_full_audit_20261106_v44/continuation_tests.log'
    text=unit_log.read_text()
    audit.check('future:12_continuation_tests', 'Ran 12 tests' in text and '\nOK\n' in text)
    audit.check('live:no_review_hold',not state['attention_required'] and not (POOL/'hold.json').exists())
    save(OUT/'incremental_checks.json',audit.checks)
    save(OUT/'incremental_problems.json',audit.problems)
    save(OUT/'live_progress.json',live)
    save(OUT/'frozen_selections.json',selections)
    save(OUT/'incremental_artifact_hashes.json',audit.manifests)
    if audit.rows:
        with (OUT/'new_completed_trials.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(audit.rows[0]));writer.writeheader();writer.writerows(audit.rows)
    if audit.problems:
        print('CLOSEOUT_REVIEW_REQUIRED',audit.problems,flush=True)
        return False
    # This file is a mutable operational bookmark. Preserve exact old bytes
    # before correcting its references; no trained result or science changes.
    with locked(POOL/'control'):
        path=POOL/'current_group.json'
        old_bytes=path.read_bytes()
        (OUT/'operational_index_before.json').write_bytes(old_bytes)
        group=read(POOL/'groups/group_000.json')
        cpu=group['active_collector']
        collector=read(POOL/'collector_status'/(cpu+'.json'))
        assert set(group['jobs'])==set(collector['accounted'])|set(collector['waiting'])
        updated=dict(group='group_000',jobs=group['jobs'],collector=cpu,target_gpus=24,
             status='ACTIVE_CPU_ACCOUNTING',updated_at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat(),
             audit_receipt=str(OUT/'operational_index_repair.json'))
        before=digest(path)
        save(path,updated)
        save(OUT/'operational_index_repair.json',dict(old_sha256=before,new_sha256=digest(path),
             only_mutable_overview_changed=True,authoritative_group_not_modified=True,
             active_collector=cpu,jobs=group['jobs'],verified=read(path)==updated))
    verify()
    result=dict(status='PASS',at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat(),
         base_checks=read(OUT/'summary.json')['checks_total'],additional_checks=len(audit.checks),
         checks_total=read(OUT/'summary.json')['checks_total']+len(audit.checks),
         completed_audited=len(original_rows)+len(audit.rows),snapshot_counts=dict(Counter(r['status'] for r in state['tasks'].values())),
         live_progress_checked=len(live),frozen_selections_checked=len(selections),
         continuation_unit_tests=12,continuation_log_sha256=digest(unit_log),
         operational_index_corrected=True,critical_findings=0,new_training=False,TEST_opened=False,
         CPU_only=True,source_sha256=digest(__file__),main_sources_unchanged=True)
    save(OUT/'closeout.json',result)
    print('CLOSEOUT',result,flush=True)
    return True


if __name__=='__main__':
    raise SystemExit(0 if main() else 1)
