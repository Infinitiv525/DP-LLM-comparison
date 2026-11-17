"""Verify actual feedback, paired trials and provenance, without opening TEST."""
import hashlib
from pathlib import Path
import sys
import zipfile
from ldc_llm_common_v29 import WORK,read,save,digest,verify
from ldc_contract_v29 import messages,canonical,parse_reply,SYSTEM_SHA256


def run():
    verify();protocol=read(WORK/'protocol.json');common=read(WORK/'common/summary.json');checks={};rows=[]
    checks['common_success']=common['status']=='LDC_CONFIGURATOR_TECHNICAL_TRIAL_COMPLETE'
    assignment=read(WORK/'device_assignment.json')
    checks['four_distinct_offline_gpus']=(assignment['status']=='FOUR_DISTINCT_GPUS_OFFLINE_READY' and len({r['uuid'] for r in assignment['rows']})==4)
    for spec in protocol['models']:
        label=spec['label'];d=WORK/'workers'/label
        if not (d/'summary.json').exists():checks[label+'_worker_exists']=False;continue
        w=read(d/'summary.json');checks[label+'_complete']=w['status']=='LDC_LLM_TWO_ROUND_PREFLIGHT_COMPLETE' and len(w['rounds'])==2
        for r in w['rounds']:
            n=r['round'];p=d/f'proposal_{n}/summary.json';fpath=d/f'feedback_{n}.json'
            if not p.exists():checks[label+f'_{n}_proposal_exists']=False;continue
            proposal=read(p);feedback=read(fpath)
            checks[label+f'_{n}_prompt_provenance']=(proposal['system_prompt_sha256']==SYSTEM_SHA256 and proposal['feedback_sha256']==digest(fpath)
                and proposal['initial_messages_sha256']==hashlib.sha256(canonical(messages(feedback)).encode()).hexdigest())
            if proposal['status']!='LDC_PROPOSAL_COMPLETE':checks[label+f'_{n}_valid_proposal']=False;continue
            final=proposal['attempts'][-1];parsed,errors,_=parse_reply(final['raw_output'],feedback)
            checks[label+f'_{n}_valid_proposal']=not errors and parsed==r['configuration']
            sp=d/f'trial_{n}/summary.json'
            if not sp.exists():checks[label+f'_{n}_trial_exists']=False;continue
            s=read(sp)
            checks[label+f'_{n}_matched_initial_weights']=s['initial_weight_sha256']==common['initial_weight_sha256']
            checks[label+f'_{n}_matched_sampling']=s['sample_sha256']==common['sample_sha256']
            checks[label+f'_{n}_full_train_coverage']=set(s['case_counts'].values())=={2} and len(s['case_counts'])==640
            checks[label+f'_{n}_true_physics_and_speed']=s['physics_updates']>0 and s['selected_speed']['pass_5ms'] and not s['main_test_opened']
            checks[label+f'_{n}_real_feedback']=r['real_feedback']['score']==s['selected_validation']['score'] and r['trial_summary_sha256']==digest(sp)
            if n==2:
                checks[label+'_sequential_feedback']=feedback['history'][-1]==w['rounds'][0]['real_feedback'] and feedback['latest_required_id'] in parsed['evidence_ids']
            rows.append(dict(model=label,round=n,configuration=parsed,measurements=s['selected_validation']['means'],
                proposal_seconds=proposal['elapsed_seconds'],training_seconds=s['training_seconds'],
                p99_max_ms=max(c['p99'] for c in s['selected_speed']['cases']),
                peak_llm_gpu_bytes=proposal['peak_sampled_nvml_gpu_bytes'],main_test_opened=False))
        for c in w['controls']:
            checks[c['method']+'_same_initial_sampling']=(c['initial_weight_sha256']==common['initial_weight_sha256'] and c['sample_sha256']==common['sample_sha256'])
    checks['all_eight_real_trials']=len(rows)==8
    checks['both_control_paths']=all(k in checks for k in ['random_same_initial_sampling','tpe_same_initial_sampling'])
    result=dict(status='FOUR_LDC_LLM_PREFLIGHTS_COMPLETE' if all(checks.values()) else 'LDC_LLM_PREFLIGHT_REQUIRES_REVIEW',
        checks=checks,rows=rows,technical_only=True,main_test_opened=False,main_experiment_started=False,llm_accuracy_benefit_established=False)
    save(WORK/'review.json',result)
    lines=['TECHNICKÁ SKÚŠKA ŠTYROCH LOKÁLNYCH LLM PRE USTÁLENÝ PINN',result['status'],
        'Každý LLM: dva nadväzujúce návrhy a skutočné krátke tréningy. Celá TRAIN/VAL sada, iba dve epochy.',
        'Výsledok potvrdzuje zapojenie, nie poradie kvality ani prínos LLM. TEST zostal uzavretý.']
    for r in rows:lines.append(f"{r['model']}, pokus {r['round']}: MSE {r['measurements']['mse_all']:.6g}, návrh {r['proposal_seconds']:.1f} s, tréning {r['training_seconds']:.1f} s, p99 {r['p99_max_ms']:.3f} ms")
    (WORK/'VYSLEDOK.txt').write_text('\n'.join(lines)+'\n')
    archive=WORK/'small_preflight.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in WORK.rglob('*'):
            if p.is_file() and p.suffix in ['.json','.txt','.log'] and p.stat().st_size<8*1024*1024:z.write(p,p.relative_to(WORK))
    save(WORK/'archive_receipt.json',dict(bytes=archive.stat().st_size,sha256=digest(archive)))
    print(result['status'],sum(checks.values()),len(checks),flush=True)
    return all(checks.values())


if __name__=='__main__':raise SystemExit(0 if run() else 1)
