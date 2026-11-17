"""Audit completed tasks against the frozen plan and actual file receipts."""
import argparse
import hashlib
from pathlib import Path
from ldc_main_common_v33 import WORK,read,save,digest,verify,phase_path
from ldc_main_contract_v33 import training_config,parse_reply,messages,canonical,system_digest,reply_schema
from ldc_main_scheduler_v33 import METHODS,choose_selection


def run(phase,job):
    verify();directory=phase_path(phase);protocol=read(WORK/'protocol.json');spec=protocol['phases'][phase]
    state=read(directory/'state.json');plan=read(directory/'plan.json')
    checks=dict(plan_matches_state=[row['task'] for row in state['tasks'].values()]==plan,
                no_infrastructure_attention=not state['attention_required'])
    assigned=read(directory/'batches'/job/'device_assignment.json')
    checks['eight_distinct_offline_gpus']=len(assigned['rows'])==8 and len({r['uuid'] for r in assigned['rows']})==8
    split=read(protocol['split_path'])['indices'];by_seed={};completed=[];failures=[];proposal_count=0
    for identifier,row in state['tasks'].items():
        if row['status']!='DONE':continue
        task=row['task'];outcome=row['outcome']
        if outcome['status']!='COMPLETE':
            failures.append(dict(task=identifier,status=outcome['status']))
            continue
        path=Path(outcome['trial_summary_path']);s=read(path);execution=path.parents[1]
        request=read(execution/'request.json');config=read(path.parent/'config.json')
        checks[identifier+':actual_config']=config==training_config(protocol['base_model'],request['parameters'],phase,task['seed'])
        checks[identifier+':registered_request']=request['task']==task and request['protocol_sha256']==digest(WORK/'protocol.json')
        checks[identifier+':actual_receipt']=outcome['trial_summary_sha256']==digest(path) and s['status']=='LDC_MAIN_TRIAL_COMPLETE'
        checks[identifier+':full_train_epochs']=set(s['case_counts'].values())=={spec['epochs']} and {int(i) for i in s['case_counts']}==set(split['train'])
        validation=read(path.parent/'selected_validation/summary.json')
        checks[identifier+':full_validation']=len(validation['cases'])==160 and {r['sample'] for r in validation['cases']}==set(split['validation'])
        checks[identifier+':native_speed_and_physics']=s['physics_updates']>0 and s['selected_speed']['pass_5ms'] and not s['main_test_opened']
        pair=(s['initial_weight_sha256'],s['sample_sha256'])
        if task['seed'] in by_seed:
            checks[identifier+':matched_initialization_and_samples']=pair==by_seed[task['seed']]
        else:by_seed[task['seed']]=pair
        history=outcome['history']
        checks[identifier+':measured_feedback']=history['score']==s['selected_validation']['score'] and history['boundary_error']==s['boundary_mean']
        checks[identifier+':no_seed_winner_selection']=s['seed']==task['seed'] and s['epochs']==spec['epochs']
        if task['kind']=='retrain':
            expected=choose_selection(state,task)
            checks[identifier+':frozen_val_selection']=read(execution/'selection.json')==expected and request['parameters']==expected['parameters']
        if task['kind']=='search' and task['candidate']>0 and task['method'] not in ['random','tpe']:
            proposal_count+=1;proposal=read(execution/'proposal/summary.json');feedback=read(execution/'feedback.json')
            parsed,errors,_=parse_reply(proposal['attempts'][-1]['raw_output'],feedback)
            checks[identifier+':local_llm_provenance']=(not errors and parsed==proposal['configuration']
                and parsed['parameters']==request['parameters']
                and proposal['system_prompt_sha256']==system_digest(phase)
                and proposal['feedback_sha256']==digest(execution/'feedback.json')
                and proposal['initial_messages_sha256']==hashlib.sha256(canonical(messages(feedback)).encode()).hexdigest()
                and proposal['attempts'][0]['messages']==messages(feedback)
                and proposal['schema']==reply_schema(feedback)
                and outcome['proposal_summary_sha256']==digest(execution/'proposal/summary.json'))
            expected_history=[state['tasks'][f'{task["method"]}_r{task["run"]:02d}_c{i:02d}']['outcome']['history'] for i in range(task['candidate'])]
            checks[identifier+':only_own_prior_actual_history']=feedback['history']==expected_history
        completed.append(dict(id=identifier,kind=task['kind'],method=task['method'],seed=task['seed'],
                              score=history['score'],velocity_error=history['velocity_error'],pressure_error=history['pressure_error']))
    all_done=all(r['status']=='DONE' for r in state['tasks'].values())
    if phase=='integration':
        checks['all_twenty_five_trainings_succeeded']=all_done and len(completed)==25 and not failures
        checks['all_four_models_two_real_proposals']=proposal_count==8
        checks['all_six_selections_retrained']=len([r for r in completed if r['kind']=='retrain'])==6
        status='LDC_MAIN_INTEGRATION_PASS' if all(checks.values()) else 'LDC_MAIN_INTEGRATION_FAILED'
    else:
        status=('LDC_MAIN_TRAINING_COMPLETE' if all_done else 'LDC_MAIN_BATCH_PARTIAL_PASS') if all(checks.values()) else 'LDC_MAIN_REVIEW_REQUIRED'
    result=dict(status=status,checks=checks,phase=phase,job=job,planned=len(plan),
        successful_completed=len(completed),all_attempts_done=sum(r['status']=='DONE' for r in state['tasks'].values()),
        failures=failures,rows=completed,all_done=all_done,main_test_opened=False,
        independent_final_test=False,test_evaluation_pending=phase=='main')
    save(directory/'batches'/job/'review.json',result);save(directory/'review.json',result)
    print(status,len(completed),'successful of',len(plan),'planned',sum(checks.values()),len(checks),flush=True)
    return all(checks.values())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['main','integration'],required=True);p.add_argument('--job',required=True);a=p.parse_args()
    raise SystemExit(0 if run(a.phase,a.job) else 1)
