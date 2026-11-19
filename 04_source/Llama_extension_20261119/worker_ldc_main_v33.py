"""One GPU per worker, persistent dependency queue, actual feedback only."""
import argparse
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback
from ldc_main_common_v33 import ROOT, WORK, read, save, digest, verify, phase_path
from ldc_main_contract_v33 import VERSION, BASELINE, validate_parameters, validate_feedback
from ldc_main_controls_v33 import propose_control
from ldc_main_scheduler_v33 import claim, finish, locked


class InfrastructureError(RuntimeError):
    pass


def bounded(command, output, seconds):
    started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with Path(output).open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        timed_out=False
        try:
            code=process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(process.pid,signal.SIGTERM)
            try:code=process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(exit_code=code,timed_out=timed_out,wall_seconds=time.monotonic()-started,
                child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)


def verify_model(directory,spec,protocol):
    folder=directory/'models'
    with locked(folder/spec['label']):
        target=folder/(spec['label']+'.json')
        proof_path=Path(protocol['preparation'])/(spec['label']+'.json')
        if target.exists():
            previous=read(target)
            if previous['manifest_sha256']!=digest(proof_path) or not previous['verified']:
                raise InfrastructureError('Model verification receipt mismatch')
            return
        proof=read(proof_path);started=time.monotonic()
        if proof['status']!='LOCAL_MODEL_VERIFIED' or proof['spec']!=spec:
            raise InfrastructureError('Model preparation mismatch')
        for row in proof['files']:
            path=Path(proof['path'])/row['file']
            if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
                raise InfrastructureError('Published model file changed')
        save(target,dict(verified=True,manifest_sha256=digest(proof_path),seconds=time.monotonic()-started,
                         compute_job=os.environ['SLURM_JOB_ID'],spec=spec))


def failed_history(task,phase,parameters,status,seconds,codes):
    return dict(id=task['id'],parameters=parameters,status=status,score=None,velocity_error=None,
        pressure_error=None,momentum_rms=None,divergence_rms=None,boundary_error=None,
        training_seconds=seconds,training_seed=task['seed'],epochs=400 if phase=='main' else 2,
        error_codes=[str(c)[:200] for c in codes][:8])


def run_task(payload,phase,directory):
    task=payload['task'];protocol=read(WORK/'protocol.json');specification=protocol['phases'][phase]
    directory.mkdir(parents=True,exist_ok=False)
    save(directory/'task.json',task)
    parameters=None;proposal=None;processes={}
    if task['kind']=='search' and task['candidate']>0:
        history=payload['history']
        feedback=dict(contract_version=VERSION,phase=phase,epochs=specification['epochs'],
            candidate_budget=specification['candidates'],remaining_attempts=specification['candidates']-task['candidate'],
            latest_required_id=history[-1]['id'],history=history)
        validate_feedback(feedback);save(directory/'feedback.json',feedback)
        control_seed=(130001 if phase=='main' else 340001)+task['run']
        if task['method'] in ['random','tpe']:
            started=time.monotonic()
            parameters,details=propose_control(task['method'],control_seed,history)
            save(directory/'control_proposal.json',dict(parameters=parameters,details=details,
                decision_seconds=time.monotonic()-started,feedback_sha256=digest(directory/'feedback.json')))
        else:
            index=next(i for i,s in enumerate(protocol['models']) if s['label']==task['method'])
            verify_model(phase_path(phase),protocol['models'][index],protocol)
            generation_seed=(120001 if phase=='main' else 350001)+task['run']*100+task['candidate']
            processes['proposal']=bounded([str(ROOT/'envs/llm-feasibility-20261021/bin/python'),'-B',
                str(ROOT/'test/llama_extension_20261119_v1/propose_ldc_main_v33.py'),'--phase',phase,'--index',str(index),
                '--seed',str(generation_seed),'--feedback',str(directory/'feedback.json'),
                '--output',str(directory/'proposal')],directory/'proposal.log',600)
            save(directory/'proposal_process.json',processes['proposal'])
            summary_path=directory/'proposal/summary.json'
            if processes['proposal']['timed_out']:
                return dict(status='INVALID_PROPOSAL',history=failed_history(task,phase,None,'INVALID_PROPOSAL',0,['PROPOSAL_TIME_LIMIT']),
                            processes=processes)
            if not summary_path.exists():
                raise InfrastructureError('Proposal process produced no receipt')
            proposal=read(summary_path)
            if proposal['status']=='LDC_PROPOSAL_INVALID':
                return dict(status='INVALID_PROPOSAL',history=failed_history(task,phase,None,'INVALID_PROPOSAL',0,
                    proposal['attempts'][-1]['errors']),processes=processes,proposal_summary_sha256=digest(summary_path))
            if proposal['status']!='LDC_PROPOSAL_COMPLETE' or processes['proposal']['exit_code']:
                raise InfrastructureError('Local model loading/generation failed; see preserved receipt')
            parameters=proposal['configuration']['parameters']
    elif task['kind']=='retrain':
        parameters=payload['selection']['parameters'];save(directory/'selection.json',payload['selection'])
    else:
        parameters=BASELINE.copy()
    validate_parameters(parameters)
    request=dict(phase=phase,task=task,parameters=parameters,protocol_sha256=digest(WORK/'protocol.json'))
    save(directory/'request.json',request)
    processes['training']=bounded([str(ROOT/'envs/training-torch-2.12-cu130/bin/python'),'-B',
        str(ROOT/'test/llama_extension_20261119_v1/train_ldc_main_v33.py'),'--request',str(directory/'request.json'),
        '--output',str(directory/'training')],directory/'training.log',specification['trial_timeout_seconds'])
    save(directory/'training_process.json',processes['training'])
    training_path=directory/'training/summary.json'
    if processes['training']['timed_out']:
        return dict(status='TRAINING_FAILED',history=failed_history(task,phase,parameters,'TRAINING_FAILED',
            processes['training']['wall_seconds'],['UNIFORM_TRAINING_TIME_LIMIT']),processes=processes)
    if processes['training']['exit_code'] or not training_path.exists():
        tail=(directory/'training.log').read_text(errors='replace')[-6000:]
        if 'Nonfinite loss' in tail or 'non-finite' in tail:
            return dict(status='TRAINING_FAILED',history=failed_history(task,phase,parameters,'TRAINING_FAILED',
                processes['training']['wall_seconds'],['NONFINITE_OPTIMIZATION']),processes=processes)
        raise InfrastructureError('PINN execution failed; preserve attempt and review before any retry')
    trained=read(training_path)
    if trained['status']!='LDC_MAIN_TRIAL_COMPLETE' or trained['epochs']!=specification['epochs'] or trained['seed']!=task['seed']:
        raise InfrastructureError('Actual training budget mismatch')
    if not trained['selected_speed']['pass_5ms']:
        return dict(status='SPEED_FAILED',history=failed_history(task,phase,parameters,'SPEED_FAILED',
                    trained['training_seconds'],['NATIVE_P99_OVER_5MS']),
                    trial_summary_sha256=digest(training_path),processes=processes)
    val=trained['selected_validation']
    if val['failure']:
        return dict(status='TRAINING_FAILED',history=failed_history(task,phase,parameters,'TRAINING_FAILED',
            trained['training_seconds'],['NONFINITE_VALIDATION']),trial_summary_sha256=digest(training_path),processes=processes)
    history=dict(id=task['id'],parameters=parameters,status='COMPLETE',score=val['score'],
        velocity_error=val['means']['velocity_relative_l2'],pressure_error=val['means']['pressure_relative_l2'],
        momentum_rms=val['mean_momentum_rms'],divergence_rms=val['mean_divergence_rms'],
        boundary_error=trained['boundary_mean'],training_seconds=trained['training_seconds'],
        training_seed=task['seed'],epochs=specification['epochs'],error_codes=[])
    return dict(status='COMPLETE',history=history,trial_summary_sha256=digest(training_path),
        trial_summary_path=str(training_path),initial_weight_sha256=trained['initial_weight_sha256'],
        sample_sha256=trained['sample_sha256'],processes=processes,
        proposal_summary_sha256=None if proposal is None else digest(directory/'proposal/summary.json'))


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['main','integration'],required=True)
    a=parser.parse_args();verify();protocol=read(WORK/'protocol.json');directory=phase_path(a.phase)
    job=os.environ['SLURM_JOB_ID'];worker=int(os.environ['SLURM_PROCID'])
    registration=read(WORK/'submissions'/(job+'.json'))
    if registration['phase']!=a.phase:
        raise InfrastructureError('Unregistered phase/job')
    if a.phase=='main' and read(WORK/'integration/review.json')['status']!='LDC_MAIN_INTEGRATION_PASS':
        raise InfrastructureError('Main run requires successful actual integration')
    started=float(os.environ['HPC_BATCH_STARTED_UNIX']);spec=protocol['phases'][a.phase]
    deadline=started+spec['batch_seconds'];reserve=spec['trial_timeout_seconds']+600+300
    completed=[]
    while time.time()+reserve < deadline:
        verify()
        payload,status=claim(directory,worker,job)
        if payload is None:
            if status in ['ALL_DONE','ATTENTION_REQUIRED']:
                break
            time.sleep(3)
            continue
        task=payload['task'];out=directory/'executions'/task['id']/job
        try:
            outcome=run_task(payload,a.phase,out)
            save(out/'outcome.json',outcome)
            finish(directory,task['id'],worker,job,outcome)
            completed.append(task['id'])
            print('TASK_COMPLETE',task['id'],outcome['status'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            outcome=dict(status='INFRASTRUCTURE_REVIEW_REQUIRED',error=error,output=str(out))
            save(out/'infrastructure_failure.json',outcome)
            finish(directory,task['id'],worker,job,outcome,attention=True)
            break
    save(directory/'batches'/job/f'worker_{worker}.json',dict(worker=worker,job=job,completed=completed,
        stopped_cleanly=True,unix=time.time(),main_test_opened=False))


if __name__=='__main__':
    run()
