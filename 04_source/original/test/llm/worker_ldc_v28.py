"""Two ask/train/feedback rounds per assigned GPU; no main experiment."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback
from ldc_llm_common_v28 import ROOT,WORK,save,read,digest,verify
from ldc_contract_v28 import VERSION,validate_parameters,validate_feedback
from ldc_controls_v28 import RandomSearch,TPESearch


def bounded(command,output,seconds):
    start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with Path(output).open('w') as log:
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        timeout=False
        try:code=p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(exit_code=code,timed_out=timeout,wall_seconds=time.monotonic()-start,
        child_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime))


def history_row(identifier,parameters,summary):
    if summary['status']!='LDC_CONFIGURATOR_TECHNICAL_TRIAL_COMPLETE':raise ValueError('Only successful real trials enter feedback')
    v=summary['selected_validation']
    if v['failure'] or not summary['selected_speed']['pass_5ms']:raise ValueError('Invalid trial score')
    return dict(id=identifier,parameters=validate_parameters(parameters),score=v['score'],
        velocity_error=v['means']['velocity_relative_l2'],pressure_error=v['means']['pressure_relative_l2'],
        momentum_rms=v['mean_momentum_rms'],divergence_rms=v['mean_divergence_rms'],
        training_seconds=summary['training_seconds'],training_seed=270001,epochs=2)


def trial(parameters,directory):
    path=directory.parent/(directory.name+'_parameters.json');save(path,parameters)
    outcome=bounded([str(ROOT/'envs/training-torch-2.12-cu130/bin/python'),'-B',str(ROOT/'test/training/train_ldc_trial_v28.py'),
        '--parameters',str(path),'--output',str(directory)],directory.parent/(directory.name+'.log'),600)
    save(directory.parent/(directory.name+'_process.json'),outcome)
    if outcome['exit_code'] or outcome['timed_out']:raise RuntimeError('Short PINN trial failed: '+str(directory))
    return read(directory/'summary.json')


def run():
    started=time.monotonic();index=int(os.environ['SLURM_PROCID']);verify();protocol=read(WORK/'protocol.json');spec=protocol['models'][index]
    directory=WORK/'workers'/spec['label'];directory.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',model=spec,index=index,rounds=[],controls=[],test_evaluated=False,technical_only=True,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(directory/'summary.json',result)
    try:
        # Large weight hashes are checked on this allocated compute node, never login.
        proof_path=Path(protocol['preparation'])/(spec['label']+'.json');proof=read(proof_path);t=time.monotonic();checked=[]
        if proof['spec']!=spec or proof['status']!='LOCAL_MODEL_VERIFIED':raise ValueError('Pinned model proof mismatch')
        for row in proof['files']:
            p=Path(proof['path'])/row['file']
            if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise ValueError('Weight/tokenizer hash mismatch: '+str(p))
            checked.append(dict(file=row['file'],sha256=row['sha256'],bytes=row['bytes']))
        save(directory/'weights_verified.json',dict(verified=True,manifest_sha256=digest(proof_path),files=checked,seconds=time.monotonic()-t))
        history=[history_row('common_0',read(WORK/'common_parameters.json'),read(WORK/'common/summary.json'))]
        for number in [1,2]:
            f=dict(contract_version=VERSION,phase='technical_preflight',epochs=2,remaining_attempts=3-number,
                latest_required_id=history[-1]['id'],history=history.copy());validate_feedback(f)
            path=directory/f'feedback_{number}.json';save(path,f)
            out=directory/f'proposal_{number}';generation_seed=270100+number
            outcome=bounded([str(ROOT/'envs/llm-feasibility-20261021/bin/python'),'-B',str(ROOT/'test/llm/propose_ldc_v28.py'),
                '--index',str(index),'--seed',str(generation_seed),'--feedback',str(path),'--output',str(out)],directory/f'proposal_{number}.log',300)
            row=dict(round=number,generation_seed=generation_seed,proposal_process=outcome,feedback_sha256=digest(path))
            result['rounds'].append(row);save(directory/'summary.json',result)
            if outcome['exit_code'] or outcome['timed_out']:raise RuntimeError('Local proposal process failed')
            proposal=read(out/'summary.json')
            if proposal['status']!='LDC_PROPOSAL_COMPLETE':raise ValueError('Invalid proposal, no manual parameter fallback')
            row['configuration']=proposal['configuration'];p=proposal['configuration']['parameters'];validate_parameters(p)
            summary=trial(p,directory/f'trial_{number}')
            h=history_row(spec['label']+'_'+str(number),p,summary);history.append(h)
            row.update(real_feedback=h,trial_summary_sha256=digest(directory/f'trial_{number}/summary.json'),
                initial_weight_sha256=summary['initial_weight_sha256'],sample_sha256=summary['sample_sha256'])
            save(directory/'summary.json',result)
            print('ROUND_COMPLETE',spec['label'],number,h['score'],flush=True)
        # The two non-LLM controls use exactly the same contract and fresh-trial path.
        if index in [0,1]:
            t=time.monotonic();initial=[history[0]];sampler=RandomSearch(270500) if index==0 else TPESearch(270500,initial)
            p=sampler.ask();decision_seconds=time.monotonic()-t
            summary=trial(p,directory/'control_trial');h=history_row('random' if index==0 else 'tpe',p,summary);sampler.tell(p,h['score'])
            result['controls'].append(dict(method=h['id'],feedback=h,initial_weight_sha256=summary['initial_weight_sha256'],
                sample_sha256=summary['sample_sha256'],decision_seconds=decision_seconds,state=sampler.export() if index==1 else None))
        result['status']='LDC_LLM_TWO_ROUND_PREFLIGHT_COMPLETE'
    except Exception:
        result['status']='LDC_LLM_PREFLIGHT_FAILED';result['error']=traceback.format_exc();print(result['error'],flush=True)
    finally:
        result['elapsed_seconds']=time.monotonic()-started;save(directory/'summary.json',result)
    return result['status']=='LDC_LLM_TWO_ROUND_PREFLIGHT_COMPLETE'


if __name__=='__main__':raise SystemExit(0 if run() else 1)
