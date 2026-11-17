"""Register held final-inference jobs before releasing any TEST evaluation."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
sys.path.insert(0,str(ROOT/'test/llm'))
from ldc_main_common_v33 import WORK,read,save,digest,fingerprint,verify

FINAL=ROOT/'results/ldc_final_test_20261108_v49'


def command(args):
    return subprocess.run(args,cwd=ROOT,capture_output=True,text=True,check=True,timeout=30).stdout.strip()


def submit(args):
    job=command(['sbatch','--parsable']+args).split(';')[0]
    if not re.fullmatch(r'\d+',job):raise ValueError('Unexpected job ID')
    return job


def main():
    verify();freeze=read(FINAL/'freeze.json')
    assert not (FINAL/'submission.json').exists() and not (FINAL/'test_access').exists()
    assert all(digest(p)==h for p,h in freeze['files'].items())
    assert all(fingerprint(p)==s for p,s in freeze['input_fingerprints'].items())
    assert not read(FINAL/'evaluation/state.json')['attention_required']
    assert all(r['status']=='PENDING' for r in read(FINAL/'evaluation/state.json')['tasks'].values())
    assert len(read(FINAL/'models.json')['models'])==155
    state=read(WORK/'main/state.json')
    assert len(state['tasks'])==395 and all(r['status']=='DONE' for r in state['tasks'].values())
    nodes=sum(int(x) for x in command(['squeue','-h','-u','lutoke823','-o','%D']).splitlines())
    assert nodes+3<=4,'Reserve GPU, accounting collector, and terminal CPU capture within four nodes'
    # Both ended CPU checks are captured now, before their scheduler history expires.
    for job,phase in [('145662','training_closeout_v49'),('145674','final_prepare_v49')]:
        path=WORK/'submissions'/(job+'.json')
        if not path.exists():save(path,dict(job=int(job),phase=phase,gpus=0,nodes=1,unix=time.time()))
        if not (WORK/'scheduler'/job/'summary.json').exists():
            command([str(ROOT/'envs/visualization/bin/python'),'-B','test/analysis/capture_ldc_main_v40.py',job])
        assert read(WORK/'scheduler'/job/'summary.json')['exit_code']=='0:0'
    gpu=submit(['--hold','test/scripts/269_ldc_final_test_v49.sbatch'])
    metadata=dict(gpu_job=gpu,models=155,test_cases=200,gpus=4,nodes=1,no_training=True,
        freeze_sha256=digest(FINAL/'freeze.json'),protocol_sha256=digest(FINAL/'protocol.json'),
        model_manifest_sha256=digest(FINAL/'models.json'),registered_before_release=True,unix=time.time())
    save(FINAL/'submission.json',metadata)
    save(WORK/'submissions'/(gpu+'.json'),dict(metadata,job=int(gpu),phase='final_test_v49'))
    cpu=submit(['--hold','test/scripts/270_ldc_final_collector_v49.sbatch'])
    save(WORK/'submissions'/(cpu+'.json'),dict(job=int(cpu),phase='final_test_collector_v49',
        gpu_job=gpu,gpus=0,nodes=1,registered_before_release=True,unix=time.time()))
    callback=submit(['--dependency=afterany:'+cpu,'test/scripts/254_ldc_main_capture_v40.sbatch',cpu])
    save(WORK/'submissions'/(callback+'.json'),dict(job=int(callback),phase='final_collector_capture_v49',
        parent=cpu,gpus=0,nodes=1,registered_before_dependency_can_complete=True,unix=time.time()))
    metadata.update(collector_job=cpu,collector_capture_job=callback)
    save(FINAL/'submission.json',metadata)
    save(WORK/'submissions'/(gpu+'.json'),dict(metadata,job=int(gpu),phase='final_test_v49'))
    command(['scontrol','release',cpu])
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        if (FINAL/'collector_ready.json').exists():
            ready=read(FINAL/'collector_ready.json')
            assert ready['gpu_job']==gpu and ready['collector']==cpu
            break
        time.sleep(2)
    else:raise RuntimeError('CPU collector not ready; GPU remains held for inspection')
    command(['scontrol','release',gpu])
    save(FINAL/'released.json',dict(gpu_job=gpu,collector_job=cpu,unix=time.time(),TEST_opened=False))
    print(json.dumps(dict(status='FINAL_INFERENCE_RELEASED',**metadata)),flush=True)


if __name__=='__main__':main()
