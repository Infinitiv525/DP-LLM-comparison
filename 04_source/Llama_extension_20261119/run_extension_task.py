from pathlib import Path
import argparse,os,time,traceback,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ldc_main_common_v33 as C
import ldc_main_scheduler_v33 as Q
import worker_ldc_main_v33 as W

def perform(payload,phase):
 task=payload['task'];directory=C.phase_path(phase);job=os.environ['SLURM_JOB_ID'];worker=0;out=directory/'executions'/task['id']/job
 try:
  result=W.run_task(payload,phase,out)
  if result['status']=='COMPLETE':
   # Random samples/initialization are paired with the original experiment, even though execution date differs.
   old=C.read(C.ROOT/'results/ldc_main_20261104_v33'/phase/'state.json')['tasks'];ref=('qwen3_8b_r%02d_c00'%task['run']) if task['kind']=='search' else ('qwen3_8b_r%02d_k%02d'%(task['run'],task['repeat']))
   baseline=old[ref]['outcome']
   for key in ['initial_weight_sha256','sample_sha256']:
    if result[key]!=baseline[key]:raise W.InfrastructureError('Paired original seed/sample mismatch: '+key)
   result['paired_initialization_and_samples_verified_against']=ref
  C.save(out/'outcome.json',result);Q.finish(directory,task['id'],worker,job,result);print('TASK_COMPLETE',task['id'],result['status'],flush=True)
  if phase=='integration' and result['status']!='COMPLETE':raise RuntimeError('Technical integration requires every actual trial COMPLETE')
 except Exception:
  error=traceback.format_exc();C.save(out/'extension_failure.json',dict(status='REVIEW_REQUIRED',traceback=error))
  with Q.locked(directory):
   state=C.read(directory/'state.json');row=state['tasks'][task['id']]
   if row['status']=='RUNNING':row.update(status='ATTENTION',outcome=dict(status='INFRASTRUCTURE_REVIEW_REQUIRED',error=error));state['attention_required']=True;C.save(directory/'state.json',state)
  raise

def exact_claim(identifier,phase):
 d=C.phase_path(phase)
 with Q.locked(d):
  state=C.read(d/'state.json');assert not state['attention_required'],'Review required';row=state['tasks'][identifier];task=row['task'];assert row['status']=='PENDING','No duplicate/resumed hidden training';assert all(state['tasks'][x]['status']=='DONE' for x in task['dependencies']),'Unfinished actual feedback'
  selection=Q.choose_selection(state,task) if task['kind']=='retrain' else None
  if task['kind']=='retrain':
   assert selection is not None,'No successful configuration';p=d/'selections'/('Llama31_8B_r%02d.json'%task['run'])
   if p.exists():assert C.read(p)==selection,'Frozen configuration changed'
   else:C.save(p,selection)
  history=[state['tasks']['Llama31_8B_r%02d_c%02d'%(task['run'],c)]['outcome']['history'] for c in range(task['candidate'])] if task['kind']=='search' else []
  row.update(status='RUNNING',worker=0,job=os.environ['SLURM_JOB_ID'],claimed_unix=time.time());C.save(d/'state.json',state)
  return dict(task=task,history=history,selection=selection)

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['main','integration'],required=True);p.add_argument('--task');a=p.parse_args();C.verify();assert os.environ.get('HF_HUB_OFFLINE')=='1'
 if a.phase=='main':
  assert C.read(C.WORK/'integration/gate.json')['status']=='LLAMA_ACTUAL_INTEGRATION_PASS';assert a.task;perform(exact_claim(a.task,'main'),'main')
 else:
  while True:
   payload,status=Q.claim(C.phase_path('integration'),0,os.environ['SLURM_JOB_ID'])
   if payload is None:
    assert status=='ALL_DONE',status;break
   perform(payload,'integration')
  s=C.read(C.WORK/'integration/state.json');assert len(s['tasks'])==4 and all(r['status']=='DONE' and r['outcome']['status']=='COMPLETE' for r in s['tasks'].values());C.verify();C.save(C.WORK/'integration/gate.json',dict(status='LLAMA_ACTUAL_INTEGRATION_PASS',actual_trainings=4,paired_initialization_and_samples_verified=True,main_training_unchanged=True,TEST_read=False,job=os.environ['SLURM_JOB_ID'],UTC=time.time()))
 C.verify()
if __name__=='__main__':main()
