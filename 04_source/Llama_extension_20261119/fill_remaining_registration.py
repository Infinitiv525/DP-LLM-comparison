from pathlib import Path
import json,subprocess,time,fcntl,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ldc_main_common_v33 as C
R=Path('/mnt/project/perun2601449/dp_llm_cfd');S=R/'test/llama_extension_20261119_v1';O=R/'results/llama_extension_20261119_v1';P=O/'main/job_registration.json'
def save(v):
 p=P.with_suffix('.tmp');p.write_text(json.dumps(v,indent=2)+'\n');p.replace(P)
with (O/'main/registration_controller.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 gate=json.loads((O/'integration/gate.json').read_text());assert gate['status']=='LLAMA_ACTUAL_INTEGRATION_PASS'
 dag=json.loads((O/'main/submission_DAG.json').read_text());state=json.loads(P.read_text());jobs={r['task']:r['job'] for r in state['jobs']};assert len(jobs)==len(state['jobs']);assert len(jobs)<=65
 if state['status']=='HELD_UNTIL_COMPLETE_REGISTRATION':
  v=subprocess.run(['scontrol','release',','.join(jobs.values())],capture_output=True,text=True);assert v.returncode==0,v.stderr;state['status']='PARTIAL_REGISTRATION_ACTIVE_AUTOFILL';state['submission_limit_note']='Perun permits50 submitted jobs. Remaining planned tasks are registered as earlier jobs finish; no logical training is removed or repeated.';save(state)
 while len(jobs)<65:
  row=dag['tasks'][len(jobs)];assert row['task'] not in jobs;C.verify();actual=C.read(O/'main/state.json');assert not actual['attention_required'];active=set(subprocess.run(['squeue','-u','lutoke823','-h','-o','%i'],capture_output=True,text=True,check=True).stdout.split());deps=[]
  for d in row['logical_dependencies']:
   if jobs[d] in active:deps.append(jobs[d])
   else:assert actual['tasks'][d]['status']=='DONE','Completed dependency lacks actual outcome: '+d
  command=['sbatch','--parsable']+(['--dependency=afterok:'+':'.join(sorted(set(deps)))] if deps else [])+['--job-name='+row['task'],str(S/'main_one_task.sbatch'),row['task']];p=subprocess.run(command,capture_output=True,text=True)
  if p.returncode:
   if 'AssocMaxSubmitJobLimit' in p.stderr:time.sleep(30);continue
   raise RuntimeError(p.stderr)
  job=p.stdout.strip().split(';')[0];assert job.isdigit();jobs[row['task']]=job;state['jobs'].append(dict(task=row['task'],job=job,lane=row['lane'],depends_on_jobs=sorted(set(deps))));state['status']='PARTIAL_REGISTRATION_ACTIVE_AUTOFILL';save(state);print('REGISTERED',row['task'],job,flush=True)
 state['status']='ALL65_REGISTERED_WITH_TECHNICAL_GATE_AND_8_LANES';save(state)
 active=set(subprocess.run(['squeue','-u','lutoke823','-h','-o','%i'],capture_output=True,text=True,check=True).stdout.split());actual=C.read(O/'main/state.json');assert not actual['attention_required'];assert all(r['job'] in active or actual['tasks'][r['task']]['status']=='DONE' for r in state['jobs']);deps=sorted(set(jobs.values())&active);close_command=['sbatch','--parsable']+(['--dependency=afterok:'+':'.join(deps)] if deps else [])+[str(S/'training_closeout.sbatch')];p=subprocess.run(close_command,capture_output=True,text=True)
 while p.returncode and 'AssocMaxSubmitJobLimit' in p.stderr:
  time.sleep(30);active=set(subprocess.run(['squeue','-u','lutoke823','-h','-o','%i'],capture_output=True,text=True,check=True).stdout.split());actual=C.read(O/'main/state.json');assert not actual['attention_required'];assert all(r['job'] in active or actual['tasks'][r['task']]['status']=='DONE' for r in state['jobs']);deps=sorted(set(jobs.values())&active);close_command=['sbatch','--parsable']+(['--dependency=afterok:'+':'.join(deps)] if deps else [])+[str(S/'training_closeout.sbatch')];p=subprocess.run(close_command,capture_output=True,text=True)
 assert p.returncode==0,p.stderr;state['closeout_job']=p.stdout.strip().split(';')[0];save(state);print('ALL65_SCHEDULED',state['closeout_job'],flush=True)
