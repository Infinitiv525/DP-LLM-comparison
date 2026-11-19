from pathlib import Path
import sys,json,hashlib,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ldc_main_common_v33 as C
C.verify();s=C.read(C.WORK/'main/state.json');assert len(s['tasks'])==65 and not s['attention_required'];assert all(v['status']=='DONE' for v in s['tasks'].values());final=[]
for key,row in s['tasks'].items():
 t=row['task'];o=row['outcome']
 if t['kind']=='retrain':
  assert o['status']=='COMPLETE',(key,o['status']);p=Path(o['trial_summary_path']);d=C.read(p);weight=p.parent/'best.pt';assert C.digest(weight)==d['checkpoint_sha256'];final.append(dict(id=key,task=t,checkpoint=str(weight),sha256=d['checkpoint_sha256'],trial_summary_sha256=C.digest(p),selected_parameters=o['history']['parameters']))
assert len(final)==25
C.save(C.WORK/'main/frozen_for_TEST.json',dict(status='ALL25_LLAMA_FINAL_CHECKPOINTS_FROZEN_FOR_SEPARATE_TEST',created_unix=time.time(),checkpoints=final,selections=[C.read(C.WORK/'main/selections'/('Llama31_8B_r%02d.json'%r)) for r in range(5)],TEST_targets_read=False,prior_original_cohort_TEST_known=True,original_dates_preserved=True));print('ALL65_TRAININGS_DONE_25_CHECKPOINTS_FROZEN',flush=True)
