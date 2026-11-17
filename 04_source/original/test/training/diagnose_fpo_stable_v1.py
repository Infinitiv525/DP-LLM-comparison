"""Readonly train-only diagnosis of V6 best/last checkpoint local vs rollout error."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from fpo_stable_v6 import StablePINN,grid,stable_step,load_cases
from fpo_common_v6 import ROOT,WORK,save,digest,verify


def metric(pred,reference,mask):
    p=pred[mask].double();r=reference[mask].double()
    p[:,2]-=p[:,2].mean();r[:,2]-=r[:,2].mean()
    n=(p-r).square().sum(0);d=r.square().sum(0)
    return dict(velocity_relative_l2=float((n[:2].sum()/d[:2].sum().clamp_min(1e-24)).sqrt()),
                pressure_relative_l2=float((n[2]/d[2].clamp_min(1e-24)).sqrt()))


def run():
    verify()
    freeze=json.loads((WORK/'diagnostic_freeze_v1.json').read_text())
    for p,h in freeze['files'].items():
        if digest(p)!=h:raise ValueError('Diagnostic input changed: '+p)
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    receipt=json.loads((WORK/'prepared.json').read_text());r=receipt['records'][0]
    if r['proposed_split']!='train':raise ValueError('TRAIN-only diagnosis')
    case=load_cases([r])[0]
    with np.load(receipt['weight_file'],allow_pickle=False) as z:case['weight']=torch.tensor(z['weight'],device='cuda')
    normal=json.loads(Path(receipt['normalization']).read_text());q=grid()
    # Predetermined equally spaced starts, all with 32 available future states.
    starts=np.linspace(1,207,16,dtype=int).tolist();rows=[];summary=[]
    out=WORK/'diagnostics'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    for config in json.loads((WORK/'protocol.json').read_text())['candidates']:
        for checkpoint in ['best','last']:
            path=WORK/'training/139784'/config['name']/(checkpoint+'.pt')
            saved=torch.load(path,weights_only=True,map_location='cuda')
            model=StablePINN(config,normal['mean'],normal['std']).cuda();model.load_state_dict(saved['state']);model.eval()
            arm=[];failures=[]
            with torch.inference_mode():
                for k in starts:
                    previous,current=case['data'][k-1],case['data'][k]
                    for step in range(1,33):
                        pred=stable_step(model,previous,current,case,q)
                        if not torch.isfinite(pred).all() or pred.abs().max()>1000:
                            failures.append(dict(start_current_frame=k+2,step=step,reason='nonfinite or abs signal above1000'))
                            break
                        if step in [1,8,32]:
                            target=case['data'][k+step]
                            result=dict(name=config['name'],checkpoint=checkpoint,start_current_frame=k+2,step=step,
                                        model=metric(pred,target,case['fluid']),
                                        persistence=metric(case['data'][k],target,case['fluid']),
                                        finite=bool(torch.isfinite(pred).all()),max_abs=float(pred.abs().max()))
                            if step==1:
                                a=(pred-case['data'][k])[case['fluid'],:2].double().flatten()
                                b=(target-case['data'][k])[case['fluid'],:2].double().flatten()
                                result['velocity_increment_norm_ratio']=float(a.norm()/b.norm().clamp_min(1e-24))
                                result['velocity_increment_cosine']=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-24))
                            arm.append(result);rows.append(result)
                        previous,current=current,pred
            scopes={str(step):{kind:{m:float(np.mean([row[kind][m] for row in arm if row['step']==step])) if any(row['step']==step for row in arm) else None
                                  for m in ['velocity_relative_l2','pressure_relative_l2']}
                              for kind in ['model','persistence']} for step in [1,8,32]}
            one=[row for row in arm if row['step']==1]
            result=dict(name=config['name'],checkpoint=checkpoint,selected_phase=saved['phase'],selected_update=saved['update'],
                        scores=scopes,checkpoint_sha256=digest(path),
                        failures=failures,completed_samples={str(step):sum(row['step']==step for row in arm) for step in [1,8,32]},
                        scores_conditional_on_completion=bool(failures),
                        mean_velocity_increment_norm_ratio=float(np.mean([row['velocity_increment_norm_ratio'] for row in one])) if one else None,
                        mean_velocity_increment_cosine=float(np.mean([row['velocity_increment_cosine'] for row in one])) if one else None)
            summary.append(result);print(json.dumps(result),flush=True)
            del model,saved;torch.cuda.empty_cache()
    save(out/'per_window.json',rows)
    save(out/'summary.json',dict(status='READONLY_KNOWN_TRAIN_CASE_DIAGNOSIS_COMPLETE',results=summary,starts=starts,
         steps=[1,8,32],source_freeze_sha256=digest(WORK/'diagnostic_freeze_v1.json'),
         scope='Known TRAIN reference states supplied only at start of each diagnostic window; later inputs own predictions',
         optimizer_used=False,checkpoint_modified=False,main_test_opened=False,independent_validation=False,
         note='Does not replace primary full238 forecast from original prefix2,3. No new training or configurations selected.'))


if __name__=='__main__':run()
