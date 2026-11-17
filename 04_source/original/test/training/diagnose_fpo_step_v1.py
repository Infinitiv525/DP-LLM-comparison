"""Read-only stability diagnosis: train-only local fit vs own 238-step rollout.

Validation future fields remain exclusively labels. Compare persistence and
unlearned two-state linear extrapolation; measure field norms and divergence.
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from fpo_step_v5a import StepPINN,grid,step,project_known_constraints,load_cases,errors
from fpo_common_v5a import WORK,save,digest,verify


def recurrence_check():
    rows=[]
    for eps in [0.,1e-6,1e-5,1e-4]:
        previous=current=0.
        for k in range(1,239):previous,current=current,2*current-previous+eps
        expected=238*239/2*eps
        if abs(current-expected)>1e-10:raise AssertionError('AR2 recurrence control failed')
        rows.append(dict(constant_additive_bias=eps,final_error=current,analytic=expected))
    return dict(scope='scalar controlled recurrence, not CFD accuracy or proof of the sole model failure cause',
                steps=238,constant_bias_amplification=238*239/2,rows=rows)


def profile(pred,reference,mask,y,x):
    p=pred[mask].astype(np.float64);r=reference[mask].astype(np.float64)
    p[:,2]-=p[:,2].mean();r[:,2]-=r[:,2].mean()
    def div(a):return (a[y,x+1,0]-a[y,x-1,0]+a[y+1,x,1]-a[y-1,x,1])*32
    return dict(kinetic_energy_ratio=float(np.sum(p[:,:2]**2)/max(np.sum(r[:,:2]**2),1e-24)),
                pressure_rms_ratio=float(np.sqrt(np.sum(p[:,2]**2)/max(np.sum(r[:,2]**2),1e-24))),
                max_speed=float(np.sqrt(np.sum(p[:,:2]**2,1)).max()),
                model_fd_divergence_rms=float(np.sqrt(np.mean(div(pred).astype(np.float64)**2))),
                reference_fd_divergence_rms=float(np.sqrt(np.mean(div(reference).astype(np.float64)**2))))


def mean_metrics(rows,key):
    values=[r[key] for r in rows]
    if any(v is None for v in values):return None
    return {metric:float(np.mean([r[metric] for r in values])) for metric in values[0]}


def run():
    verify();torch.set_num_threads(4)
    freeze=json.loads((WORK/'diagnostic_freeze_v1.json').read_text())
    for path,checksum in freeze['files'].items():
        if digest(path)!=checksum:raise ValueError('Diagnostic source/checkpoint changed')
    records=json.loads((WORK/'prepared.json').read_text())['records'];selected=[];selection=[]
    for family in ['harmonics','nurbs','skelenton']:
        rows=sorted([r for r in records if r['proposed_split']=='train' and r['family']==family],key=lambda r:(r['reynolds'],r['path']))
        for r in [rows[0],rows[-1]]:
            selected.append(r);selection.append(dict(path=r['path'],reason='lowest/highest training Re within family, chosen without outcome'))
    selected += [r for r in records if r['proposed_split']=='validation']
    if any(r['proposed_split']=='test' for r in selected):raise ValueError('Main test sealed')
    cases=load_cases(selected);qgrid=grid()
    checkpoint=WORK/'training/139549/best.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    normal=json.loads((WORK/'training/139549/normalization.json').read_text())
    model=StepPINN(state['config'],normal['mean'],normal['std']).cuda();model.load_state_dict(state['state']);model.eval()
    out=WORK/'diagnostics'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    save(out/'recurrence_control.json',recurrence_check());save(out/'selection.json',selection)
    summaries=[]
    with torch.inference_mode():
        for case in cases:
            training=case['record']['proposed_split']=='train'
            if training:
                previous,current=case['data'][0],case['data'][1]
                reference=np.load(case['record']['field'],mmap_mode='r',allow_pickle=False)
            else:
                previous,current=case['prefix'];reference=case['reference']
            linear_previous,linear_current=previous,current
            persistence=current.cpu().numpy()[case['fluid_np']]
            rng=np.random.default_rng(20261031);y,x=case['interior'][rng.integers(len(case['interior']),size=2048)].T
            rows=[];failed=None
            for offset in range(1,239):
                linear=project_known_constraints(2*linear_current-linear_previous,case['fluid'])
                linear_previous,linear_current=linear_current,linear
                lp=linear.cpu().numpy();truth=np.asarray(reference[offset+3]);fluid_truth=truth[case['fluid_np']]
                row=dict(frame=offset+3,offset=offset,model=None,profile=None,
                         linear_extrapolation=errors(lp[case['fluid_np']],fluid_truth),
                         persistence=errors(persistence,fluid_truth),train_only_one_step=None)
                if failed is None:
                    predicted=step(model,previous,current,case['geometry'],case['record']['reynolds'],case['fluid'],qgrid)
                    p=predicted.cpu().numpy()
                    if not np.isfinite(p).all() or np.max(np.abs(p))>1000:
                        failed=dict(offset=offset,reason='nonfinite or abs signal above1000')
                    else:
                        row['model']=errors(p[case['fluid_np']],fluid_truth)
                        row['profile']=profile(p,truth,case['fluid_np'],y,x)
                        previous,current=current,predicted
                if training:
                    # True subsequent inputs permitted ONLY for this training
                    # fit diagnostic. Never executed for a validation case.
                    pred=step(model,case['data'][offset-1],case['data'][offset],case['geometry'],
                              case['record']['reynolds'],case['fluid'],qgrid).cpu().numpy()
                    row['train_only_one_step']=errors(pred[case['fluid_np']],fluid_truth)
                rows.append(row)
            horizons={'all_future':rows,'steps_1_10':rows[:10],'steps_11_40':rows[10:40],
                      'steps_41_120':rows[40:120],'steps_121_238':rows[120:]}
            summary=dict(path=case['record']['path'],role=case['record']['proposed_split'],divergence=failed,
                         metrics={name:{key:mean_metrics(rr,key) for key in ['model','linear_extrapolation','persistence']}
                                  for name,rr in horizons.items()},
                         train_only_local_fit=mean_metrics(rows,'train_only_one_step') if training else None,
                         profiles={name:mean_metrics(rr,'profile') for name,rr in horizons.items()})
            folder=out/case['record']['family']/case['record']['geometry_id']/str(case['record']['reynolds'])
            folder.mkdir(parents=True,exist_ok=False);save(folder/'per_frame.json',rows);save(folder/'summary.json',summary)
            summaries.append(summary);print('DIAGNOSED',case['record']['path'],case['record']['proposed_split'],flush=True)
    save(out/'summary.json',dict(status='FROZEN_MODEL_STABILITY_DIAGNOSED',checkpoint_sha256=digest(checkpoint),
         weights_changed=False,results=summaries,selection=selection,
         validation_future_as_input=False,training_true_window_inputs_used_only_for_fit_diagnostic=True,
         limitations=['Training-case fit is not independent accuracy evidence',
                      'FD divergence is a discretized interior diagnostic, not an exact PDE residual',
                      'A scalar recurrence control does not prove the sole cause of all model errors']))


if __name__=='__main__':run()
