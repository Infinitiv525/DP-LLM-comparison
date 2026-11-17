"""Read-only diagnosis of frozen v2 checkpoints: fitting vs transfer and PDE error."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from fpo_conditioned_v2 import (WORK, ConditionalPINN, load_case, residual, coords,
                                digest, save, verify)
from train_fpo_conditioned_v2 import full_evaluation, tensor, context


def fd_terms(a, k, y, x, re):
    h = 1/64
    f = np.asarray(a[k,y,x],dtype=np.float64)
    dx = (a[k,y,x+1]-a[k,y,x-1])/(2*h)
    dy = (a[k,y+1,x]-a[k,y-1,x])/(2*h)
    lap = (a[k,y,x+1]-2*f+a[k,y,x-1]+a[k,y+1,x]-2*f+a[k,y-1,x])/(h*h)
    dt = (a[k+1,y,x,:2]-a[k-1,y,x,:2])/.1
    advection = f[:,0,None]*dx[:,:2]+f[:,1,None]*dy[:,:2]
    pressure = np.column_stack((dx[:,2],dy[:,2]))
    diffusion = -lap[:,:2]/re
    return dt+advection+pressure+diffusion, dx[:,0]+dy[:,1], [dt,advection,pressure,diffusion], dx[:,:2], dy[:,:2]


def check_fd():
    # Known Poiseuille solution: quadratic velocity, linear pressure, constant
    # in time. Central differences and analytic NS residual are both exact.
    y,x=np.meshgrid(np.arange(8)/64,np.arange(8)/64,indexing='ij')
    for re in (100.,382.,1000.):
        f=np.stack((1-y*y,np.zeros_like(x),-2*x/re),-1)
        a=np.broadcast_to(f,(4,)+f.shape)
        yy,xx=np.meshgrid(np.arange(1,7),np.arange(1,7),indexing='ij')
        r,d,*_=fd_terms(a,2,yy.ravel(),xx.ravel(),re)
        if np.max(np.abs(r))>1e-10 or np.max(np.abs(d))>1e-10:
            raise AssertionError('Finite difference diagnostic failed manufactured solution')


def physics(model, case):
    rng=np.random.default_rng(20261030)
    rows=[]
    with torch.no_grad():
        c=context(model,case)
        if c is not None:c=c.detach()
    for k in [12,32,52,72,112,152,192,220]:
        y,x=case['interior_yx'][rng.integers(len(case['interior_yx']),size=1024)].T
        reference_r,reference_d,terms,dx,dy=fd_terms(case['field'],k,y,x,case['record']['reynolds'])
        q=tensor(coords(np.full(len(y),k),y,x)).requires_grad_(True)
        r=residual(q,model(q,c),case['record']['reynolds']).detach().cpu().numpy().astype(np.float64)
        momentum_scale=max(float(np.sqrt(sum(np.mean(np.sum(t*t,axis=1)) for t in terms))),1e-12)
        gradient_scale=max(float(np.sqrt(np.mean(np.sum(dx*dx+dy*dy,axis=1)))),1e-12)
        def metrics(m,d):
            mr=float(np.sqrt(np.mean(np.sum(m*m,axis=1))))
            dr=float(np.sqrt(np.mean(d*d)))
            return dict(momentum_rms=mr,continuity_rms=dr,
                        momentum_over_reference_term_scale=mr/momentum_scale,
                        continuity_over_reference_gradient_scale=dr/gradient_scale)
        rows.append(dict(frame=k,samples=len(y),momentum_reference_scale=momentum_scale,
                         gradient_reference_scale=gradient_scale,
                         model=metrics(r[:,:2],r[:,2]),reference_fd=metrics(reference_r,reference_d)))
    return dict(path=case['record']['path'],role=case['record']['proposed_split'],frames=rows,
                mean={kind:{key:float(np.mean([r[kind][key] for r in rows])) for key in rows[0][kind]}
                      for kind in ['model','reference_fd']})


def run():
    verify();check_fd();torch.set_num_threads(4)
    freeze=json.loads((WORK/'diagnostic_freeze_v1.json').read_text())
    for path,checksum in freeze['files'].items():
        if digest(path)!=checksum:raise ValueError('Diagnostic input changed: '+path)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    out=WORK/'diagnostics'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    records=json.loads((WORK/'prepared.json').read_text())['records']
    if any(r['proposed_split']=='test' for r in records):raise ValueError('Main test sealed')
    cases=[load_case(r) for r in records]
    for case in cases:
        for name in ['geometry.npz','samples.npz']:
            path=str(Path(case['record']['cache'])/name)
            if digest(path)!=case['record']['hashes'][path]:raise ValueError('Cache changed')
        case['gpu_geometry']=tensor(case['geometry'][None])
    root=WORK/'training/139135'
    specifications=[('conditional_best',root/'conditional/conditional_fourier_50000/best.pt',cases,False),
                    ('conditional_last',root/'conditional/conditional_fourier_50000/last.pt',cases,False),
                    ('single_fourier_pde',root/'ablations/fourier_20000/best.pt',[cases[0]],True),
                    ('single_fourier_data_only',root/'ablations/fourier_data_only_20000/best.pt',[cases[0]],True)]
    summaries=[]
    for label,path,evaluation_cases,temporal in specifications:
        state=torch.load(path,map_location='cpu',weights_only=True)
        config=state['config'];model=ConditionalPINN(config,[0.,0.,0.],[1.,1.,1.]).cuda()
        model.load_state_dict(state['state']);model.eval()
        folder=out/label;folder.mkdir()
        # All training as well as validation grids for conditional checkpoints.
        accuracy=full_evaluation(model,evaluation_cases,folder,temporal)
        diagnostics=[physics(model,c) for c in evaluation_cases]
        summary=dict(name=label,checkpoint=str(path),checkpoint_sha256=digest(path),step=state['step'],
                     accuracy=accuracy,physics=diagnostics,weights_changed=False)
        save(folder/'summary.json',summary);summaries.append(summary)
        print('DIAGNOSED',label,flush=True)
        del model;torch.cuda.empty_cache()
    result=dict(status='CHECKPOINT_DIAGNOSIS_COMPLETE',results=summaries,manufactured_fd_check_passed=True,
                first_two_frames_excluded=True,main_test_opened=False,weights_changed=False,
                normalized_residual_policy='RMS momentum divided by RMS energy scale of reference time/advection/pressure/diffusion terms; continuity by reference velocity-gradient RMS',
                limitations=['Reference finite differences are a discretized consistency check, not zero-error ground truth derivatives',
                             'Point diagnostics sample eight times and 1024 interior fluid points per time',
                             'This diagnoses development checkpoints, not final hypothesis significance'])
    save(out/'summary.json',result)


if __name__=='__main__':run()
