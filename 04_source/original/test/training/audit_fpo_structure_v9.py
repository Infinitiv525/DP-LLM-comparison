"""Readonly gradients, discretization consistency, reference floor, input sensitivity."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from fpo_stable_v6 import StablePINN,grid,stable_step,load_cases,features,conditioning,grid_physics
from train_fpo_stable_v6 import continuous_physics
from diagnose_fpo_stable_v1 import metric
from fpo_common_v9 import WORK,save,digest,verify


def diagnostic(model,case,previous,current,target,reference_current,qgrid,rng):
    model.zero_grad(set_to_none=True)
    pred=stable_step(model,previous,current,case,qgrid)
    data=((pred[case['fluid']]-target[case['fluid']])/(model.std*.05)).square().mean()
    mom,div=grid_physics(current,pred,case,model.std[:2].mean())
    raw=features(previous,current);c=conditioning(model,raw,case['geometry'],case['record']['reynolds'])
    pde,bc=continuous_physics(model,raw,c,case,rng,True,model.config_for_audit)
    terms=dict(data=data,pde=.01*pde,boundary=.01*bc,momentum=.1*mom,continuity=.1*div)
    params=list(model.parameters());vectors={};scalars={}
    for name,loss in terms.items():
        gradients=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        vector=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for p,g in zip(params,gradients)])
        finite=bool(torch.isfinite(vector).all());vectors[name]=vector
        scalars[name]=dict(weighted_loss=float(loss.detach()),gradient_norm=float(vector.norm()) if finite else None,finite_gradient=finite)
    cosines={}
    for a in vectors:
        for b in vectors:
            if a>=b:continue
            denominator=vectors[a].norm()*vectors[b].norm()
            cosines[a+'__'+b]=float(torch.dot(vectors[a],vectors[b])/denominator) if denominator>1e-16 and torch.isfinite(denominator) else None
    yy,xx=case['interior'][rng.integers(len(case['interior']),size=512)].T
    q=qgrid.reshape(256,1024,3)[yy,xx].clone().requires_grad_(True)
    continuous=model(q,c,raw,case['weight'])
    discrete=pred[yy,xx]
    # Compare raw continuous output to raw grid, avoiding pressure-gauge offset.
    difference=continuous-discrete;difference=torch.cat((difference[:,:2],difference[:,2:3]-difference[:,2:3].mean()),1)
    mismatch=(difference/model.std).square().mean(0).sqrt().detach().cpu().tolist()
    with torch.no_grad():
        rm,rd=grid_physics(reference_current,target,case,model.std[:2].mean())
        base=metric(pred,target,case['fluid'])
        perturbations=[]
        for amplitude in [1e-5,1e-4,1e-3]:
            noise=torch.randn(current.shape,device='cuda')*model.std*amplitude*case['fluid'][:,:,None]
            value=stable_step(model,previous+noise,current+noise,case,qgrid)
            input_norm=(noise[case['fluid']]/model.std).square().mean().sqrt()
            output_norm=((value-pred)[case['fluid']]/model.std).square().mean().sqrt()
            perturbations.append(dict(relative_noise=amplitude,normalized_gain=float(output_norm/input_norm),error=metric(value,target,case['fluid'])))
    return dict(terms=scalars,gradient_cosines=cosines,continuous_vs_grid_std_normalized_rms=mismatch,
                continuous_vs_grid_note='Same physical cell centers at localdt; velocity compares analytic curl versus gridFD curl. Pressure difference centered on sampled points.',
                reference_fd_floor=dict(momentum=float(rm),continuity=float(rd)),model_error=base,
                perturbations=perturbations,perturbation_note='Same random noise on previous/current keeps their difference unchanged; may introduce nonphysical divergence. Diagnostic sensitivity, not proof of illposedness.')


def run():
    verify();torch.set_num_threads(4);torch.manual_seed(20261031)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    receipt=json.loads((WORK/'prepared.json').read_text());record=receipt['records'][0]
    if record['proposed_split']!='train':raise ValueError('TRAIN only')
    case=load_cases([record])[0]
    with np.load(receipt['weight_file'],allow_pickle=False) as z:case['weight']=torch.tensor(z['weight'],device='cuda')
    case['interior_gpu']=torch.zeros_like(case['fluid']);y,x=case['interior'].T;case['interior_gpu'][y,x]=True
    normal=json.loads(Path(receipt['normalization']).read_text());qgrid=grid()
    out=WORK/'diagnostics'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False);rows=[]
    protocol=json.loads((WORK/'protocol.json').read_text())
    for source in protocol['checkpoints']:
        saved=torch.load(source,weights_only=True,map_location='cuda');config=saved['config']
        model=StablePINN(config,normal['mean'],normal['std']).cuda();model.load_state_dict(saved['state']);model.eval();model.config_for_audit=config
        rng=np.random.default_rng(20261031);torch.manual_seed(20261031)
        for k in [1,35,69,103,137,171,205,237]:
            result=diagnostic(model,case,case['data'][k-1],case['data'][k],case['data'][k+1],case['data'][k],qgrid,rng)
            rows.append(dict(checkpoint=source,role='TRUE_TRAIN_WINDOW',current_frame=k+2,diagnostic=result))
            save(out/'partial.json',rows)
        previous,current=case['data'][0],case['data'][1]
        for offset in range(238):
            if offset in [0,31,119,237]:
                result=diagnostic(model,case,previous,current,case['data'][offset+2],case['data'][offset+1],qgrid,rng)
                rows.append(dict(checkpoint=source,role='OWN_ROLLOUT_STATE',predicted_offset=offset+1,diagnostic=result));save(out/'partial.json',rows)
            with torch.no_grad():pred=stable_step(model,previous,current,case,qgrid)
            if not torch.isfinite(pred).all() or pred.abs().max()>1000:raise RuntimeError('Frozen rollout unexpectedly diverged')
            previous,current=current,pred
        del model,saved;torch.cuda.empty_cache();print('AUDITED',source,flush=True)
    save(out/'summary.json',dict(status='READONLY_STRUCTURE_AUDIT_COMPLETE',rows=rows,optimizer_used=False,
         main_test_opened=False,independent_validation=False,
         limitations=protocol['limitations']))


if __name__=='__main__':run()
