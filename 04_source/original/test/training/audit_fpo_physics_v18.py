"""Frozen-checkpoint diagnosis; no optimizer, fitting, or future model inputs."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from fpo_common_v18 import WORK,save,digest,verify
from fpo_direct_curl_v15 import DirectPINN
from fpo_step_v5a import grid,conditioning,fingerprint
from train_fpo_direct_curl_v15 import query
from evaluate_fpo_transfer_v16 import reference_state
from fpo_state_v4 import errors
from fpo_stable_v6 import grid_physics
from fpo_physics_audit_v18 import continuous_terms,grid_terms,time_stencil,temporal_difference,describe,compare


def detach(terms):return {k:v.detach() for k,v in terms.items()}


def known_case(record):
    if record['proposed_split'] not in ['train','validation']:raise ValueError('Sealed role')
    if fingerprint(record['field'])!=record['field_fingerprint']:raise ValueError('Changed field')
    if digest(record['state_cache'])!=record['state_cache_sha256']:raise ValueError('Changed prefix')
    for name in ['geometry.npz','samples.npz']:
        p=str(Path(record['cache'])/name)
        if digest(p)!=record['hashes'][p]:raise ValueError('Changed geometry cache')
    with np.load(Path(record['cache'])/'geometry.npz',allow_pickle=False) as z:
        geometry=z['encoded'].copy();fluid=z['fluid'].copy();interior=z['interior_yx'].copy()
    with np.load(record['state_cache'],allow_pickle=False) as z:raw=z['features'].copy()
    # Future TRAIN trajectories are not loaded either: this is a prefix-only audit.
    known=dict(raw=torch.tensor(raw,device='cuda'),geometry=torch.tensor(geometry[None],device='cuda'),
               fluid=torch.tensor(fluid,device='cuda'),record={'reynolds':record['reynolds']})
    mask=torch.zeros_like(known['fluid']);y,x=interior.T;mask[y,x]=True
    scoring=dict(record=record,reference=np.load(record['field'],mmap_mode='r',allow_pickle=False),
                 fluid_np=fluid,interior_gpu=mask)
    return known,scoring


def field_query(model,known,qbase,t):
    with torch.no_grad():value=query(model,known,qbase,t)
    if not torch.isfinite(value).all() or value.abs().max()>1000:raise RuntimeError('Invalid prediction; no clipping')
    return value


def audit_case(model,known,scoring,sampling,protocol,out,qbase):
    out.mkdir(parents=True,exist_ok=False)
    yx=torch.tensor(sampling['yx'],device='cuda',dtype=torch.long);y,x=yx.T;cpu_yx=yx.cpu()
    jitter=torch.tensor(sampling['jitter_cells'],device='cuda',dtype=torch.float32)
    with torch.no_grad():
        context=conditioning(model,known['raw'],known['geometry'],known['record']['reynolds'])
        initial=field_query(model,known,qbase,0.)
        torch.testing.assert_close(initial,known['raw'][:,:,:3],rtol=1e-5,atol=1e-6)
    cpu_mask=torch.from_numpy(scoring['fluid_np']);scale=model.std[:2].mean();cpu_scale=scale.cpu()
    cpu_scoring=dict(record=known['record'],interior_gpu=scoring['interior_gpu'].cpu())
    reference=scoring['reference'];re=known['record']['reynolds'];rows=[];pointwise={}
    for offset in protocol['offsets']:
        t=offset*.05;frame=offset+3
        current=field_query(model,known,qbase,t)
        q=torch.stack((4+(x+.5)/64,6+(y+.5)/64,torch.full_like(x,t,dtype=torch.float32)),1).requires_grad_()
        ad=detach(continuous_terms(q,model(q,context,known['raw']),re))
        qj=q.detach().clone();qj[:,:2]+=jitter/64;qj.requires_grad_()
        jittered=detach(continuous_terms(qj,model(qj,context,known['raw']),re))
        qinfo=dict(continuous_centre=describe(ad,scale),continuous_jitter=describe(jittered,scale))
        fd={}
        for step in protocol['neural_fd_time_steps']:
            times,backward=time_stencil(t,step)
            minus=field_query(model,known,qbase,times[0]);other=field_query(model,known,qbase,times[1])
            temporal=temporal_difference(current,minus,other,step,backward)
            terms=grid_terms(current,temporal,yx,re);fd[str(step)]=terms
            qinfo['fd_'+str(step)]=dict(terms=describe(terms,scale),versus_continuous=compare(ad,terms),
                time_stencil=times,backward_second_order=backward)
        qinfo['coarse_to_fine_fd']=compare(fd[str(protocol['neural_fd_time_steps'][0])],fd[str(protocol['neural_fd_time_steps'][1])])
        target=np.asarray(reference[frame]);ref=reference_state(reference,frame,cpu_mask)
        backward=offset==238
        ref_minus=reference_state(reference,frame-1,cpu_mask)
        ref_other=reference_state(reference,frame-2 if backward else frame+1,cpu_mask)
        ref_dt=temporal_difference(ref,ref_minus,ref_other,.05,backward)
        ref_terms=grid_terms(ref,ref_dt,cpu_yx,re)
        previous=field_query(model,known,qbase,t-.05)
        midpoint_momentum,_=grid_physics(previous,current,scoring,scale)
        ref_midpoint_momentum,_=grid_physics(ref_minus,ref,cpu_scoring,cpu_scale)
        row=dict(offset=offset,frame=frame,time=t,scope=scoring['record']['proposed_split'],path=scoring['record']['path'],
                 field_errors=errors(current.cpu().numpy()[scoring['fluid_np']],target[scoring['fluid_np']]),
                 reference_instantaneous_fd=describe(ref_terms,cpu_scale),reference_fd_time_step=.05,
                 reference_backward_second_order=backward,
                 v17_midpoint_fd_momentum_loss_full_interior=float(midpoint_momentum),
                 reference_midpoint_fd_momentum_loss_full_interior=float(ref_midpoint_momentum),
                 midpoint_time=t-.025,**qinfo)
        rows.append(row)
        pointwise['offset_'+str(offset)+'_q']=q.detach().cpu().numpy()
        pointwise['offset_'+str(offset)+'_q_jitter']=qj.detach().cpu().numpy()
        for source,terms in [('ad',ad),('ad_jitter',jittered),('reference_fd',ref_terms)]+[('fd_'+step,terms) for step,terms in fd.items()]:
            for key,value in terms.items():pointwise[f'offset_{offset}_{source}_{key}']=value.detach().cpu().numpy()
        print('AUDIT_TIME',out.parent.name,out.name,offset,flush=True)
    if fingerprint(scoring['record']['field'])!=scoring['record']['field_fingerprint']:raise ValueError('Reference changed')
    save(out/'summary.json',dict(path=scoring['record']['path'],scope=scoring['record']['proposed_split'],rows=rows))
    np.savez_compressed(out/'pointwise.npz',yx=np.asarray(sampling['yx']),**pointwise)
    return rows


def aggregate(rows):
    def mean(f):return float(np.mean([f(r) for r in rows]))
    return dict(case_time_count=len(rows),field_mean_errors={k:mean(lambda r:r['field_errors'][k]) for k in rows[0]['field_errors']},
        continuous_centre_momentum_loss=mean(lambda r:r['continuous_centre']['momentum']['mean_square_over_velocity_scale2']),
        continuous_jitter_momentum_loss=mean(lambda r:r['continuous_jitter']['momentum']['mean_square_over_velocity_scale2']),
        fine_fd_momentum_loss=mean(lambda r:r['fd_0.0025']['terms']['momentum']['mean_square_over_velocity_scale2']),
        coarse_fd_momentum_loss=mean(lambda r:r['fd_0.025']['terms']['momentum']['mean_square_over_velocity_scale2']),
        fine_fd_ad_momentum_difference_rms=mean(lambda r:r['fd_0.0025']['versus_continuous']['momentum']['difference_rms']),
        fine_fd_ad_velocity_difference_rms=mean(lambda r:r['fd_0.0025']['versus_continuous']['velocity']['difference_rms']),
        reference_instantaneous_fd_momentum_loss=mean(lambda r:r['reference_instantaneous_fd']['momentum']['mean_square_over_velocity_scale2']),
        v17_midpoint_fd_momentum_loss_full_interior=mean(lambda r:r['v17_midpoint_fd_momentum_loss_full_interior']),
        reference_midpoint_fd_momentum_loss_full_interior=mean(lambda r:r['reference_midpoint_fd_momentum_loss_full_interior']))


def run():
    verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    protocol=json.loads((WORK/'protocol.json').read_text());receipt=json.loads((WORK/'prepared.json').read_text())
    samples=json.loads((WORK/'sampling.json').read_text());records=receipt['records']
    if (sum(r['proposed_split']=='train' for r in records),sum(r['proposed_split']=='validation' for r in records))!=(3,6):raise ValueError('Expected3TRAIN6VALaudit')
    cases=[known_case(r) for r in records];qbase=grid()
    normal=json.loads(Path(receipt['normalization']).read_text())
    out=WORK/'audit'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False);results=[];outputs={}
    for c in protocol['checkpoints']:
        chosen=torch.load(c['path'],weights_only=True,map_location='cuda')
        if chosen['update']!=c['update']:raise ValueError('Unexpected checkpoint update')
        model=DirectPINN(chosen['config'],normal['mean'],normal['std']).cuda()
        model.load_state_dict(chosen['state']);model.eval();model.requires_grad_(False)
        rows=[]
        for i,(known,scoring) in enumerate(cases):
            rows+=audit_case(model,known,scoring,samples['cases'][scoring['record']['path']],protocol,out/c['name']/str(i),qbase)
        result=dict(checkpoint=c,by_scope={role:aggregate([r for r in rows if r['scope']==role]) for role in ['train','validation']},
            by_scope_horizon={role:{label:aggregate([r for r in rows if r['scope']==role and r['offset'] in offsets])
                for label,offsets in [('early',[1,5,10]),('middle',[20,40]),('late',[80,160,238])]}
                for role in ['train','validation']},optimizer_updates=0,main_test_opened=False)
        save(out/c['name']/'summary.json',result);results.append(result)
        del model,chosen;torch.cuda.empty_cache()
    verify()
    for p in out.rglob('*'):
        if p.is_file():outputs[str(p.relative_to(out))]=dict(bytes=p.stat().st_size,sha256=digest(p))
    save(out/'output_manifest.json',outputs)
    save(out/'summary.json',dict(status='READ_ONLY_PHYSICS_AUDIT_COMPLETE',results=results,
        cases=9,offsets=protocol['offsets'],points_per_case=samples['points_per_case'],optimizer_updates=0,
        full_238frame_assessment=False,main_test_opened=False,independent_final_test=False,llm_ready=False))
    print('READ_ONLY_PHYSICS_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':run()
