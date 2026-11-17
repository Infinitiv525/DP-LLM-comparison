"""Matched PDE-weight fine-tuning on3knownTRAINcases, architecture fixed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from fpo_state_v4 import StatePINN,residual,errors
from fpo_step_v5a import load_cases,grid,conditioning,project_known_constraints,fingerprint
from fpo_common_v14 import WORK,save,digest,verify
from fpo_own_v13 import make_model
from fpo_stable_v6 import grid_physics
from fpo_physics_selection_v14 import checkpoint_key


def query(model,case,qbase,time_value):
    q=qbase.clone();q[:,2]=time_value
    context=conditioning(model,case['raw'],case['geometry'],case['record']['reynolds'])
    value=model(q,context,cached_features=case['raw'].reshape(-1,6)).reshape(256,1024,3)
    return project_known_constraints(value,case['fluid'])


def evaluate(model,case,out,qbase):
    out.mkdir(parents=True,exist_ok=False);rows=[];times=[];model.eval()
    with torch.inference_mode():
        initial=case['raw'][:,:,:3]
        previous=initial
        for offset in range(1,239):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();value=query(model,case,qbase,offset*.05);b.record();b.synchronize();times.append(a.elapsed_time(b))
            if not torch.isfinite(value).all() or value.abs().max()>1000:
                save(out/'failure.json',dict(offset=offset,reason='nonfinite or abs>1000'));model.train()
                result=dict(score=1e6,failure=True,predicted_frames=len(rows));save(out/'summary.json',result);return result
            # Future validation labels stay CPU-side and never enter query().
            target=np.asarray(case['reference'][offset+3])
            mask=case['interior_gpu'];h=1/64
            u,v=value[:,:,0],value[:,:,1]
            div=(torch.roll(u,-1,1)-torch.roll(u,1,1)+torch.roll(v,-1,0)-torch.roll(v,1,0))/(2*h)
            yy,xx=case['interior'].T
            refdiv=(target[yy,xx+1,0]-target[yy,xx-1,0]+target[yy+1,xx,1]-target[yy-1,xx,1])*32
            mom,_=grid_physics(previous,value,case,model.std[:2].mean())
            refmom,_=grid_physics(case['data'][offset],case['data'][offset+1],case,model.std[:2].mean())
            rows.append(dict(offset=offset,frame=offset+3,time=offset*.05,model=errors(value.cpu().numpy()[case['fluid_np']],target[case['fluid_np']]),
                             persistence=errors(initial.cpu().numpy()[case['fluid_np']],target[case['fluid_np']]),
                             interior_div_rms=float(div[mask].square().mean().sqrt()),reference_div_rms=float(np.sqrt(np.mean(refdiv**2))),
                             fd_momentum_loss=float(mom),reference_fd_momentum_loss=float(refmom)))
            previous=value
    means={k:float(np.mean([r['model'][k] for r in rows])) for k in rows[0]['model']}
    result=dict(score=sum(means.values())/2,failure=False,predicted_frames=len(rows),mean_errors=means,
                interior_div_rms=float(np.mean([r['interior_div_rms'] for r in rows])),
                direct_query_gpu_ms={k:float(np.percentile(times,p)) for k,p in [('p50',50),('p95',95),('p99',99)]},
                causal_step_latency=False,scope=case['record']['proposed_split'],path=case['record']['path'])
    u,v=initial[:,:,0],initial[:,:,1];h=1/64
    initial_div=(torch.roll(u,-1,1)-torch.roll(u,1,1)+torch.roll(v,-1,0)-torch.roll(v,1,0))/(2*h)
    limit=max(3*float(initial_div[mask].square().mean().sqrt()),3*float(np.mean([r['reference_div_rms'] for r in rows])),1e-3)
    result['mass_limit']=limit;result['mass_pass']=result['interior_div_rms']<=limit
    result['mean_fd_momentum_loss']=float(np.mean([r['fd_momentum_loss'] for r in rows]))
    result['mean_reference_fd_momentum_loss']=float(np.mean([r['reference_fd_momentum_loss'] for r in rows]))
    if fingerprint(case['record']['field'])!=case['record']['field_fingerprint']:raise ValueError('Reference changed')
    save(out/'per_frame.json',rows);save(out/'summary.json',result);model.train();return result


def train(config,train_cases,val_cases,out,qbase,normal):
    out.mkdir(parents=True,exist_ok=False);save(out/'config.json',config)
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    model=make_model(config,normal['mean'],normal['std']).cuda()
    source=torch.load(config['warm_start'],weights_only=True,map_location='cuda');model.load_state_dict(source['state']);del source
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['updates'],eta_min=config['lr']*.05)
    history=[];coverage=np.zeros((len(train_cases),238),dtype=bool);sampling=hashlib.sha256()
    orders=[rng.permutation(238) for _ in train_cases];cursors=np.zeros(len(train_cases),dtype=int)
    case_order=rng.permutation(len(train_cases));case_cursor=0;counts=np.zeros(len(train_cases),dtype=int);start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        case=train_cases[0]
        initial=query(model,case,qbase,0.)
        torch.testing.assert_close(initial,case['raw'][:,:,:3],rtol=1e-5,atol=1e-6)
    initial_reports=[evaluate(model,c,out/'assessments'/'initial'/str(i),qbase) for i,c in enumerate(val_cases)]
    best=checkpoint_key(initial_reports);best_data=float(np.mean([r['score'] for r in initial_reports]))
    initial_state=dict(state=model.state_dict(),config=config,update=0,score=best_data,source_freeze=digest(WORK/'freeze.json'))
    torch.save(initial_state,out/'best.pt');torch.save(initial_state,out/'best_accuracy.pt')
    for update in range(1,config['updates']+1):
        if case_cursor==len(case_order):case_order=rng.permutation(len(train_cases));case_cursor=0
        ci=int(case_order[case_cursor]);case_cursor+=1;case=train_cases[ci];counts[ci]+=1
        if cursors[ci]==238:orders[ci]=rng.permutation(238);cursors[ci]=0
        offset=int(orders[ci][cursors[ci]])+1;cursors[ci]+=1;coverage[ci,offset-1]=True
        yy,xx=case['fluid_yx'][rng.integers(len(case['fluid_yx']),size=config['batch'])].T
        q=torch.tensor(np.column_stack((4+(xx+.5)/64,6+(yy+.5)/64,np.full(len(xx),offset*.05))),device='cuda',dtype=torch.float32)
        context=conditioning(model,case['raw'],case['geometry'],case['record']['reynolds'])
        pred=model(q,context,cached_features=case['raw'][yy,xx]);target=case['data'][offset+1,yy,xx]
        data=((pred-target)/model.std).square().mean()
        pde=torch.zeros((),device='cuda');pde_parts=torch.zeros(3,device='cuda');ramp=1.
        if ramp>0:
            y,x=case['interior'][rng.integers(len(case['interior']),size=config['collocation'])].T
            qc=torch.tensor(np.column_stack((4+(x+.5+rng.uniform(-.4,.4,len(x)))/64,6+(y+.5+rng.uniform(-.4,.4,len(y)))/64,rng.uniform(0,11.9,len(x)))),device='cuda',dtype=torch.float32,requires_grad=True)
            pde_parts=residual(qc,model(qc,context,case['raw']),case['record']['reynolds']).square().mean(0)
            pde=pde_parts.sum()
        boundary=case['boundary'][rng.integers(len(case['boundary']),size=128)]
        qb=torch.tensor(np.column_stack((boundary,rng.uniform(0,11.9,len(boundary)))),device='cuda',dtype=torch.float32)
        bc=(model(qb,context,case['raw'])[:,:2]/model.std[:2]).square().mean()
        loss=data+config['pde_weight']*pde+.01*bc
        sampling.update(json.dumps(rng.bit_generator.state,sort_keys=True).encode())
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite reconstruction loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step()
        if update==1 or update%config['assess_every']==0 or update==config['updates']:
            reports=[evaluate(model,c,out/'assessments'/f'update_{update:06d}'/str(i),qbase) for i,c in enumerate(val_cases)]
            result=dict(score=float(np.mean([r['score'] for r in reports])),cases=reports)
            row=dict(update=update,data=float(data.detach()),pde=float(pde.detach()),pde_parts=pde_parts.detach().cpu().tolist(),weighted_pde=float(config['pde_weight']*pde.detach()),boundary=float(bc.detach()),gradient_norm_before_clip=float(gradient_norm),assessment=result,elapsed_seconds=time.monotonic()-start)
            history.append(row);save(out/'history.json',history)
            saved=dict(state=model.state_dict(),config=config,update=update,score=result['score'],source_freeze=digest(WORK/'freeze.json'))
            torch.save(saved,out/'last.pt')
            key=checkpoint_key(reports)
            if key<best:best=key;torch.save(saved,out/'best.pt')
            if result['score']<best_data:best_data=result['score'];torch.save(saved,out/'best_accuracy.pt')
            print(config['name'],update,json.dumps(result),flush=True)
    chosen=torch.load(out/'best.pt',weights_only=True,map_location='cuda');model.load_state_dict(chosen['state'])
    selected=[evaluate(model,c,out/'selected_checkpoint'/str(i),qbase) for i,c in enumerate(val_cases)]
    train_monitor=[next(c for c in train_cases if c['record']['family']==family) for family in sorted({c['record']['family'] for c in train_cases})]
    training_accuracy=[evaluate(model,c,out/'training_monitor'/str(i),qbase) for i,c in enumerate(train_monitor)]
    result=dict(status='PHYSICS_WEIGHT_TRAIN_FIT_EVALUATED',config=config,accuracy=selected,training_monitor=training_accuracy,selected_update=chosen['update'],
                all238_training_frames_visited=bool(coverage.all()),training_seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                checkpoint_sha256=digest(out/'best.pt'),independent_validation=False,causal_stepper=False,llm_ready=False,
                train_scenarios=len(train_cases),validation_scenarios=0,train_selection_scenarios=len(val_cases),training_case_updates=counts.tolist(),
                sampling_sha256=sampling.hexdigest(),best_accuracy_checkpoint_sha256=digest(out/'best_accuracy.pt'),
                working_gate=dict(accuracy_pass=all(not r['failure'] and r['mean_errors']['velocity_relative_l2']<=.1 and r['mean_errors']['pressure_relative_l2']<=.2 for r in selected),
                    mass_pass=all(not r['failure'] and r['mass_pass'] for r in selected),
                    direct_query_latency_pass=all(not r['failure'] and r['direct_query_gpu_ms']['p99']<=5 for r in selected)),
                limitations=['Same TRAIN cases used for training and fit-checkpoint selection; not validation/generalization',
                             'Only fixed PDE coefficient differs across arms; physics-first selection may select a less accurate field fit',
                             'Absolute time query, no recurrent own states; not directly comparable causal latency',
                             'Primitive output PINN soft physics; different representation bundle from curl stepper'])
    save(out/'summary.json',result);return result


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    receipt=json.loads((WORK/'prepared.json').read_text());records=receipt['records']
    if any(r['proposed_split']!='train' for r in records):raise ValueError('TRAIN probes only')
    for r in records:
        if r['proposed_split']=='test' or digest(r['state_cache'])!=r['state_cache_sha256']:raise ValueError('Sealed test or changed prefix')
        for name in ['geometry.npz','samples.npz']:
            p=str(Path(r['cache'])/name)
            if digest(p)!=r['hashes'][p]:raise ValueError('Cache changed')
    cases=load_cases(records)
    for case in cases:
        r=case['record']
        with np.load(r['state_cache'],allow_pickle=False) as z:case['raw']=torch.tensor(z['features'],device='cuda')
        case['reference']=np.load(r['field'],mmap_mode='r',allow_pickle=False)
        case['interior_gpu']=torch.zeros_like(case['fluid']);y,x=case['interior'].T;case['interior_gpu'][y,x]=True
    train_cases=cases
    if len(cases)!=3:raise ValueError('Expected3TRAINcases')
    normal=json.loads(Path(receipt['normalization']).read_text());qbase=grid();results=[]
    out=WORK/('smoke' if args.smoke else 'training')/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    for c in json.loads((WORK/'protocol.json').read_text())['candidates']:
        if args.smoke and c['pde_weight']!=1.:continue
        if args.smoke:c=dict(c,updates=3,assess_every=3)
        selected_cases=train_cases[:c['train_count']]
        results.append(train(c,selected_cases,selected_cases,out/c['name'],qbase,normal));torch.cuda.empty_cache()
    save(out/'summary.json',dict(status='SMOKE_COMPLETE' if args.smoke else 'PHYSICS_WEIGHT_PROBES_COMPLETE',results=results,
         matched_sampling=len({r['sampling_sha256'] for r in results})==1,causal_stepper=False,independent_validation=False))


if __name__=='__main__':run()
