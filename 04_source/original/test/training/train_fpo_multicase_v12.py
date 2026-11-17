"""69 TRAIN /6 geometry-disjoint development validation, direct future time queries."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from fpo_state_v4 import StatePINN,residual,errors
from fpo_step_v5a import load_cases,grid,conditioning,project_known_constraints,fingerprint
from fpo_common_v12 import WORK,save,digest,verify


def query(model,case,qbase,time_value):
    q=qbase.clone();q[:,2]=time_value
    context=conditioning(model,case['raw'],case['geometry'],case['record']['reynolds'])
    value=model(q,context,cached_features=case['raw'].reshape(-1,6)).reshape(256,1024,3)
    return project_known_constraints(value,case['fluid'])


def evaluate(model,case,out,qbase):
    out.mkdir(parents=True,exist_ok=False);rows=[];times=[];model.eval()
    with torch.inference_mode():
        initial=case['raw'][:,:,:3]
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
            rows.append(dict(offset=offset,frame=offset+3,time=offset*.05,model=errors(value.cpu().numpy()[case['fluid_np']],target[case['fluid_np']]),
                             persistence=errors(initial.cpu().numpy()[case['fluid_np']],target[case['fluid_np']]),
                             interior_div_rms=float(div[mask].square().mean().sqrt()),reference_div_rms=float(np.sqrt(np.mean(refdiv**2)))))
    means={k:float(np.mean([r['model'][k] for r in rows])) for k in rows[0]['model']}
    result=dict(score=sum(means.values())/2,failure=False,predicted_frames=len(rows),mean_errors=means,
                interior_div_rms=float(np.mean([r['interior_div_rms'] for r in rows])),
                direct_query_gpu_ms={k:float(np.percentile(times,p)) for k,p in [('p50',50),('p95',95),('p99',99)]},
                causal_step_latency=False,scope=case['record']['proposed_split'],path=case['record']['path'])
    u,v=initial[:,:,0],initial[:,:,1];h=1/64
    initial_div=(torch.roll(u,-1,1)-torch.roll(u,1,1)+torch.roll(v,-1,0)-torch.roll(v,1,0))/(2*h)
    limit=max(3*float(initial_div[mask].square().mean().sqrt()),3*float(np.mean([r['reference_div_rms'] for r in rows])),1e-3)
    result['mass_limit']=limit;result['mass_pass']=result['interior_div_rms']<=limit
    if fingerprint(case['record']['field'])!=case['record']['field_fingerprint']:raise ValueError('Reference changed')
    save(out/'per_frame.json',rows);save(out/'summary.json',result);model.train();return result


def train(config,train_cases,val_cases,out,qbase,normal):
    out.mkdir(parents=True,exist_ok=False);save(out/'config.json',config)
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    model=StatePINN(config,normal['mean'],normal['std']).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['updates'],eta_min=config['lr']*.05)
    history=[];best=float('inf');coverage=np.zeros((len(train_cases),238),dtype=bool)
    orders=[rng.permutation(238) for _ in train_cases];cursors=np.zeros(len(train_cases),dtype=int)
    case_order=rng.permutation(len(train_cases));case_cursor=0;counts=np.zeros(len(train_cases),dtype=int);start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        case=train_cases[0]
        initial=query(model,case,qbase,0.)
        torch.testing.assert_close(initial,case['raw'][:,:,:3],rtol=1e-5,atol=1e-6)
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
        pde=torch.zeros((),device='cuda');ramp=min(1.,max(0.,(update-config['warmup'])/1000))
        if ramp>0:
            y,x=case['interior'][rng.integers(len(case['interior']),size=config['collocation'])].T
            qc=torch.tensor(np.column_stack((4+(x+.5+rng.uniform(-.4,.4,len(x)))/64,6+(y+.5+rng.uniform(-.4,.4,len(y)))/64,rng.uniform(0,11.9,len(x)))),device='cuda',dtype=torch.float32,requires_grad=True)
            pde=residual(qc,model(qc,context,case['raw']),case['record']['reynolds']).square().sum(1).mean()
        boundary=case['boundary'][rng.integers(len(case['boundary']),size=128)]
        qb=torch.tensor(np.column_stack((boundary,rng.uniform(0,11.9,len(boundary)))),device='cuda',dtype=torch.float32)
        bc=(model(qb,context,case['raw'])[:,:2]/model.std[:2]).square().mean()
        loss=data+ramp*.01*pde+.01*bc
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite reconstruction loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step()
        if update==1 or update%config['assess_every']==0 or update==config['updates']:
            reports=[evaluate(model,c,out/'assessments'/f'update_{update:06d}'/str(i),qbase) for i,c in enumerate(val_cases)]
            result=dict(score=float(np.mean([r['score'] for r in reports])),cases=reports)
            row=dict(update=update,data=float(data.detach()),pde=float(pde.detach()),boundary=float(bc.detach()),assessment=result,elapsed_seconds=time.monotonic()-start)
            history.append(row);save(out/'history.json',history)
            saved=dict(state=model.state_dict(),config=config,update=update,score=result['score'],source_freeze=digest(WORK/'freeze.json'))
            torch.save(saved,out/'last.pt')
            if result['score']<best:best=result['score'];torch.save(saved,out/'best.pt')
            print(config['name'],update,json.dumps(result),flush=True)
    chosen=torch.load(out/'best.pt',weights_only=True,map_location='cuda');model.load_state_dict(chosen['state'])
    selected=[evaluate(model,c,out/'selected_checkpoint'/str(i),qbase) for i,c in enumerate(val_cases)]
    train_monitor=[next(c for c in train_cases if c['record']['family']==family) for family in sorted({c['record']['family'] for c in train_cases})]
    training_accuracy=[evaluate(model,c,out/'training_monitor'/str(i),qbase) for i,c in enumerate(train_monitor)]
    result=dict(status='DIRECT_MULTICASE_DEVELOPMENT_EVALUATED',config=config,accuracy=selected,training_monitor=training_accuracy,selected_update=chosen['update'],
                all238_training_frames_visited=bool(coverage.all()),training_seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                checkpoint_sha256=digest(out/'best.pt'),independent_validation=False,causal_stepper=False,llm_ready=False,
                train_scenarios=len(train_cases),validation_scenarios=len(val_cases),training_case_updates=counts.tolist(),
                working_gate=dict(accuracy_pass=all(not r['failure'] and r['mean_errors']['velocity_relative_l2']<=.1 and r['mean_errors']['pressure_relative_l2']<=.2 for r in selected),
                    mass_pass=all(not r['failure'] and r['mass_pass'] for r in selected),
                    direct_query_latency_pass=all(not r['failure'] and r['direct_query_gpu_ms']['p99']<=5 for r in selected)),
                limitations=['Geometry-disjoint development validation used for checkpoint selection; not final independent test',
                             'Absolute time query, no recurrent own states; not directly comparable causal latency',
                             'Primitive output PINN soft physics; different representation bundle from curl stepper'])
    save(out/'summary.json',result);return result


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    receipt=json.loads((WORK/'prepared.json').read_text());records=receipt['records']
    if args.smoke:records=[next(r for r in records if r['proposed_split']==role) for role in ['train','validation']]
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
    train_cases=[c for c in cases if c['record']['proposed_split']=='train'];val_cases=[c for c in cases if c['record']['proposed_split']=='validation']
    key=lambda c:(c['record']['family'],c['record']['geometry_id'])
    if {key(c) for c in train_cases}&{key(c) for c in val_cases}:raise ValueError('Geometry leakage')
    if not args.smoke and (len(train_cases),len(val_cases))!=(69,6):raise ValueError('Expected69/6')
    normal=json.loads(Path(receipt['normalization']).read_text());qbase=grid();results=[]
    out=WORK/('smoke' if args.smoke else 'training')/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    for c in json.loads((WORK/'protocol.json').read_text())['candidates']:
        if args.smoke:c=dict(c,updates=3,warmup=0,assess_every=3)
        results.append(train(c,train_cases,val_cases,out/c['name'],qbase,normal));torch.cuda.empty_cache()
    save(out/'summary.json',dict(status='SMOKE_COMPLETE' if args.smoke else 'MULTICASE_DEVELOPMENT_COMPLETE',results=results,causal_stepper=False,independent_validation=False))


if __name__=='__main__':run()
