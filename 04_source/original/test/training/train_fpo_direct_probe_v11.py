"""Known-case direct space-time reconstruction diagnostic; NOT a causal stepper."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from fpo_state_v4 import StatePINN,residual
from fpo_step_v5a import load_cases,grid,conditioning,project_known_constraints
from diagnose_fpo_stable_v1 import metric
from fpo_common_v11 import WORK,save,digest,verify


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
            target=case['data'][offset+1]
            mask=case['interior_gpu'];h=1/64
            u,v=value[:,:,0],value[:,:,1]
            div=(torch.roll(u,-1,1)-torch.roll(u,1,1)+torch.roll(v,-1,0)-torch.roll(v,1,0))/(2*h)
            rows.append(dict(offset=offset,frame=offset+3,time=offset*.05,model=metric(value,target,case['fluid']),
                             persistence=metric(initial,target,case['fluid']),interior_div_rms=float(div[mask].square().mean().sqrt())))
    means={k:float(np.mean([r['model'][k] for r in rows])) for k in rows[0]['model']}
    result=dict(score=sum(means.values())/2,failure=False,predicted_frames=len(rows),mean_errors=means,
                interior_div_rms=float(np.mean([r['interior_div_rms'] for r in rows])),
                direct_query_gpu_ms={k:float(np.percentile(times,p)) for k,p in [('p50',50),('p95',95),('p99',99)]},
                causal_step_latency=False,scope='KNOWN_TRAIN_RECONSTRUCTION_FROM_PREFIX_AND_TIME')
    save(out/'per_frame.json',rows);save(out/'summary.json',result);model.train();return result


def train(config,case,out,qbase,normal):
    out.mkdir(parents=True,exist_ok=False);save(out/'config.json',config)
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    model=StatePINN(config,normal['mean'],normal['std']).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['updates'],eta_min=config['lr']*.05)
    history=[];best=float('inf');coverage=np.zeros(238,dtype=bool);order=rng.permutation(238);cursor=0;start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        initial=query(model,case,qbase,0.)
        torch.testing.assert_close(initial,case['raw'][:,:,:3],rtol=1e-5,atol=1e-6)
    for update in range(1,config['updates']+1):
        if cursor==238:order=rng.permutation(238);cursor=0
        offset=int(order[cursor])+1;cursor+=1;coverage[offset-1]=True
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
            result=evaluate(model,case,out/'assessments'/f'update_{update:06d}',qbase)
            row=dict(update=update,data=float(data.detach()),pde=float(pde.detach()),boundary=float(bc.detach()),assessment=result,elapsed_seconds=time.monotonic()-start)
            history.append(row);save(out/'history.json',history)
            saved=dict(state=model.state_dict(),config=config,update=update,score=result['score'],source_freeze=digest(WORK/'freeze.json'))
            torch.save(saved,out/'last.pt')
            if result['score']<best:best=result['score'];torch.save(saved,out/'best.pt')
            print(config['name'],update,json.dumps(result),flush=True)
    chosen=torch.load(out/'best.pt',weights_only=True,map_location='cuda');model.load_state_dict(chosen['state'])
    selected=evaluate(model,case,out/'selected_checkpoint',qbase)
    result=dict(status='DIRECT_RECONSTRUCTION_PROBE_EVALUATED',config=config,accuracy=selected,selected_update=chosen['update'],
                all238_training_frames_visited=bool(coverage.all()),training_seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                checkpoint_sha256=digest(out/'best.pt'),independent_validation=False,causal_stepper=False,llm_ready=False,
                limitations=['Known trajectory used for training and selection; not heldout forecasting',
                             'Absolute time query, no recurrent own states; not directly comparable causal latency',
                             'Primitive output PINN soft physics; different representation bundle from curl stepper'])
    save(out/'summary.json',result);return result


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    receipt=json.loads((WORK/'prepared.json').read_text());r=receipt['records'][0]
    if r['proposed_split']!='train':raise ValueError('Known TRAIN only')
    case=load_cases([r])[0]
    with np.load(r['state_cache'],allow_pickle=False) as z:case['raw']=torch.tensor(z['features'],device='cuda')
    case['interior_gpu']=torch.zeros_like(case['fluid']);y,x=case['interior'].T;case['interior_gpu'][y,x]=True
    normal=json.loads(Path(receipt['normalization']).read_text());qbase=grid();results=[]
    out=WORK/('smoke' if args.smoke else 'training')/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    for c in json.loads((WORK/'protocol.json').read_text())['candidates']:
        if args.smoke:c=dict(c,updates=3,warmup=0,assess_every=3)
        results.append(train(c,case,out/c['name'],qbase,normal));torch.cuda.empty_cache()
    save(out/'summary.json',dict(status='SMOKE_COMPLETE' if args.smoke else 'DIRECT_RECONSTRUCTION_PROBES_COMPLETE',results=results,causal_stepper=False,independent_validation=False))


if __name__=='__main__':run()
