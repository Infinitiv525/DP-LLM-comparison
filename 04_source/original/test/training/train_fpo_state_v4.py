"""Prospective correction: known prefix is input; every scored frame is later."""
import argparse
import json
import os
import time
import numpy as np
import torch
from torch import nn
from fpo_state_v4 import StatePINN,load,residual,errors,coords
from fpo_common_v4 import WORK,save,digest,verify


def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device='cuda')


def context(model,case):
    return model.context(case['geo_gpu'],case['low_gpu'],case['record']['reynolds'])


def local(case,q):
    x=((q[:,0]-4)*64-.5).round().long();y=((q[:,1]-6)*64-.5).round().long()
    return case['state_gpu'][y,x]


def predict(model,case,q,c,cached=True):
    return model(q,c,case['state_gpu'],local(case,q) if cached else None)


def validate(model,cases):
    rows=[];model.eval()
    with torch.inference_mode():
        for case in cases:
            q,y,k=case['pools']['validation'];c=context(model,case)
            p=np.concatenate([predict(model,case,tensor(q[j:j+8192]),c).cpu().numpy() for j in range(0,len(q),8192)])
            frames=[errors(p[k==frame],y[k==frame]) for frame in np.unique(k)]
            rows.append(dict(path=case['record']['path'],**{key:float(np.mean([r[key] for r in frames])) for key in frames[0]}))
    model.train()
    return float(np.mean([(r['velocity_relative_l2']+r['pressure_relative_l2'])/2 for r in rows])),rows


def evaluate(model,cases,out):
    model.eval();all_rows=[];summary=[]
    x,y=np.meshgrid(4+(np.arange(1024)+.5)/64,6+(np.arange(256)+.5)/64)
    q=tensor(np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size))))
    with torch.inference_mode():
        for case in cases:
            c=context(model,case);mask=case['fluid'];rows=[]
            a=np.load(case['record']['field'],mmap_mode='r',allow_pickle=False)
            initial=case['initial'][:,:,:3][mask]
            for k in range(4,242):
                q[:,2]=(k-3)*.05
                p=torch.cat([predict(model,case,chunk,c) for chunk in q.split(65536)]).cpu().numpy().reshape(256,1024,3)
                ref=np.asarray(a[k])[mask]
                row=dict(path=case['record']['path'],frame=k,relative_time=(k-3)*.05,
                         model=errors(p[mask],ref),persistence=errors(initial,ref))
                rows.append(row)
            scopes={'all_future':rows,'steps_1_10':rows[:10],'steps_11_40':rows[10:40],
                    'steps_41_120':rows[40:120],'steps_121_238':rows[120:]}
            scores={scope:{kind:{key:float(np.mean([r[kind][key] for r in group])) for key in group[0][kind]}
                            for kind in ['model','persistence']} for scope,group in scopes.items()}
            summary.append(dict(path=case['record']['path'],frames=238,scores=scores))
            all_rows.extend(rows);print('EVALUATED_FUTURE_ONLY',case['record']['path'],flush=True)
    save(out/'per_frame_accuracy.json',all_rows)
    return summary


def latency(model,case):
    model.eval();x,y=np.meshgrid(4+(np.arange(1024)+.5)/64,6+(np.arange(256)+.5)/64)
    host=torch.tensor(np.column_stack((x.ravel(),y.ravel(),np.full(x.size,6.))),dtype=torch.float32)
    q=host.cuda();rows={};cold=[];host_times=[]
    with torch.inference_mode():
        c=context(model,case);features=case['state_gpu'].reshape(-1,6)
        for cached in [True,False]:
            for _ in range(30):model(q,c,case['state_gpu'],features if cached else None)
            rounds=[]
            for _ in range(3):
                values=[]
                for i in range(200):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();model(q,c,case['state_gpu'],features if cached else None);end.record();end.synchronize()
                    values.append(start.elapsed_time(end))
                rounds.append({key:float(np.percentile(values,p)) for key,p in [('p50',50),('p95',95),('p99',99)]})
            rows['grid_prefix_features_cached' if cached else 'bicubic_features_computed_each_query']=rounds
        for _ in range(30):
            torch.cuda.synchronize();start=time.perf_counter();c=context(model,case)
            torch.cuda.synchronize();cold.append((time.perf_counter()-start)*1000)
            torch.cuda.synchronize();start=time.perf_counter()
            model(host.cuda(),c,case['state_gpu'],features).cpu()
            torch.cuda.synchronize();host_times.append((time.perf_counter()-start)*1000)
    return dict(gpu_ms=rows,grid=[1024,256],samples_per_round=200,
                context_gpu_ms={key:float(np.percentile(cold,p)) for key,p in [('p50',50),('p99',99)]},
                host_to_host_cached_grid_ms={key:float(np.percentile(host_times,p)) for key,p in [('p50',50),('p99',99)]},
                cold_samples=30,rendering_included=False,prefix_preprocessing_included=False,
                encoder_and_prefix_tensors_already_on_gpu=True,causal_rollout=False)


def train(config,train_cases,val_cases,out):
    out.mkdir(parents=True,exist_ok=False);save(out/'config.json',config)
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    targets=np.concatenate([c['pools']['all'][1] for c in train_cases])
    mean=targets.mean(0);std=np.maximum(targets.std(0),1e-4)
    model=StatePINN(config,mean,std).cuda();pools=[]
    for case in train_cases:pools.append((tensor(case['pools']['all'][0]),tensor(case['pools']['all'][1])))
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['steps'],eta_min=config['lr']*.05)
    save(out/'normalization.json',dict(mean=mean.tolist(),std=std.tolist(),source='future observations of training cases only'))
    history=[];best=float('inf');start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for step in range(1,config['steps']+1):
        ci=int(rng.integers(len(train_cases)));case=train_cases[ci];qp,yp=pools[ci]
        ids=torch.randint(len(qp),(config['batch'],),device='cuda');q,target=qp[ids],yp[ids]
        c=context(model,case);optimizer.zero_grad(set_to_none=True)
        data=((predict(model,case,q,c)-target)/model.std).square().mean()
        ramp=min(1.,max(0.,(step-config['warmup'])/config['ramp']))
        pde=torch.zeros((),device='cuda')
        if ramp:
            yy,xx=case['interior'][rng.integers(len(case['interior']),size=config['collocation'])].T
            pq=tensor(np.column_stack((4+(xx+.5+rng.uniform(-.4,.4,len(xx)))/64,
                                       6+(yy+.5+rng.uniform(-.4,.4,len(yy)))/64,
                                       rng.uniform(0,11.9,len(xx))))).requires_grad_(True)
            pde=residual(pq,model(pq,c,case['state_gpu']),case['record']['reynolds']).square().sum(1).mean()
        boundary=case['boundary'][rng.integers(len(case['boundary']),size=256)]
        bq=tensor(np.column_stack((boundary,rng.uniform(0,11.9,len(boundary)))))
        bc=(model(bq,c,case['state_gpu'])[:,:2]/model.std[:2]).square().mean()
        # Train-case initial supervision is legal in both formulations; only
        # the state-aware formulation receives the prefix at validation time.
        yy,xx=case['fluid_yx'][rng.integers(len(case['fluid_yx']),size=256)].T
        iq=tensor(coords(np.full(len(xx),3),yy,xx));iy=case['state_gpu'][yy,xx,:3]
        ic=((predict(model,case,iq,c)-iy)/model.std).square().mean()
        loss=data+config['pde_weight']*ramp*pde+config['bc_weight']*bc+config['ic_weight']*ic
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();schedule.step()
        if step==1 or step%config['validate_every']==0 or step==config['steps']:
            score,values=validate(model,val_cases)
            row=dict(step=step,data=float(data.detach()),pde=float(pde.detach()),boundary=float(bc.detach()),
                     initial=float(ic.detach()),ramp=ramp,validation_score=score,cases=values,elapsed_seconds=time.monotonic()-start)
            history.append(row);save(out/'history.json',history)
            state=dict(config=config,state=model.state_dict(),step=step,validation_score=score,source_freeze=digest(WORK/'freeze.json'))
            torch.save(state,out/'last.pt')
            if score<best:best=score;torch.save(state,out/'best.pt')
            print(config['name'],step,'VAL',score,'DATA',row['data'],'PDE',row['pde'],flush=True)
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    chosen=torch.load(out/'best.pt',weights_only=True);model.load_state_dict(chosen['state'])
    accuracy=evaluate(model,val_cases,out);speed=latency(model,val_cases[0])
    accurate=all(r['scores']['all_future']['model']['velocity_relative_l2']<=.1 and
                 r['scores']['all_future']['model']['pressure_relative_l2']<=.2 for r in accuracy)
    beats_persistence=all(all(r['scores']['all_future']['model'][key]<r['scores']['all_future']['persistence'][key]
                              for key in ['velocity_relative_l2','pressure_relative_l2']) for r in accuracy)
    fast=all(r['p99']<=5 for r in speed['gpu_ms']['grid_prefix_features_cached'])
    result=dict(status='PREFIX_CONDITIONED_DEVELOPMENT_EVALUATED',config=config,selected_step=chosen['step'],
                train_scenarios=len(train_cases),validation_scenarios=len(val_cases),input_frames=[2,3],target_frames=[4,241],
                targets_per_scenario=238,accuracy=accuracy,latency=speed,training_seconds=elapsed,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),
                checkpoint_sha256=digest(out/'best.pt'),future_fields_in_model_input=False,main_test_opened=False,
                working_gate=dict(accuracy_pass=accurate,latency_pass=fast,beats_persistence=beats_persistence),
                llm_comparison_ready=False,causal_rollout=False,
                interpretation='direct future trajectory queries conditioned on two known prefix states; hard initial condition in state-aware variant',
                limitations=['Not a full-domain initial-boundary value solver: crop exterior data are unavailable',
                             'State-aware vs no-state comparison changes the conditioning and initial-condition enforcement together',
                             'New input protocol, not a retroactive reinterpretation of v2/v3 geometry-only results',
                             'No final test, multiple-seed confirmation or hypothesis significance claim'])
    save(out/'summary.json',result);return result


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    verify();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    records=json.loads((WORK/'prepared.json').read_text())['records'];cases=[]
    if args.smoke:
        records=[next(r for r in records if r['proposed_split']==role) for role in ['train','validation']]
    for r in records:
        if r['proposed_split']=='test' or digest(r['state_cache'])!=r['state_cache_sha256']:raise ValueError('Sealed test or changed input')
        for name in ['geometry.npz','samples.npz']:
            path=str(r['cache'])+'/'+name
            if digest(path)!=r['hashes'][path]:raise ValueError('Original cache changed')
        case=load(r);case['geo_gpu']=tensor(case['geometry'][None]);case['low_gpu']=tensor(case['low'][None])
        case['state_gpu']=tensor(case['initial']);case['fluid_yx']=np.column_stack(np.where(case['fluid']))
        cases.append(case)
    train_cases=[c for c in cases if c['record']['proposed_split']=='train']
    val_cases=[c for c in cases if c['record']['proposed_split']=='validation']
    protocol=json.loads((WORK/'protocol.json').read_text())
    out=WORK/('smoke' if args.smoke else 'training')/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    results=[]
    for config in protocol['candidates']:
        if args.smoke:config=dict(config,steps=3,validate_every=3,warmup=0,ramp=1)
        results.append(train(config,train_cases,val_cases,out/config['name']));torch.cuda.empty_cache()
    save(out/'summary.json',dict(status='SMOKE_COMPLETE' if args.smoke else 'COMPLETE',results=results))


if __name__=='__main__':run()
