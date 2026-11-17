"""Matched fresh primitive/curl direct PINNs on69TRAIN/6developmentVAL."""
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
from fpo_common_v17 import WORK,save,digest,verify
from fpo_direct_curl_v15 import DirectPINN
from evaluate_fpo_transfer_v16 import evaluate as score_case


def query(model,case,qbase,time_value):
    q=qbase.clone();q[:,2]=time_value
    context=conditioning(model,case['raw'],case['geometry'],case['record']['reynolds'])
    value=model.grid_prediction(q,context,case['raw'])
    return project_known_constraints(value,case['fluid'])


def evaluate(model,case,out,qbase):
    result=score_case(model,case,out,qbase)
    result.setdefault('score',1e6)
    model.train()
    return result

def train(config,train_cases,val_cases,out,qbase,normal):
    out.mkdir(parents=True,exist_ok=False);save(out/'config.json',config)
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    model=DirectPINN(config,normal['mean'],normal['std']).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['updates'],eta_min=config['lr']*.05)
    sampling=hashlib.sha256();history=[];best=float('inf');coverage=np.zeros((len(train_cases),238),dtype=bool)
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
        sampling.update(np.asarray([ci,offset],dtype='<i8').tobytes())
        yy,xx=case['fluid_yx'][rng.integers(len(case['fluid_yx']),size=config['batch'])].T
        q=torch.tensor(np.column_stack((4+(xx+.5)/64,6+(yy+.5)/64,np.full(len(xx),offset*.05))),device='cuda',dtype=torch.float32)
        context=conditioning(model,case['raw'],case['geometry'],case['record']['reynolds'])
        pred=model.sampled_prediction(q,context,case['raw'],torch.as_tensor(yy,device='cuda'),torch.as_tensor(xx,device='cuda'));target=case['data'][offset+1,yy,xx]
        data=((pred-target)/model.std).square().mean()
        pde=torch.zeros((),device='cuda');parts=torch.zeros(3,device='cuda');ramp=min(1.,max(0.,(update-config['warmup'])/1000))
        if ramp>0:
            y,x=case['interior'][rng.integers(len(case['interior']),size=config['collocation'])].T
            qc=torch.tensor(np.column_stack((4+(x+.5+rng.uniform(-.4,.4,len(x)))/64,6+(y+.5+rng.uniform(-.4,.4,len(y)))/64,rng.uniform(0,11.9,len(x)))),device='cuda',dtype=torch.float32,requires_grad=True)
            parts=residual(qc,model(qc,context,case['raw']),case['record']['reynolds']).square().mean(0);pde=parts.sum()
        boundary=case['boundary'][rng.integers(len(case['boundary']),size=128)]
        qb=torch.tensor(np.column_stack((boundary,rng.uniform(0,11.9,len(boundary)))),device='cuda',dtype=torch.float32,requires_grad=True)
        bc=(model(qb,context,case['raw'])[:,:2]/model.std[:2]).square().mean()
        loss=data+ramp*.01*pde+.01*bc
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite reconstruction loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step()
        if update==1 or update%config['assess_every']==0 or update==config['updates']:
            reports=[evaluate(model,c,out/'assessments'/f'update_{update:06d}'/str(i),qbase) for i,c in enumerate(val_cases)]
            result=dict(score=float(np.mean([r['score'] for r in reports])),cases=reports)
            row=dict(update=update,data=float(data.detach()),pde=float(pde.detach()),pde_parts=parts.detach().cpu().tolist(),boundary=float(bc.detach()),assessment=result,elapsed_seconds=time.monotonic()-start)
            if update%100000==0 or update==config['updates']:
                monitors=[next(c for c in train_cases if c['record']['family']==family) for family in sorted({c['record']['family'] for c in train_cases})]
                row['train_fit_monitor']=[evaluate(model,c,out/'fit_history'/f'update_{update:06d}'/str(i),qbase) for i,c in enumerate(monitors)]
            history.append(row);save(out/'history.json',history)
            saved=dict(state=model.state_dict(),config=config,update=update,score=result['score'],source_freeze=digest(WORK/'freeze.json'))
            torch.save(saved,out/'last.pt')
            if result['score']<best:best=result['score'];torch.save(saved,out/'best.pt')
            torch.save(dict(checkpoint=saved,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                orders=[o.tolist() for o in orders],cursors=cursors.tolist(),case_order=case_order.tolist(),
                case_cursor=case_cursor,counts=counts.tolist(),coverage=coverage.tolist(),best_score=best,
                case_frame_schedule_sha256=sampling.hexdigest()),out/'training_state.pt')
            print(config['name'],update,json.dumps(result),flush=True)
    chosen=torch.load(out/'best.pt',weights_only=True,map_location='cuda');model.load_state_dict(chosen['state'])
    selected=[evaluate(model,c,out/'selected_checkpoint'/str(i),qbase) for i,c in enumerate(val_cases)]
    train_monitor=[next(c for c in train_cases if c['record']['family']==family) for family in sorted({c['record']['family'] for c in train_cases})]
    training_accuracy=[evaluate(model,c,out/'training_monitor'/str(i),qbase) for i,c in enumerate(train_monitor)]
    result=dict(status='MULTICASE_DIRECT_REPRESENTATION_EVALUATED',config=config,accuracy=selected,training_monitor=training_accuracy,selected_update=chosen['update'],
                all238_training_frames_visited=bool(coverage.all()),training_seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                checkpoint_sha256=digest(out/'best.pt'),independent_validation=False,causal_stepper=False,llm_ready=False,
                train_scenarios=len(train_cases),validation_scenarios=len(val_cases),train_selection_scenarios=0,training_case_updates=counts.tolist(),case_frame_schedule_sha256=sampling.hexdigest(),
                working_gate=dict(accuracy_pass=all(not r['failure'] and r['mean_errors']['velocity_relative_l2']<=.1 and r['mean_errors']['pressure_relative_l2']<=.2 for r in selected),
                    mass_pass=all(not r['failure'] and r['mass_pass'] for r in selected),
                    direct_query_latency_pass=all(not r['failure'] and r['direct_query_gpu_ms']['p99']<=5 for r in selected)),
                limitations=['69TRAIN and6geometry-disjoint developmentVAL; VAL checkpoint selection only, no backprop/future input; not final independent test',
                             'Same trunk/CNN but final primitive3 versus streamfunction-pressure2 head differs',
                             'Curl increment preserves initial discrete div only on eroded fluid interior after masking',
                             'Absolute time query, no recurrent own states; not directly comparable causal latency',
                             'Direct time query, continuous-autograd PDE versus matching FD-data/grid output'])
    save(out/'summary.json',result);return result


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('representation',choices=['primitive','curl'])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    receipt=json.loads((WORK/'prepared.json').read_text());records=receipt['records']
    if (sum(r['proposed_split']=='train' for r in records),sum(r['proposed_split']=='validation' for r in records))!=(69,6):raise ValueError('Expected69TRAIN6VAL')
    key=lambda r:(r['family'],r['geometry_id'])
    if {key(r) for r in records if r['proposed_split']=='train'}&{key(r) for r in records if r['proposed_split']=='validation'}:raise ValueError('Geometry overlap')
    if args.smoke:records=[next(r for r in records if r['proposed_split']==role) for role in ['train','validation']]
    for r in records:
        if r['proposed_split']=='test' or digest(r['state_cache'])!=r['state_cache_sha256']:raise ValueError('Sealed test or changed prefix')
        for name in ['geometry.npz','samples.npz']:
            p=str(Path(r['cache'])/name)
            if digest(p)!=r['hashes'][p]:raise ValueError('Cache changed')
    cases=load_cases(records)
    for case in cases:
        r=case['record']
        if r['proposed_split']=='validation' and 'data' in case:raise ValueError('Future VAL loaded to GPU')
        with np.load(r['state_cache'],allow_pickle=False) as z:case['raw']=torch.tensor(z['features'],device='cuda')
        case['reference']=np.load(r['field'],mmap_mode='r',allow_pickle=False)
        case['interior_gpu']=torch.zeros_like(case['fluid']);y,x=case['interior'].T;case['interior_gpu'][y,x]=True
    train_cases=[c for c in cases if c['record']['proposed_split']=='train']
    val_cases=[c for c in cases if c['record']['proposed_split']=='validation']
    normal=json.loads(Path(receipt['normalization']).read_text());qbase=grid()
    config=next(c for c in json.loads((WORK/'protocol.json').read_text())['candidates'] if c['representation']==args.representation)
    if args.smoke:config=dict(config,updates=3,warmup=0,assess_every=3)
    out=WORK/('smoke' if args.smoke else 'training')/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    result=train(config,train_cases,val_cases,out/config['name'],qbase,normal)
    verify()
    save(out/'summary.json',dict(status='SMOKE_COMPLETE' if args.smoke else 'MULTICASE_REPRESENTATION_COMPLETE',results=[result],causal_stepper=False,independent_final_test=False))


if __name__=='__main__':run()
