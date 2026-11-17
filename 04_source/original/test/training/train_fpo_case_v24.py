"""Fresh independent PINN per development scenario; temporal labels held out."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from fpo_common_v24 import WORK,save,digest,verify
from fpo_case_v24 import load_case,split_frames,training_target,known_inputs
from fpo_direct_curl_v15 import DirectPINN
from fpo_state_v4 import residual
from fpo_step_v5a import grid,conditioning,fingerprint
from evaluate_fpo_case_v24 import query,evaluate,reserved_audit_from_full


def readiness(model,case,out):
    q=grid();known=known_inputs(case);times=[];model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(query(model,known,q,0.),case['raw'][:,:,:3],rtol=1e-5,atol=1e-6)
        for _ in range(30):query(model,known,q,.05)
        torch.cuda.synchronize()
        for frame in range(4,242):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();value=query(model,known,q,(frame-3)*.05);b.record();b.synchronize()
            times.append(a.elapsed_time(b))
            if not torch.isfinite(value).all():raise ValueError('Nonfinite readiness output')
    result=dict(p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95)),
        p99=float(np.percentile(times,99)),full_field=[256,1024],queries=238,
        reads_future_reference=False,includes_context_curl_gauge_mask=True,inputs_gpu=True)
    save(out/'readiness.json',result);model.train()
    if result['p99']>5:raise RuntimeError('Full-field query p99 exceeds5ms; no training')
    return result


def train(config,case,out,smoke=False):
    out.mkdir(parents=True,exist_ok=False);save(out/'config.json',config)
    normal=case['normalization'];save(out/'normalization.json',normal)
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    model=DirectPINN(config,normal['mean'],normal['std']).cuda()
    initial=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):initial.update(k.encode());initial.update(v.detach().cpu().numpy().tobytes())
    gate=readiness(model,case,out)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['updates'],eta_min=config['lr']*.05)
    frames=split_frames()['train'];order=rng.permutation(frames);cursor=0
    counts={f:0 for f in frames};sampling=hashlib.sha256();history=[];best=float('inf')
    statistics=torch.zeros(4,device='cuda');window_count=0;start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    qbase=grid()

    def assess(update):
        nonlocal best,window_count
        report=evaluate(model,case,out/'validation'/f'update_{update:06d}','validation')
        row=dict(update=update,validation=report,elapsed_seconds=time.monotonic()-start,
            window_updates=window_count,lr=optimizer.param_groups[0]['lr'],
            window_mean=dict(zip(['data','continuous_ns','obstacle_bc','gradient_norm_before_clip'],
                (statistics/max(window_count,1)).cpu().tolist())) if window_count else None)
        history.append(row);save(out/'history.json',history)
        checkpoint=dict(state=model.state_dict(),config=config,update=update,score=report['score'],
            normalization=normal,record_path=case['record']['path'],source_freeze=digest(WORK/'freeze.json'))
        torch.save(checkpoint,out/'last.pt')
        if report['score']<best:best=report['score'];torch.save(checkpoint,out/'best.pt')
        torch.save(dict(checkpoint=checkpoint,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            frame_order=order.tolist(),cursor=cursor,frame_counts=counts,sample_sha256=sampling.hexdigest(),best_score=best),out/'training_state.pt')
        print('VALIDATE',config['name'],update,report['mean_errors'] if not report['failure'] else report,flush=True)
        statistics.zero_();window_count=0

    assess(0)
    for update in range(1,config['updates']+1):
        if cursor==len(order):order=rng.permutation(frames);cursor=0
        frame=int(order[cursor]);cursor+=1;counts[frame]+=1
        sampling.update(np.asarray([frame],dtype='<i8').tobytes())
        y,x=case['fluid_yx'][rng.integers(len(case['fluid_yx']),size=config['batch'])].T
        sampling.update(np.asarray(np.column_stack((y,x)),dtype='<i8').tobytes())
        q=torch.tensor(np.column_stack((4+(x+.5)/64,6+(y+.5)/64,np.full(len(x),(frame-3)*.05))),device='cuda',dtype=torch.float32)
        context=conditioning(model,case['raw'],case['geometry'],case['record']['reynolds'])
        pred=model.sampled_prediction(q,context,case['raw'],torch.as_tensor(y,device='cuda'),torch.as_tensor(x,device='cuda'))
        target=training_target(case,frame,y,x)
        data=((pred-target)/model.std).square().mean()
        ramp=min(1.,max(0.,(update-config['warmup'])/1000))
        pde=torch.zeros((),device='cuda')
        if ramp>0:
            yy,xx=case['interior'][rng.integers(len(case['interior']),size=config['collocation'])].T
            qc_np=np.column_stack((4+(xx+.5+rng.uniform(-.4,.4,len(xx)))/64,6+(yy+.5+rng.uniform(-.4,.4,len(yy)))/64,rng.uniform(0,11.9,len(xx))))
            sampling.update(np.asarray(qc_np,dtype='<f8').tobytes())
            qc=torch.tensor(qc_np,device='cuda',dtype=torch.float32,requires_grad=True)
            pde=residual(qc,model(qc,context,case['raw']),case['record']['reynolds']).square().mean(0).sum()
        boundary=case['boundary'][rng.integers(len(case['boundary']),size=128)]
        qb_np=np.column_stack((boundary,rng.uniform(0,11.9,len(boundary))));sampling.update(np.asarray(qb_np,dtype='<f8').tobytes())
        qb=torch.tensor(qb_np,device='cuda',dtype=torch.float32,requires_grad=True)
        bc=(model(qb,context,case['raw'])[:,:2]/model.std[:2]).square().mean()
        loss=data+ramp*.01*pde+.01*bc
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; no field clipping')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step();statistics.add_(torch.stack((data,pde,bc,grad)).detach());window_count+=1
        if update%config['assess_every']==0 or update==config['updates']:assess(update)
    chosen=torch.load(out/'best.pt',weights_only=True,map_location='cuda');model.load_state_dict(chosen['state'])
    selected=evaluate(model,case,out/'selected_validation','validation',checkpoint_selected=True)
    full=None if smoke else evaluate(model,case,out/'full_reconstruction','postfit_full',checkpoint_selected=True)
    audit=None if smoke else (reserved_audit_from_full(full,out/'full_reconstruction',out/'reserved_time_audit')
        if not full['failure'] else dict(full,phase='audit'))
    assert set(counts)==set(frames) and sum(counts.values())==config['updates']
    if not smoke and min(counts.values())<1:raise ValueError('Incomplete supervised time coverage')
    usable=not smoke and all(not r['failure'] and r['accuracy_pass'] and r['mass_pass'] for r in [selected,audit]) and not full['failure'] and full['latency_pass']
    result=dict(status='CASE_SPECIFIC_SMOKE_COMPLETE' if smoke else 'CASE_SPECIFIC_EVALUATED',
        config=config,record_path=case['record']['path'],selected_update=chosen['update'],
        validation=selected,reserved_time_audit=audit,full_reconstruction=full,
        train_frame_counts=counts,sample_sha256=sampling.hexdigest(),initial_state_sha256=initial.hexdigest(),
        checkpoint_sha256=digest(out/'best.pt'),normalization_sha256=digest(out/'normalization.json'),
        training_seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        readiness=gate,engineering_gates_pass=usable,main_test_opened=False,independent_final_test=False,
        task='Per-case data-assisted temporal reconstruction/interpolation; not unseen-case forecast',
        audit_evaluated_only_after_checkpoint_selection=not smoke,causal_stepper=False,llm_ready=False)
    save(out/'summary.json',result);return result


def run():
    p=argparse.ArgumentParser();p.add_argument('worker',type=int,choices=[0,1]);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    protocol=json.loads((WORK/'protocol.json').read_text());records=json.loads((WORK/'prepared.json').read_text())['records']
    if not args.smoke:
        s=json.loads((WORK/'smoke'/os.environ['SLURM_JOB_ID']/'summary.json').read_text())
        if s['status']!='WORKER_SMOKE_COMPLETE' or s['worker']!=args.worker:raise ValueError('Missing smoke')
    out=WORK/('smoke' if args.smoke else 'training')/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    results=[]
    for index in protocol['workers'][str(args.worker)]:
        record=records[index];case=load_case(record)
        config=dict(protocol['model'],name=f'case_{index}',seed=20261104+index)
        if args.smoke:config.update(updates=3,assess_every=3,warmup=-1000)
        results.append(train(config,case,out/config['name'],args.smoke));del case;torch.cuda.empty_cache()
    verify();save(out/'summary.json',dict(status='WORKER_SMOKE_COMPLETE' if args.smoke else 'CASE_SPECIFIC_WORKER_COMPLETE',
        worker=args.worker,results=results,all_engineering_gates_pass=all(r['engineering_gates_pass'] for r in results),
        main_test_opened=False,independent_final_test=False,llm_ready=False))


if __name__=='__main__':run()
