"""One shared stationary PINN over640flowcases; authors standard split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from ldc_common_v26 import ROOT,WORK,save,digest,verify
from ldc_shared_pinn_v26 import SharedSteadyPINN,steady_residual
from ldc_store_v26 import Store,query,conditioning
from fpo_state_v4 import errors
sys.path.insert(0,str(ROOT/'test/analysis'))
from ldc_data_v25 import reference_physics


def benchmark(model,store,out):
    out.mkdir(parents=True,exist_ok=False);rows=[];model.eval()
    with torch.inference_mode():
        for index in [store.train_ids[0],store.validation_ids[0]]:
            known=store.known(index)
            for _ in range(30):query(model,known,store.qgrid)
            torch.cuda.synchronize();times=[]
            for _ in range(200):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();value=query(model,known,store.qgrid);b.record();b.synchronize();times.append(a.elapsed_time(b))
            if not torch.isfinite(value).all():raise ValueError('Nonfinite readiness prediction')
            rows.append(dict(sample=index,p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95)),p99=float(np.percentile(times,99))))
    result=dict(cases=rows,pass_5ms=all(r['p99']<=5 for r in rows),fullfield=[512,512],inputs_gpu=True,
        includes='Native input pooling, geometry encoder, coordinate MLP, spatial interpolation, curl, gauge, mask; no input-dependent feature cache',
        excludes='CPUtransfer/rendering,30warmup,staticcoordinateweightsetup')
    save(out/'summary.json',result);model.train()
    if not result['pass_5ms']:raise RuntimeError('Nativefullfieldp99>5ms: no training')
    return result


def evaluate(model,store,out):
    out.mkdir(parents=True,exist_ok=False);rows=[];model.eval()
    with torch.inference_mode():
        for index in store.validation_ids:
            known=store.known(index)
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();pred=query(model,known,store.qgrid);b.record();b.synchronize()
            if not torch.isfinite(pred).all() or pred.abs().max()>1000:
                result=dict(score=1e6,failure=True,failure_sample=index,cases=rows);save(out/'summary.json',result);model.train();return result
            prediction=pred.cpu().numpy();target=np.moveaxis(store.validation_reference(index),0,2)
            fluid=np.asarray(store.fluid_np[index]);p=prediction[fluid];r=target[fluid]
            metric=errors(p,r);difference=np.asarray(prediction,dtype=np.float64)-target
            field_l2=float(np.sqrt(np.sum(difference**2)/max(float(np.sum(np.asarray(target,dtype=np.float64)**2)),1e-24)))
            solid=np.column_stack(np.where(~fluid));lo=solid.min(0);hi=solid.max(0)
            margin=np.maximum(np.ceil((hi-lo+1)*.1).astype(int),1);lo=np.maximum(lo-margin,0);hi=np.minimum(hi+margin,511)
            region=np.zeros_like(fluid);region[lo[0]:hi[0]+1,lo[1]:hi[1]+1]=True;region &= fluid
            near=errors(prediction[region],target[region])
            physics=reference_physics(np.moveaxis(prediction,2,0),fluid,known['reynolds'])
            baseline=reference_physics(np.moveaxis(target,2,0),fluid,known['reynolds'])
            rows.append(dict(sample=index,reynolds=known['reynolds'],**metric,
                mse_all=float(np.mean(difference**2)),relative_l2_all=field_l2,near_obstacle_errors=near,
                physics=physics,reference_physics=baseline,query_gpu_ms=float(a.elapsed_time(b))))
    keys=['velocity_relative_l2','pressure_relative_l2','mse_all','relative_l2_all']
    means={k:float(np.mean([r[k] for r in rows])) for k in keys}
    result=dict(failure=False,score=means['mse_all'],means=means,cases=rows,
        mean_momentum_rms=float(np.mean([r['physics']['momentum_rms'] for r in rows])),
        mean_divergence_rms=float(np.mean([r['physics']['divergence_rms'] for r in rows])),
        timing_note='One queryperVALcase is diagnostic; use200repeatbenchmarkforp99',
        main_test_opened=False,geometry_disjoint=False)
    save(out/'summary.json',result);model.train();return result


def train(config,store,out,smoke=False):
    out.mkdir(parents=True,exist_ok=False);save(out/'config.json',config)
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    model=SharedSteadyPINN(config,store.normalization).cuda()
    gate=benchmark(model,store,out/'readiness')
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    steps_per_epoch=len(store.train_ids)//config['batch_cases']
    if steps_per_epoch*config['batch_cases']!=len(store.train_ids):raise ValueError('ExactfullTRAINepoch required')
    updates=3 if smoke else steps_per_epoch*config['epochs']
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=updates,eta_min=config['lr']*.05)
    history=[];counts={i:0 for i in store.train_ids};sampling=hashlib.sha256();best=float('inf');start=time.monotonic()
    save(out/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()),
        seed=config['seed'],precision='float32',allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,source_freeze=digest(WORK/'freeze.json')))
    order=rng.permutation(store.train_ids);cursor=0;statistics=torch.zeros(4,device='cuda');window=0
    torch.cuda.reset_peak_memory_stats()

    def assess(update):
        nonlocal best,window
        report=evaluate(model,store,out/'validation'/f'update_{update:06d}')
        row=dict(update=update,equivalent_epoch=update/steps_per_epoch,validation=report,
            physics_coefficient=config['physics_weight']*(1. if smoke else min(1.,max(0.,(update-config['warmup'])/config['ramp']))),
            window_updates=window,window_mean=None if not window else dict(zip(['data','steady_ns','bc','gradient_norm_before_clip'],(statistics/window).cpu().tolist())),
            elapsed_seconds=time.monotonic()-start)
        history.append(row);save(out/'history.json',history)
        checkpoint=dict(state=model.state_dict(),config=config,normalization=store.normalization,update=update,
            score=report['score'],source_freeze=digest(WORK/'freeze.json'))
        torch.save(checkpoint,out/'last.pt')
        if report['score']<best:best=report['score'];torch.save(checkpoint,out/'best.pt')
        torch.save(dict(checkpoint=checkpoint,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            case_counts=counts,case_order=order.tolist(),cursor=cursor,sample_sha256=sampling.hexdigest()),out/'training_state.pt')
        print('ASSESS',update,report.get('means',{'failure':True}),flush=True);statistics.zero_();window=0

    assess(0)
    for update in range(1,updates+1):
        if cursor==len(order):order=rng.permutation(store.train_ids);cursor=0
        batch=order[cursor:cursor+config['batch_cases']];cursor+=len(batch)
        losses=[];parts=[]
        coefficient=config['physics_weight']*(1. if smoke else min(1.,max(0.,(update-config['warmup'])/config['ramp'])))
        for index in batch:
            index=int(index);counts[index]+=1;known=store.known(index)
            sampling.update(np.asarray([index],dtype='<i8').tobytes())
            context=conditioning(model,known)
            y,x=store.fluid_cells[index][rng.integers(len(store.fluid_cells[index]),size=config['data_points'])].T
            sampling.update(np.asarray(np.column_stack((y,x)),dtype='<i8').tobytes())
            yy=torch.as_tensor(y.astype('int64'),device='cuda');xx=torch.as_tensor(x.astype('int64'),device='cuda')
            q=torch.stack(((xx+.5)/512,(yy+.5)/512),1)
            pred=model.sampled_prediction(q,context,yy,xx)
            data=((pred-store.training_target(index,yy,xx))/model.std).square().mean()
            pde=torch.zeros((),device='cuda')
            if coefficient:
                cells=store.interiors[index];cy,cx=cells[rng.integers(len(cells),size=config['physics_points'])].T
                qc_np=np.column_stack(((cx+.5+rng.uniform(-.4,.4,len(cx)))/512,(cy+.5+rng.uniform(-.4,.4,len(cy)))/512))
                sampling.update(np.asarray(qc_np,dtype='<f8').tobytes())
                qc=torch.tensor(qc_np,device='cuda',dtype=torch.float32,requires_grad=True)
                pde=steady_residual(qc,model(qc,context),known['reynolds']).square().mean(0).sum()
            n=config['wall_points_per_side'];s=rng.uniform(1/512,1-1/512,size=(4,n))
            wall=np.concatenate((np.column_stack((s[0],np.zeros(n))),np.column_stack((s[1],np.ones(n))),
                np.column_stack((np.zeros(n),s[2])),np.column_stack((np.ones(n),s[3]))))
            body=store.body[index][rng.integers(len(store.body[index]),size=config['body_points'])]
            qb_np=np.concatenate((wall,body));sampling.update(np.asarray(qb_np,dtype='<f8').tobytes())
            qb=torch.tensor(qb_np,device='cuda',dtype=torch.float32,requires_grad=True)
            target=torch.zeros(len(qb),2,device='cuda');target[n:2*n,0]=1.
            bc=((model(qb,context)[:,:2]-target)/model.std[:2]).square().mean()
            losses.append(data+coefficient*pde+config['bc_weight']*bc);parts.append(torch.stack((data,pde,bc)).detach())
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss, no fieldclipping')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step();statistics[:3]+=torch.stack(parts).mean(0);statistics[3]+=grad;window+=1
        if update%(steps_per_epoch*config['assess_epochs'])==0 or update==updates:assess(update)
    selected=torch.load(out/'best.pt',weights_only=True,map_location='cuda');model.load_state_dict(selected['state'])
    result=evaluate(model,store,out/'selected_validation');speed=benchmark(model,store,out/'selected_speed')
    if not smoke and set(counts.values())!={config['epochs']}:raise ValueError('Unequalcasecoverage')
    summary=dict(status='SHARED_STEADY_SMOKE_COMPLETE' if smoke else 'SHARED_STEADY_PILOT_COMPLETE',
        selected_update=selected['update'],selected_validation=result,selected_speed=speed,initial_speed=gate,
        case_counts=counts,sample_sha256=sampling.hexdigest(),checkpoint_sha256=digest(out/'best.pt'),
        training_seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        train_cases=len(store.train_ids),validation_cases=len(store.validation_ids),shared_weights=True,
        main_test_opened=False,independent_final_test=False,llm_ready=False)
    save(out/'summary.json',summary);return summary


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    protocol=json.loads((WORK/'protocol.json').read_text());prepared=json.loads(Path(protocol['prepared_path']).read_text())
    if prepared['status']!='LDC_DATA_READY' or not prepared['physics_ready']:raise ValueError('Reference-scaleauditnotpassed')
    split=json.loads(Path(protocol['split_path']).read_text())['indices']
    if not args.smoke:
        s=json.loads((WORK/'smoke'/os.environ['SLURM_JOB_ID']/'summary.json').read_text())
        if s['status']!='SHARED_STEADY_SMOKE_COMPLETE':raise ValueError('Missing successful smoke')
    store=Store(prepared,split,args.smoke)
    out=WORK/('smoke' if args.smoke else 'training')/os.environ['SLURM_JOB_ID']
    train(protocol['model'],store,out,args.smoke);verify()


if __name__=='__main__':run()
