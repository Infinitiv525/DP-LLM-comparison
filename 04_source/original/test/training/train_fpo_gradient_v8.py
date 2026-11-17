"""Matched8vs32 gradients: same32-step update cadence, data, losses and warm weights."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from fpo_stable_v6 import StablePINN,grid,stable_step,load_cases,features,conditioning,grid_physics
from train_fpo_stable_v6 import assess,benchmark,continuous_physics
from fpo_gradient_v8 import matched_window,RolloutFailure
from fpo_common_v8 import ROOT,WORK,save,digest,verify


def state_record(model,config,phase,epoch,updates,score=None):
    return dict(state=model.state_dict(),config=config,phase=phase,epoch=epoch,update=updates,
                score=score,source_freeze=digest(WORK/'freeze.json'))


def train(config,case,out,qgrid,normal):
    out.mkdir(parents=True,exist_ok=False);save(out/'config.json',config)
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    model=StablePINN(config,normal['mean'],normal['std']).cuda()
    source=torch.load(config['warm_start'],weights_only=True,map_location='cuda');model.load_state_dict(source['state']);del source
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    total_updates=config['epochs']*math.ceil(config['horizon']/32)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total_updates,eta_min=config['lr']*.1)
    history=[];assessments=[];sampling=[];best_key=None;updates=0;terminal_failure=None;coverage=np.zeros(238,dtype=bool)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()

    def evaluate(label,phase,epoch):
        nonlocal best_key
        result=assess(model,case,out/'assessments'/label,qgrid,config)
        key=(0 if result['divergence'] is None and result['mass']['pass_gate'] else 1,result['score'])
        state=state_record(model,config,phase,epoch,updates,result['score']);torch.save(state,out/'last.pt')
        if best_key is None or key<best_key:
            best_key=key;torch.save(state,out/'best.pt')
        row=dict(label=label,phase=phase,epoch=epoch,updates=updates,score=result['score'],mass=result['mass'],
                 divergence=result['divergence'],elapsed_seconds=time.monotonic()-started)
        assessments.append(row);save(out/'assessment_history.json',assessments)
        print(config['name'],label,'SCORE',result['score'],'MASS',result['mass'],flush=True)

    evaluate('initial','initial',0)
    phase='full238_matched_updates'
    for epoch in range(1,config['epochs']+1):
        previous,current=case['prefix'];totals=np.zeros(6);trained_steps=0;failure=None;max_signal=0.;norms=[]
        for offset in range(0,config['horizon'],32):
            count=min(32,config['horizon']-offset);stats=[];block_maxima=[]
            rng_before=hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True).encode()).hexdigest()
            optimizer.zero_grad(set_to_none=True)

            def block_objective(local_offset,previous,current,predictions):
                with torch.no_grad():
                    signals=torch.stack([p.detach() for p in predictions]);finite=bool(torch.isfinite(signals).all())
                    maximum=float(signals.abs().max()) if finite else None
                if not finite or maximum>config['divergence_limit']:
                    raise RolloutFailure(dict(offset=offset+local_offset,reason='nonfinite or abs signal above1000',max_abs=maximum))
                block_maxima.append(maximum)
                losses=[];momenta=[];divergences=[];energies=[];old=current
                for local,pred in enumerate(predictions,1):
                    target=case['data'][offset+local_offset+local+1]
                    losses.append(((pred[case['fluid']]-target[case['fluid']])/(model.std*config['data_loss_scale'])).square().mean())
                    m,d=grid_physics(old,pred,case,model.std[:2].mean());momenta.append(m);divergences.append(d)
                    with torch.no_grad():
                        energies.append(pred[case['fluid'],:2].square().mean()/target[case['fluid'],:2].square().mean().clamp_min(1e-12))
                    old=pred
                data=torch.stack(losses).mean();momentum=torch.stack(momenta).mean();divergence=torch.stack(divergences).mean()
                raw=features(previous,current);context=conditioning(model,raw,case['geometry'],case['record']['reynolds'])
                pde,bc=continuous_physics(model,raw,context,case,rng,True,config)
                loss=data+config['pde_weight']*pde+config['bc_weight']*bc+config['own_momentum_weight']*momentum+config['own_divergence_weight']*divergence
                if not torch.isfinite(loss):raise RolloutFailure(dict(offset=offset+local_offset,reason='nonfinite loss'))
                stats.append(len(predictions)*np.array([float(t.detach()) for t in [data,pde,bc,momentum,divergence,torch.stack(energies).mean()]]))
                return loss

            try:
                next_previous,next_current,loss=matched_window(previous,current,
                     lambda a,b:stable_step(model,a,b,case,qgrid),block_objective,count,config['gradient_horizon'])
            except RolloutFailure as exc:failure=exc.record;break
            loss.backward()
            try:norm=nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
            except RuntimeError:
                failure=dict(offset=offset,reason='nonfinite gradient');terminal_failure=dict(phase=phase,epoch=epoch,**failure);break
            optimizer.step();scheduler.step();updates+=1;norms.append(float(norm))
            previous,current=next_previous,next_current
            coverage[offset:offset+count]=True;trained_steps+=count;totals+=np.sum(stats,axis=0);max_signal=max(max_signal,*block_maxima)
            sampling.append(dict(epoch=epoch,offset=offset,count=count,rng_before=rng_before,
                      rng_after=hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True).encode()).hexdigest()))
        row=dict(phase=phase,horizon=config['horizon'],epoch=epoch,updates=updates,trained_steps=trained_steps,failure=failure,
                 max_abs=max_signal,elapsed_seconds=time.monotonic()-started,
                 mean_gradient_norm=float(np.mean(norms)) if norms else None,max_gradient_norm=max(norms) if norms else None,
                 clipped_updates=sum(n>config['gradient_clip'] for n in norms))
        row.update({k:float(v/max(trained_steps,1)) for k,v in zip(['data','pde','boundary','own_momentum','own_divergence','velocity_energy_ratio'],totals)})
        history.append(row);save(out/'training_history.json',history);save(out/'sampling_trace.json',sampling)
        if epoch==1 or epoch%config['assess_every_epochs']==0 or epoch==config['epochs'] or terminal_failure:
            evaluate(f'epoch_{epoch:04d}',phase,epoch)
            continuation=state_record(model,config,phase,epoch,updates)
            continuation.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
            torch.save(continuation,out/'continuation.pt')
            print(config['name'],'TRAIN',json.dumps(row),flush=True)
        if terminal_failure:break
    elapsed=time.monotonic()-started
    if terminal_failure:save(out/'failure.json',terminal_failure)
    chosen=torch.load(out/'best.pt',weights_only=True,map_location='cuda');model.load_state_dict(chosen['state'])
    selected=assess(model,case,out/'selected_checkpoint',qgrid,config);timing=benchmark(model,case,qgrid)
    accurate=selected['divergence'] is None and selected['scores']['all_future']['model']['velocity_relative_l2']<=.1 and selected['scores']['all_future']['model']['pressure_relative_l2']<=.2
    fast=selected['gpu_step_ms']['p99']<=5 and all(r['p99']<=5 for r in timing['whole_gpu_step_rounds_ms'])
    result=dict(status='KNOWN_CASE_MATCHED_GRADIENT_EVALUATED' if terminal_failure is None else 'TRAINING_ABORTED_NONFINITE_GRADIENT',
                config=config,accuracy=selected,latency=timing,selected_phase=chosen['phase'],selected_epoch=chosen['epoch'],
                actual_optimizer_updates=updates,expected_optimizer_updates=total_updates,
                all238_training_targets_visited=bool(coverage.all()),completed_episodes=sum(r['trained_steps']==config['horizon'] for r in history),
                sampling_trace_sha256=digest(out/'sampling_trace.json'),
                training_seconds=elapsed,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),
                working_gate=dict(accuracy_pass=bool(accurate),latency_pass=bool(fast),mass_growth_pass=selected['mass']['pass_gate']),
                checkpoint_sha256=digest(out/'best.pt'),training_failure=terminal_failure,
                independent_validation=False,main_test_opened=False,llm_comparison_ready=False,
                limitations=['Known TRAIN positive control only; same trajectory used for checkpoint selection',
                             'Both update weights every32 own steps; detach gradient every8 versus32',
                             'Common8step objectives/collocation schedule; own numerical states carry onward',
                             'Only interior curl increment has divergence identity; not global NS stability'])
    save(out/'summary.json',result);return result


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    receipt=json.loads((WORK/'prepared.json').read_text());record=receipt['records'][0]
    if record['proposed_split']!='train' or (record['family'],record['geometry_id'],record['reynolds'])!=('harmonics','1',382):raise ValueError('Known TRAIN case only')
    case=load_cases([record])[0]
    if digest(receipt['weight_file'])!=receipt['weight_sha256']:raise ValueError('Geometry weight changed')
    with np.load(receipt['weight_file'],allow_pickle=False) as z:case['weight']=torch.tensor(z['weight'],device='cuda')
    case['interior_gpu']=torch.zeros_like(case['fluid']);y,x=case['interior'].T;case['interior_gpu'][y,x]=True
    with np.load(record['state_cache'],allow_pickle=False) as z:raw=torch.tensor(z['features'],device='cuda')
    case['prefix']=(raw[:,:,:3]-raw[:,:,3:],raw[:,:,:3]);case['reference']=np.load(record['field'],mmap_mode='r',allow_pickle=False)
    normal=json.loads(Path(receipt['normalization']).read_text());qgrid=grid()
    configs=json.loads((WORK/'protocol.json').read_text())['candidates']
    out=WORK/('smoke' if args.smoke else 'training')/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    save(out/'normalization.json',normal);results=[]
    for config in configs:
        if args.smoke:config=dict(config,horizon=46,epochs=1,assess_every_epochs=1)
        results.append(train(config,case,out/config['name'],qgrid,normal));torch.cuda.empty_cache()
        if args.smoke and (results[-1]['training_failure'] is not None or results[-1]['actual_optimizer_updates']!=results[-1]['expected_optimizer_updates']):
            raise RuntimeError('Smoke did not finish all expected finite gradient updates')
    matched=(len({x['config']['warm_start_sha256'] for x in results})==1 and
             len({x['sampling_trace_sha256'] for x in results})==1 and
             all(x['actual_optimizer_updates']==x['expected_optimizer_updates'] and x['completed_episodes']==x['config']['epochs'] for x in results))
    save(out/'summary.json',dict(status='SMOKE_COMPLETE' if args.smoke else 'KNOWN_CASE_MATCHED_GRADIENT_COMPLETE',
             results=results,matched_execution=matched,independent_validation=False))
    if args.smoke and not matched:raise RuntimeError('Smoke comparison not fully matched')


if __name__=='__main__':run()
