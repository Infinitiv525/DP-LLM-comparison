"""One-step PINN pretraining, short own-prediction rollout training, full causal validation."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from fpo_step_v5a import (StepPINN,DT,grid,features,conditioning,step,trajectory,
                          load_cases,residual,errors,fingerprint)
from fpo_common_v5a import WORK,save,digest,verify


def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device='cuda')


def evaluate(model,cases,folder,qgrid,config):
    folder.mkdir(parents=True,exist_ok=False);model.eval();all_rows=[];summaries=[]
    with torch.inference_mode():
        for case in cases:
            if fingerprint(case['record']['field'])!=case['record']['field_fingerprint']:raise ValueError('Changed validation labels')
            previous,current=case['prefix'];initial=current.cpu().numpy()[case['fluid_np']]
            times=[];rows=[];failure=None
            # Only the closure below can produce subsequent inputs. The scorer
            # reads reference[k] after prediction and never passes it to step.
            def advance(a,b):
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record();pred=step(model,a,b,case['geometry'],case['record']['reynolds'],case['fluid'],qgrid)
                end.record();end.synchronize();times.append(begin.elapsed_time(end));return pred
            for offset,pred in enumerate(trajectory(previous,current,advance,238),1):
                p=pred.cpu().numpy()
                if not np.isfinite(p).all() or np.max(np.abs(p))>config['divergence_limit']:
                    failure=dict(step=offset,reason='nonfinite or signal magnitude above prospective safety limit')
                    break
                k=offset+3;reference=np.asarray(case['reference'][k])[case['fluid_np']]
                rows.append(dict(frame=k,offset=offset,time=offset*DT,
                                 model=errors(p[case['fluid_np']],reference),persistence=errors(initial,reference)))
            scores={}
            if failure is None:
                groups={'all_future':rows,'steps_1_10':rows[:10],'steps_11_40':rows[10:40],
                        'steps_41_120':rows[40:120],'steps_121_238':rows[120:]}
                scores={scope:{kind:{key:float(np.mean([r[kind][key] for r in rr])) for key in rr[0][kind]}
                               for kind in ['model','persistence']} for scope,rr in groups.items()}
            summary=dict(path=case['record']['path'],predicted_steps=len(rows),divergence=failure,scores=scores,
                         actual_rollout_step_gpu_ms={key:float(np.percentile(times,p)) for key,p in [('p50',50),('p95',95),('p99',99)]})
            summaries.append(summary);all_rows.append(dict(path=case['record']['path'],rows=rows))
            print('CAUSAL_VALIDATION',case['record']['path'],'STEPS',len(rows),'FAILED',bool(failure),flush=True)
    failed=any(r['divergence'] is not None for r in summaries)
    score=config['divergence_score_penalty'] if failed else float(np.mean([
        (r['scores']['all_future']['model']['velocity_relative_l2']+r['scores']['all_future']['model']['pressure_relative_l2'])/2
        for r in summaries]))
    result=dict(score=score,any_diverged=failed,cases=summaries,reference_future_used_as_input=False)
    save(folder/'per_frame_accuracy.json',all_rows);save(folder/'summary.json',result)
    model.train();return result


def physics_losses(model,raw,c,case,rng,active,config):
    pde=torch.zeros((),device='cuda')
    if active:
        yy,xx=case['interior'][rng.integers(len(case['interior']),size=config['collocation'])].T
        q=tensor(np.column_stack((4+(xx+.5+rng.uniform(-.4,.4,len(xx)))/64,
                                  6+(yy+.5+rng.uniform(-.4,.4,len(yy)))/64,
                                  rng.uniform(0,DT,len(xx))))).requires_grad_(True)
        pde=residual(q,model(q,c,raw),case['record']['reynolds']).square().sum(1).mean()
    boundary=case['boundary'][rng.integers(len(case['boundary']),size=256)]
    q=tensor(np.column_stack((boundary,rng.uniform(0,DT,len(boundary)))))
    bc=(model(q,c,raw)[:,:2]/model.std[:2]).square().mean()
    return pde,bc


def timing(model,case,qgrid):
    model.eval();previous,current=case['prefix'];rounds=[];host=[]
    with torch.inference_mode():
        for _ in range(30):step(model,previous,current,case['geometry'],case['record']['reynolds'],case['fluid'],qgrid)
        for _ in range(3):
            samples=[]
            for i in range(200):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();value=step(model,previous,current,case['geometry'],case['record']['reynolds'],case['fluid'],qgrid)
                b.record();b.synchronize();samples.append(a.elapsed_time(b))
            rounds.append({key:float(np.percentile(samples,p)) for key,p in [('p50',50),('p95',95),('p99',99)]})
        for _ in range(30):
            torch.cuda.synchronize();start=time.perf_counter()
            step(model,previous,current,case['geometry'],case['record']['reynolds'],case['fluid'],qgrid).cpu()
            torch.cuda.synchronize();host.append((time.perf_counter()-start)*1000)
    return dict(whole_gpu_step_rounds_ms=rounds,samples_per_round=200,grid=[1024,256],
                includes_state_feature_rebuild=True,includes_encoder=True,includes_prediction_pressure_gauge=True,
                includes_solid_mask_update=True,initial_tensors_already_gpu=True,rendering_included=False,
                gpu_step_plus_output_copy_ms={key:float(np.percentile(host,p)) for key,p in [('p50',50),('p99',99)]})


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    verify();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    if 'H200' not in torch.cuda.get_device_name():raise RuntimeError('Frozen benchmark requires H200')
    receipt=json.loads((WORK/'prepared.json').read_text());config=json.loads((WORK/'protocol.json').read_text())['training']
    if args.smoke:config=dict(config,one_step_updates=3,rollout_updates=2,validate_every=3,warmup=0,ramp=1)
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    cases=load_cases(receipt['records'],args.smoke)
    train_cases=[c for c in cases if c['record']['proposed_split']=='train'];val_cases=[c for c in cases if c['record']['proposed_split']=='validation']
    normal=json.loads(Path(receipt['normalization']).read_text());model=StepPINN(config,normal['mean'],normal['std']).cuda()
    qgrid=grid();best=float('inf');history=[];start=time.monotonic()
    out=WORK/('smoke' if args.smoke else 'training')/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',config);save(out/'normalization.json',normal)
    coverage=np.zeros((len(train_cases),238),dtype=bool);torch.cuda.reset_peak_memory_stats()
    for phase,updates,length,lr in [('one_step',config['one_step_updates'],1,config['lr']),
                                    ('short_rollout',config['rollout_updates'],config['rollout_length'],config['rollout_lr'])]:
        if phase=='short_rollout' and (out/'best.pt').exists():
            model.load_state_dict(torch.load(out/'best.pt',weights_only=True)['state'])
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=updates,eta_min=lr*.05)
        windows=[(ci,k) for ci in range(len(train_cases)) for k in range(1,240-length)]
        order=rng.permutation(len(windows));cursor=0
        for update in range(1,updates+1):
            if cursor==len(order):order=rng.permutation(len(windows));cursor=0
            ci,k=windows[int(order[cursor])];cursor+=1;case=train_cases[ci]
            previous,current=case['data'][k-1],case['data'][k];target=case['data'][k+1]
            raw=features(previous,current);c=conditioning(model,raw,case['geometry'],case['record']['reynolds'])
            optimizer.zero_grad(set_to_none=True)
            if length==1:
                yy,xx=case['fluid_yx'][rng.integers(len(case['fluid_yx']),size=config['batch'])].T
                q=tensor(np.column_stack((4+(xx+.5)/64,6+(yy+.5)/64,np.full(len(xx),DT))))
                pred=model(q,c,cached_features=raw[yy,xx])
                data=((pred-target[yy,xx])/(model.std*config['data_loss_scale'])).square().mean()
                coverage[ci,k-1]=True
            else:
                losses=[]
                for offset,pred in enumerate(trajectory(previous,current,
                        lambda a,b:step(model,a,b,case['geometry'],case['record']['reynolds'],case['fluid'],qgrid),length),1):
                    losses.append(((pred[case['fluid']]-case['data'][k+offset][case['fluid']])/
                                   (model.std*config['data_loss_scale'])).square().mean())
                data=torch.stack(losses).mean()
            ramp=1. if phase=='short_rollout' else min(1.,max(0.,(update-config['warmup'])/config['ramp']))
            pde,bc=physics_losses(model,raw,c,case,rng,ramp>0,config)
            loss=data+config['pde_weight']*ramp*pde+config['bc_weight']*bc
            if not torch.isfinite(loss):
                save(out/'failure.json',dict(phase=phase,update=update,reason='nonfinite loss'));raise RuntimeError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();schedule.step()
            if update==1 or update%config['validate_every']==0 or update==updates:
                label=f'{phase}_{update:06d}'
                validation=evaluate(model,val_cases,out/'validation'/label,qgrid,config)
                row=dict(phase=phase,update=update,data=float(data.detach()),pde=float(pde.detach()),boundary=float(bc.detach()),
                         validation_score=validation['score'],any_diverged=validation['any_diverged'],elapsed_seconds=time.monotonic()-start)
                history.append(row);save(out/'history.json',history)
                state=dict(config=config,state=model.state_dict(),phase=phase,update=update,
                           score=validation['score'],source_freeze=digest(WORK/'freeze.json'))
                torch.save(state,out/'last.pt')
                if validation['score']<best:best=validation['score'];torch.save(state,out/'best.pt')
                print('STEP_PINN',label,'VAL',validation['score'],'DATA',row['data'],'PDE',row['pde'],flush=True)
    elapsed=time.monotonic()-start;chosen=torch.load(out/'best.pt',weights_only=True);model.load_state_dict(chosen['state'])
    final=evaluate(model,val_cases,out/'selected_checkpoint',qgrid,config);speed=timing(model,val_cases[0],qgrid)
    accurate=not final['any_diverged'] and all(r['scores']['all_future']['model']['velocity_relative_l2']<=.1 and
                  r['scores']['all_future']['model']['pressure_relative_l2']<=.2 for r in final['cases'])
    fast_standard=all(r['p99']<=5 for r in speed['whole_gpu_step_rounds_ms'])
    fast_rollout=all(r['actual_rollout_step_gpu_ms']['p99']<=5 for r in final['cases'])
    beats_persistence=not final['any_diverged'] and all(all(
        r['scores']['all_future']['model'][key]<r['scores']['all_future']['persistence'][key]
        for key in ['velocity_relative_l2','pressure_relative_l2']) for r in final['cases'])
    save(out/'coverage.json',dict(possible_transitions_each_case=238,one_step_observed_counts=coverage.sum(1).tolist(),
                                all_one_step_transitions_visited=bool(coverage.all()),
                                cases=[c['record']['path'] for c in train_cases]))
    save(out/'summary.json',dict(status='SMOKE_COMPLETE' if args.smoke else 'AUTOREGRESSIVE_PINN_EVALUATED',
          config=config,selected_phase=chosen['phase'],selected_update=chosen['update'],accuracy=final,
          latency=speed,training_seconds=elapsed,gpu=torch.cuda.get_device_name(),
          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),train_scenarios=len(train_cases),validation_scenarios=len(val_cases),
          input_frames=[2,3],future_frames=[4,241],rollout_steps=238,dt=DT,
          main_test_opened=False,reference_future_used_as_validation_input=False,causal_rollout=True,
          working_gate=dict(accuracy_pass=accurate,whole_step_latency_pass=fast_standard and fast_rollout,
                            benchmark_latency_pass=fast_standard,actual_rollout_latency_pass=fast_rollout,
                            beats_persistence=beats_persistence),llm_comparison_ready=False,
          checkpoint_sha256=digest(out/'best.pt'),
          limitations=['Data-assisted time-local PINN; crop lacks full external boundary histories',
                       'Training may use true states from training trajectories; validation gets only two prefix frames',
                       'No final independent test, multiple seeds or hypothesis significance claim']))


if __name__=='__main__':run()
