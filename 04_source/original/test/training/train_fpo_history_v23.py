"""Matched shared PINNs, last2 versus20 observations, full220-frame future."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from fpo_common_v23 import WORK,save,digest,verify
from fpo_history_v23 import HistoryPINN, conditioning, load_cases, TARGET_COUNT, ANCHOR, END_TIME
from fpo_state_v4 import residual
from fpo_step_v5a import grid,fingerprint
from fpo_history_v23 import query
from evaluate_fpo_history_v23 import evaluate
from fpo_grid_physics_v19 import sampled_grid_terms,momentum_loss



def state_digest(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(str(tuple(value.shape)).encode());h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(config,train_cases,val_cases,out,qbase,normal,source_checkpoint,smoke=False):
    out.mkdir(parents=True,exist_ok=False);save(out/'config.json',config)
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    model=HistoryPINN(config,normal['mean'],normal['std']).cuda()
    chosen=torch.load(source_checkpoint['path'],weights_only=True,map_location='cuda')
    if chosen['update']!=source_checkpoint['update'] or chosen['config']['representation']!='curl' or not chosen['config']['local_context']:raise ValueError('Wrong warm start')
    torch.testing.assert_close(model.std,chosen['state']['std'],rtol=0,atol=0)
    torch.testing.assert_close(model.mean,chosen['state']['mean'],rtol=0,atol=0)
    model.load_source(chosen['state']);del chosen;initial_sha=state_digest(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['updates'],eta_min=config['lr']*.05)
    sampling=hashlib.sha256();case_hash=hashlib.sha256();history=[];best=float('inf')
    coverage=np.zeros((len(train_cases),220),dtype=bool);counts=np.zeros(len(train_cases),dtype=int)
    orders=[rng.permutation(220) for _ in train_cases];cursors=np.zeros(len(train_cases),dtype=int)
    case_order=rng.permutation(len(train_cases));case_cursor=0;start=time.monotonic()
    statistics=torch.zeros(6,device='cuda');window_count=0;torch.cuda.reset_peak_memory_stats()
    monitors=[next(c for c in train_cases if c['record']['family']==f) for f in sorted({c['record']['family'] for c in train_cases})]
    with torch.no_grad():
        for c in train_cases[:1]:torch.testing.assert_close(query(model,c,qbase,0.),c['raw'][:,:,:3],rtol=1e-5,atol=1e-6)

    def assess(update):
        nonlocal best,window_count
        reports=[evaluate(model,c,out/'assessments'/f'update_{update:06d}'/str(i),qbase) for i,c in enumerate(val_cases)]
        score=float(np.mean([r['score'] for r in reports]))
        row=dict(update=update,assessment=dict(score=score,cases=reports),elapsed_seconds=time.monotonic()-start,
            lr=optimizer.param_groups[0]['lr'],window_updates=window_count,
            window_mean=dict(zip(['data','continuous_ns','fd_momentum_normalized','boundary','gradient_norm_before_clip','fd_coefficient'],
                                 (statistics/max(window_count,1)).cpu().tolist())) if window_count else None)
        if update%config['monitor_every']==0 or update==config['updates']:
            row['train_fit_monitor']=[evaluate(model,c,out/'fit_history'/f'update_{update:06d}'/str(i),qbase) for i,c in enumerate(monitors)]
        model.train()
        history.append(row);save(out/'history.json',history)
        saved=dict(state=model.state_dict(),config=config,update=update,score=score,source_freeze=digest(WORK/'freeze.json'),
                   source_checkpoint_sha256=source_checkpoint['sha256'],source_checkpoint_update=source_checkpoint['update'],source_checkpoint_stage='V21')
        torch.save(saved,out/'last.pt')
        if score<best:best=score;torch.save(saved,out/'best.pt')
        torch.save(dict(checkpoint=saved,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            orders=[o.tolist() for o in orders],cursors=cursors.tolist(),case_order=case_order.tolist(),case_cursor=case_cursor,
            counts=counts.tolist(),coverage=coverage.tolist(),best_score=best,
            sample_schedule_sha256=sampling.hexdigest(),case_frame_schedule_sha256=case_hash.hexdigest()),out/'training_state.pt')
        print('ASSESS',config['name'],update,json.dumps(row),flush=True)
        statistics.zero_();window_count=0

    assess(0)
    for update in range(1,config['updates']+1):
        if case_cursor==len(case_order):case_order=rng.permutation(len(train_cases));case_cursor=0
        ci=int(case_order[case_cursor]);case_cursor+=1;case=train_cases[ci];counts[ci]+=1
        if cursors[ci]==220:orders[ci]=rng.permutation(220);cursors[ci]=0
        offset=int(orders[ci][cursors[ci]])+1;cursors[ci]+=1;coverage[ci,offset-1]=True
        key=np.asarray([ci,offset],dtype='<i8').tobytes();sampling.update(key);case_hash.update(key)
        yy,xx=case['fluid_yx'][rng.integers(len(case['fluid_yx']),size=config['batch'])].T
        q_np=np.column_stack((4+(xx+.5)/64,6+(yy+.5)/64,np.full(len(xx),offset*.05)))
        sampling.update(np.asarray(np.column_stack((yy,xx)),dtype='<i8').tobytes())
        q=torch.tensor(q_np,device='cuda',dtype=torch.float32)
        context=conditioning(model,case)
        pred=model.sampled_prediction(q,context,case['raw'],torch.as_tensor(yy,device='cuda'),torch.as_tensor(xx,device='cuda'))
        data=((pred-case['data'][offset+19,yy,xx])/model.std).square().mean()
        y,x=case['interior'][rng.integers(len(case['interior']),size=config['collocation'])].T
        qc_np=np.column_stack((4+(x+.5+rng.uniform(-.4,.4,len(x)))/64,6+(y+.5+rng.uniform(-.4,.4,len(y)))/64,rng.uniform(0,11.0,len(x))))
        sampling.update(np.asarray(np.column_stack((y,x)),dtype='<i8').tobytes());sampling.update(np.asarray(qc_np,dtype='<f8').tobytes())
        qc=torch.tensor(qc_np,device='cuda',dtype=torch.float32,requires_grad=True)
        pde=residual(qc,model(qc,context,case['raw']),case['record']['reynolds']).square().mean(0).sum()
        n=config['fd_points'];tf=qc.detach()[:n,2].clamp(config['fd_dt'],11.0-config['fd_dt'])
        yf=torch.as_tensor(y[:n],device='cuda');xf=torch.as_tensor(x[:n],device='cuda')
        # Both arms backpropagate identical continuous and grid physics objectives.
        if config['fd_weight']:
            fd=momentum_loss(sampled_grid_terms(model,case['raw'],context,yf,xf,tf,case['record']['reynolds'],config['fd_dt']),model.std[:2].mean())
        else:
            with torch.no_grad():fd=momentum_loss(sampled_grid_terms(model,case['raw'],context,yf,xf,tf,case['record']['reynolds'],config['fd_dt']),model.std[:2].mean())
        boundary=case['boundary'][rng.integers(len(case['boundary']),size=128)]
        qb_np=np.column_stack((boundary,rng.uniform(0,11.0,len(boundary))));sampling.update(np.asarray(qb_np,dtype='<f8').tobytes())
        qb=torch.tensor(qb_np,device='cuda',dtype=torch.float32,requires_grad=True)
        bc=(model(qb,context,case['raw'])[:,:2]/model.std[:2]).square().mean()
        coefficient=config['fd_weight']*min(1.,update/config['fd_ramp_updates'])
        loss=data+config['continuous_weight']*pde+config['boundary_weight']*bc+coefficient*fd
        if not torch.isfinite(loss) or not torch.isfinite(fd):raise RuntimeError('Nonfinite loss; no clipping of fields')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step()
        statistics.add_(torch.stack((data,pde,fd,bc,grad,loss.new_tensor(coefficient))).detach());window_count+=1
        if update%config['assess_every']==0 or update==config['updates']:assess(update)
    last_diagnostic=None
    chosen=torch.load(out/'best.pt',weights_only=True,map_location='cuda');model.load_state_dict(chosen['state'])
    selected=[evaluate(model,c,out/'selected_checkpoint'/str(i),qbase) for i,c in enumerate(val_cases)]
    train_accuracy=[evaluate(model,c,out/'training_monitor'/str(i),qbase) for i,c in enumerate(monitors)]
    selected_diagnostic=None
    result=dict(status='KNOWN_HISTORY_EVALUATED',config=config,accuracy=selected,training_monitor=train_accuracy,
        source_checkpoint_sha256=source_checkpoint['sha256'],source_checkpoint_update=source_checkpoint['update'],source_checkpoint_stage='V21',initial_state_sha256=initial_sha,
        initial_validation_score=history[0]['assessment']['score'],selected_added_updates=chosen['update'],
        added_training_case_updates=counts.tolist(),all220_training_frames_visited=bool(coverage.all()),
        sample_schedule_sha256=sampling.hexdigest(),case_frame_schedule_sha256=case_hash.hexdigest(),
        train_scenarios=len(train_cases),validation_scenarios=len(val_cases),
        checkpoint_sha256=digest(out/'best.pt'),training_seconds=time.monotonic()-start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),physics_last=last_diagnostic,physics_selected=selected_diagnostic,
        anchor_frame=21,predicted_frames_per_case=220,history_length=20 if config['history_enabled'] else 2,
        diagnostic_scope='Full220-frame finite-difference momentum and divergence plus recorded continuous-NS training loss; no separate V18 pointwise audit',
        working_gate=dict(accuracy_pass=all(not r['failure'] and r['accuracy_pass'] for r in selected),
            mass_pass=all(not r['failure'] and r['mass_pass'] for r in selected),
            direct_query_latency_pass=all(not r['failure'] and r['latency_pass'] for r in selected)),
        independent_final_test=False,main_test_opened=False,causal_stepper=False,llm_ready=False)
    save(out/'summary.json',result);return result


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    verify();torch.set_num_threads(4)
    gate=json.loads((WORK/'benchmark'/os.environ['SLURM_JOB_ID']/'summary.json').read_text())
    if not gate['readiness_pass']:raise RuntimeError('History readiness gate failed')
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    receipt=json.loads((WORK/'prepared.json').read_text());records=receipt['records']
    key=lambda r:(r['family'],r['geometry_id'])
    if len(records)!=75 or (sum(r['proposed_split']=='train' for r in records),sum(r['proposed_split']=='validation' for r in records))!=(69,6):raise ValueError('Expected69TRAIN6VAL')
    if {key(r) for r in records if r['proposed_split']=='train'}&{key(r) for r in records if r['proposed_split']=='validation'}:raise ValueError('Geometry overlap')
    if args.smoke:records=[next(r for r in records if r['proposed_split']==role) for role in ['train','validation']]
    for r in records:
        if r['proposed_split']=='test' or digest(r['state_cache'])!=r['state_cache_sha256']:raise ValueError('Sealed or changed prefix')
        for name in ['geometry.npz','samples.npz']:
            p=str(Path(r['cache'])/name)
            if digest(p)!=r['hashes'][p]:raise ValueError('Changed geometry cache')
    cases=load_cases(records)
    for c in cases:
        if c['record']['proposed_split']=='validation' and 'data' in c:raise ValueError('Future VAL onGPU')
        # Known20-frame history and anchor are already attached by V23 loader.
        c['reference']=np.load(c['record']['field'],mmap_mode='r',allow_pickle=False)
        c['interior_gpu']=torch.zeros_like(c['fluid']);y,x=c['interior'].T;c['interior_gpu'][y,x]=True
    train_cases=[c for c in cases if c['record']['proposed_split']=='train'];val_cases=[c for c in cases if c['record']['proposed_split']=='validation']
    normal=json.loads(Path(receipt['normalization']).read_text());protocol=json.loads((WORK/'protocol.json').read_text());results=[]
    out=WORK/('smoke' if args.smoke else 'training')/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    for config in protocol['candidates']:
        if args.smoke:config=dict(config,updates=3,assess_every=3,monitor_every=3,fd_ramp_updates=1)
        results.append(train(config,train_cases,val_cases,out/config['name'],grid(),normal,protocol['source_checkpoint'],args.smoke))
        torch.cuda.empty_cache()
    for field in ['initial_state_sha256','sample_schedule_sha256','case_frame_schedule_sha256','added_training_case_updates']:
        if results[0][field]!=results[1][field]:raise ValueError('Unmatched comparison: '+field)
    if not args.smoke and not all(r['all220_training_frames_visited'] for r in results):raise ValueError('Incomplete frame coverage')
    if not np.isclose(results[0]['initial_validation_score'],results[1]['initial_validation_score'],rtol=1e-6,atol=1e-6):raise ValueError('Unmatched initial validation')
    for c in cases:
        if fingerprint(c['record']['field'])!=c['record']['field_fingerprint']:raise ValueError('Reference changed')
    viable=[r['config']['name'] for r in results if all(r['working_gate'].values())]
    verify();save(out/'summary.json',dict(status='SMOKE_COMPLETE' if args.smoke else 'KNOWN_HISTORY_COMPLETE',
        results=results,matched_initial_state=True,matched_full_sampling=True,main_test_opened=False,independent_final_test=False,
        viable_shared_variants=viable,decision='SMOKE_ONLY' if args.smoke else
            ('SHARED_MODEL_PASSES_ENGINEERING_GATES' if viable else 'REVIEW_THEN_AUTHORIZED_CASE_SPECIFIC_FALLBACK'),
        fallback_authorized_by_user=True,automatic_fallback_submitted=False))


if __name__=='__main__':run()
