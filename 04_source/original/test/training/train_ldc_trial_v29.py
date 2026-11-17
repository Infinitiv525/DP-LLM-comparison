"""Fresh technical PINN trial, derived from frozen V26; no main-search execution.
New: five bounded hyperparameters and independent, matched data/PDE/BC RNG streams.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'llm'))
import numpy as np
import torch
from ldc_llm_common_v29 import ROOT,WORK,save,read,digest,verify
from ldc_contract_v29 import training_config
from ldc_shared_pinn_v26 import SharedSteadyPINN,steady_residual
from ldc_store_v26 import Store,conditioning
from train_ldc_shared_v26 import benchmark,evaluate


def train(config,store,out):
    out.mkdir(parents=True,exist_ok=False);save(out/'config.json',config)
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    model=SharedSteadyPINN(config,store.normalization).cuda()
    gate=benchmark(model,store,out/'readiness')
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    steps_per_epoch=len(store.train_ids)//config['batch_cases']
    if steps_per_epoch*config['batch_cases']!=len(store.train_ids):raise ValueError('ExactfullTRAINepoch required')
    updates=steps_per_epoch*config['epochs']
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=updates,eta_min=config['lr']*.05)
    data_rng=np.random.default_rng(config['seed']+100000);pde_rng=np.random.default_rng(config['seed']+200000);bc_rng=np.random.default_rng(config['seed']+300000)
    history=[];counts={i:0 for i in store.train_ids};sampling=hashlib.sha256();best=float('inf');start=time.monotonic()
    save(out/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()),
        seed=config['seed'],precision='float32',allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,source_freeze=digest(WORK/'freeze.json')))
    weight_hash=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):weight_hash.update(name.encode());weight_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    initial_weight_sha256=weight_hash.hexdigest();physics_updates=0
    order=rng.permutation(store.train_ids);cursor=0;statistics=torch.zeros(4,device='cuda');window=0
    torch.cuda.reset_peak_memory_stats()

    def assess(update):
        nonlocal best,window
        report=evaluate(model,store,out/'validation'/f'update_{update:06d}')
        row=dict(update=update,equivalent_epoch=update/steps_per_epoch,validation=report,
            physics_coefficient=config['physics_weight']*min(1.,max(0.,(update-config['warmup'])/config['ramp'])),
            window_updates=window,window_mean=None if not window else dict(zip(['data','steady_ns','bc','gradient_norm_before_clip'],(statistics/window).cpu().tolist())),
            elapsed_seconds=time.monotonic()-start)
        history.append(row);save(out/'history.json',history)
        checkpoint=dict(state=model.state_dict(),config=config,normalization=store.normalization,update=update,
            score=report['score'],source_freeze=digest(WORK/'freeze.json'))
        torch.save(checkpoint,out/'last.pt')
        if report['score']<best:best=report['score'];torch.save(checkpoint,out/'best.pt')
        torch.save(dict(checkpoint=checkpoint,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            numpy_rng=rng.bit_generator.state,data_rng=data_rng.bit_generator.state,pde_rng=pde_rng.bit_generator.state,
            bc_rng=bc_rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            case_counts=counts,case_order=order.tolist(),cursor=cursor,sample_sha256=sampling.hexdigest()),out/'training_state.pt')
        print('ASSESS',update,report.get('means',{'failure':True}),flush=True);statistics.zero_();window=0

    assess(0)
    for update in range(1,updates+1):
        if cursor==len(order):order=rng.permutation(store.train_ids);cursor=0
        batch=order[cursor:cursor+config['batch_cases']];cursor+=len(batch)
        losses=[];parts=[]
        coefficient=config['physics_weight']*min(1.,max(0.,(update-config['warmup'])/config['ramp']))
        for index in batch:
            index=int(index);counts[index]+=1;known=store.known(index)
            sampling.update(np.asarray([index],dtype='<i8').tobytes())
            context=conditioning(model,known)
            y,x=store.fluid_cells[index][data_rng.integers(len(store.fluid_cells[index]),size=config['data_points'])].T
            sampling.update(np.asarray(np.column_stack((y,x)),dtype='<i8').tobytes())
            yy=torch.as_tensor(y.astype('int64'),device='cuda');xx=torch.as_tensor(x.astype('int64'),device='cuda')
            q=torch.stack(((xx+.5)/512,(yy+.5)/512),1)
            pred=model.sampled_prediction(q,context,yy,xx)
            weights=pred.new_tensor([1.,1.,config['pressure_weight']])
            data=(((pred-store.training_target(index,yy,xx))/model.std).square()*weights).mean()
            cells=store.interiors[index];cy,cx=cells[pde_rng.integers(len(cells),size=config['physics_points'])].T
            qc_np=np.column_stack(((cx+.5+pde_rng.uniform(-.4,.4,len(cx)))/512,(cy+.5+pde_rng.uniform(-.4,.4,len(cy)))/512))
            # Consume and hash the same samples even during PDE warmup.
            sampling.update(np.asarray(qc_np,dtype='<f8').tobytes())
            pde=torch.zeros((),device='cuda')
            if coefficient:
                qc=torch.tensor(qc_np,device='cuda',dtype=torch.float32,requires_grad=True)
                pde=steady_residual(qc,model(qc,context),known['reynolds']).square().mean(0).sum()
            n=config['wall_points_per_side'];s=bc_rng.uniform(1/512,1-1/512,size=(4,n))
            wall=np.concatenate((np.column_stack((s[0],np.zeros(n))),np.column_stack((s[1],np.ones(n))),
                np.column_stack((np.zeros(n),s[2])),np.column_stack((np.ones(n),s[3]))))
            body=store.body[index][bc_rng.integers(len(store.body[index]),size=config['body_points'])]
            qb_np=np.concatenate((wall,body));sampling.update(np.asarray(qb_np,dtype='<f8').tobytes())
            qb=torch.tensor(qb_np,device='cuda',dtype=torch.float32,requires_grad=True)
            target=torch.zeros(len(qb),2,device='cuda');target[n:2*n,0]=1.
            bc=((model(qb,context)[:,:2]-target)/model.std[:2]).square().mean()
            losses.append(data+coefficient*pde+config['bc_weight']*bc);parts.append(torch.stack((data,pde,bc)).detach())
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss, no fieldclipping')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        physics_updates+=int(coefficient>0)
        optimizer.step();scheduler.step();statistics[:3]+=torch.stack(parts).mean(0);statistics[3]+=grad;window+=1
        if update%(steps_per_epoch*config['assess_epochs'])==0 or update==updates:assess(update)
    selected=torch.load(out/'best.pt',weights_only=True,map_location='cuda');model.load_state_dict(selected['state'])
    result=evaluate(model,store,out/'selected_validation');speed=benchmark(model,store,out/'selected_speed')
    if set(counts.values())!={config['epochs']}:raise ValueError('Unequalcasecoverage')
    summary=dict(status='LDC_CONFIGURATOR_TECHNICAL_TRIAL_COMPLETE',
        physics_updates=physics_updates,initial_weight_sha256=initial_weight_sha256,
        selected_update=selected['update'],selected_validation=result,selected_speed=speed,initial_speed=gate,
        case_counts=counts,sample_sha256=sampling.hexdigest(),checkpoint_sha256=digest(out/'best.pt'),
        training_seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        train_cases=len(store.train_ids),validation_cases=len(store.validation_ids),shared_weights=True,
        main_test_opened=False,independent_final_test=False,llm_ready=False)
    if physics_updates<1:raise ValueError('Physics must be exercised in technical training')
    save(out/'summary.json',summary);return summary



def run():
    p=argparse.ArgumentParser();p.add_argument('--parameters',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise ValueError('H200 required')
    if torch.cuda.device_count()!=1:raise ValueError('Exactly one assigned GPU required for each independent trial')
    protocol=read(WORK/'protocol.json')
    if protocol['phase']!='technical_preflight' or protocol['technical_epochs']!=2:raise ValueError('Technical preflight only')
    prepared=read(protocol['prepared_path']);split=read(protocol['split_path'])['indices']
    if prepared['status']!='LDC_DATA_READY' or not prepared['physics_ready']:raise ValueError('Data scale audit required')
    config=training_config(protocol['base_model'],read(a.parameters),2,protocol['training_seed'])
    store=Store(prepared,split,False);train(config,store,a.output);verify()


if __name__=='__main__':run()
