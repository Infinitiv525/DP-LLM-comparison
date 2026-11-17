"""Read-only transfer scoring. Future development labels stay on CPU."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from fpo_common_v16 import ROOT, WORK, save, digest, verify
from fpo_direct_curl_v15 import DirectPINN
from train_fpo_direct_curl_v15 import query
from fpo_step_v5a import load_cases, grid, fingerprint
from fpo_stable_v6 import grid_physics
from fpo_state_v4 import errors


def reference_state(reference, frame, fluid):
    # CPU-only scorer, matching the float32/gauge/mask convention of TRAIN.
    a=torch.from_numpy(np.array(reference[frame],dtype=np.float32,copy=True))
    a[:,:,2]-=a[:,:,2][fluid].mean()
    return a*fluid[:,:,None]


def evaluate(model,case,out,qbase):
    out.mkdir(parents=True,exist_ok=False)
    # Only these known-input keys reach the predictor. No reference/data key.
    known={k:case[k] for k in ['raw','geometry','record','fluid']}
    known['record']={'reynolds':case['record']['reynolds']}
    cpu_mask=torch.from_numpy(case['fluid_np'])
    interior_cpu=case['interior_gpu'].cpu()
    scorer={'record':known['record'],'interior_gpu':interior_cpu}
    scale=model.std[:2].mean(); cpu_scale=scale.cpu()
    initial=case['raw'][:,:,:3]; initial_cpu=initial.cpu().numpy()
    y,x=case['interior'].T
    previous=initial
    previous_ref=reference_state(case['reference'],3,cpu_mask)
    u,v=initial[:,:,0],initial[:,:,1]
    initial_div=(torch.roll(u,-1,1)-torch.roll(u,1,1)+torch.roll(v,-1,0)-torch.roll(v,1,0))*32
    prefix_div=float(initial_div[case['interior_gpu']].square().mean().sqrt())
    rows=[]; times=[]; model.eval()
    with torch.inference_mode():
        ic=query(model,known,qbase,0.)
        torch.testing.assert_close(ic,initial,rtol=1e-5,atol=1e-6)
        # Same warm-up for both checkpoints; excluded from timing percentiles.
        for _ in range(20):query(model,known,qbase,.05)
        torch.cuda.synchronize()
        for offset in range(1,239):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();value=query(model,known,qbase,offset*.05);b.record();b.synchronize()
            times.append(a.elapsed_time(b))
            if not torch.isfinite(value).all() or value.abs().max()>1000:
                result=dict(failure=True,predicted_frames=len(rows),failure_offset=offset,path=case['record']['path'])
                save(out/'per_frame.json',rows);save(out/'summary.json',result);return result
            target=np.asarray(case['reference'][offset+3])
            ref=reference_state(case['reference'],offset+3,cpu_mask)
            reference_momentum,_=grid_physics(previous_ref,ref,scorer,cpu_scale)
            momentum,_=grid_physics(previous,value,case,scale)
            u,v=value[:,:,0],value[:,:,1]
            div=(torch.roll(u,-1,1)-torch.roll(u,1,1)+torch.roll(v,-1,0)-torch.roll(v,1,0))*32
            refdiv=(target[y,x+1,0]-target[y,x-1,0]+target[y+1,x,1]-target[y-1,x,1])*32
            rows.append(dict(offset=offset,frame=offset+3,time=offset*.05,
                model=errors(value.cpu().numpy()[case['fluid_np']],target[case['fluid_np']]),
                persistence=errors(initial_cpu[case['fluid_np']],target[case['fluid_np']]),
                interior_div_rms=float(div[case['interior_gpu']].square().mean().sqrt()),
                reference_div_rms=float(np.sqrt(np.mean(refdiv**2))),
                fd_momentum_loss=float(momentum),reference_fd_momentum_loss=float(reference_momentum)))
            previous=value;previous_ref=ref
    if fingerprint(case['record']['field'])!=case['record']['field_fingerprint']:raise ValueError('Reference changed')
    means={k:float(np.mean([r['model'][k] for r in rows])) for k in rows[0]['model']}
    persistence={k:float(np.mean([r['persistence'][k] for r in rows])) for k in rows[0]['persistence']}
    limit=max(3*prefix_div,3*float(np.mean([r['reference_div_rms'] for r in rows])),1e-3)
    div=float(np.mean([r['interior_div_rms'] for r in rows]))
    timing={k:float(np.percentile(times,p)) for k,p in [('p50',50),('p95',95),('p99',99)]}
    result=dict(failure=False,predicted_frames=len(rows),path=case['record']['path'],scope=case['record']['proposed_split'],
        mean_errors=means,score=sum(means.values())/2,persistence_mean_errors=persistence,
        interior_div_rms=div,prefix_div_rms=prefix_div,mass_limit=limit,mass_pass=div<=limit,
        direct_query_gpu_ms=timing,causal_step_latency=False,
        mean_fd_momentum_loss=float(np.mean([r['fd_momentum_loss'] for r in rows])),
        mean_reference_fd_momentum_loss=float(np.mean([r['reference_fd_momentum_loss'] for r in rows])),
        accuracy_pass=means['velocity_relative_l2']<=.1 and means['pressure_relative_l2']<=.2,
        latency_pass=timing['p99']<=5)
    save(out/'per_frame.json',rows);save(out/'summary.json',result);return result


def run():
    verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    receipt=json.loads((WORK/'prepared.json').read_text());records=receipt['records']
    if len(records)!=6 or any(r['proposed_split']!='validation' for r in records):raise ValueError('Six development VAL only')
    for r in records:
        if digest(r['state_cache'])!=r['state_cache_sha256']:raise ValueError('Changed prefix')
        for name in ['geometry.npz','samples.npz']:
            p=str(Path(r['cache'])/name)
            if digest(p)!=r['hashes'][p]:raise ValueError('Changed cache')
    cases=load_cases(records)
    for case in cases:
        assert 'data' not in case
        with np.load(case['record']['state_cache'],allow_pickle=False) as z:case['raw']=torch.tensor(z['features'],device='cuda')
        case['interior_gpu']=torch.zeros_like(case['fluid']);y,x=case['interior'].T;case['interior_gpu'][y,x]=True
    normal=json.loads(Path(receipt['normalization']).read_text());protocol=json.loads((WORK/'protocol.json').read_text())
    out=WORK/'evaluation'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    results=[];qbase=grid()
    for c in protocol['checkpoints']:
        chosen=torch.load(c['path'],weights_only=True,map_location='cuda')
        model=DirectPINN(chosen['config'],normal['mean'],normal['std']).cuda()
        model.load_state_dict(chosen['state']);model.eval();model.requires_grad_(False)
        reports=[evaluate(model,case,out/c['name']/str(i),qbase) for i,case in enumerate(cases)]
        result=dict(name=c['name'],checkpoint_sha256=digest(c['path']),checkpoint_update=chosen['update'],cases=reports,
            optimizer_updates=0,checkpoint_selected_on_validation=False,main_test_opened=False,
            independent_final_test=False,causal_stepper=False,llm_ready=False)
        if all(not r['failure'] for r in reports):
            result['mean_errors']={k:float(np.mean([r['mean_errors'][k] for r in reports])) for k in reports[0]['mean_errors']}
            result['gates']={k:all(r[k] for r in reports) for k in ['accuracy_pass','mass_pass','latency_pass']}
        save(out/c['name']/'summary.json',result);results.append(result)
        print('EVALUATED',json.dumps(result),flush=True)
        del model,chosen;torch.cuda.empty_cache()
    verify()
    save(out/'summary.json',dict(status='READ_ONLY_TRANSFER_EVALUATED',results=results,
        validation_scenarios=6,source_train_scenarios=3,optimizer_updates=0,
        main_test_opened=False,independent_final_test=False))


if __name__=='__main__':run()
