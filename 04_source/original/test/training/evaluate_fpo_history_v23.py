"""Read-only transfer scoring. Future development labels stay on CPU."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from fpo_common_v23 import ROOT, WORK, save, digest, verify
from fpo_history_v23 import query, TARGET_COUNT, ANCHOR, DT
from fpo_step_v5a import fingerprint
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
    known={k:case[k] for k in ['raw','history','geometry','record','fluid']}
    known['record']={'reynolds':case['record']['reynolds']}
    cpu_mask=torch.from_numpy(case['fluid_np'])
    interior_cpu=case['interior_gpu'].cpu()
    scorer={'record':known['record'],'interior_gpu':interior_cpu}
    scale=model.std[:2].mean(); cpu_scale=scale.cpu()
    initial=case['raw'][:,:,:3]; initial_cpu=initial.cpu().numpy()
    y,x=case['interior'].T
    previous=initial
    previous_ref=reference_state(case['reference'],ANCHOR,cpu_mask)
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
        for offset in range(1,TARGET_COUNT+1):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();value=query(model,known,qbase,offset*.05);b.record();b.synchronize()
            times.append(a.elapsed_time(b))
            if not torch.isfinite(value).all() or value.abs().max()>1000:
                result=dict(failure=True,predicted_frames=len(rows),failure_offset=offset,path=case['record']['path'])
                save(out/'per_frame.json',rows);save(out/'summary.json',result);return result
            target=np.asarray(case['reference'][offset+ANCHOR])
            ref=reference_state(case['reference'],offset+ANCHOR,cpu_mask)
            reference_momentum,_=grid_physics(previous_ref,ref,scorer,cpu_scale)
            momentum,_=grid_physics(previous,value,case,scale)
            u,v=value[:,:,0],value[:,:,1]
            div=(torch.roll(u,-1,1)-torch.roll(u,1,1)+torch.roll(v,-1,0)-torch.roll(v,1,0))*32
            refdiv=(target[y,x+1,0]-target[y,x-1,0]+target[y+1,x,1]-target[y-1,x,1])*32
            rows.append(dict(offset=offset,frame=offset+ANCHOR,time=offset*.05,
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


