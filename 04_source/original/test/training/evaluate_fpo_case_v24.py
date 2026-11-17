"""Score only declared time partitions; diagnostic audit opens after selection."""
import numpy as np
import torch
from fpo_common_v24 import save
from fpo_case_v24 import split_frames, known_inputs
from fpo_step_v5a import grid, conditioning, project_known_constraints, fingerprint
from fpo_stable_v6 import grid_physics
from fpo_state_v4 import errors
from evaluate_fpo_transfer_v16 import reference_state


def query(model, known, qbase, t):
    if not 0 <= t <= 11.9+1e-6:
        raise ValueError('Query outside retained trajectory')
    q=qbase.clone();q[:,2]=t
    context=conditioning(model,known['raw'],known['geometry'],known['record']['reynolds'])
    return project_known_constraints(model.grid_prediction(q,context,known['raw']),known['fluid'])


def evaluate(model,case,out,phase,checkpoint_selected=False):
    partitions=split_frames()
    if phase=='validation':frames=partitions['validation']
    elif phase in ['audit','postfit_full'] and checkpoint_selected:
        frames=partitions['audit'] if phase=='audit' else list(range(4,242))
    else:raise ValueError('Reserved audit/full scoring allowed only after checkpoint selection')
    out.mkdir(parents=True,exist_ok=False)
    known=known_inputs(case);qbase=grid();cpu_mask=torch.from_numpy(case['fluid_np'])
    interior_cpu=case['interior_gpu'].cpu()
    scorer=dict(record=known['record'],interior_gpu=interior_cpu)
    scale=model.std[:2].mean();cpu_scale=scale.cpu()
    initial=known['raw'][:,:,:3];initial_cpu=initial.cpu().numpy()
    y,x=case['interior'].T
    u,v=initial[:,:,0],initial[:,:,1]
    initial_div=(torch.roll(u,-1,1)-torch.roll(u,1,1)+torch.roll(v,-1,0)-torch.roll(v,1,0))*32
    prefix_div=float(initial_div[case['interior_gpu']].square().mean().sqrt())
    rows=[];times=[];model.eval()
    previous_frame=None;previous=None;previous_ref=None
    with torch.inference_mode():
        torch.testing.assert_close(query(model,known,qbase,0.),initial,rtol=1e-5,atol=1e-6)
        for _ in range(30):query(model,known,qbase,.05)
        torch.cuda.synchronize()
        for frame in frames:
            t=(frame-3)*.05
            # FD residual always spans ONE original .05 interval. The preceding
            # selected validation frame may be six intervals away and is invalid.
            if previous_frame!=frame-1:
                previous=query(model,known,qbase,t-.05)
                previous_ref=reference_state(case['reference'],frame-1,cpu_mask)
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();value=query(model,known,qbase,t);b.record();b.synchronize()
            times.append(a.elapsed_time(b))
            if not torch.isfinite(value).all() or value.abs().max()>1000:
                result=dict(failure=True,score=1e6,phase=phase,predicted_frames=len(rows),failure_frame=frame,path=case['record']['path'])
                save(out/'per_frame.json',rows);save(out/'summary.json',result);model.train();return result
            # CPU scorer only; reference never reaches query/model/loss.
            target=np.asarray(case['reference'][frame])
            ref=reference_state(case['reference'],frame,cpu_mask)
            reference_momentum,_=grid_physics(previous_ref,ref,scorer,cpu_scale)
            momentum,_=grid_physics(previous,value,case,scale)
            u,v=value[:,:,0],value[:,:,1]
            div=(torch.roll(u,-1,1)-torch.roll(u,1,1)+torch.roll(v,-1,0)-torch.roll(v,1,0))*32
            refdiv=(target[y,x+1,0]-target[y,x-1,0]+target[y+1,x,1]-target[y-1,x,1])*32
            rows.append(dict(frame=frame,time=t,fd_previous_frame=frame-1,
                query_gpu_ms=times[-1],
                model=errors(value.cpu().numpy()[case['fluid_np']],target[case['fluid_np']]),
                persistence=errors(initial_cpu[case['fluid_np']],target[case['fluid_np']]),
                interior_div_rms=float(div[case['interior_gpu']].square().mean().sqrt()),
                reference_div_rms=float(np.sqrt(np.mean(refdiv**2))),
                fd_momentum_loss=float(momentum),reference_fd_momentum_loss=float(reference_momentum)))
            previous_frame=frame;previous=value;previous_ref=ref
    if fingerprint(case['record']['field'])!=case['record']['field_fingerprint']:
        raise ValueError('Reference changed')
    means={k:float(np.mean([r['model'][k] for r in rows])) for k in rows[0]['model']}
    persistence={k:float(np.mean([r['persistence'][k] for r in rows])) for k in rows[0]['persistence']}
    limit=max(3*prefix_div,3*float(np.mean([r['reference_div_rms'] for r in rows])),1e-3)
    div=float(np.mean([r['interior_div_rms'] for r in rows]))
    timing={k:float(np.percentile(times,p)) for k,p in [('p50',50),('p95',95),('p99',99)]}
    result=dict(failure=False,phase=phase,predicted_frames=len(rows),frames=frames,path=case['record']['path'],
        mean_errors=means,score=sum(means.values())/2,persistence_mean_errors=persistence,
        interior_div_rms=div,prefix_div_rms=prefix_div,mass_limit=limit,mass_pass=div<=limit,
        direct_query_gpu_ms=timing,causal_step_latency=False,
        mean_fd_momentum_loss=float(np.mean([r['fd_momentum_loss'] for r in rows])),
        mean_reference_fd_momentum_loss=float(np.mean([r['reference_fd_momentum_loss'] for r in rows])),
        accuracy_pass=means['velocity_relative_l2']<=.1 and means['pressure_relative_l2']<=.2,
        latency_pass=timing['p99']<=5,checkpoint_selected=checkpoint_selected,
        full_trajectory_reconstruction=phase=='postfit_full',independent_final_test=False)
    save(out/'per_frame.json',rows);save(out/'summary.json',result);model.train();return result


def reserved_audit_from_full(full_report,full_directory,out):
    """Report the39 reserved times from ONE frozen post-selection evaluation."""
    import json
    if full_report['failure'] or not full_report['checkpoint_selected'] or full_report['phase']!='postfit_full':
        raise ValueError('Successful frozen full evaluation required')
    all_rows=json.loads((full_directory/'per_frame.json').read_text())
    wanted=split_frames()['audit'];rows=[r for r in all_rows if r['frame'] in set(wanted)]
    if [r['frame'] for r in rows]!=wanted:raise ValueError('Reserved audit coverage differs')
    result=dict(full_report)
    means={k:float(np.mean([r['model'][k] for r in rows])) for k in rows[0]['model']}
    persistence={k:float(np.mean([r['persistence'][k] for r in rows])) for k in rows[0]['persistence']}
    limit=max(3*result['prefix_div_rms'],3*float(np.mean([r['reference_div_rms'] for r in rows])),1e-3)
    div=float(np.mean([r['interior_div_rms'] for r in rows]))
    timing={k:float(np.percentile([r['query_gpu_ms'] for r in rows],p)) for k,p in [('p50',50),('p95',95),('p99',99)]}
    result.update(phase='audit',frames=wanted,predicted_frames=len(rows),mean_errors=means,
        persistence_mean_errors=persistence,score=sum(means.values())/2,interior_div_rms=div,
        mass_limit=limit,mass_pass=div<=limit,direct_query_gpu_ms=timing,latency_pass=timing['p99']<=5,
        mean_fd_momentum_loss=float(np.mean([r['fd_momentum_loss'] for r in rows])),
        mean_reference_fd_momentum_loss=float(np.mean([r['reference_fd_momentum_loss'] for r in rows])),
        accuracy_pass=means['velocity_relative_l2']<=.1 and means['pressure_relative_l2']<=.2,
        full_trajectory_reconstruction=False,summarized_from_single_frozen_full_evaluation=True)
    out.mkdir(parents=True,exist_ok=False);save(out/'per_frame.json',rows);save(out/'summary.json',result)
    return result
