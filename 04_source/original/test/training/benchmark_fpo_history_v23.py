"""Native full-field history-query speed gate before matched training."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from fpo_common_v23 import WORK, save, verify, digest
from fpo_history_v23 import HistoryPINN, known_case, query, conditioning, TARGET_COUNT
from fpo_spatial_context_v21 import SpatialPINN
from fpo_step_v5a import grid, conditioning as old_conditioning, project_known_constraints


def run():
    verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():
        raise RuntimeError('H200 required')
    protocol=json.loads((WORK/'protocol.json').read_text())
    receipt=json.loads((WORK/'prepared.json').read_text())
    records=[next(r for r in receipt['records'] if r['proposed_split']==role) for role in ['train','validation']]
    cases=[known_case(r) for r in records]
    normal=json.loads(Path(receipt['normalization']).read_text());q=grid()
    source=torch.load(protocol['source_checkpoint']['path'],weights_only=True,map_location='cuda')
    original=SpatialPINN(source['config'],normal['mean'],normal['std']).cuda()
    original.load_state_dict(source['state']);original.eval()
    out=WORK/'benchmark'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False);results=[]
    with torch.inference_mode():
        for config in protocol['candidates']:
            model=HistoryPINN(config,normal['mean'],normal['std']).cuda()
            model.load_source(source['state']);model.eval();checks=[]
            for record,known in zip(records,cases):
                torch.testing.assert_close(query(model,known,q,0.),known['raw'][:,:,:3],rtol=1e-5,atol=1e-6)
                for t in [.05,1.,11.]:
                    qt=q.clone();qt[:,2]=t
                    old=old_conditioning(original,known['raw'],known['geometry'],known['record']['reynolds'])
                    expected=project_known_constraints(original.grid_prediction(qt,old,known['raw']),known['fluid'])
                    torch.testing.assert_close(query(model,known,q,t),expected,rtol=2e-4,atol=2e-5)
                for _ in range(30):query(model,known,q,.05)
                torch.cuda.synchronize();times=[]
                for offset in range(1,TARGET_COUNT+1):
                    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                    a.record();value=query(model,known,q,offset*.05);b.record();b.synchronize();times.append(a.elapsed_time(b))
                if not torch.isfinite(value).all():raise ValueError('Nonfinite gate output')
                checks.append(dict(path=record['path'],query_gpu_ms={k:float(np.percentile(times,p)) for k,p in [('p50',50),('p95',95),('p99',99)]}))
            numerical=None
            if config['history_enabled']:
                model.history_projection.weight[:]=.01
                known=cases[0];context=conditioning(model,known);qt=q.clone();qt[:,2]=1.
                whole=model.grid_prediction(qt,context,known['raw'])
                y=torch.tensor([0,0,255,255,32,100,200],device='cuda')
                x=torch.tensor([0,1023,0,1023,100,500,800],device='cuda')
                sparse=model.sampled_prediction(qt[y*1024+x],context,known['raw'],y,x)
                error=float((sparse-whole[y,x]).abs().max())
                numerical=dict(nonzero_temporal_weight=.01,max_abs_sparse_grid_error=error,tolerance=.0005)
                if error>.0005:raise ValueError('History float32 sparse/grid mismatch')
            result=dict(name=config['name'],cases=checks,numerical=numerical,p99_pass=all(c['query_gpu_ms']['p99']<=5 for c in checks))
            results.append(result);save(out/(config['name']+'.json'),result)
            del model;torch.cuda.empty_cache()
    summary=dict(status='KNOWN_HISTORY_READINESS_EVALUATED',results=results,
        readiness_pass=all(r['p99_pass'] for r in results),prefix_frames=list(range(2,22)),
        target_frames=[22,241],initial_fields_gpu=True,
        includes_history_spatial_pooling_and_encoder=True,
        cached='Only input-independent coordinate interpolation matrices; no case-dependent features cached',
        excludes='CPUtransfer/rendering,30warmup,static coordinate stencil setup',
        source_checkpoint_sha256=digest(protocol['source_checkpoint']['path']))
    save(out/'summary.json',summary);print(json.dumps(summary),flush=True)
    if not summary['readiness_pass']:raise RuntimeError('History p99>5ms: stop before training')


if __name__=='__main__':run()
