"""GPU readiness gate before any training; fixed stencil weights may be cached."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from fpo_common_v21 import WORK,save,verify,digest
from fpo_spatial_context_v21 import SpatialPINN
from fpo_direct_curl_v15 import DirectPINN
from fpo_step_v5a import grid,conditioning
from audit_fpo_physics_v18 import known_case
from train_fpo_multicurl_v17 import query


def run():
    verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    protocol=json.loads((WORK/'protocol.json').read_text());receipt=json.loads((WORK/'prepared.json').read_text())
    records=[next(r for r in receipt['records'] if r['proposed_split']==role) for role in ['train','validation']]
    cases=[known_case(r)[0] for r in records]
    normal=json.loads(Path(receipt['normalization']).read_text());q=grid()
    source=torch.load(protocol['source_checkpoint']['path'],weights_only=True,map_location='cuda')
    original=DirectPINN(source['config'],normal['mean'],normal['std']).cuda();original.load_state_dict(source['state']);original.eval()
    out=WORK/'benchmark'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False);results=[]
    with torch.inference_mode():
        for config in protocol['candidates']:
            model=SpatialPINN(config,normal['mean'],normal['std']).cuda();model.load_source(source['state']);model.eval()
            checks=[]
            for record,known in zip(records,cases):
                torch.testing.assert_close(query(model,known,q,0.),known['raw'][:,:,:3],rtol=1e-5,atol=1e-6)
                for t in [.05,1.,11.9]:torch.testing.assert_close(query(model,known,q,t),query(original,known,q,t),rtol=1e-4,atol=1e-5)
                times=[]
                for _ in range(30):query(model,known,q,.05)
                torch.cuda.synchronize()
                for offset in range(1,239):
                    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                    a.record();value=query(model,known,q,offset*.05);b.record();b.synchronize();times.append(a.elapsed_time(b))
                if not torch.isfinite(value).all():raise ValueError('Nonfinite gate output')
                checks.append(dict(path=record['path'],query_gpu_ms={k:float(np.percentile(times,p)) for k,p in [('p50',50),('p95',95),('p99',99)]}))
            # Exercise nonzero spatial input weights as a numerical kernel check.
            numerical=None
            if config['local_context']:
                model.net[0].weight[:,model.original_input_features:]=.01
                known=cases[0];context=conditioning(model,known['raw'],known['geometry'],known['record']['reynolds'])
                qt=q.clone();qt[:,2]=1.;whole=model.grid_prediction(qt,context,known['raw'])
                y=torch.tensor([0,0,255,255,32,100,200],device='cuda');x=torch.tensor([0,1023,0,1023,100,500,800],device='cuda')
                sparse=model.sampled_prediction(qt[y*1024+x],context,known['raw'],y,x)
                error=float((sparse-whole[y,x]).abs().max())
                numerical=dict(nonzero_spatial_weight=.01,max_abs_sparse_grid_error=error,tolerance=.0005)
                if error>.0005:raise ValueError('Sparse/grid float32 mismatch')
            result=dict(name=config['name'],cases=checks,numerical=numerical,
                p99_pass=all(c['query_gpu_ms']['p99']<=5 for c in checks))
            results.append(result);save(out/(config['name']+'.json'),result)
            del model;torch.cuda.empty_cache()
    summary=dict(status='SPATIAL_CONTEXT_READINESS_EVALUATED',results=results,readiness_pass=all(r['p99_pass'] for r in results),
        initial_fields_gpu=True,includes_cnn_and_spatial_feature_computation=True,
        cached='Only input-independent regular-grid interpolation matrices; every query recomputes CNN and feature-map contractions',
        excludes='CPUtransfer/rendering,30warmup,one-time constantstencilsetup',source_checkpoint_sha256=digest(protocol['source_checkpoint']['path']))
    save(out/'summary.json',summary);print(json.dumps(summary),flush=True)
    if not summary['readiness_pass']:raise RuntimeError('p99>5ms: no smoke or main training authorized by this readiness gate')


if __name__=='__main__':run()
