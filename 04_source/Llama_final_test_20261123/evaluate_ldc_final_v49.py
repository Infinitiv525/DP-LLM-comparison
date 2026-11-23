"""Four-worker final TEST inference only; no learning and no configuration changes."""
import argparse
import os
import resource
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from ldc_final_v49 import (ROOT,WORK,FINAL,read,save,digest,verify,EvaluationInputs,
    load_model,predict_case,aggregate,claim,finish)


def run(mode):
    protocol=verify();rank=int(os.environ['SLURM_PROCID']);job=os.environ['SLURM_JOB_ID']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    if torch.__version__!='2.12.0+cu130' or np.__version__!='2.5.4':raise ValueError('Frozen evaluation runtime required')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name():
        raise ValueError('Exactly one assigned H200 required')
    prop=torch.cuda.get_device_properties(0)
    env=dict(job=job,rank=rank,gpu=prop.name,uuid=str(prop.uuid),torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,precision='float32',allow_tf32=False,cudnn_allow_tf32=True,
        protocol_sha256=digest(FINAL/'protocol.json'),source_freeze_sha256=digest(FINAL/'freeze.json'))
    models=read(FINAL/'models.json')['models']
    if mode=='preflight':
        store=EvaluationInputs(protocol,'validation');model=load_model(models[0],store.normalization)
        index=store.ids[0];row,_,_=predict_case(model,store,index,False)
        expected=next(r for r in read(Path(models[0]['training_summary']).parent/'selected_validation/summary.json')['cases'] if r['sample']==index)
        checks={k:bool(np.isclose(row[k],expected[k],rtol=1e-3,atol=1e-9))
                for k in ['mse_all','velocity_relative_l2','pressure_relative_l2','relative_l2_all']}
        if not all(checks.values()):raise ValueError('VAL correctness mismatch: '+str(checks))
        save(FINAL/'preflight'/f'rank_{rank}.json',dict(status='VALIDATION_CORRECTNESS_PASS',environment=env,
            model_id=models[0]['id'],sample=index,checks=checks,metrics={k:row[k] for k in checks},TEST_opened=False))
        print('VALIDATION_PREFLIGHT_PASS',rank,flush=True)
        return
    if mode!='test':raise ValueError('Unknown mode')
    if read(FINAL/'preflight/review.json')['status']!='FOUR_GPU_VALIDATION_PREFLIGHT_PASS':
        raise ValueError('All four validation checks must pass before TEST access')
    store=EvaluationInputs(protocol,'test')
    save(FINAL/'test_access'/f'rank_{rank}.json',dict(environment=env,unix=time.time(),
         role='Final sealed TEST only after frozen25model manifest and all four VALIDATION preflights',TEST_opened=True))
    save(FINAL/'environment'/f'rank_{rank}.json',env)
    while True:
        verify();item=claim(rank,job)
        if item is None:break
        out=FINAL/'evaluation/models'/item['id']
        try:
            out.mkdir(parents=True,exist_ok=False)
            start=time.monotonic();cpu=time.process_time()
            model=load_model(item,store.normalization);torch.cuda.reset_peak_memory_stats()
            rows=[];raw=[]
            fields=np.lib.format.open_memmap(out/'predicted_fields.npy',mode='w+',dtype=np.float32,shape=(200,512,512,3))
            for index in store.ids:
                row,times,prediction=predict_case(model,store,index,True)
                fields[len(rows)]=prediction
                rows.append(row);raw.append(times)
                if len(rows)%50==0:
                    save(out/'progress.json',dict(completed_cases=len(rows),total_cases=200,
                         last_case=index,elapsed_seconds=time.monotonic()-start,TEST_opened=True))
                    print('FINAL_MODEL_PROGRESS',item['id'],len(rows),'of200',flush=True)
            if [r['sample'] for r in rows]!=protocol['test_ids']:raise ValueError('Incomplete TEST coverage')
            if digest(item['checkpoint'])!=item['checkpoint_sha256']:raise ValueError('Weights changed during evaluation')
            fields.flush();del fields
            fields_sha256=digest(out/'predicted_fields.npy')
            save(out/'fields_manifest.json',dict(path='predicted_fields.npy',shape=[200,512,512,3],dtype='float32',axes=['case','y','x','channel'],channels=['u','v','p'],case_ids=protocol['test_ids'],sha256=fields_sha256,origin='Exactly the last measured native prediction also used to compute metrics; no additional query',model_checkpoint_sha256=item['checkpoint_sha256']))
            np.save(out/'timings_ms.npy',np.asarray(raw,dtype=np.float32),allow_pickle=False)
            summary=dict(status='FINAL_TEST_MODEL_COMPLETE',model=item,role='TEST',main_test_opened=True,
                cases=rows,means=aggregate(rows),test_cases=len(rows),
                worst_case_p99_ms=max(r['timing']['p99'] for r in rows),
                pass_5ms=all(r['timing']['p99']<=5 for r in rows),
                mean_momentum_rms=float(np.mean([r['physics']['momentum_rms'] for r in rows])),
                mean_divergence_rms=float(np.mean([r['physics']['divergence_rms'] for r in rows])),
                mean_boundary_rms=float(np.mean([r['boundary_mean'] for r in rows])),
                predicted_fields_sha256=fields_sha256,predicted_fields_shape=[200,512,512,3],
                timing_shape=[200,200],timings_sha256=digest(out/'timings_ms.npy'),
                elapsed_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                environment=env,no_weight_updates=True,no_TEST_selection=True)
            save(out/'summary.json',summary);finish(item['id'],rank,summary=out/'summary.json')
            print('FINAL_MODEL_COMPLETE',item['id'],summary['means'],flush=True)
            del model;torch.cuda.empty_cache()
        except Exception:
            error=traceback.format_exc();save(out/'failure.json',dict(error=error,model=item,TEST_opened=True))
            finish(item['id'],rank,error=error);raise
    save(FINAL/'workers'/f'rank_{rank}.json',dict(job=job,rank=rank,clean_exit=True,unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['preflight','test'],required=True)
    run(parser.parse_args().mode)
