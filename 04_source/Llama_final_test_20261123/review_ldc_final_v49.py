"""Final inference integrity checks and predeclared numerical summaries."""
import argparse
from collections import Counter
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from ldc_final_v49 import FINAL,read,save,digest,verify


def preflight():
    protocol=verify();rows=[read(FINAL/'preflight'/f'rank_{r}.json') for r in range(4)]
    assert all(r['status']=='VALIDATION_CORRECTNESS_PASS' and not r['TEST_opened'] and all(r['checks'].values()) for r in rows)
    assert len({r['environment']['uuid'] for r in rows})==4
    save(FINAL/'preflight/review.json',dict(status='FOUR_GPU_VALIDATION_PREFLIGHT_PASS',
         rows=rows,TEST_opened=False,protocol_sha256=digest(FINAL/'protocol.json')))
    print('ALL_FOUR_GPU_CORRECTNESS_PREFLIGHTS_PASS',flush=True)


def completed():
    protocol=verify();models=read(FINAL/'models.json')['models'];state=read(FINAL/'evaluation/state.json')
    assert not state['attention_required'] and len(state['tasks'])==25
    assert all(r['status']=='DONE' for r in state['tasks'].values())
    metrics=[];by_model={};timing_max=0.
    for model in models:
        folder=FINAL/'evaluation/models'/model['id'];s=read(folder/'summary.json')
        assert digest(folder/'summary.json')==state['tasks'][model['id']]['summary_sha256']
        assert s['status']=='FINAL_TEST_MODEL_COMPLETE' and s['model']==model and s['test_cases']==200
        assert [r['sample'] for r in s['cases']]==protocol['test_ids']
        assert s['no_weight_updates'] and s['no_TEST_selection']
        assert digest(model['checkpoint'])==model['checkpoint_sha256']
        raw=np.load(folder/'timings_ms.npy',allow_pickle=False)
        assert raw.shape==(200,200) and np.isfinite(raw).all() and (raw>0).all()
        assert digest(folder/'timings_ms.npy')==s['timings_sha256']
        for j,p in enumerate([50,95,99]):
            assert np.allclose(np.percentile(raw,p,axis=1),[r['timing'][f'p{p}'] for r in s['cases']],rtol=1e-7,atol=1e-8)
        assert s['pass_5ms']==all(r['timing']['p99']<=5 for r in s['cases'])
        timing_max=max(timing_max,s['worst_case_p99_ms']);by_model[model['id']]=s
        for case in s['cases']:
            metrics.append(dict(model=model['id'],method=model['method'],search_run=model['search_run'],seed=model['seed'],
                sample=case['sample'],reynolds=case['reynolds'],geometry=case['geometry'],
                **{k:case[k] for k in ['mse_all','velocity_relative_l2','pressure_relative_l2','relative_l2_all']},
                momentum_rms=case['physics']['momentum_rms'],divergence_rms=case['physics']['divergence_rms'],
                reference_momentum_rms=case['reference_physics']['momentum_rms'],
                boundary_rms=case['boundary_mean'],p50_ms=case['timing']['p50'],p95_ms=case['timing']['p95'],
                p99_ms=case['timing']['p99'],maximum_ms=case['timing']['maximum_ms'],
                **{side+'_bc_rms':case['boundary'][side]['velocity_error_rms'] for side in ['bottom','top','left','right','obstacle']}))
    assert len(metrics)==5000
    report=FINAL/'report';report.mkdir(exist_ok=False)
    with (report/'case_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    methods=[]
    for method in protocol['methods']:
        selected=[s for s in by_model.values() if s['model']['method']==method]
        row=dict(method=method,models=len(selected),test_cases_per_model=200,
             **{k:float(np.mean([s['means'][k] for s in selected])) for k in ['mse_all','velocity_relative_l2','pressure_relative_l2','relative_l2_all']},
             mean_momentum_rms=float(np.mean([s['mean_momentum_rms'] for s in selected])),
             mean_divergence_rms=float(np.mean([s['mean_divergence_rms'] for s in selected])),
             mean_boundary_rms=float(np.mean([s['mean_boundary_rms'] for s in selected])),
             worst_p99_ms=max(s['worst_case_p99_ms'] for s in selected),
             speed_pass_models=sum(s['pass_5ms'] for s in selected),
             evaluation_seconds=sum(s['elapsed_seconds'] for s in selected))
        assert row['models']==(5 if method=='fixed' else 25)
        methods.append(row)
    with (report/'method_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(methods[0]));writer.writeheader();writer.writerows(methods)
    result=dict(status='LLAMA_FINAL_TEST_EVALUATION_COMPLETE',models=25,case_model_pairs=5000,raw_timing_queries=1000000,worst_case_p99_ms=timing_max,methods=methods,no_training=True,no_config_selection_on_TEST=True,original_primary_results_unchanged=True,source_freeze_sha256=digest(FINAL/'freeze.json'))
    for model in models:
        folder=FINAL/'evaluation/models'/model['id'];s=by_model[model['id']];assert digest(folder/'predicted_fields.npy')==s['predicted_fields_sha256'];fields=np.load(folder/'predicted_fields.npy',mmap_mode='r',allow_pickle=False);assert fields.shape==(200,512,512,3) and fields.dtype==np.float32
    save(report/'summary.json',result)
    from compare_with_original import main as compare
    compare();verify();print('LLAMA_TEST_AND_COMBINED_COMPARISON_COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','completed']);globals()[parser.parse_args().mode]()
