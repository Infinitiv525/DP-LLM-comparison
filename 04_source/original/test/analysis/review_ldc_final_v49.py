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
    assert not state['attention_required'] and len(state['tasks'])==155
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
    assert len(metrics)==31000
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
    # Average the five fresh final seeds before uncertainty over the five
    # independent searches. TEST geometry clusters are resampled together.
    arrays={m:np.full((5,5,200),np.nan) for m in protocol['search_methods']}
    for m in models:
        if m['method']=='fixed':continue
        arrays[m['method']][m['search_run'],m['repeat']]=[r['mse_all'] for r in by_model[m['id']]['cases']]
    assert all(np.isfinite(a).all() for a in arrays.values())
    mean_seeds={m:a.mean(1) for m,a in arrays.items()}
    geometry=protocol['test_geometries'];unique=sorted(set(geometry))
    clusters=[np.where(np.asarray(geometry)==g)[0] for g in unique]
    rng=np.random.default_rng(protocol['analysis']['bootstrap_seed'])
    labels=protocol['llm_methods'];draws=np.empty((10000,4))
    for b in range(10000):
        repeats=rng.integers(0,5,5)
        indices=np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))])
        control=mean_seeds['tpe'][repeats][:,indices].mean()
        for j,label in enumerate(labels):draws[b,j]=mean_seeds[label][repeats][:,indices].mean()-control
    contrasts=[]
    control=float(mean_seeds['tpe'].mean())
    for j,label in enumerate(labels):
        effect=float(mean_seeds[label].mean()-control)
        low,high=np.quantile(draws[:,j],[.00625,.99375],method='linear')
        contrasts.append(dict(method=label,control='tpe',delta_test_mse=effect,
            relative_delta_percent=100*effect/control,interval_percent=98.75,
            interval=[float(low),float(high)],interval_below_zero=bool(high<0)))
    np.save(report/'bootstrap_deltas.npy',draws,allow_pickle=False)
    result=dict(status='FINAL_TEST_EVALUATION_COMPLETE',at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat(),
        models=155,case_model_pairs=31000,raw_timing_queries=6200000,worst_case_p99_ms=timing_max,
        methods=methods,primary_contrasts=contrasts,bootstrap=protocol['analysis'],
        source_freeze_sha256=digest(FINAL/'freeze.json'),model_manifest_sha256=digest(FINAL/'models.json'),
        TEST_opened=True,no_training=True,no_config_selection_on_TEST=True,
        uncertainty_note='Approximate paired crossed-bootstrap intervals; only5 independent searches. Final seeds averaged, not25 independent searches. Known-geometry flowcase split.')
    save(report/'summary.json',result)
    (report/'VYSLEDOK.txt').write_text('Finálny TEST:155sietí x200prípadov, bez učenia a výberu podľa TEST.\n'
       'Čísla po prípadoch:case_metrics.csv; metódy:method_metrics.csv; intervaly a zdroje:summary.json.\n'
       'Štatistické intervaly sú približné;5hľadaní, nie25nezávislých optimalizácií.\n',encoding='utf-8')
    verify();print('FINAL_TEST_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','completed'])
    globals()[parser.parse_args().mode]()
