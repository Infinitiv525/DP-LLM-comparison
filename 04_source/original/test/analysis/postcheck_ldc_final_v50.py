"""CPU closeout of completed frozen TEST outputs; no fitting or new predictions."""
from collections import Counter,defaultdict
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import numpy as np
import torch

ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
sys.path.insert(0,str(ROOT/'test/llm'))
from ldc_main_common_v33 import WORK,read,save,digest,fingerprint,verify
from ldc_main_scheduler_v33 import choose_selection

FINAL=ROOT/'results/ldc_final_test_20261108_v49'
OUT=ROOT/'results/ldc_final_closeout_20261108_v50'


def main():
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.is_available():
        raise RuntimeError('Allocated CPU-only closeout required')
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    checks={};hashes={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise ValueError('Closeout check failed: '+name)
    def hashed(path):
        value=digest(path);hashes[str(path)]=value;return value
    verify();freeze=read(FINAL/'freeze.json')
    for path,expected in freeze['files'].items():check('frozen:'+path,hashed(path)==expected)
    for path,expected in freeze['input_fingerprints'].items():check('input:'+path,fingerprint(path)==expected)
    protocol=read(FINAL/'protocol.json');manifest=read(FINAL/'models.json');models=manifest['models']
    main_state=read(WORK/'main/state.json');state=read(FINAL/'evaluation/state.json')
    check('main_state_unchanged',hashed(WORK/'main/state.json')==manifest['main_state_sha256'])
    check('395_completed',len(main_state['tasks'])==395 and all(r['status']=='DONE' for r in main_state['tasks'].values()))
    check('155_completed',len(models)==len(state['tasks'])==155 and all(r['status']=='DONE' for r in state['tasks'].values()) and not state['attention_required'])
    check('TEST_once_no_fitting',read(FINAL/'report/summary.json')['no_training'] and protocol['no_selection_on_TEST'])
    model_rows=[];by_method=defaultdict(list);cases_by_model={};all_times=[]
    caseids=protocol['test_ids'];base_config=next(m['config'] for m in models if m['method']=='fixed').copy();base_config.pop('seed')
    fixed_by_seed={};parameter_groups=defaultdict(list)
    for model in models:
        identifier=model['id'];folder=FINAL/'evaluation/models'/identifier;s=read(folder/'summary.json')
        check(identifier+':result_hash',hashed(folder/'summary.json')==state['tasks'][identifier]['summary_sha256'])
        check(identifier+':model_identity',s['model']==model and s['test_cases']==200 and s['role']=='TEST')
        check(identifier+':no_fitting',s['no_weight_updates'] and s['no_TEST_selection'])
        check(identifier+':case_coverage',[r['sample'] for r in s['cases']]==caseids and len(set(caseids))==200)
        check(identifier+':checkpoint_hash',hashed(model['checkpoint'])==model['checkpoint_sha256'])
        check(identifier+':training_hash',hashed(model['training_summary'])==model['training_summary_sha256'])
        cp=torch.load(model['checkpoint'],map_location='cpu',weights_only=True)
        check(identifier+':checkpoint_configuration',cp['config']==model['config'] and cp['normalization']==manifest['normalization'])
        if model['kind']=='retrain':
            check(identifier+':VAL_only_selection',choose_selection(main_state,main_state['tasks'][identifier]['task'])==model['selected_by_VAL'])
        raw=np.load(folder/'timings_ms.npy',allow_pickle=False)
        check(identifier+':raw_times',raw.shape==(200,200) and np.isfinite(raw).all() and (raw>0).all() and hashed(folder/'timings_ms.npy')==s['timings_sha256'])
        all_times.append(raw)
        for percentile in [50,95,99]:
            check(identifier+':p'+str(percentile),np.allclose(np.percentile(raw,percentile,axis=1),[r['timing']['p'+str(percentile)] for r in s['cases']],rtol=1e-7,atol=1e-8))
        check(identifier+':raw_maximum',np.array_equal(raw.max(1),[r['timing']['maximum_ms'] for r in s['cases']]))
        check(identifier+':raw_over_limit',np.array_equal((raw>5).sum(1),[r['timing']['above_5ms'] for r in s['cases']]))
        check(identifier+':speed_flag',s['pass_5ms']==bool((np.percentile(raw,99,axis=1)<=5).all()))
        keys=['mse_all','velocity_relative_l2','pressure_relative_l2','relative_l2_all']
        means={k:float(np.mean([r[k] for r in s['cases']])) for k in keys}
        check(identifier+':mean_errors',all(np.isclose(means[k],s['means'][k],rtol=1e-13,atol=1e-18) for k in keys))
        config=dict(model['config']);config.pop('seed')
        row=dict(model=identifier,method=model['method'],search_run=model['search_run'],seed=model['seed'],
            selected_source=None if not model['selected_by_VAL'] else model['selected_by_VAL']['source_task'],
            config_is_fixed=config==base_config,config_json=json.dumps(config,sort_keys=True),
            **means,mean_momentum_rms=s['mean_momentum_rms'],mean_boundary_rms=s['mean_boundary_rms'],
            worst_case_p99_ms=s['worst_case_p99_ms'],pass_5ms=s['pass_5ms'])
        model_rows.append(row);by_method[model['method']].append(s);cases_by_model[identifier]=s['cases']
        # Equal configurations and matched seeds can yield identical fitted states;
        # keep every independently executed training in provenance, not as IID searches.
        import hashlib
        h=hashlib.sha256()
        for name,value in sorted(cp['state'].items()):h.update(name.encode());h.update(value.contiguous().numpy().tobytes())
        parameter_groups[h.hexdigest()].append(identifier)
    report=read(FINAL/'report/summary.json');hashed(FINAL/'report/summary.json')
    with (FINAL/'report/case_metrics.csv').open(newline='') as stream:flat=list(csv.DictReader(stream))
    check('31000_unique_csv_rows',len(flat)==len({(r['model'],int(r['sample'])) for r in flat})==31000)
    for identifier in cases_by_model:
        selected=[r for r in flat if r['model']==identifier]
        check(identifier+':CSV_matches_JSON',len(selected)==200 and all(
            int(a['sample'])==b['sample'] and all(float(a[k])==b[k] for k in ['mse_all','velocity_relative_l2','pressure_relative_l2','relative_l2_all'])
            for a,b in zip(selected,cases_by_model[identifier])))
    for row in report['methods']:
        group=by_method[row['method']]
        check(row['method']+':report_aggregate',len(group)==row['models'] and all(np.isclose(row[k],np.mean([s['means'][k] for s in group]),rtol=1e-13,atol=1e-18)
              for k in ['mse_all','velocity_relative_l2','pressure_relative_l2','relative_l2_all']))
    draws=np.load(FINAL/'report/bootstrap_deltas.npy',allow_pickle=False)
    check('bootstrap_shape',draws.shape==(10000,4) and np.isfinite(draws).all())
    tpe=next(r['mse_all'] for r in report['methods'] if r['method']=='tpe')
    for j,contrast in enumerate(report['primary_contrasts']):
        mse=next(r['mse_all'] for r in report['methods'] if r['method']==contrast['method'])
        check('contrast:'+contrast['method'],np.isclose(contrast['delta_test_mse'],mse-tpe,rtol=1e-12,atol=1e-16)
             and np.allclose(np.quantile(draws[:,j],[.00625,.99375]),contrast['interval'],rtol=1e-13,atol=1e-18))
    selection=[]
    for method in protocol['search_methods']:
        selected=[r for r in model_rows if r['method']==method and r['seed']==150001]
        selection.append(dict(method=method,independent_searches=5,
            distinct_selected_configs=len({r['config_json'] for r in selected}),
            searches_selecting_fixed_config=sum(r['config_is_fixed'] for r in selected)))
    times=np.concatenate(all_times,axis=0)
    timing=dict(queries=int(times.size),case_model_pairs=int(times.shape[0]),
        pooled_quantiles_ms={str(p):float(np.percentile(times,p)) for p in [50,95,99]},
        maximum_single_query_ms=float(times.max()),single_queries_above_5ms=int((times>5).sum()),
        worst_case_p99_ms=float(np.percentile(times,99,axis=1).max()),
        failed_case_model_p99=int((np.percentile(times,99,axis=1)>5).sum()))
    check('raw_query_total',timing['queries']==6200000)
    check('global_p99_matches',timing['worst_case_p99_ms']==report['worst_case_p99_ms'])
    account=read(FINAL/'accounting_closeout.json')
    check('GPU_completed_and_accounted',account['status']=='FINAL_TEST_ACCOUNTED' and account['accounting']['exit_code']=='0:0' and account['report_ready'])
    for job in ['145677','145678']:
        a=read(WORK/'scheduler'/job/'summary.json');check('job:'+job,a['state']=='COMPLETED' and a['exit_code']=='0:0');hashed(WORK/'scheduler'/job/'summary.json')
    for source in [FINAL/'evaluation/state.json',FINAL/'report/case_metrics.csv',FINAL/'report/method_metrics.csv',FINAL/'report/bootstrap_deltas.npy']:
        hashed(source)
    with (OUT/'model_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(model_rows[0]));writer.writeheader();writer.writerows(model_rows)
    save(OUT/'selected_configuration_summary.json',selection)
    save(OUT/'identical_parameter_state_groups.json',dict(groups=list(parameter_groups.values()),
         explanation='Independent executed trainings may have identical tensor states when selected configurations and training seeds agree. Final seeds are averaged before five-search inference.'))
    save(OUT/'timing_summary.json',timing);save(OUT/'checks.json',checks);save(OUT/'artifact_hashes.json',hashes)
    result=dict(status='FINAL_OUTPUTS_AUDITED',at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat(),
        checks=len(checks),problems=0,models=155,test_cases=200,case_model_pairs=31000,
        timing=timing,selected_configs=selection,distinct_parameter_states=len(parameter_groups),
        primary_metric='Mean unweighted all-cell TEST MSE',primary_best_method=min(report['methods'],key=lambda r:r['mse_all'])['method'],
        LLM_beats_TPE_primary=False,all_four_primary_intervals_above_zero=all(r['interval'][0]>0 for r in report['primary_contrasts']),
        no_new_training=True,no_new_GPU_queries=True,source_sha256=digest(__file__),report_sha256=digest(FINAL/'report/summary.json'),
        limitations=['Five independent searches; bootstrap coverage approximate.','Known-geometry flowcase split.','Steady fields, not temporal rollout.','Speed is GPU query with inputs resident; excludes transfer/rendering.','Physics residuals remain nonzero.'])
    save(OUT/'summary.json',result);verify();print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
