"""Prospective LLM-search sample-size and measured-cost planning. CPU only."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import scipy
from scipy import stats,integrate

ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
METHODS=['qwen3_8b','ministral3_8b','phi4_mini','granite33_8b','Llama31_8B','tpe','random']
LLMS=METHODS[:5]
METRICS=['mse_all','velocity_relative_l2','pressure_relative_l2']
PROOFS={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 p=Path(p);PROOFS[str(p)]=sha(p)
 return json.loads(p.read_text())
def save(p,d):
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def moments(D):
 # D[r,k]: paired TEST means, search r and common final-training seed k.
 R,K=D.shape
 mu=D.mean();a=D.mean(1);b=D.mean(0)
 e=float(np.sum((D-a[:,None]-b[None,:]+mu)**2)/((R-1)*(K-1)))
 vr=float(a.var(ddof=1));vc=float(b.var(ddof=1))
 return dict(row_means=a.tolist(),column_means=b.tolist(),row_variance=vr,column_variance=vc,
  residual_interaction_mean_square=e,
  search_variance_raw=vr-e/K,final_seed_variance_raw=vc-e/R,
  fresh_block_MOM_variance=max(0.,vr-e/K)+max(0.,vc-e/R)/K+e/K,
  fresh_block_variance_planning_proxy=vr+vc/K,
  proxy_excess_in_expectation='interaction_variance/(R*K)',
  conditional_on_old_five_final_seeds_sd=math.sqrt(vr),
  fresh_block_planning_sd=math.sqrt(vr+vc/K),
  mean_difference=float(mu))
def collect(out):
 cubes=np.full((7,5,5,200,3),np.nan);modelrows=[];costrows=[];oldsummary=None
 for cohort,folder,N in [('original',ROOT/'results/ldc_final_test_20261108_v49',155),
                          ('llama',ROOT/'results/llama_final_test_20261123_v1',25)]:
  manifest=read(folder/'models.json');state=read(folder/'evaluation/state.json')
  assert len(manifest['models'])==N and not state['attention_required']
  protocol=read(folder/'protocol.json')
  ids=protocol['test_ids'];assert len(ids)==200
  if oldsummary is None:oldsummary=protocol
  else:assert ids==oldsummary['test_ids']
  for m in manifest['models']:
   p=folder/'evaluation/models'/m['id']/'summary.json';d=read(p)
   assert d['model']==m and d['test_cases']==200 and d['no_weight_updates'] and d['no_TEST_selection']
   assert state['tasks'][m['id']]['status']=='DONE' and state['tasks'][m['id']]['summary_sha256']==sha(p)
   assert [x['sample'] for x in d['cases']]==ids
   if m['method']=='fixed':continue
   i=METHODS.index(m['method']);r=m['search_run'];k=m['repeat']
   assert m['seed']==150001+k
   rows=np.asarray([[x[q] for q in METRICS] for x in d['cases']])
   assert rows.shape==(200,3) and np.isfinite(rows).all()
   assert np.isnan(cubes[i,r,k]).all();cubes[i,r,k]=rows
   for j,metric in enumerate(METRICS):
    assert np.isclose(rows[:,j].mean(),d['means'][metric],rtol=2e-12,atol=1e-14)
   modelrows.append(dict(method=m['method'],search=r,final_seed_index=k,id=m['id'],
    summary_path=str(p),summary_sha256=sha(p),TEST_elapsed_GPU_h=d['elapsed_seconds']/3600))
 assert np.isfinite(cubes).all() and len(modelrows)==175
 combined=read(ROOT/'results/llama_final_test_20261123_v1/combined_report/summary.json')
 for m in METHODS:
  d=next(v for v in combined['methods'] if v['method']==m)
  for j,q in enumerate(METRICS):assert np.isclose(cubes[METHODS.index(m),:,:,:,j].mean(),d[q],rtol=2e-12,atol=1e-14)
 assert len(set(oldsummary['test_geometries']))==85
 oldtasks=None
 for rel,expected in [('results/ldc_main_20261104_v33',395),('results/llama_extension_20261119_v1',65)]:
  state=read(ROOT/rel/'main/state.json');assert len(state['tasks'])==expected and not state['attention_required']
  if expected==395:oldtasks=state['tasks']
  for key,v in state['tasks'].items():
   t=v['task'];o=v['outcome']
   assert v['status']=='DONE' and o['status']=='COMPLETE'
   assert o['processes']['training']['exit_code']==0 and not o['processes']['training']['timed_out']
   p=Path(o['trial_summary_path']);d=read(p);assert sha(p)==o['trial_summary_sha256']
   assert d['epochs']==400 and d['train_cases']==640 and d['validation_cases']==160
   assert set(d['case_counts'].values())=={400} and d['main_test_opened'] is False
   assert abs(d['training_seconds']-o['history']['training_seconds'])<1e-6
   assert d['initial_weight_sha256']==o['initial_weight_sha256']
   procs=o['processes'];train=procs['training']['wall_seconds'];prop=procs.get('proposal',{}).get('wall_seconds',0.)
   window=v['finished_unix']-v['claimed_unix']
   assert window>=train+prop-1
   costrows.append(dict(id=key,method=t['method'],run=t.get('run'),kind=t['kind'],job=v['job'],
    training_GPU_h=train/3600,proposal_GPU_h=prop/3600,
    active_task_GPU_h=window/3600,summary_sha256=sha(p),summary_path=str(p)))
 assert len(costrows)==460
 # Actual old pooled reservations: allocate idle overhead only as a planning calibration.
 reserved=0.;queries=[]
 for job in sorted({v['job'] for v in oldtasks.values()}):
  p=ROOT/'results/ldc_main_20261104_v33/scheduler'/job/'summary.json';d=read(p)
  query=read(p.parent/'query.json')
  assert query['returncode']==0 and hashlib.sha256(query['stdout'].encode()).hexdigest()==d['raw_query_sha256']
  assert d['state']=='COMPLETED' and d['exit_code']=='0:0'
  assert d['allocated_gpu_seconds']==d['runtime_seconds']*d['gpu_count']
  reserved+=d['allocated_gpu_seconds']/3600;queries.append(d)
 oldactive=sum(x['active_task_GPU_h'] for x in costrows if x['method']!='Llama31_8B')
 assert reserved>=oldactive
 idle_factor=reserved/oldactive
 test_ratios={}
 for tag,folder in [('original',ROOT/'results/ldc_final_test_20261108_v49'),('llama',ROOT/'results/llama_final_test_20261123_v1')]:
  d=read(folder/'accounting_closeout.json')
  if tag=='original':reserved_test=d['accounting']['allocated_gpu_seconds']/3600
  else:
   parts=list(map(int,d['controller_fields']['RunTime'].split(':')))
   reserved_test=(parts[0]*3600+parts[1]*60+parts[2])*4/3600
  all_models=read(folder/'models.json')['models']
  active_test=sum(read(folder/'evaluation/models'/m['id']/'summary.json')['elapsed_seconds']/3600 for m in all_models)
  test_ratios[tag]=dict(reserved_GPU_h=reserved_test,active_model_GPU_h=active_test,
   scheduler_overhead_factor=max(1.,reserved_test/active_test))
 costs={}
 for method in METHODS:
  trials=[x for x in costrows if x['method']==method]
  assert len(trials)==65
  units=[]
  for run in range(5):
   group=[x for x in trials if x['run']==run];assert len(group)==13
   assert sum(x['kind']=='search' for x in group)==8 and sum(x['kind']=='retrain' for x in group)==5
   tests=[x for x in modelrows if x['method']==method and x['search']==run];assert len(tests)==5
   active=sum(x['active_task_GPU_h'] for x in group);test=sum(x['TEST_elapsed_GPU_h'] for x in tests)
   units.append(dict(run=run,training_GPU_h=sum(x['training_GPU_h'] for x in group),
    proposal_GPU_h=sum(x['proposal_GPU_h'] for x in group),active_task_GPU_h=active,active_TEST_GPU_h=test))
  tag='llama' if method=='Llama31_8B' else 'original'
  # Llama has individually queued jobs, so old pool factor is an overhead scenario, not its billing.
  meanactive=float(np.mean([x['active_task_GPU_h'] for x in units]))
  meantest=float(np.mean([x['active_TEST_GPU_h'] for x in units]))
  costs[method]=dict(per_search=units,mean_active_search_plus_5_refits_GPU_h=meanactive,
   mean_proposal_only_GPU_h=float(np.mean([x['proposal_GPU_h'] for x in units])),
   mean_active_TEST_5_models_GPU_h=meantest,
   expected_reserved_GPU_h=(meanactive*idle_factor+meantest*test_ratios[tag]['scheduler_overhead_factor']),
   expected_cost_model='Observed active windows times pooled scheduling overhead, plus measured TEST overhead. Estimate, not final portal billing.',
   proposed_training_jobs_cap_GPU_h=13*6.5,proposed_TEST_reserve_GPU_h=.5)
 np.savez_compressed(out/'analysis_inputs.npz',errors=cubes,methods=np.asarray(METHODS),metrics=np.asarray(METRICS),
  case_ids=np.asarray(ids),geometry=np.asarray(oldsummary['test_geometries']))
 save(out/'audited_training_costs.json',costrows);save(out/'audited_final_models.json',modelrows)
 save(out/'costs_by_search_method.json',costs)
 save(out/'observed_reservation_calibration.json',dict(old395_reserved_GPU_h=reserved,old395_active_task_GPU_h=oldactive,
  old_pool_overhead_factor=idle_factor,old6_jobs=queries,TEST=test_ratios,
  note='No direct per-method allocation billing in pooled jobs; active GPU windows assigned by task and idle calibration shared. New individual jobs avoid old idle pool tails.'))
 return cubes,costs,np.asarray(oldsummary['test_geometries'])

_nodes,_weights=np.polynomial.legendre.leggauss(192)
_u=(_nodes+1)/2;_weights=_weights/2;_bounds={}
def power(n,delta,sd,alpha):
 if sd<=0:return 1. if delta else alpha
 c=stats.t.isf(alpha/2,n-1);lam=abs(delta)*math.sqrt(n)/sd
 if n<2000:
  v=float(stats.nct.sf(c,n-1,lam)+stats.nct.cdf(-c,n-1,lam))
  if np.isfinite(v):return float(np.clip(v,0,1))
 key=(n,alpha)
 if key not in _bounds:_bounds[key]=c*np.sqrt(stats.chi2.ppf(_u,n-1)/(n-1))
 b=_bounds[key]
 return float(np.sum(_weights*(stats.norm.sf(b-lam)+stats.norm.cdf(-b-lam))))
def required(delta,sd,alpha,target):
 lo,hi=2,1000000
 if power(hi,delta,sd,alpha)<target:return None
 while lo<hi:
  mid=(lo+hi)//2
  if power(mid,delta,sd,alpha)>=target:hi=mid
  else:lo=mid+1
 assert power(lo,delta,sd,alpha)>=target-1e-12 and (lo==2 or power(lo-1,delta,sd,alpha)<target+1e-12)
 return lo
def priced(counts,costs):
 searches=sum(counts.values());models=13*searches
 return dict(new_searches_by_method=counts,new_independent_searches_total=searches,
  candidate_trainings=8*searches,selected_PINN_refits=5*searches,total_PINN_trainings=models,
  expected_active_search_and_refits_GPU_h=sum(counts[m]*costs[m]['mean_active_search_plus_5_refits_GPU_h'] for m in counts),
  expected_proposal_GPU_h_included=sum(counts[m]*costs[m]['mean_proposal_only_GPU_h'] for m in counts),
  expected_active_TEST_GPU_h=sum(counts[m]*costs[m]['mean_active_TEST_5_models_GPU_h'] for m in counts),
  expected_reserved_total_GPU_h=sum(counts[m]*costs[m]['expected_reserved_GPU_h'] for m in counts),
  proposed_6h30_per_training_job_cap_plus_TEST_reserve_GPU_h=85.*searches,
  cap_is_proposal_not_enqueued=True)
def geometry_components(D,geometry):
 # Diagnostic only: old final-seed panel fixed; search x geometry model.
 groups=sorted(set(geometry.tolist()));w=np.array([(geometry==g).sum()/200 for g in groups])
 M=np.stack([D[:,geometry==g].mean(1) for g in groups],1);S,G=M.shape
 mu=M.mean();a=M.mean(1);b=M.mean(0)
 e=float(np.sum((M-a[:,None]-b[None,:]+mu)**2)/((S-1)*(G-1)))
 sv=max(0.,float(G*np.sum((a-mu)**2)/(S-1)-e)/G)
 gv=max(0.,float(S*np.sum((b-mu)**2)/(G-1)-e)/S)
 return dict(geometry_variance_floor=gv*float((w*w).sum()),search_reducible=sv+e*float((w*w).sum()),
  geometry_SE_floor=math.sqrt(gv*float((w*w).sum())),diagnostic_only=True,
  limitations='Method-of-moments crossed approximation, old final-seed panel held fixed, 85 known geometries; no guarantee for new geometries.')
def main(out):
 started=time.monotonic();assert not out.exists();out.mkdir(parents=True)
 c,costs,geometry=collect(out);means=c.mean((1,2,3));m=c.mean(3);ti=METHODS.index('tpe')
 factor=math.sqrt(4/stats.chi2.ppf(.025,4))
 designlist=[dict(key='five_LLM_vs_TPE_MSE',llms=LLMS,metrics=[0],family=5),
  dict(key='five_LLM_vs_TPE_three_metrics',llms=LLMS,metrics=[0,1,2],family=15),
  dict(key='original_four_vs_TPE_MSE',llms=LLMS[:4],metrics=[0],family=4),
  dict(key='Ministral_vs_TPE_MSE_exploratory_targeted',llms=['ministral3_8b'],metrics=[0],family=1)]
 variances={};geometryrows=[]
 for method in LLMS:
  i=METHODS.index(method)
  for j,metric in enumerate(METRICS):
   d=m[i,:,:,j]-m[ti,:,:,j];mmt=moments(d)
   mmt.update(method=method,control='tpe',metric=metric,
    TPE_mean=float(means[ti,j]),method_mean=float(means[i,j]),
    relative_observed_difference_vs_TPE=float((means[i,j]-means[ti,j])/means[ti,j]),
    upper95_per_contrast_planning_sd=mmt['fresh_block_planning_sd']*factor,
    upper95_bound_definition='Bonferroni upper97.5% normal variance bounds for row and column means; joint>=95% per contrast, not across contrasts; depends on Gaussian crossed model.',
    variance_2component_upper_multiplier=factor)
   variances[(method,j)]=mmt
   geo=geometry_components((c[i,:,:,:,j]-c[ti,:,:,:,j]).mean(1),geometry)
   geometryrows.append(dict(method=method,metric=metric,**geo))
 contrasts=[];designs=[]
 for des in designlist:
  alpha=.05/des['family']
  for rel in [.05,.10,.20]:
   for target in [.8,.9]:
    for mode in ['fresh_independent_blocks_point_proxy','fresh_independent_blocks_upper95_SD','old_final_seed_panel_conditional']:
     rows=[]
     for method in des['llms']:
      for j in des['metrics']:
       v=variances[(method,j)]
       sd=v['fresh_block_planning_sd'] if mode=='fresh_independent_blocks_point_proxy' else v['upper95_per_contrast_planning_sd'] if mode=='fresh_independent_blocks_upper95_SD' else v['conditional_on_old_five_final_seeds_sd']
       delta=rel*means[ti,j];n=required(delta,sd,alpha,target)
       row=dict(design=des['key'],method=method,metric=METRICS[j],family_size=des['family'],alpha_per_contrast=alpha,
        target_relative_difference_vs_TPE=rel,absolute_target_difference=float(delta),target_power=target,sd_scenario=mode,
        assumed_sd=sd,required_search_pairs=n,estimated_power_at_n=power(n,delta,sd,alpha) if n else None,
        power_at_n_minus_1=power(n-1,delta,sd,alpha) if n and n>2 else None)
       contrasts.append(row);rows.append(row)
     maxn=max(x['required_search_pairs'] for x in rows)
     perllm={method:max(x['required_search_pairs'] for x in rows if x['method']==method) for method in des['llms']}
     counts={**perllm,'tpe':maxn}
     balanced={method:maxn for method in des['llms']+['tpe']}
     item=dict(design=des['key'],family_size=des['family'],target_relative_difference_vs_TPE=rel,target_power=target,
      sd_scenario=mode,balanced_search_blocks=maxn,required_searches_each_LLM=perllm,
      limiting_contrasts=[dict(method=x['method'],metric=x['metric']) for x in rows if x['required_search_pairs']==maxn],
      minimum_exact_two_sided_signflip_resolution_n=math.ceil(math.log2(2/alpha)))
     if mode=='old_final_seed_panel_conditional':
      item['exploratory_add_to_old_five_same_final_seed_panel']=priced({method:max(0,maxn-5) for method in balanced},costs)
      item['scope']='Power conditional on the exact old five final seeds and fixed TEST cases. Existing+new analysis is exploratory; not generalization over fresh training seeds.'
     else:
      item['balanced_all_new']=priced(balanced,costs);item['unequal_all_new_shared_TPE']=priced(counts,costs)
      item['scope']='All NEW independent search blocks. Five fresh refit seeds per block, paired across methods but not reused across blocks. Old five not pooled into primary confirmation.'
     designs.append(item)
 save(out/'variance_components_and_search_means.json',list(variances.values()))
 save(out/'sample_size_by_contrast.json',contrasts);save(out/'sample_size_designs.json',designs)
 save(out/'geometry_diagnostic.json',geometryrows)
 # Explicit finite-benchmark vs geometry-generalization diagnostic for primary MSE.
 gd=[]
 for rel in [.05,.10,.20]:
  for v in geometryrows:
   if v['metric']!='mse_all':continue
   z=stats.norm.isf(.01/2);delta=rel*means[ti,0];se=v['geometry_SE_floor']
   ceiling=1. if se==0 else float(stats.norm.sf(z-delta/se)+stats.norm.cdf(-z-delta/se))
   gd.append(dict(method=v['method'],relative_target=rel,geometry_model_infinite_search_power_ceiling=ceiling,
    diagnostic_not_exact=True))
 save(out/'geometry_power_ceiling_diagnostic.json',gd)
 # Independently integrate the noncentral t defining normal/chi-square mixture.
 checks=[]
 unique=set()
 for row in contrasts:
  n=row['required_search_pairs'];alpha=row['alpha_per_contrast'];d=row['absolute_target_difference']/row['assumed_sd']
  key=(n,round(d,10),alpha)
  if key in unique:continue
  unique.add(key)
  for nn in [n]+([n-1] if n>2 else []):
   critical=stats.t.isf(alpha/2,nn-1);lam=d*math.sqrt(nn)
   def f(u):
    b=critical*math.sqrt(stats.chi2.ppf(u,nn-1)/(nn-1))
    return stats.norm.sf(b-lam)+stats.norm.cdf(-b-lam)
   q,err=integrate.quad(f,0,1,epsabs=2e-10,epsrel=2e-9,limit=150)
   p=power(nn,d,1.,alpha);assert abs(q-p)<1e-7,(nn,d,q,p)
  checks.append(dict(required_n=n,verified_n=nn,d=d,alpha=alpha,independent_power=q,computed_power=p,absolute_error=abs(q-p)))
 # Monte Carlo verifies conditional power only, not the true unobserved future variance.
 rng=np.random.default_rng(241124);mc=[]
 for rel in [.05,.10,.20]:
  row=next(x for x in contrasts if x['design']=='five_LLM_vs_TPE_MSE' and x['method']=='qwen3_8b'
   and x['target_power']==.9 and x['target_relative_difference_vs_TPE']==rel
   and x['sd_scenario']=='fresh_independent_blocks_point_proxy')
  n=row['required_search_pairs'];d=row['absolute_target_difference']/row['assumed_sd'];crit=stats.t.isf(.005,n-1)
  draws=rng.normal(size=(100000,n))+d;t=draws.mean(1)/(draws.std(1,ddof=1)/math.sqrt(n))
  ph=float((abs(t)>crit).mean());p=power(n,d,1.,.01);se=math.sqrt(p*(1-p)/100000)
  assert abs(ph-p)<6*se
  mc.append(dict(n=n,relative_effect=rel,empirical_power=ph,model_power=p,MC_SE=se))
 # Diagnose calibration under non-Gaussian skew, not used to adjust the answer.
 nulls=[]
 for shape in ['normal','standardized_t3','centered_exponential']:
  n=15
  if shape=='normal':x=rng.normal(size=(100000,n))
  elif shape=='standardized_t3':x=rng.standard_t(3,size=(100000,n))/math.sqrt(3)
  else:x=rng.exponential(size=(100000,n))-1
  tt=x.mean(1)/(x.std(1,ddof=1)/math.sqrt(n))
  nulls.append(dict(noise=shape,N=n,alpha=.01,rejection_rate=float((abs(tt)>stats.t.isf(.005,n-1)).mean())))
 save(out/'numerical_verification.json',dict(status='PASS',independent_integrations=checks,Monte_Carlo=mc,
  null_diagnostics=nulls,normal_assumption_required=True))
 save(out/'input_manifest.json',dict(files=PROOFS,source_sha256=sha(Path(__file__)),
  unchanged_original_results=True,weights_or_fields_loaded=False))
 summary=dict(status='LLM_PROSPECTIVE_POWER_AND_GPU_COST_ANALYSIS_COMPLETE',methods=METHODS,
  primary_family='five_LLM_vs_TPE_MSE',independent_pilot_search_blocks=5,final_refits_per_selected_configuration=5,
  audited_main_training_records=460,audited_search_method_final_models=175,fixed_final_models_excluded_from_power=5,
  primary_relative_effects_vs_TPE=[.05,.1,.2],target_power=[.8,.9],design_scenarios=len(designs),
  numeric_contrast_rows=len(contrasts),new_GPU_jobs_launched=False,GPU_used_for_analysis=False,
  variance_upper95_multiplier=factor,environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
  elapsed_seconds=time.monotonic()-started,finished_at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat(),
  caveats=['Prospective planning for hypothetical 5/10/20% effects, never observed post-hoc power.',
   'Only five search blocks and five shared final seeds: variance and required N uncertain.',
   'Four LLM families repeatedly selected the common fixed candidate; future changes in selection behaviour are not identified by these data.',
   'Gaussian crossed random effects assumed for variance planning; independent paired t power assumes Gaussian new block means.',
   '95% variance bounds are per contrast only, not simultaneous assurance for five or fifteen contrasts.',
   '90% target is marginal power per contrast, not 90% probability all comparisons reject.',
   'Known fixed 200-case benchmark. More searches do not add unseen geometries.',
   'No power guarantee that LLM is superior. More runs can confirm it is worse.',
   'New independent confirmation does not pool the five old searches; N-5 applies only to explicitly conditional exploratory extension.',
   'Costs are measured-runtime projections, not quotation or guaranteed final portal billing.'])
 save(out/'summary.json',summary)
 save(out/'OUTPUT_MANIFEST.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()})
 print(json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
