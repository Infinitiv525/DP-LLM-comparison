"""Independent arithmetic/provenance audit of LLM power plan. No GPU."""
import hashlib,json,math,statistics,time
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
OUT=ROOT/'results/llm_power_20261124_v2'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
t=time.monotonic();manifest=read(OUT/'input_manifest.json')
for path,h in manifest['files'].items():assert sha(Path(path))==h,path
assert sha(ROOT/'test/llm_power_20261124_v1/power_analysis_v2.py')==manifest['source_sha256']
bundle=np.load(OUT/'analysis_inputs.npz',allow_pickle=False)
cube=bundle['errors'];methods=bundle['methods'].tolist();metrics=bundle['metrics'].tolist()
assert cube.shape==(7,5,5,200,3)
rawmodels=read(OUT/'audited_final_models.json')
for row in rawmodels:
 d=read(Path(row['summary_path']))
 i=methods.index(row['method']);r=row['search'];k=row['final_seed_index']
 x=np.array([[v[key] for key in metrics] for v in d['cases']])
 assert np.array_equal(x,cube[i,r,k])
vs=read(OUT/'variance_components_and_search_means.json');lookup={}
for v in vs:
 i=methods.index(v['method']);ti=methods.index('tpe');j=metrics.index(v['metric'])
 D=[[statistics.mean(cube[i,r,k,:,j].tolist())-statistics.mean(cube[ti,r,k,:,j].tolist()) for k in range(5)] for r in range(5)]
 row=[statistics.mean(z) for z in D];col=[statistics.mean([D[r][k] for r in range(5)]) for k in range(5)]
 vr=statistics.variance(row);vc=statistics.variance(col);mu=statistics.mean(row)
 e=sum((D[r][k]-row[r]-col[k]+mu)**2 for r in range(5) for k in range(5))/16
 for a,b in [(vr,v['row_variance']),(vc,v['column_variance']),(e,v['residual_interaction_mean_square']),
             (math.sqrt(vr+vc/5),v['fresh_block_planning_sd'])]:
  assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-24),(v['method'],v['metric'],a,b)
 upper=math.sqrt((vr+vc/5)*4/stats.chi2.ppf(.025,4))
 assert math.isclose(upper,v['upper95_per_contrast_planning_sd'],rel_tol=1e-10)
 lookup[(v['method'],v['metric'])]=v
rows=read(OUT/'sample_size_by_contrast.json')
for v in rows:
 n=v['required_search_pairs'];alpha=v['alpha_per_contrast'];effect=v['absolute_target_difference'];sd=v['assumed_sd']
 # All requested N here are small. A separate direct CDF path audits the planner's survival-function path.
 assert n<2000
 def power(nn):
  q=stats.t.ppf(1-alpha/2,nn-1);nc=effect*math.sqrt(nn)/sd
  return float(1-stats.nct.cdf(q,nn-1,nc)+stats.nct.cdf(-q,nn-1,nc))
 assert power(n)>=v['target_power']-1e-10,(v,power(n))
 assert n==2 or power(n-1)<v['target_power']+1e-10
 assert math.isclose(power(n),v['estimated_power_at_n'],rel_tol=1e-8,abs_tol=1e-10)
costs=read(OUT/'costs_by_search_method.json');trials=read(OUT/'audited_training_costs.json')
for method,v in costs.items():
 if method not in methods:continue
 group=[x for x in trials if x['method']==method]
 assert len(group)==65
 for run in range(5):
  g=[x for x in group if x['run']==run];u=v['per_search'][run]
  assert len(g)==13 and sum(x['kind']=='search' for x in g)==8
  for key in ['active_task_GPU_h','training_GPU_h','proposal_GPU_h']:
   assert math.isclose(sum(x[key] for x in g),u[key],rel_tol=1e-12)
des=read(OUT/'sample_size_designs.json')
for v in des:
 matches=[x for x in rows if all(x[k]==v[k] for k in ['design','target_relative_difference_vs_TPE','target_power','sd_scenario'])]
 assert len(matches)==v['family_size']
 assert max(x['required_search_pairs'] for x in matches)==v['balanced_search_blocks']
 for key in ['balanced_all_new','unequal_all_new_shared_TPE','exploratory_add_to_old_five_same_final_seed_panel']:
  if key not in v:continue
  price=v[key];counts=price['new_searches_by_method'];S=sum(counts.values())
  assert price['new_independent_searches_total']==S and price['total_PINN_trainings']==13*S
  assert price['candidate_trainings']==8*S and price['selected_PINN_refits']==5*S
  assert price['proposed_6h30_per_training_job_cap_plus_TEST_reserve_GPU_h']==85*S
  calc=sum(counts[m]*costs[m]['expected_reserved_GPU_h'] for m in counts)
  assert math.isclose(calc,price['expected_reserved_total_GPU_h'],rel_tol=1e-12)
  if key=='balanced_all_new':assert all(x==v['balanced_search_blocks'] for x in counts.values())
  if key=='exploratory_add_to_old_five_same_final_seed_panel':assert all(x==max(0,v['balanced_search_blocks']-5) for x in counts.values())
# A collapsed search-row panel must not be interpreted as independent final refits.
identical=[]
for method in ['qwen3_8b','phi4_mini','granite33_8b','Llama31_8B']:
 i=methods.index(method);spread=float(np.max(np.abs(cube[i]-cube[i,:1])))
 identical.append(dict(method=method,max_abs_difference_across_old_search_rows=spread))
summary=dict(status='INDEPENDENT_PROVENANCE_VARIANCE_POWER_AND_COST_AUDIT_PASS',
 input_files_verified=len(manifest['files']),raw_final_models_reproduced=len(rawmodels),
 training_records=len(trials),variance_contrasts=len(vs),sample_size_rows_verified=len(rows),
 design_rows_verified=len(des),shared_seed_panel_diagnostic=identical,
 elapsed_seconds=time.monotonic()-t,no_GPU=True,no_new_training=True)
save(OUT/'independent_verification.json',summary);print(json.dumps(summary),flush=True)
