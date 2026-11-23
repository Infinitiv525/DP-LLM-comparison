"""Paired additive analysis; keep original four contrasts intact."""
from pathlib import Path
import argparse,json,datetime
import numpy as np
from ldc_final_v49 import ROOT,FINAL,read,save,digest,verify
OLD=ROOT/'results/ldc_final_test_20261108_v49'

def load_cohort(folder,expected):
 manifest=read(folder/'models.json');state=read(folder/'evaluation/state.json');models=manifest['models'];assert len(models)==expected and not state['attention_required'];by_model={}
 for model in models:
  assert state['tasks'][model['id']]['status']=='DONE';p=folder/'evaluation/models'/model['id']/'summary.json';assert digest(p)==state['tasks'][model['id']]['summary_sha256'];s=read(p);assert s['model']==model and s['test_cases']==200 and s['no_weight_updates'] and s['no_TEST_selection'];assert [v['sample'] for v in s['cases']]==read(FINAL/'protocol.json')['test_ids'];by_model[model['id']]=s
 return models,by_model

def calculate(models,by_model,labels,protocol):
 methods=list(dict.fromkeys(m['method'] for m in models if m['method']!='fixed'));arrays={m:np.full((5,5,200),np.nan) for m in methods}
 for m in models:
  if m['method']=='fixed':continue
  arrays[m['method']][m['search_run'],m['repeat']]=[r['mse_all'] for r in by_model[m['id']]['cases']]
 assert all(np.isfinite(a).all() for a in arrays.values());mean_seeds={m:a.mean(1) for m,a in arrays.items()};geometry=protocol['test_geometries'];unique=sorted(set(geometry));assert len(unique)==85;clusters=[np.where(np.asarray(geometry)==g)[0] for g in unique];rng=np.random.default_rng(510001);draws=np.empty((10000,len(labels)))
 for b in range(10000):
  repeats=rng.integers(0,5,5);indices=np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))]);control=mean_seeds['tpe'][repeats][:,indices].mean()
  for j,label in enumerate(labels):draws[b,j]=mean_seeds[label][repeats][:,indices].mean()-control
 return mean_seeds,draws

def main(original_only=False):
 protocol=verify();original=read(OLD/'protocol.json');old_models,old_rows=load_cohort(OLD,155);labels=original['llm_methods'];assert len(labels)==4
 if original_only:models,by_model=old_models,old_rows
 else:
  new_models,new_rows=load_cohort(FINAL,25);assert not(set(old_rows)&set(new_rows));models=old_models+new_models;by_model={**old_rows,**new_rows};labels=labels+['Llama31_8B']
 mean_seeds,draws=calculate(models,by_model,labels,protocol);old_draws=np.load(OLD/'report/bootstrap_deltas.npy',allow_pickle=False);assert np.array_equal(draws[:,:4],old_draws),'Archived four contrast draws must remain exactly identical'
 if original_only:
  save(FINAL/'original_analysis_reproduction.json',dict(status='ORIGINAL_FOUR_BOOTSTRAP_COLUMNS_REPRODUCED_EXACTLY',iterations=10000,seed=510001,TEST_Llama_read=False,original_report_sha256=digest(OLD/'report/summary.json')));print('ORIGINAL_BOOTSTRAP_REPRODUCTION_PASS',flush=True);return
 out=FINAL/'combined_report';out.mkdir(exist_ok=False);rows=[]
 for method in original['methods']+['Llama31_8B']:
  group=[s for s in by_model.values() if s['model']['method']==method];row=dict(method=method,models=len(group),test_cases_per_model=200,**{k:float(np.mean([s['means'][k] for s in group])) for k in ['mse_all','velocity_relative_l2','pressure_relative_l2','relative_l2_all']},mean_momentum_rms=float(np.mean([s['mean_momentum_rms'] for s in group])),mean_divergence_rms=float(np.mean([s['mean_divergence_rms'] for s in group])),mean_boundary_rms=float(np.mean([s['mean_boundary_rms'] for s in group])),worst_p99_ms=max(s['worst_case_p99_ms'] for s in group),speed_pass_models=sum(s['pass_5ms'] for s in group),cohort='additive_20261119' if method=='Llama31_8B' else 'original_20261104');assert row['models']==(5 if method=='fixed' else 25);rows.append(row)
 control=float(mean_seeds['tpe'].mean());family=[]
 for j,label in enumerate(labels):
  lo,hi=np.quantile(draws[:,j],[.005,.995],method='linear');effect=float(mean_seeds[label].mean()-control);family.append(dict(method=label,control='tpe',delta_test_mse=effect,relative_delta_percent=100*effect/control,interval_percent=99.0,interval=[float(lo),float(hi)],interval_below_zero=bool(hi<0),role='Additive exploratory five-contrast family, not a replacement of original primary contrasts'))
 low,high=np.quantile(draws[:,4],[.025,.975],method='linear');llama=dict(method='Llama31_8B',control='tpe',delta_test_mse=float(mean_seeds['Llama31_8B'].mean()-control),interval_percent=95.,interval=[float(low),float(high)],interval_below_zero=bool(high<0),role='Single exploratory additive contrast')
 np.save(out/'bootstrap_deltas_five_LLM.npy',draws,allow_pickle=False)
 result=dict(status='ADDITIVE_LLAMA_FULL_COMPARISON_COMPLETE',at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=180,case_model_pairs=36000,methods=rows,original_primary_contrasts=read(OLD/'report/summary.json')['primary_contrasts'],original_four_bootstrap_columns_exactly_reproduced=True,Llama_exploratory_contrast=llama,additive_five_LLM_family=family,original_report_sha256=digest(OLD/'report/summary.json'),new_report_sha256=digest(FINAL/'report/summary.json'),analysis=protocol['analysis'],no_new_training=True,no_TEST_selection=True,limitations=['Llama added after original TEST outcomes were known; actual dates retained.','Only five independent searches. Five final seeds are averaged before inference.','Known-geometry flowcase split; steady fields, not temporal rollout.','Original and new latency observations were collected on different dates with the same frozen query and H200 precision protocol.','Bootstrap coverage is approximate; no causal proof that a declared explanation used feedback.'])
 save(out/'summary.json',result);save(out/'method_metrics.json',rows)
 lines=['DODATOČNÉ PLNÉ POROVNANIE LLAMY','', '180 finálnych modelov ×200TEST prípadov; pôvodných155 modelov sa nepretrénovalo ani znova nevyberalo.','Pôvodné štyri bootstrap kontrasty sa reprodukovali presne. Ich pôvodné98.75%intervaly zostávajú zachované.','Llama má samostatný exploratívny95%interval protiTPE; osobitne je uvedená doplnená päťčlenná rodina s99%intervalmi.','Metóda | modely | TEST MSE | relatívna chyba rýchlosti | relatívna chyba tlaku | najhoršie p99 [ms]']
 for r in rows:lines.append('%s | %d | %.12g | %.6g | %.6g | %.6g'%(r['method'],r['models'],r['mse_all'],r['velocity_relative_l2'],r['pressure_relative_l2'],r['worst_p99_ms']))
 lines+=['','Kompletné čísla a intervaly: summary.json. Predikované polia všetkých25 nových modelov: evaluation/models/<model>/predicted_fields.npy na Perune.','Päť finálnych semien nie je päť nezávislých hľadaní. Dodatočný charakter Llamy a odlišné dátumy sa priznávajú.'];(out/'VYSLEDOK.txt').write_text(chr(10).join(lines)+chr(10),encoding='utf-8');print('ADDITIVE_FULL_COMPARISON_COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--original-only',action='store_true');main(p.parse_args().original_only)
