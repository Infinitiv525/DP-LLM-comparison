import sys,os,json,pathlib,hashlib,time,traceback,platform,collections
R=pathlib.Path('/mnt/project/perun2601449/dp_llm_cfd');OUT=R/'results/llm_native_fields_20261118_v1';OUT.mkdir(exist_ok=True,parents=True)
for p in ['training','analysis','llm']:sys.path.insert(0,str(R/'test'/p))
import numpy as np,torch
import ldc_final_v49 as f
from ldc_store_v26 import query

def save(p,d):f.save(p,d)
def quick(pred,ref,fluid):
 difference=pred.astype(np.float64)-ref
 result=f.errors(pred[fluid],ref[fluid]);result.update(mse_all=float(np.mean(difference**2)),relative_l2_all=float(np.sqrt(np.sum(difference**2)/max(np.sum(ref.astype(np.float64)**2),1e-24))))
 return {k:float(result[k]) for k in ['mse_all','relative_l2_all','velocity_relative_l2','pressure_relative_l2']}

def run():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 assert torch.__version__=='2.12.0+cu130' and np.__version__=='2.5.4'
 assert torch.cuda.is_available() and torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name()
 protocol=f.verify();models=f.read(f.FINAL/'models.json')['models'];assert len(models)==155
 groups=collections.OrderedDict()
 for row in models:groups.setdefault(row['checkpoint_sha256'],[]).append(row)
 assert len(groups)==101
 env=dict(job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(0),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),python=platform.python_version(),precision='FP32',allow_tf32=False,cudnn_allow_tf32=True,no_optimizer_or_weight_updates=True,original_source_freeze_sha256=f.digest(f.FINAL/'freeze.json'),original_models_sha256=f.digest(f.FINAL/'models.json'),script_sha256=f.digest(__file__))
 save(OUT/'environment.json',env)
 # Reuse the original validation-only correctness gate before reconstructing TEST fields.
 val=f.EvaluationInputs(protocol,'validation');model=f.load_model(models[0],val.normalization);index=val.ids[0]
 with torch.inference_mode():pred=query(model,val.known[index],val.qgrid).cpu().numpy()
 measured=quick(pred,val.reference[index],np.asarray(val.fluid[index]));old=next(v for v in f.read(pathlib.Path(models[0]['training_summary']).parent/'selected_validation/summary.json')['cases'] if v['sample']==index)
 assert all(np.isclose(measured[k],old[k],rtol=1e-3,atol=1e-9) for k in measured)
 save(OUT/'validation_preflight.json',dict(status='PASS',model=models[0]['id'],sample=index,measured=measured,original={k:old[k] for k in measured},rtol=1e-3,atol=1e-9));del model,val;torch.cuda.empty_cache()
 store=f.EvaluationInputs(protocol,'test');assert store.ids==protocol['test_ids'] and len(store.ids)==200
 normal=store.normalization;completed=[];start=time.monotonic()
 save(OUT/'PLAN.json',dict(logical_models=155,unique_checkpoint_groups=101,cases_per_model=200,shape=[200,512,512,3],axes=['case','y','x','field'],fields=['u','v','p'],rtol=1e-3,atol=1e-9,reason_for_tolerance='Same tolerance as the frozen original VAL correctness gate; float32 runtime reproduction only, not an accuracy threshold or a new result-selection rule',no_training=True,no_new_benchmark_timing=True,groups=[dict(id=v[0]['id'],checkpoint_sha256=k,models=[x['id'] for x in v]) for k,v in groups.items()]))
 for gi,(sha,aliases) in enumerate(groups.items()):
  identifier=aliases[0]['id'];directory=OUT/'models'/identifier;done=directory/'complete.json'
  if done.exists():
   record=f.read(done);assert f.digest(directory/'predictions.npy')==record['predictions_sha256'];completed.append(record);continue
  directory.mkdir(parents=True,exist_ok=False);originals={}
  for row in aliases:
   assert row['config']==aliases[0]['config'] and f.digest(row['checkpoint'])==sha
   path=f.FINAL/'evaluation/models'/row['id']/'summary.json';summary=f.read(path);assert summary['status']=='FINAL_TEST_MODEL_COMPLETE' and summary['model']['checkpoint_sha256']==sha
   assert [x['sample'] for x in summary['cases']]==store.ids
   originals[row['id']]=dict(path=str(path),sha256=f.digest(path),cases={x['sample']:x for x in summary['cases']})
  model=f.load_model(aliases[0],normal);prediction_path=directory/'predictions.incomplete.npy';arr=np.lib.format.open_memmap(prediction_path,mode='w+',dtype='<f4',shape=(200,512,512,3));checks=[];begin=time.monotonic()
  for slot,index in enumerate(store.ids):
   with torch.inference_mode():prediction=query(model,store.known[index],store.qgrid).cpu().numpy()
   assert prediction.shape==(512,512,3) and prediction.dtype==np.float32 and np.isfinite(prediction).all() and np.abs(prediction).max()<=1000
   metrics=quick(prediction,store.reference[index],np.asarray(store.fluid[index]))
   comparisons={}
   for row in aliases:
    old=originals[row['id']]['cases'][index];assert old['geometry']==store.inputs['geometry_hashes'][index] and old['reynolds']==store.known[index]['reynolds']
    comparisons[row['id']]={k:dict(original=old[k],reconstructed=metrics[k],abs_difference=abs(old[k]-metrics[k])) for k in metrics}
    if not all(np.isclose(metrics[k],old[k],rtol=1e-3,atol=1e-9) for k in metrics):
     save(directory/'metric_mismatch.json',dict(model=row['id'],sample=index,comparison=comparisons[row['id']]));raise ValueError('Reconstruction differs from frozen original metrics')
   arr[slot]=prediction;checks.append(dict(sample=index,metrics=metrics,comparisons=comparisons))
   if (slot+1)%50==0:
    arr.flush();save(OUT/'status.json',dict(stage='reconstructing_and_verifying_TEST_fields',completed_unique_groups=len(completed),total_unique_groups=101,current_model=identifier,completed_cases=slot+1,total_cases=200));print('FIELDS_PROGRESS',gi+1,identifier,slot+1,flush=True)
  arr.flush();del arr;prediction_path.rename(directory/'predictions.npy');del model;torch.cuda.empty_cache()
  record=dict(status='ALL_200_CASES_MATCH_ORIGINAL_METRICS',id=identifier,aliases=[x['id'] for x in aliases],checkpoint_sha256=sha,checkpoint=aliases[0]['checkpoint'],predictions_sha256=f.digest(directory/'predictions.npy'),bytes=(directory/'predictions.npy').stat().st_size,shape=[200,512,512,3],fields=['u','v','p'],case_indices=store.ids,source='Reconstructed from unchanged frozen final checkpoints using the original V49 query; not an originally saved prediction file',original_summaries=[dict(id=k,path=v['path'],sha256=v['sha256']) for k,v in originals.items()],rtol=1e-3,atol=1e-9,elapsed_seconds=time.monotonic()-begin,environment=env)
  save(directory/'metric_verification.json',dict(status='PASS',cases=checks));save(directory/'aliases.json',dict(models=aliases));save(directory/'normalization.json',normal);save(done,record);completed.append(record)
 f.verify()
 for row in models:assert f.digest(row['checkpoint'])==row['checkpoint_sha256']
 summary=dict(status='ALL_155_MODELS_NATIVE_TEST_FIELDS_RECONSTRUCTED_AND_VERIFIED',logical_models=155,unique_field_arrays=101,cases_per_model=200,logical_case_fields=31000,unique_case_fields=20200,groups=completed,environment=env,elapsed_seconds=time.monotonic()-start,no_training=True,no_new_benchmark_timing=True,original_results_unchanged=True,fields_are_steady_cases_not_time_sequence=True)
 save(OUT/'complete.json',summary);save(OUT/'status.json',dict(stage='complete',completed_unique_groups=101,total_unique_groups=101));print('ALL_FIELDS_COMPLETE',155,101,flush=True)
if __name__=='__main__':
 try:run()
 except Exception as exc:
  save(OUT/'failure.json',dict(exception=type(exc).__name__,traceback=traceback.format_exc(),no_training=True));raise
