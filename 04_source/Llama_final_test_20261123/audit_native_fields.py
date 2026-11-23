"""Independent CPU audit of all recorded fields against reference metrics; no GPU queries."""
from pathlib import Path
import json,datetime
import numpy as np
from ldc_final_v49 import FINAL,ROOT,read,save,digest,verify,canonical_target,errors
p=verify();state=read(FINAL/'evaluation/state.json');assert not state['attention_required'] and all(r['status']=='DONE' for r in state['tasks'].values());assert len(state['tasks'])==25
prepared=read(p['prepared_path']);fluid=np.load(Path(prepared['cache'])/'fluid.npy',mmap_mode='r',allow_pickle=False);raw=np.load(prepared['rawtest_targets_sealed_in'],mmap_mode='r',allow_pickle=False);refs={i:np.moveaxis(canonical_target(np.asarray(raw[i]),np.asarray(fluid[i])),0,2) for i in p['test_ids']};models=read(FINAL/'models.json')['models'];rows=[]
for m in models:
 folder=FINAL/'evaluation/models'/m['id'];s=read(folder/'summary.json');assert digest(folder/'summary.json')==state['tasks'][m['id']]['summary_sha256'];assert digest(folder/'predicted_fields.npy')==s['predicted_fields_sha256'];a=np.load(folder/'predicted_fields.npy',mmap_mode='r',allow_pickle=False);assert a.shape==(200,512,512,3) and a.dtype==np.float32;worst=0.
 for j,index in enumerate(p['test_ids']):
  pred=np.asarray(a[j]);ref=refs[index];assert np.isfinite(pred).all();mse=float(np.mean((pred.astype(np.float64)-ref)**2));expected=s['cases'][j];assert expected['sample']==index;assert np.isclose(mse,expected['mse_all'],rtol=1e-12,atol=1e-16),(m['id'],index,'Recorded field does not match reported MSE');e=errors(pred[np.asarray(fluid[index])],ref[np.asarray(fluid[index])]);assert all(np.isclose(e[k],expected[k],rtol=1e-6,atol=1e-9) for k in ['velocity_relative_l2','pressure_relative_l2']);worst=max(worst,abs(mse-expected['mse_all']))
 rows.append(dict(model=m['id'],cases=200,maximum_MSE_difference=worst,predicted_fields_sha256=s['predicted_fields_sha256'],checkpoint_sha256=m['checkpoint_sha256']));print('FIELDS_MATCH_METRICS',m['id'],flush=True)
result=dict(status='ALL5000_RECORDED_FIELDS_INDEPENDENTLY_AUDITED',at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=25,cases_per_model=200,total_fields=5000,problems=0,no_new_training=True,no_new_GPU_queries=True,rows=rows,comparison_summary_sha256=digest(FINAL/'combined_report/summary.json'));save(FINAL/'native_field_audit/summary.json',result);verify();print(result['status'],flush=True)
