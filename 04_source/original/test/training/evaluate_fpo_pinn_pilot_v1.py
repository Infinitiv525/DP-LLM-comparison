"""Posthoc full-grid pilot accuracy, no weight update or independent-case claim."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd');WORK=ROOT/'results/fpo_pinn_pilot_20261030_v1';OUT=ROOT/'results/fpo_pinn_accuracy_20261030_v1'
def roles():
 train=set(range(2,242,6));val=set(range(5,242,6));held=set(range(2,242))-train-val
 return {'train':train,'validation':val,'heldout_time':held}
def errors(pred,ref):
 p=pred.astype(np.float64).copy();r=ref.astype(np.float64).copy()
 raw=np.sum((p-r)**2,axis=0)
 p[:,2]-=p[:,2].mean();r[:,2]-=r[:,2].mean()
 sq=np.sum((p-r)**2,axis=0);norm=np.sum(r*r,axis=0)
 return dict(velocity_relative_l2=float(np.sqrt(sq[:2].sum()/max(norm[:2].sum(),1e-24))),pressure_gauge_aligned_relative_l2=float(np.sqrt(sq[2]/max(norm[2],1e-24))),component_relative_l2=np.sqrt(sq/np.maximum(norm,1e-24)).tolist(),component_rmse=np.sqrt(sq/len(p)).tolist(),raw_component_rmse=np.sqrt(raw/len(p)).tolist())
class Model(nn.Module):
 def __init__(self):
  super().__init__();layers=[];last=3
  for _ in range(4):layers.extend([nn.Linear(last,64),nn.Tanh()]);last=64
  layers.append(nn.Linear(last,3));self.net=nn.Sequential(*layers)
  for k in ['lo','span','mean','std']:self.register_buffer(k,torch.zeros(3))
 def forward(self,q):return self.net(2*(q-self.lo)/self.span-1)*self.std+self.mean
def run():
 import hashlib
 freeze=json.loads((OUT/'freeze.json').read_text())
 for p,h in freeze['files'].items():
  if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Accuracy source/checkpoint changed')
 torch.set_num_threads(4)
 if not torch.cuda.is_available():raise RuntimeError('GPU required')
 state=torch.load(WORK/'pilot/138875/best.pt',map_location='cpu',weights_only=True);model=Model();model.load_state_dict(state,strict=True);model.to('cuda').eval()
 a=np.load(WORK/'data/data.npy',mmap_mode='r',allow_pickle=False)
 with np.load(WORK/'data/input_geometry.npz',allow_pickle=False) as z:m=z['mask']
 solid=(m>.5) if np.mean(m>.5)<.5 else (m<=.5);mask=~solid
 x,y=np.meshgrid(4+(np.arange(1024)+.5)*16/1024,6+(np.arange(256)+.5)*4/256,indexing='xy')
 q=torch.tensor(np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)]),dtype=torch.float32,device='cuda');split=roles();rows=[]
 baseline=state['mean'].numpy().astype(float)
 with torch.inference_mode():
  for k in range(2,242):
   q[:,2]=(k-2)*.05;prediction=model(q).cpu().numpy().reshape(256,1024,3);ref=np.asarray(a[k]);role=next(name for name,ids in split.items() if k in ids)
   row=dict(frame=k,time=(k-2)*.05,role=role,metrics=errors(prediction[mask],ref[mask]),constant_training_mean_baseline=errors(np.broadcast_to(baseline,(int(mask.sum()),3)),ref[mask]));rows.append(row)
   if k%40==0:print('EVALUATED',k,flush=True)
 summary={}
 for role,ids in split.items():
  rr=[r for r in rows if r['role']==role];summary[role]=dict(frames=len(rr),mean_velocity_relative_l2=float(np.mean([r['metrics']['velocity_relative_l2'] for r in rr])),mean_pressure_relative_l2=float(np.mean([r['metrics']['pressure_gauge_aligned_relative_l2'] for r in rr])),mean_v_relative_l2=float(np.mean([r['metrics']['component_relative_l2'][1] for r in rr])),constant_baseline_velocity=float(np.mean([r['constant_training_mean_baseline']['velocity_relative_l2'] for r in rr])),constant_baseline_pressure=float(np.mean([r['constant_training_mean_baseline']['pressure_gauge_aligned_relative_l2'] for r in rr])))
 (OUT/'per_frame_accuracy.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
 result=dict(status='PINN_PILOT_FULL_GRID_ACCURACY_EVALUATED',checkpoint='pilot/138875/best.pt',checkpoint_sha256=hashlib.sha256((WORK/'pilot/138875/best.pt').read_bytes()).hexdigest(),frame_sets={k:sorted(v) for k,v in split.items()},metrics=summary,first_two_frames_excluded=True,spatial_mask='fluid-only, minority mask interpreted as solid as in pilot',pressure_gauge_aligned=True,checkpoint_changed=False,no_independent_scenario_test=True,posthoc_pilot_diagnostic=True,limitations=['Heldout_time is interpolation in the same case, not independent geometry','Validation checkpoint selection already used validation frames','No predefined final accuracy acceptance threshold for this feasibility pilot','Not a final LLM comparison'])
 (OUT/'accuracy_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(summary,flush=True)
if __name__=='__main__':run()
