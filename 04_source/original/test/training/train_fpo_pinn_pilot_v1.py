"""Single-case technical data-assisted PINN, NOT a final test or LLM ranking."""
import json,time,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from audit_fpo_deep_v1 import verify,sha
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd');WORK=ROOT/'results/fpo_pinn_pilot_20261030_v1'
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def ns_residual(q,pred):
 def grad(v):return torch.autograd.grad(v,q,torch.ones_like(v),create_graph=True)[0]
 u,v,p=pred.unbind(1);gu,gv,gp=grad(u),grad(v),grad(p)
 lu=grad(gu[:,0])[:,0]+grad(gu[:,1])[:,1];lv=grad(gv[:,0])[:,0]+grad(gv[:,1])[:,1]
 return torch.stack([gu[:,2]+u*gu[:,0]+v*gu[:,1]+gp[:,0]-lu/382,gv[:,2]+u*gv[:,0]+v*gv[:,1]+gp[:,1]-lv/382,gu[:,0]+gv[:,1]],1)
def run():
 verify()
 audit=json.loads((WORK/'deep_audit/audit.json').read_text())
 if not audit['pilot_ready']:raise RuntimeError('Audit did not qualify pilot')
 if sha(WORK/'data/data.npy')!=audit['data_sha256']:raise RuntimeError('Data changed after audit')
 out=WORK/'pilot'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
 torch.manual_seed(20261030);rng=np.random.default_rng(20261030);torch.set_num_threads(4)
 if not torch.cuda.is_available():raise RuntimeError('GPU required')
 device='cuda';a=np.load(WORK/'data/data.npy',mmap_mode='r',allow_pickle=False)
 with np.load(WORK/'data/input_geometry.npz',allow_pickle=False) as z:m=z['mask']
 solid=(m>.5) if np.mean(m>.5)<.5 else (m<=.5);mask=~solid
 yy,xx=np.where(mask);interior=mask.copy()
 for sy in range(-3,4):
  for sx in range(-3,4):interior &= np.roll(mask,(sy,sx),(0,1))
 interior[:4]=False;interior[-4:]=False;interior[:,:4]=False;interior[:,-4:]=False
 iy,ix=np.where(interior)
 train_frames=np.arange(2,242,6);val_frames=np.arange(5,242,6)
 def batch(n,frames,inside=False):
  cy,cx=(iy,ix) if inside else (yy,xx);j=rng.integers(len(cy),size=n);y=cy[j];x=cx[j];k=rng.choice(frames,size=n)
  coords=np.column_stack([4+(x+.5)*16/1024,6+(y+.5)*4/256,(k-2)*.05]).astype('float32')
  values=np.asarray(a[k,y,x],dtype='float32')
  return torch.tensor(coords,device=device),torch.tensor(values,device=device)
 _,sample=batch(20000,train_frames);mean=sample.mean(0);std=sample.std(0).clamp_min(1e-4)
 class Model(nn.Module):
  def __init__(self):
   super().__init__();layers=[];last=3
   for _ in range(4):layers += [nn.Linear(last,64),nn.Tanh()];last=64
   layers += [nn.Linear(last,3)];self.net=nn.Sequential(*layers)
   self.register_buffer('lo',torch.tensor([4.,6.,0.]));self.register_buffer('span',torch.tensor([16.,4.,11.95]));self.register_buffer('mean',mean);self.register_buffer('std',std)
  def forward(self,q):return self.net(2*(q-self.lo)/self.span-1)*self.std+self.mean
 model=Model().to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
 def physics(q):
  q=q.detach().requires_grad_(True);return ns_residual(q,model(q)).square().sum(1).mean()
 valq,valy=batch(8192,val_frames);history=[];best=float('inf');start=time.monotonic()
 for step in range(1,2001):
  q,y=batch(1024,train_frames);cq,_=batch(512,train_frames,True)
  opt.zero_grad();data=((model(q)-y)/std).square().mean();pde=physics(cq);loss=data+.01*pde
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite pilot loss')
  loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if step==1 or step%100==0:
   with torch.no_grad():vl=float((((model(valq)-valy)/std)**2).mean())
   row=dict(step=step,data_loss=float(data.detach()),pde_loss=float(pde.detach()),validation_normalized_mse=vl);history.append(row);print(row,flush=True)
   if vl<best:best=vl;torch.save(model.state_dict(),out/'best.pt')
 model.load_state_dict(torch.load(out/'best.pt',weights_only=True));model.eval()
 gx,gy=np.meshgrid(4+(np.arange(1024)+.5)*16/1024,6+(np.arange(256)+.5)*4/256,indexing='xy')
 grid=torch.tensor(np.column_stack([gx.ravel(),gy.ravel(),np.full(gx.size,6.)]),dtype=torch.float32,device=device)
 lat=[]
 with torch.inference_mode():
  for i in range(70):
   torch.cuda.synchronize();t=time.perf_counter();pred=model(grid);torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-t)
   if i>=20:lat.append(elapsed)
 save(out/'history.json',history)
 save(out/'summary.json',dict(status='TECHNICAL_DATA_ASSISTED_PINN_PILOT_COMPLETE',steps=2000,best_validation_normalized_mse=best,elapsed_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),latency_ms_gpu_resident_full_grid={k:float(np.percentile(lat,v)) for k,v in [('p50',50),('p95',95),('p99',99)]},latency_samples=len(lat),grid=[1024,256],dt_assumption=.05,time_units='solver-relative, not SIseconds',scope='One case reconstruction; no independent generalization test, no LLM comparison',boundary_policy='No wall imposed at crop edges; observed training data constrain reconstruction; not pure forward boundary-value solver',physical_startup_claim=False,final_scientific_result=False))
if __name__=='__main__':run()
