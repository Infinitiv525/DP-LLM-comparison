"""Archive entire repository inventory/text; one pinned numeric readiness case."""
import concurrent.futures as cf
import hashlib,json,re,urllib.request,zipfile,shutil,ast
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
WORK=ROOT/'results/fpo_pinn_pilot_20261030_v1'
COMMIT='6877099b8883232138c73b2b6056ff01f7d3697b'
REV='937c672604d4c9f39aceedf1fa2f0a603695264b'
PATH='FPO_NS_2D_1024x256/harmonics/1/'
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fetch(url):
 with urllib.request.urlopen(url,timeout=90) as r:return r.read()
def verify():
 f=json.loads((WORK/'freeze.json').read_text())
 for p,h in f['files'].items():
  if sha(Path(p))!=h:raise ValueError('Source freeze changed')
def derivative_check(a,mask,stride=1):
 vals=[];h=stride;dx=16/1024;dy=4/256
 for k in [12,32,52,72,112,152,192,220]:
  f=a[k];u=f[:,:,0];v=f[:,:,1];p=f[:,:,2]
  ok=mask.copy();ok[:4*h]=False;ok[-4*h:]=False;ok[:,:4*h]=False;ok[:,-4*h:]=False
  for sy in range(-3*h,3*h+1):
   for sx in range(-3*h,3*h+1):ok &= np.roll(mask,(sy,sx),(0,1))
  def x(q):return (np.roll(q,-h,1)-np.roll(q,h,1))/(2*h*dx)
  def y(q):return (np.roll(q,-h,0)-np.roll(q,h,0))/(2*h*dy)
  def lap(q):return (np.roll(q,-h,1)-2*q+np.roll(q,h,1))/(h*dx)**2+(np.roll(q,-h,0)-2*q+np.roll(q,h,0))/(h*dy)**2
  A=((a[k+1,:,:,:2]-a[k-1,:,:,:2])/2)[ok]
  B=np.stack([u*x(u)+v*y(u)+x(p)-lap(u)/382,u*x(v)+v*y(v)+y(p)-lap(v)/382],-1)[ok]
  if len(A)<100 or not np.isfinite(A).all() or not np.isfinite(B).all():raise ValueError('Bad derivative sample')
  aa=float(np.sum(A*A));ab=float(np.sum(A*B));bb=float(np.sum(B*B))
  coef=-ab/max(aa,1e-30);dt=1/coef if coef>0 else None
  res=float(np.linalg.norm(coef*A+B)/max(np.linalg.norm(B),1e-12))
  vals.append(dict(frame=k,estimated_snapshot_dt=dt,relative_residual=res,points=len(A),derivative_energy=aa))
 return vals
def run():
 verify();out=WORK/'deep_audit';out.mkdir(exist_ok=False)
 tree=json.loads(fetch(f'https://api.github.com/repos/baskargroup/flowbench-tools/git/trees/{COMMIT}?recursive=1'))
 save(out/'repository_inventory.json',tree)
 entries=[e for e in tree['tree'] if e['type']=='blob'];selected=[e for e in entries if Path(e['path']).suffix in ['.py','.md','.txt','.yaml','.sh','.ipynb'] and e.get('size',0)<10*1024*1024]
 def archive(e):
  b=fetch(f'https://raw.githubusercontent.com/baskargroup/flowbench-tools/{COMMIT}/'+e['path']);p=out/'repository'/e['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
  text=b.decode('utf8',errors='replace');hits=[dict(line=i+1,text=line[:500]) for i,line in enumerate(text.splitlines()) if re.search(r'dt_V|OutputSpan|totalT|time.?step|time.?interval|SamplingBounds|ignore the first',line,re.I)]
  return dict(path=e['path'],sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),hits=hits)
 with cf.ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(archive,selected))
 save(out/'text_inventory_and_hits.json',records)
 for label,url in [('issues','https://api.github.com/repos/baskargroup/flowbench-tools/issues?state=all&per_page=100'),('dataset_root',f'https://huggingface.co/api/datasets/BGLab/FlowBench/tree/{REV}?limit=1000')]:
  save(out/(label+'.json'),json.loads(fetch(url)))
 configs=[]
 for family in ['Harmonic','Nurbs','Skelneton']:
  s=(out/'repository'/f'CaseGenerator/{family}/FPO/config_generator.py').read_text()
  def val(key):return ast.literal_eval(re.search(r'(?m)^'+key+r'\s*=\s*(\[[^\n]+?\])',s)[1])
  configs.append(dict(family=family,dt=val('dt_V'),span=val('OutputSpan_V'),end=val('totalT_V')))
 assert all(c['dt'][-1]==.005 and c['span'][-1]==10 and c['end'][-2:]==[392,404] for c in configs)
 data=WORK/'data';data.mkdir(exist_ok=False)
 listing=json.loads(fetch(f'https://huggingface.co/api/datasets/BGLab/FlowBench/tree/{REV}/{PATH}?limit=100'))
 for name in ['Re_382.npz','input_geometry.npz']:
  e=next(e for e in listing if e['path']==PATH+name);p=data/name
  with urllib.request.urlopen(f'https://huggingface.co/datasets/BGLab/FlowBench/resolve/{REV}/'+e['path'],timeout=120) as r,p.open('xb') as f:shutil.copyfileobj(r,f,4*1024*1024)
  if p.stat().st_size!=e['size'] or sha(p)!=e['lfs']['oid']:raise ValueError('Full file SHA mismatch')
  print('DOWNLOADED_VERIFIED',name,flush=True)
 with zipfile.ZipFile(data/'Re_382.npz') as z,z.open('data.npy') as f,(data/'data.npy').open('xb') as g:shutil.copyfileobj(f,g,4*1024*1024)
 a=np.load(data/'data.npy',mmap_mode='r',allow_pickle=False);assert a.shape==(242,256,1024,3)
 finite=all(np.isfinite(a[k]).all() for k in range(242))
 with np.load(data/'input_geometry.npz',allow_pickle=False) as z:m=z['mask']
 unique=np.unique(m);assert set(unique)<=set([0.,1.])
 solid=(m>0.5) if np.mean(m>0.5)<.5 else (m<=.5);fluid=~solid
 checks={str(s):derivative_check(a,fluid,s) for s in [1,2]}
 checks_flat=[r for v in checks.values() for r in v]
 positive=all(r['estimated_snapshot_dt'] is not None for r in checks_flat)
 median_dt=float(np.median([r['estimated_snapshot_dt'] for r in checks_flat])) if positive else None
 residual=float(np.median([r['relative_residual'] for r in checks_flat]))
 ready=bool(finite and positive and abs(median_dt/.05-1)<=.25 and residual<=.25)
 result=dict(status='DEEP_FPO_AUDIT_COMPLETE',commit=COMMIT,dataset_revision=REV,repository_files=len(entries),archived_text_files=len(records),skipped_files=[e['path'] for e in entries if e not in selected],author_configs=configs,solver_dt=.005,source_supported_snapshot_dt=.05,time_units='solver units, not established SI seconds',absolute_frame_times_confirmed=False,proposed_relative_time='(frame_index-2)*0.05 for retained frames2..241; not physical startup',crop_bounds_assumption=[4,20,6,10],crop_provenance='author downsampler example; numeric consistency screened, not per-file metadata',derivative_checks=checks,median_inferred_dt=median_dt,median_relative_residual=residual,pilot_ready=ready,data_sha256=sha(data/'data.npy'),finite_all_frames=finite,physical_startup_claim=False,full_dataset_audited=False,pilot_role='single-case data-assisted reconstruction and residual/latency feasibility; no generalization or LLM ranking')
 save(out/'audit.json',result);print('AUDIT_COMPLETE_PILOT_READY',ready,median_dt,residual,flush=True)
 if not ready:raise RuntimeError('Physical consistency gate failed; do not launch pilot with guessed scaling')
if __name__=='__main__':run()
