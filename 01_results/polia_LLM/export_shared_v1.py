import pathlib,json,sys,os,hashlib,time,traceback,urllib.request,collections
R=pathlib.Path('/mnt/project/perun2601449/dp_llm_cfd');OUT=R/'results/llm_native_fields_20261118_v1/shared_data';OUT.mkdir(parents=True,exist_ok=True)
for p in ['training','analysis','llm']:sys.path.insert(0,str(R/'test'/p))
import numpy as np
import ldc_final_v49 as f

def run():
 protocol=f.verify();prepared=f.read(protocol['prepared_path']);split=f.read(protocol['split_path'])['indices'];cache=pathlib.Path(prepared['cache']);inputs=f.read(cache/'inputs.json');schema=inputs['schema'];fluid=np.load(cache/'fluid.npy',mmap_mode='r',allow_pickle=False);original_x=np.load(cache.parent/'raw/X_original.npy',mmap_mode='r',allow_pickle=False);original_y=np.load(prepared['rawtest_targets_sealed_in'],mmap_mode='r',allow_pickle=False);ytrain=np.load(cache/'Y_train.npy',mmap_mode='r',allow_pickle=False);yval=np.load(cache/'Y_validation.npy',mmap_mode='r',allow_pickle=False);assert [len(split[v]) for v in ['train','validation','test']]==[640,160,200]
 chunks=[]
 for role in ['train','validation','test']:
  ids=split[role]
  for offset in range(0,len(ids),100):
   chunk_ids=ids[offset:offset+100];name=role+'_%03d'%(offset//100);directory=OUT/name;directory.mkdir(exist_ok=False);n=len(chunk_ids);shape=(n,512,512)
   targets=np.lib.format.open_memmap(directory/'reference_uvp.npy',mode='w+',dtype='<f4',shape=shape+(3,));sdf=np.lib.format.open_memmap(directory/'geometry_sdf.npy',mode='w+',dtype='<f4',shape=shape);mask=np.lib.format.open_memmap(directory/'fluid_mask.npy',mode='w+',dtype='bool',shape=shape)
   for slot,index in enumerate(chunk_ids):
    fluid_one=np.asarray(fluid[index],dtype=bool);mask[slot]=fluid_one;sdf[slot]=np.asarray(original_x[index,schema['sdf_channel']],dtype=np.float32)*schema['sdf_orientation']
    target=np.moveaxis(np.asarray(ytrain[offset+slot] if role=='train' else yval[offset+slot]),0,2) if role!='test' else np.moveaxis(f.canonical_target(np.asarray(original_y[index]),fluid_one),0,2)
    assert target.shape==(512,512,3) and np.isfinite(target).all();targets[slot]=target
   targets.flush();sdf.flush();mask.flush();del targets,sdf,mask
   np.save(directory/'case_indices.npy',np.asarray(chunk_ids,dtype='<i8'),allow_pickle=False);np.save(directory/'reynolds.npy',np.asarray([inputs['reynolds'][v] for v in chunk_ids],dtype='<f8'),allow_pickle=False)
   metadata=dict(role=role,case_indices=chunk_ids,geometry_hashes=[inputs['geometry_hashes'][v] for v in chunk_ids],shape=[n,512,512,3],target_convention='Original actual cached TRAIN/VAL targets, or original canonical_target TEST; u/v/p float32; known solid zeros and own-fluid pressure gauge',input_convention='Native fluid mask, signed distance field with the frozen orientation, Reynolds; geometry pooling remains inside original model query')
   f.save(directory/'cases.json',metadata);files=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=f.digest(p)) for p in sorted(directory.iterdir()) if p.is_file()];chunks.append(dict(id=name,role=role,cases=n,files=files));f.save(OUT/'status.json',dict(stage='exporting_task_used_data',completed_chunks=len(chunks),current=name))
 metadata=OUT/'metadata';metadata.mkdir(exist_ok=False)
 for name,path in [('inputs.json',cache/'inputs.json'),('normalization.json',cache/'normalization.json'),('split.json',pathlib.Path(protocol['split_path'])),('prepared.json',pathlib.Path(protocol['prepared_path'])),('original_evaluation_protocol.json',f.FINAL/'protocol.json'),('original_models.json',f.FINAL/'models.json'),('original_source_freeze.json',f.FINAL/'freeze.json')]:
  (metadata/name).write_bytes(path.read_bytes())
 # Preserve the original evaluation/model dependencies, without binary environments or downloaded foundation models.
 sources=[]
 for name,module in sorted(sys.modules.copy().items()):
  p=getattr(module,'__file__',None)
  if not p:continue
  p=pathlib.Path(p)
  if p.suffix!='.py':continue
  try:relative=p.relative_to(R/'test')
  except ValueError:continue
  target=OUT/'source'/'test'/relative;target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():target.write_bytes(p.read_bytes());sources.append(dict(path=str(target.relative_to(OUT)),sha256=f.digest(target)))
 request=urllib.request.Request('https://huggingface.co/datasets/BGLab/FlowBench/resolve/'+prepared['revision']+'/README.md',headers={'User-Agent':'research-data-preservation'})
 with urllib.request.urlopen(request,timeout=45) as response:card=response.read(1000000)
 assert b'cc-by-nc-4.0' in card.lower();(metadata/'ORIGINAL_FLOWBENCH_README.md').write_bytes(card)
 note='Task-used subset of BGLab/FlowBench, LDC_NS_2D/512x512 harmonics, revision '+prepared['revision']+'. Original authors retain attribution. Dataset license: CC BY NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/ . See ORIGINAL_FLOWBENCH_README.md for original authors and citation. This export preserves the 1000 cases actually used (640 TRAIN,160 VAL,200 TEST), canonical u/v/p targets and model inputs. It is not the entire FlowBench dataset and contains no new CFD.\n'
 (metadata/'DATASET_ATTRIBUTION.txt').write_text(note);f.verify();result=dict(status='ALL_1000_TASK_USED_INPUTS_AND_REFERENCES_EXPORTED',cases=1000,split_counts={k:len(v) for k,v in split.items()},chunks=chunks,sources=sources,metadata_files=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=f.digest(p)) for p in sorted(metadata.iterdir())],license='Original FlowBench CC BY NC 4.0; no relicensing of source data or external code',dataset_revision=prepared['revision'],job=os.environ['SLURM_JOB_ID'],no_new_CFD=True,no_training=True,original_data_unchanged=True)
 f.save(OUT/'complete.json',result);f.save(OUT/'status.json',dict(stage='complete',completed_chunks=len(chunks)));print('ALL_SHARED_DATA_COMPLETE',len(chunks),flush=True)
if __name__=='__main__':
 try:run()
 except Exception as e:f.save(OUT/'failure.json',dict(exception=type(e).__name__,traceback=traceback.format_exc()));raise
