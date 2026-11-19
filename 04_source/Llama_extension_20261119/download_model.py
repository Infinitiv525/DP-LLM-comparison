from pathlib import Path
import json,hashlib,os,traceback,time
from huggingface_hub import snapshot_download
R=Path('/mnt/project/perun2601449/dp_llm_cfd');O=R/'results/llama_extension_20261119_v1';revision='0e9e39f249a16976918f6564b8830bc894c89659';dest=R/'public_models/llm_feasibility/Llama-3.1-8B-Instruct'/revision

def save(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
try:
 start=time.perf_counter();snapshot_download('meta-llama/Llama-3.1-8B-Instruct',revision=revision,local_dir=dest,allow_patterns=['*.json','*.safetensors','tokenizer.model','*.txt','*.md','LICENSE'],max_workers=4)
 paths=sorted([p for p in dest.rglob('*') if p.is_file() and '.cache' not in p.parts]);rows=[]
 for p in paths:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
  rows.append(dict(file=p.relative_to(dest).as_posix(),bytes=p.stat().st_size,sha256=h.hexdigest()))
 assert any(p['file'].endswith('.safetensors') for p in rows)
 save('model_ready.json',dict(status='PINNED_LLAMA_MODEL_DOWNLOADED_ON_PERUN',repo='meta-llama/Llama-3.1-8B-Instruct',revision=revision,model_path=str(dest),files=rows,elapsed_seconds=time.perf_counter()-start,job=os.environ.get('SLURM_JOB_ID'),redistribution='Meta model weights are not uploaded to GitHub; users obtain their own authorized copy'))
 print(json.dumps({'status':'MODEL_READY','files':len(rows),'bytes':sum(p['bytes'] for p in rows)}))
except Exception as e:
 save('download_failure.json',dict(error_type=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None)));raise
