import hashlib
import json
from pathlib import Path
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
WORK=ROOT/'results/ldc_pinn_prepare_20261104_v25'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def verify():
    s=json.loads((WORK/'freeze.json').read_text())
    for p,h in s['files'].items():
        if digest(p)!=h:raise ValueError('Frozen source changed: '+p)
