import hashlib
import json
from pathlib import Path
import sys
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
WORK=ROOT/'results/ldc_llm_preflight_20261104_v29'
for directory in ['training','analysis','llm']:
    sys.path.insert(0,str(ROOT/'test'/directory))


def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def fingerprint(path):
    s=Path(path).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)


def verify():
    f=json.loads((WORK/'freeze.json').read_text())
    for p,h in f['files'].items():
        if digest(p)!=h:raise ValueError('Frozen source/metadata changed: '+p)
    for key in ['cache_fingerprints','model_fingerprints']:
        for p,expected in f[key].items():
            if fingerprint(p)!=expected:raise ValueError('Audited data/model changed: '+p)


def read(path):return json.loads(Path(path).read_text())
