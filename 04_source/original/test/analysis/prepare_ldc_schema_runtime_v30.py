"""Install two tiny pinned pure-Python wheels without changing existing environments."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
TARGET=ROOT/'test/runtime/ldc_schema_v30'
PACKAGES=[('lm-format-enforcer','0.11.3','cf586350875def1ae7a8fba84fcbbfc8371424b6c9d05c1fcba70aa233fbf06f'),
    ('interegular','0.3.3','b0c07007d48c89d6d19f7204972d369b2a77222722e126b6aa63aa721dc3b19c')]


def run():
    TARGET.mkdir(parents=True,exist_ok=False);wheels=TARGET/'wheels';wheels.mkdir();rows=[]
    for name,version,sha in PACKAGES:
        with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30) as r:metadata=json.load(r)
        wheel=next(u for u in metadata['urls'] if u['filename'].endswith('.whl'))
        if wheel['digests']['sha256']!=sha or wheel['size']>1024*1024:raise ValueError('Pinned tiny wheel required')
        with urllib.request.urlopen(wheel['url'],timeout=30) as r:data=r.read()
        if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Wheel SHA mismatch')
        p=wheels/wheel['filename'];p.write_bytes(data)
        rows.append(dict(name=name,version=version,sha256=sha,bytes=len(data),path=str(p),url=wheel['url']))
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-compile','--target',str(TARGET/'site'),*[r['path'] for r in rows]],check=True)
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (TARGET/'site').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (TARGET/'manifest.json').write_text(json.dumps(dict(status='ISOLATED_SCHEMA_RUNTIME_READY',packages=rows,files=files),indent=2)+'\n')
    print('SCHEMA_RUNTIME_READY',len(files),flush=True)


if __name__=='__main__':run()
