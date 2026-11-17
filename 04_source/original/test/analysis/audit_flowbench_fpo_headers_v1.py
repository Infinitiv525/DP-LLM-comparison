"""Bounded HTTP-range audit of one pinned FPO case. No full dataset download."""
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile
import numpy as np

ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
WORK=ROOT/'results/flowbench_2d_audit_20261030_v1'
REV='937c672604d4c9f39aceedf1fa2f0a603695264b'
BASE='https://huggingface.co/datasets/BGLab/FlowBench/resolve/'+REV+'/'


def get_json(url):
    with urllib.request.urlopen(url,timeout=40) as r:return json.load(r)


class RangeFile(io.RawIOBase):
    def __init__(self,url,size,budget=40*1024*1024):
        self.url=url;self.size=size;self.pos=0;self.used=0;self.budget=budget;self.requests=[]
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset if whence==2 else -1
        if target<0:raise ValueError('Invalid seek')
        self.pos=target;return target
    def read(self,n=-1):
        if n<0:
            if self.size-self.pos>1024*1024:raise ValueError('Unbounded remote read refused')
            n=max(0,self.size-self.pos)
        n=min(n,max(0,self.size-self.pos))
        if not n:return b''
        if n>20*1024*1024 or self.used+n>self.budget:raise ValueError('Remote read budget exceeded')
        start=self.pos;end=start+n-1
        url=self.url+f'?audit_range={start}-{end}'
        req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
        with urllib.request.urlopen(req,timeout=90) as r:
            expected=f'bytes {start}-{end}/{self.size}'
            if r.status!=206 or r.headers.get('Content-Range')!=expected:raise ValueError('Server did not honor bounded range; body not read')
            data=r.read(n+1)
        if len(data)!=n:raise ValueError('Range length differs')
        self.requests.append(dict(start=start,end=end,bytes=n,sha256=hashlib.sha256(data).hexdigest()))
        self.used+=n;self.pos+=n;return data


def header(stream):
    version=np.lib.format.read_magic(stream)
    if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
    elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
    else:raise ValueError('Unsupported NPY version')
    if dtype.hasobject:raise ValueError('Object arrays refused')
    return shape,order,dtype


def inspect(path,size,first_frames=False):
    remote=RangeFile(BASE+path,size);arrays=[]
    with zipfile.ZipFile(remote) as z:
        for item in z.infolist():
            if not item.filename.endswith('.npy'):continue
            with z.open(item) as stream:
                shape,order,dtype=header(stream)
                record=dict(name=item.filename,shape=list(shape),fortran_order=order,dtype=str(dtype),compressed_bytes=item.compress_size,uncompressed_bytes=item.file_size)
                if any(k in item.filename.lower() for k in ['time','dt','coord']) and item.file_size<1024*1024:
                    raw=stream.read();a=np.frombuffer(raw,dtype=dtype)
                    record['small_metadata_values']=a.tolist() if len(a)<300 else dict(min=float(a.min()),max=float(a.max()))
                if first_frames and not order and len(shape)==4 and shape[0] in [240,242] and shape[-1]==3 and tuple(shape[1:3]) in [(256,1024),(1024,256)]:
                    count=int(np.prod(shape[1:]));need=2*count*dtype.itemsize
                    raw=stream.read(need)
                    if len(raw)!=need:raise ValueError('Incomplete first two frames')
                    a=np.frombuffer(raw,dtype=dtype).reshape((2,)+shape[1:]);stats=[]
                    for frame in a:
                        stats.append(dict(finite=bool(np.isfinite(frame).all()),channel_min=frame.min(axis=(0,1)).tolist(),channel_max=frame.max(axis=(0,1)).tolist(),channel_rms=np.sqrt(np.mean(frame.astype(float)**2,axis=(0,1))).tolist()))
                    record['first_two_frame_stats']=stats
                    record['first_two_frames_equal']=bool(np.array_equal(a[0],a[1]))
                    record['no_physical_t0_inferred_from_index']=True
                arrays.append(record)
    return dict(path=path,file_bytes=size,arrays=arrays,range_bytes_read=remote.used,range_requests=remote.requests,full_file_downloaded=False)


def run():
    WORK.mkdir(parents=True,exist_ok=True);target=WORK/'audit.json'
    if target.exists():raise RuntimeError('Never overwrite audit')
    freeze=json.loads((WORK/'source_freeze.json').read_text())
    for name,sha in freeze['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Audit source changed')
    entries=get_json('https://huggingface.co/api/datasets/BGLab/FlowBench/tree/'+REV+'/FPO_NS_2D_1024x256/harmonics/1?limit=100')
    selected=[e for e in entries if e['path'].endswith('/Re_382.npz') or e['path'].endswith('/input_geometry.npz')]
    if len(selected)!=2:raise ValueError('Expected exact data/geometry files')
    results=[]
    for e in selected:
        print('INSPECT',e['path'],flush=True)
        r=inspect(e['path'],e['size'],e['path'].endswith('/Re_382.npz'));r['expected_lfs_sha256']=e['lfs']['oid'];r['whole_file_sha256_verified']=False;results.append(r)
    commit=get_json('https://api.github.com/repos/baskargroup/flowbench-tools/commits/master')['sha']
    source_url='https://raw.githubusercontent.com/baskargroup/flowbench-tools/'+commit+'/DataPrep/fpo2d.py'
    with urllib.request.urlopen(source_url,timeout=40) as r:source=r.read()
    (WORK/'official_fpo2d_source.txt').write_bytes(source)
    result=dict(status='FPO_BOUNDED_SAMPLE_AUDIT_COMPLETE',date='2026-10-30',dataset_revision=REV,sample_role='readiness sample, not assigned test data',files=results,code_revision=commit,code_url=source_url,code_sha256=hashlib.sha256(source).hexdigest(),source_says_ignore_first_two=b'ignore the first two' in source,limitations=['One geometry and oneRe; not a whole-dataset integrity audit','First frame statistics do not establish physical initialization or time units','Partial range reads do not validate full LFS file SHA','No wall boundary condition inferred at cropped edges','No PINN training performed'])
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('FPO_BOUNDED_SAMPLE_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':run()
