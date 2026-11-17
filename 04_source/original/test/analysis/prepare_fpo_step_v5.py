"""CPU integrity receipts for existing reference files; no new CFD or download."""
import json
import os
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from fpo_common_v5 import ROOT,WORK,save,digest,verify


def fingerprint(p):
    s=Path(p).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)


def run():
    verify();previous=ROOT/'results/fpo_pinn_state_20261030_v4'
    source=json.loads((previous/'prepared.json').read_text());records=[]
    out=WORK/'preparation'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    for r in source['records']:
        if r['proposed_split']=='test' or r['input_frames']!=[2,3]:raise ValueError('Unexpected data access')
        paths=[r['field'],str(Path(r['cache'])/'geometry.npz'),str(Path(r['cache'])/'samples.npz')]
        for p in paths:
            if digest(p)!=r['hashes'][p]:raise ValueError('Reference/cache checksum mismatch')
        if digest(r['state_cache'])!=r['state_cache_sha256']:raise ValueError('Prefix checksum mismatch')
        a=np.load(r['field'],mmap_mode='r',allow_pickle=False)
        if a.shape!=(242,256,1024,3) or a.dtype!=np.float64:raise ValueError('Unexpected format')
        records.append(dict(r,field_fingerprint=fingerprint(r['field'])))
        save(out/'progress.json',dict(completed=len(records),total=len(source['records'])))
        print('REHASHED_EXISTING_REFERENCE',r['path'],flush=True)
    save(WORK/'prepared.json',dict(status='EXISTING_ASSETS_REHASHED',records=records,
          normalization=str(previous/'training/139486/prefix_conditioned/normalization.json'),
          numeric_values_for_new_test_set_read=False,main_test_opened=False,new_CFD=False,new_downloads=False))


if __name__=='__main__':run()
