"""No downloads or CFD: derive state input ONLY from existing frames 2 and 3."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from fpo_common_v4 import ROOT,WORK,save,digest,verify


def prefix_features(field,fluid):
    # This function is separately tested with a reader that rejects any
    # attempt to access a future frame.
    first=np.asarray(field[2],dtype='float32').copy()
    last=np.asarray(field[3],dtype='float32').copy()
    first[:,:,2]-=first[:,:,2][fluid].mean(dtype=np.float64)
    last[:,:,2]-=last[:,:,2][fluid].mean(dtype=np.float64)
    features=np.concatenate((last,last-first),2)
    features[~fluid]=0
    low=features.reshape(32,8,128,8,6).mean((1,3)).transpose(2,0,1).astype('float32')
    return features,low


def run():
    verify()
    previous=ROOT/'results/fpo_pinn_conditioned_20261030_v3/prepared.json'
    records=json.loads(previous.read_text())['records'];ready=[]
    out=WORK/'preparation'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    for r in records:
        started=time.monotonic()
        if r['proposed_split']=='test':raise ValueError('Main test sealed')
        geometry=Path(r['cache'])/'geometry.npz'
        if digest(geometry)!=r['hashes'][str(geometry)]:raise ValueError('Geometry changed')
        with np.load(geometry,allow_pickle=False) as z:fluid=z['fluid']
        field=np.load(r['field'],mmap_mode='r',allow_pickle=False)
        # Intentionally explicit: no future frame is read into input features.
        features,low=prefix_features(field,fluid)
        cache=WORK/'state_cache'/r['family']/r['geometry_id']/str(r['reynolds'])
        cache.mkdir(parents=True,exist_ok=False)
        path=cache/'prefix.npz';np.savez_compressed(path,features=features,low=low)
        item=dict(r,state_cache=str(path),state_cache_sha256=digest(path),input_frames=[2,3],
                  target_frames=list(range(4,242)),relative_time_origin_frame=3,
                  prefix_preparation_and_compression_seconds=time.monotonic()-started)
        ready.append(item)
        save(out/'progress.json',dict(completed=len(ready),total=len(records)))
        print('PREFIX_ONLY_PREPARED',r['path'],flush=True)
    save(WORK/'prepared.json',dict(status='PREFIX_INPUTS_PREPARED',records=ready,
                                  input_frames=[2,3],future_fields_in_inputs=False,main_test_opened=False))


if __name__=='__main__':run()
