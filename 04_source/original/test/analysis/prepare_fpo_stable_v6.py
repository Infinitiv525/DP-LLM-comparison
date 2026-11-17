"""Single existing training case; derive geometry-only streamfunction taper."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from fpo_common_v6 import WORK,save,digest,verify


def run():
    verify();receipt=json.loads((WORK/'pilot_input.json').read_text());r=receipt['record']
    if r['proposed_split']!='train' or (r['family'],r['geometry_id'],r['reynolds'])!=('harmonics','1',382):raise ValueError('Unexpected positive control')
    s=Path(r['field']).stat();fp=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)
    if fp!=r['field_fingerprint'] or digest(r['field'])!=r['hashes'][r['field']]:raise ValueError('Reference changed')
    for name in ['geometry.npz','samples.npz']:
        p=str(Path(r['cache'])/name)
        if digest(p)!=r['hashes'][p]:raise ValueError('Cache changed')
    if digest(r['state_cache'])!=r['state_cache_sha256']:raise ValueError('Prefix changed')
    with np.load(str(Path(r['cache'])/'geometry.npz'),allow_pickle=False) as z:fluid=z['fluid']
    distance=ndimage.distance_transform_edt(fluid,sampling=(1/64,1/64))
    weight=((1-np.exp(-distance/.5))**2).astype('float32')
    if weight.shape!=(256,1024) or not np.isfinite(weight).all() or np.max(weight[~fluid])!=0:raise ValueError('Invalid taper')
    if min(weight[0].min(),weight[-1].min(),weight[:,0].min(),weight[:,-1].min())<=0:raise ValueError('Crop edges must not be obstacle walls')
    out=WORK/'preparation'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    path=out/'geometry_weight.npz';np.savez_compressed(path,weight=weight)
    save(WORK/'prepared.json',dict(status='EXISTING_PILOT_AND_GEOMETRY_READY',records=[r],normalization=receipt['normalization'],
         weight_file=str(path),weight_sha256=digest(path),recipe='(1-exp(-EDT(fluid)/0.5))^2, physical dx=dy=1/64',
         main_test_opened=False,new_CFD=False,new_downloads=False))
    print('PILOT_READY',r['path'],flush=True)


if __name__=='__main__':run()
