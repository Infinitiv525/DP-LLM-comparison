"""User amendment: run ONLY the already frozen prefix-conditioned variant."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from fpo_common_v4 import WORK,save,digest,verify
from fpo_state_v4 import load
from train_fpo_state_v4 import train,tensor


def run():
    verify()
    for path,checksum in json.loads((WORK/'state_only_freeze_v1.json').read_text())['files'].items():
        if digest(path)!=checksum:raise ValueError('Amendment changed')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    configs=[c for c in json.loads((WORK/'protocol.json').read_text())['candidates'] if c['state_conditioning']]
    if len(configs)!=1:raise ValueError('Expected exactly one prefix-conditioned configuration')
    records=json.loads((WORK/'prepared.json').read_text())['records'];cases=[]
    for r in records:
        if r['proposed_split']=='test' or r['input_frames']!=[2,3]:raise ValueError('Unexpected data access')
        if digest(r['state_cache'])!=r['state_cache_sha256']:raise ValueError('Prefix changed')
        for name in ['geometry.npz','samples.npz']:
            path=str(r['cache'])+'/'+name
            if digest(path)!=r['hashes'][path]:raise ValueError('Cache changed')
        case=load(r);case['geo_gpu']=tensor(case['geometry'][None]);case['low_gpu']=tensor(case['low'][None])
        case['state_gpu']=tensor(case['initial']);case['fluid_yx']=np.column_stack(np.where(case['fluid']))
        cases.append(case)
    train_cases=[c for c in cases if c['record']['proposed_split']=='train']
    val_cases=[c for c in cases if c['record']['proposed_split']=='validation']
    out=WORK/'training'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    result=train(configs[0],train_cases,val_cases,out/configs[0]['name'])
    save(out/'summary.json',dict(status='COMPLETE',amendment='Known prefix always supplied; no geometry-only control',results=[result]))


if __name__=='__main__':run()
