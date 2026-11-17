"""Require eight physically distinct H200 GPUs and propagated offline flags."""
import argparse
import os
from ldc_main_common_v33 import WORK,phase_path,read,save,verify


def run():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['main','integration']);p.add_argument('--review',action='store_true');a=p.parse_args()
    verify();directory=phase_path(a.phase)/'batches'/os.environ['SLURM_JOB_ID']
    expected=read(WORK/'submissions'/(os.environ['SLURM_JOB_ID']+'.json'))['gpus']
    if a.review:
        rows=[read(directory/'devices'/f'rank_{i}.json') for i in range(expected)]
        if len({r['uuid'] for r in rows})!=expected:raise ValueError('All assigned physical GPUs must be distinct')
        save(directory/'device_assignment.json',dict(rows=rows));return
    import torch
    if os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('TRANSFORMERS_OFFLINE')!='1':raise ValueError('Offline flags absent')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('One GPU per worker required')
    prop=torch.cuda.get_device_properties(0)
    if 'H200' not in prop.name or not str(prop.uuid):raise ValueError('H200 GPU UUID required')
    rank=int(os.environ['SLURM_PROCID'])
    save(directory/'devices'/f'rank_{rank}.json',dict(rank=rank,uuid=str(prop.uuid),device=prop.name,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),offline=True))


if __name__=='__main__':run()
