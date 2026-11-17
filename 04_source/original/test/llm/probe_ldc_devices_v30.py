"""Compute-side preflight: propagated offline settings and four distinct CUDA GPUs."""
import argparse
import os
from ldc_llm_common_v30 import WORK,save,read,verify


def run():
    p=argparse.ArgumentParser();p.add_argument('--review',action='store_true');a=p.parse_args();verify()
    if a.review:
        rows=[read(WORK/'devices'/f'rank_{i}.json') for i in range(4)]
        if len({r['uuid'] for r in rows})!=4:raise ValueError('Four workers must have different physical GPUs')
        save(WORK/'device_assignment.json',dict(status='FOUR_DISTINCT_GPUS_OFFLINE_READY',rows=rows));return
    import torch
    rank=int(os.environ['SLURM_PROCID'])
    if os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('TRANSFORMERS_OFFLINE')!='1':raise ValueError('Offline settings not propagated by srun')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Exactly one available CUDA GPU per worker')
    properties=torch.cuda.get_device_properties(0)
    if 'H200' not in properties.name:raise ValueError('H200 required')
    uuid=str(getattr(properties,'uuid',''))
    if not uuid:raise ValueError('GPU UUID required for physical binding check')
    save(WORK/'devices'/f'rank_{rank}.json',dict(rank=rank,uuid=uuid,device=properties.name,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),slurm_step_gpus=os.environ.get('SLURM_STEP_GPUS'),
        hf_hub_offline=os.environ['HF_HUB_OFFLINE'],transformers_offline=os.environ['TRANSFORMERS_OFFLINE']))


if __name__=='__main__':run()
