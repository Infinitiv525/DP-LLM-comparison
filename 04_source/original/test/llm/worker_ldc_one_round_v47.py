"""Run the unchanged V33 worker for at most one original queued attempt."""
import os
from pathlib import Path
import worker_ldc_main_v33 as original
from ldc_main_common_v33 import ROOT, WORK, read, save, digest, verify

POLICY=ROOT/'results/ldc_tail_20261107_v47/freeze.json'


class OneRoundClaim:
    def __init__(self, claim):
        self.claim=claim
        self.finished_claiming=False
        self.task=None
        self.stop_reason=None

    def __call__(self,*args):
        if self.finished_claiming:
            self.stop_reason='ONE_ATTEMPT_ASSIGNED'
            # This is the original worker's local exit signal. It does not
            # modify the global state or claim that the experiment is done.
            return None,'ALL_DONE'
        payload,status=self.claim(*args)
        self.finished_claiming=True
        if payload is not None:
            self.task=payload['task']['id']
            return payload,status
        self.stop_reason=status
        return None,'ATTENTION_REQUIRED' if status=='ATTENTION_REQUIRED' else 'ALL_DONE'


def run():
    verify()
    for path,expected in read(POLICY)['files'].items():
        if digest(path)!=expected:raise ValueError('Tail operational source changed: '+path)
    gate=OneRoundClaim(original.claim)
    original.claim=gate
    original.run()
    job=os.environ['SLURM_JOB_ID'];rank=int(os.environ['SLURM_PROCID'])
    save(WORK/'main/batches'/job/'one_round'/f'rank_{rank}.json',dict(
        policy='At most one original queued attempt per worker',task=gate.task,
        rank=rank,job=job,stop_reason=gate.stop_reason,
        operational_freeze_sha256=digest(POLICY),main_test_opened=False))


if __name__=='__main__':run()
