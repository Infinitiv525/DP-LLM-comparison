"""Attach accounting before any neural computation in a new GPU batch."""
import os
from ldc24_ops_v36 import bootstrap

if __name__=='__main__':bootstrap(os.environ['SLURM_JOB_ID'])
