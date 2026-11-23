#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --job-name=LLM-config-chain-audit
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/results/llm_config_integrity_20261123_v1/audit-%j.out
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/project/perun2601449/dp_llm_cfd/envs/training-torch-2.12-cu130/bin/python -B /mnt/project/perun2601449/dp_llm_cfd/test/llm_config_integrity_20261123_v1/audit_chain.py
