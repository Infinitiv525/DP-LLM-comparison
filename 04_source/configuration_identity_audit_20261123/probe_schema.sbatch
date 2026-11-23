#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --job-name=LLM-schema-identity-audit
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/results/llm_config_integrity_20261123_v1/schema-%j.out
set -euo pipefail
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/project/perun2601449/dp_llm_cfd/envs/llm-feasibility-20261021/bin/python -B /mnt/project/perun2601449/dp_llm_cfd/test/llm_config_integrity_20261123_v1/probe_schema.py
