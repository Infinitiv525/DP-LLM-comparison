#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=Llama31-accounting_collector
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/results/llama_final_test_20261123_v1/accounting_collector-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/project/perun2601449/dp_llm_cfd/envs/training-torch-2.12-cu130/bin/python -B /mnt/project/perun2601449/dp_llm_cfd/test/llama_final_test_20261123_v1/capture_test_accounting.py
