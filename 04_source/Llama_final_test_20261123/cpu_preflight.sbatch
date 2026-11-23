#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --job-name=Llama31-TEST-CPU-checks
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/results/llama_final_test_20261123_v1/cpu-preflight-%j.out
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/project/perun2601449/dp_llm_cfd/envs/training-torch-2.12-cu130/bin/python
"$PY" -B /mnt/project/perun2601449/dp_llm_cfd/test/llama_final_test_20261123_v1/test_ldc_final_v49.py
"$PY" -B /mnt/project/perun2601449/dp_llm_cfd/test/llama_final_test_20261123_v1/compare_with_original.py --original-only
