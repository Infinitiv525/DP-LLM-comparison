#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --job-name=Llama31-native_field_audit
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/results/llama_final_test_20261123_v1/native_field_audit-%j.out
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/project/perun2601449/dp_llm_cfd/envs/training-torch-2.12-cu130/bin/python -B /mnt/project/perun2601449/dp_llm_cfd/test/llama_final_test_20261123_v1/audit_native_fields.py
