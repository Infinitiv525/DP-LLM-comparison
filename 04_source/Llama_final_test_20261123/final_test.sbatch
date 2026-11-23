#!/bin/bash
#SBATCH --job-name=Llama31-final-TEST
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --export=NIL
#SBATCH --no-requeue
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/results/llama_final_test_20261123_v1/test-%j.out
#SBATCH --error=/mnt/project/perun2601449/dp_llm_cfd/results/llama_final_test_20261123_v1/test-%j.err
set -euo pipefail
cd /mnt/project/perun2601449/dp_llm_cfd
ROOT=$PWD
export USER="${SLURM_JOB_USER:-lutoke823}" LOGNAME="${SLURM_JOB_USER:-lutoke823}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 SLURM_EXPORT_ENV=ALL
PY="$ROOT/envs/training-torch-2.12-cu130/bin/python"
export PYTHONPATH="/mnt/project/perun2601449/dp_llm_cfd/test/llama_final_test_20261123_v1:$ROOT/test/training:$ROOT/test/analysis:$ROOT/test/llm:$ROOT/test/runtime/ldc_schema_v30/site"
srun --export=ALL --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    "$PY" -B -u test/llama_final_test_20261123_v1/evaluate_ldc_final_v49.py --mode preflight
"$PY" -B test/llama_final_test_20261123_v1/review_ldc_final_v49.py preflight
srun --export=ALL --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    "$PY" -B -u test/llama_final_test_20261123_v1/evaluate_ldc_final_v49.py --mode test
"$PY" -B test/llama_final_test_20261123_v1/review_ldc_final_v49.py completed
