#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:30:00
#SBATCH --job-name=Llama31-main
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/results/llama_extension_20261119_v1/main-job-%j.out
set -euo pipefail
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
/mnt/project/perun2601449/dp_llm_cfd/envs/llm-feasibility-20261021/bin/python -B /mnt/project/perun2601449/dp_llm_cfd/test/llama_extension_20261119_v1/run_extension_task.py --phase main --task "$1"
