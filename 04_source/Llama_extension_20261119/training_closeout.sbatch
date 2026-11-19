#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=Llama31-training_closeout
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/results/llama_extension_20261119_v1/training_closeout-%j.out
set -euo pipefail
/mnt/project/perun2601449/dp_llm_cfd/envs/llm-feasibility-20261021/bin/python -B /mnt/project/perun2601449/dp_llm_cfd/test/llama_extension_20261119_v1/training_closeout.py
