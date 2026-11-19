#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --job-name=Llama31-model-download
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/results/llama_extension_20261119_v1/download-%j.out
/mnt/project/perun2601449/dp_llm_cfd/envs/llm-feasibility-20261021/bin/python /mnt/project/perun2601449/dp_llm_cfd/test/llama_extension_20261119_v1/download_model.py
