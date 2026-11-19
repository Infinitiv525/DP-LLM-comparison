#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:15:00
#SBATCH --job-name=LLM-behavior-audit-v2
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/results/research_review_20261119_v1/LLM-v2-%j.out
/mnt/project/perun2601449/dp_llm_cfd/envs/architecture-pilot-20261110-v1/bin/python /mnt/project/perun2601449/dp_llm_cfd/test/research_review_20261119_v1/analyze_llm_behavior_v2.py
