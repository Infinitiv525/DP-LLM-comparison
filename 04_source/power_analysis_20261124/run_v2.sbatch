#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=dp-LLM-power
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/test/llm_power_20261124_v1/analysis-%j.out
#SBATCH --error=/mnt/project/perun2601449/dp_llm_cfd/test/llm_power_20261124_v1/analysis-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/project/perun2601449/dp_llm_cfd/envs/architecture-pilot-20261110-v1/bin/python -B /mnt/project/perun2601449/dp_llm_cfd/test/llm_power_20261124_v1/power_analysis_v2.py --output /mnt/project/perun2601449/dp_llm_cfd/results/llm_power_20261124_v2
