#!/bin/bash
#SBATCH --account=perun2601449
#SBATCH --qos=perun2601449
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=dp-LLM-power-audit
#SBATCH --output=/mnt/project/perun2601449/dp_llm_cfd/test/llm_power_20261124_v1/verify-%j.out
#SBATCH --error=/mnt/project/perun2601449/dp_llm_cfd/test/llm_power_20261124_v1/verify-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/project/perun2601449/dp_llm_cfd/envs/architecture-pilot-20261110-v1/bin/python
"$PY" -B /mnt/project/perun2601449/dp_llm_cfd/test/llm_power_20261124_v1/verify_power_v3.py
"$PY" -B /mnt/project/perun2601449/dp_llm_cfd/test/llm_power_20261124_v1/build_report.py
