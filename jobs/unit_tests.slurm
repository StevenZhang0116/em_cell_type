#!/bin/bash
#SBATCH --job-name=celltype_tests
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=log/unit_tests.txt
# pytest unit tests for the celltype package (about 2 minutes on one CPU).
# When cpu-g2 is blocked by the group quota (AssocGrpCpuLimit), use the checkpoint partition:
#   sbatch --account=ckpt-amath --partition=ckpt --qos=ckpt --output=log/unit_tests_ckpt.txt jobs/unit_tests.slurm
set -euo pipefail
source jobs/env.sh
$PYTHON -m pytest tests -q -p no:cacheprovider
