#!/bin/bash
#SBATCH --job-name=celltype_tests
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=log/unit_tests.txt
# pytest unit tests for the celltype package.
set -euo pipefail
source jobs/env.sh
$PYTHON -m pytest tests -q -p no:cacheprovider
