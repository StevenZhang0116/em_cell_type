#!/bin/bash
#SBATCH --job-name=split_half_pairs
#SBATCH --account=cpu-g2-amath
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=log/split_half_pairs_%j.txt
set -euo pipefail
source jobs/env.sh
$PYTHON -m pytest tests/test_split_half.py tests/test_split_half_pairs.py -q -p no:cacheprovider
$PYTHON -u scripts/analyze_split_half_pairs.py --side dendrite
