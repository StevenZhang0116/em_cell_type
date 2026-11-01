#!/bin/bash
#SBATCH --job-name=gmm_matches
#SBATCH --account=cpu-g2-amath
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=log/gmm_matches_%j.txt
set -euo pipefail
source jobs/env.sh
$PYTHON -m pytest tests/test_gmm_matches.py -q -p no:cacheprovider
$PYTHON -u scripts/analyze_gmm_matches.py --side dendrite
