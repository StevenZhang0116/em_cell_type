#!/bin/bash
#SBATCH --job-name=celltype_cluster
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=log/cluster_%j.txt
# Stage C. Usage: same --export variables as distances.slurm
set -euo pipefail
source jobs/env.sh
$PYTHON -u scripts/03_cluster_neurons.py --dataset "${DATASET:-microns}" --side "$SIDE" \
  --method "${METHOD:-points}" --coords "${COORDS:-xyz}" --depth-mode "${DEPTH:-relative}"
