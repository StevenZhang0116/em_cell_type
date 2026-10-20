#!/bin/bash
#SBATCH --job-name=celltype_dist
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=log/dist_%A_%a.txt
# Stage B, one bootstrap replicate per array task.
# Usage: sbatch --export=ALL,DATASET=microns,SIDE=axon,METHOD=points,DEPTH=relative jobs/distances.slurm
# Optional: COORDS=cylindrical (default xyz)
set -euo pipefail
source jobs/env.sh
$PYTHON -u scripts/02_compute_distances.py --dataset "${DATASET:-microns}" --side "$SIDE" \
  --method "${METHOD:-points}" --coords "${COORDS:-xyz}" --depth-mode "${DEPTH:-relative}" --rep "$SLURM_ARRAY_TASK_ID"
