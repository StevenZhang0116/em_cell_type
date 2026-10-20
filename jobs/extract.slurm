#!/bin/bash
#SBATCH --job-name=celltype_extract
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=log/extract_%x_%j.txt
# Stage A. Usage: sbatch --export=ALL,DATASET=microns jobs/extract.slurm
set -euo pipefail
source jobs/env.sh
$PYTHON -u scripts/01_extract_clouds.py --dataset "${DATASET:-microns}" --lateral-cutoff "${CUTOFF:-250}"
