#!/bin/bash
#SBATCH --job-name=celltype_pipeline
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=log/pipeline_%j.txt
# Whole pipeline (Stage A, then B + C + D for every side) in one job.
# Memory is sized for Stage A; CPUs are used by Stages B and D.
# Usage: sbatch jobs/pipeline.slurm [run_pipeline.py options]
#   sbatch jobs/pipeline.slurm --dataset microns
#   sbatch jobs/pipeline.slurm --sides dendrite --n-neurons 3000
# Finished outputs are skipped, so resubmitting continues an interrupted run.
set -euo pipefail
source jobs/env.sh
$PYTHON -u scripts/run_pipeline.py "$@"
