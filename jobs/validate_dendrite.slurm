#!/bin/bash
#SBATCH --job-name=validate_dendrite
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=log/validate_dendrite.txt
#SBATCH --open-mode=append
# Split-half validation of the dendrite consensus clustering (docs/split_cluster.md):
# 200 dendrites with >= 2000 synapses, halves of 500 synapses, 20 outer splits,
# 50 inner bootstrap replicates per half. About 200 CPU hours (10 per outer
# split), i.e. 5 to 6 hours on 40 CPUs. Every finished split is saved to
# output/microns/trials/validate_dendrite/m500.npz, so if the job runs out of
# time, submit it again and it continues after the last finished split.
#
#   sbatch jobs/validate_dendrite.slurm
#
# When cpu-g2 is blocked by the group quota (AssocGrpCpuLimit), the checkpoint
# partition works too; a preempted job is simply resubmitted:
#   sbatch --account=ckpt-amath --partition=ckpt --qos=ckpt jobs/validate_dendrite.slurm
set -euo pipefail
source jobs/env.sh
echo "== $(date) job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
$PYTHON -u scripts/run_pipeline.py --sides dendrite --stages validate \
    --n-neurons 200 --min-synapses 2000 --min-border-distance 50 \
    --validation-points-per-half 500 --validation-reps 20 --n-reps 50 \
    --k-min 2 --k-max 10 --min-cluster-size 5
echo "== $(date) done"
