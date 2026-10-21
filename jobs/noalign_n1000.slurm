#!/bin/bash
#SBATCH --job-name=noalign_n1000
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=log/noalign_n1000_%j.txt
# Control for jobs/align_n1000.slurm: identical settings (1000 neurons per
# side, 10 bootstrap replicates, soma at least 100 um from the volume border,
# Stage C at depth weight 0 and 0.05) but on the clouds as extracted, without
# the canonical in-plane rotation. Stage B replicate 0 prints the "alignment
# check" line: here the median ratio of the given W2 to the minimum over 36
# in-plane rotations shows how much the arbitrary orientation inflates the
# distances when nothing is done about it.
#
# Trial folder:
#   output/microns/trials/n1000_rep10_pt200_min100_border100_weighted_seed20261019_average_k2-20_mcs5
#
# Usage (submit from the project root so log/ resolves):
#   sbatch jobs/noalign_n1000.slurm                 # both sides, axon first
#   sbatch jobs/noalign_n1000.slurm --sides axon    # later options override the defaults below
# If cpu-g2 is blocked by the group quota (AssocGrpCpuLimit), use the checkpoint partition:
#   sbatch --account=ckpt-amath --partition=ckpt --qos=ckpt jobs/noalign_n1000.slurm
# Finished replicates and clusterings are skipped, so a preempted job is simply resubmitted.
#
# Afterwards, compare the two trials:
#   $PYTHON scripts/compare_trials.py \
#       --trial-a n1000_rep10_pt200_min100_border100_weighted_seed20261019_average_k2-20_mcs5 \
#       --trial-b n1000_rep10_pt200_aligned_min100_border100_weighted_seed20261019_average_k2-20_mcs5
set -euo pipefail
source jobs/env.sh
echo "job ${SLURM_JOB_ID:-?} on $(hostname -s), ${SLURM_CPUS_PER_TASK:-1} cpus, started $(date)"
$PYTHON -u scripts/run_pipeline.py \
    --sides axon dendrite \
    --stages distances cluster \
    --n-neurons 1000 \
    --min-border-distance 100 \
    --depth-weight 0 0.05 \
    "$@"
