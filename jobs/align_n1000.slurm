#!/bin/bash
#SBATCH --job-name=align_n1000
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=log/align_n1000_%j.txt
# Aligned, border-filtered trial on 1000 neurons per side: Stage B (10
# bootstrap replicates of the W2 distances between canonically rotated clouds)
# and Stage C (consensus clustering), keeping only neurons whose soma is at
# least 100 um from the volume border (about the lowest 5 % of dist_border,
# i.e. the arbors most truncated by the border). Stage C runs at depth weight
# 0 and 0.05 so the result can be compared with the earlier unaligned trial
# at either weight. Stage B replicate 0 prints the "alignment check" line:
# the median ratio of the aligned W2 to the minimum over 36 in-plane rotations
# should be close to 1 if canonical alignment removes the orientation effect.
#
# Trial folder:
#   output/microns/trials/n1000_rep10_pt200_aligned_min100_border100_weighted_seed20261019_average_k2-20_mcs5
#
# Usage (submit from the project root so log/ resolves):
#   sbatch jobs/align_n1000.slurm                 # both sides, axon first
#   sbatch jobs/align_n1000.slurm --sides axon    # later options override the defaults below
# If cpu-g2 is blocked by the group quota (AssocGrpCpuLimit), use the checkpoint partition:
#   sbatch --account=ckpt-amath --partition=ckpt --qos=ckpt jobs/align_n1000.slurm
# Finished replicates and clusterings are skipped, so a preempted job is simply resubmitted.
#
# Afterwards, compare with the control run of jobs/noalign_n1000.slurm:
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
    --align \
    --min-border-distance 100 \
    --depth-weight 0 0.05 \
    "$@"
