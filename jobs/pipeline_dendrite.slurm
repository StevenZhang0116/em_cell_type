#!/bin/bash
#SBATCH --job-name=pipeline_dendrite
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=log/pipeline_dendrite.txt
#SBATCH --open-mode=append
# The dendrite pipeline on 200 neurons with >= 2000 synapses: split-half validation
# (docs/split_cluster.md: halves of 500 synapses, 20 outer splits, 50 inner
# bootstrap replicates per half), then Stage B (exact W2 of the complete clouds
# plus 50 bootstrap replicates) and Stage C (consensus clustering) on the same
# cohort, so summary.json of the validation also reports how often the full
# fit's k was chosen from half A.
#
# Cost on 40 CPUs: validation about 200 CPU hours (10 per outer split), i.e.
# 5 to 6 hours; Stage B about 300 CPU hours (these clouds have a median of about
# 3000 synapses, 3 s per exact pair), i.e. 7 to 8 hours; Stage C minutes.
# Everything is resumable: validation keeps every finished split in
# output/microns/trials/validate_dendrite/m500.npz and Stage B every finished
# replicate in trials/dist_dendrite/, so if the job runs out of time, submit
# it again and it continues where it stopped.
#
#   sbatch jobs/pipeline_dendrite.slurm
#
# When cpu-g2 is blocked by the group quota (AssocGrpCpuLimit), the checkpoint
# partition works too; a preempted job is simply resubmitted:
#   sbatch --account=ckpt-amath --partition=ckpt --qos=ckpt jobs/pipeline_dendrite.slurm
set -euo pipefail
source jobs/env.sh
echo "== $(date) job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
$PYTHON -u scripts/run_pipeline.py --sides dendrite --stages validate distances cluster \
    --n-neurons 200 --min-synapses 2000 --min-border-distance 50 \
    --validation-points-per-half 500 --validation-reps 20 --n-reps 50 \
    --k-min 2 --k-max 10 --min-cluster-size 5
echo "== $(date) done"
