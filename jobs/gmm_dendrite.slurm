#!/bin/bash
#SBATCH --job-name=gmm_dendrite
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=log/gmm_dendrite.txt
#SBATCH --open-mode=append
# Per-neuron GMMs of the dendrite side (docs/neuron_gmm.md): 1000 random
# dendrites with >= 2000 synapses (seed as the pipeline; a random subset of
# another size is not nested in this one, so only about 20 of the validation
# job's 200 dendrites are in it), the cylindrical diagonal_constraint mixture
# of ../mixture (one lateral centre shared by all components, one lateral
# variance per component, components differ along depth), components 1-19 by
# 2 (19 is what the guard allows every cloud of 2000 synapses with this
# model, so all neurons have the same candidates), 30 train/test repetitions
# per neuron, best of 3 EM starts per fit, the number of components chosen by
# the weight-normalised held-out log-likelihood with a tolerance of 0.1 nats
# per synapse: the smallest k whose mean held-out log-likelihood over the
# repetitions is within 0.1 of the best candidate's (an estimated mean loss,
# not a population bound; the paired test of ../mixture is only recorded as a
# diagnostic). Without the tolerance every dendrite of the diagonal pilot
# chose the largest candidate it was allowed (the held-out log-likelihood
# never plateaus); the value is a resolution choice, re-check it on the stored
# curves (models.json) after the run. 10000 random pairs of neurons give the
# between-neuron distance reference.
# About 900 EM fits per neuron, roughly 300 CPU-seconds each; neurons run in
# parallel, so about 2 hours on 40 CPUs (the 200-neuron pilot with 20
# repetitions took 10 minutes). Outputs go to output/microns/gmm/dendrite/ and
# every run overwrites the previous one, so copy that folder first if you want
# to keep a run (e.g. the pilot, or before --select-by bic).
#
#   sbatch jobs/gmm_dendrite.slurm
#
# When cpu-g2 is blocked by the group quota (AssocGrpCpuLimit):
#   sbatch --account=ckpt-amath --partition=ckpt --qos=ckpt jobs/gmm_dendrite.slurm
set -euo pipefail
source jobs/env.sh
echo "== $(date) job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite \
    --n-neurons 1000 --min-synapses 2000 --min-border-distance 50 \
    --covariance diagonal_constraint --components-max 19 --components-step 2 \
    --select-tolerance 0.1 --repeats 30 --starts 3 --between-pairs 10000
echo "== $(date) done"
