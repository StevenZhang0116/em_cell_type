#!/bin/bash
#SBATCH --job-name=dendrite_1000
#SBATCH --account=cpu-g2-amath
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00
#SBATCH --output=log/pipeline_dendrite_%j.txt
#SBATCH --open-mode=append
# 1000 dendrites with >=2000 synapses and soma border distance >=50 um.
# Use all observed synapses; first compute full-cloud distances and consensus,
# then validate the same cohort with 20 disjoint splits of 500 synapses/half.
# TRIAL_REPS sets both full-fit and per-half bootstrap counts (default 20).
# This is an exploratory count, supported by the 200-neuron sensitivity check;
# stability at 1000 neurons still needs to be assessed from the new outputs.
#
# 499500 neuron pairs, versus 19900 at N=200 (~25.1x). Extrapolating the old
# 40-CPU log (exact 1518 s, ~560 s/bootstrap, ~11.2 h including validation)
# gives roughly 75 h on 64 CPUs with 20 bootstraps; cohort and scaling effects
# can change this substantially. Request 5 days with 96G for concurrent OT.
# Completed exact matrices, bootstrap matrices and outer splits are reused;
# an interrupted matrix/split restarts. Prefer cpu-g2 for this long run.
#
#   sbatch jobs/pipeline_dendrite.slurm
# To retain 50 bootstraps (~176 h by the same extrapolation):
#   sbatch --time=9-00:00:00 --export=ALL,TRIAL_REPS=50 jobs/pipeline_dendrite.slurm
#
# One output location is used by the pipeline. Before changing cohort size,
# preserve the existing run as trials_<old size>; never overwrite an archive.
# Resubmission with the same 1000-neuron settings resumes trials/ normally.
set -euo pipefail
source jobs/env.sh
TRIAL_REPS=${TRIAL_REPS:-20}
if ! [[ "$TRIAL_REPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRIAL_REPS must be a positive integer" >&2
    exit 1
fi
echo "== $(date) job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
"$PYTHON" - <<'PY'
import json
from pathlib import Path

trials = Path("output/microns/trials")
config = trials / "config.json"
if config.exists():
    previous_size = json.loads(config.read_text()).get("n_neurons")
    if previous_size != 1000:
        archive = trials.with_name(f"trials_{previous_size if previous_size is not None else 'all'}")
        if archive.exists():
            raise SystemExit(f"Archive already exists: {archive}; choose another archive name before submitting.")
        trials.rename(archive)
        print(f"Preserved previous run: {archive}", flush=True)
PY

args=(--sides dendrite --n-neurons 1000 --min-synapses 2000 --min-border-distance 50
      --validation-points-per-half 500 --validation-reps 20 --n-reps "$TRIAL_REPS"
      --k-min 2 --k-max 10 --min-cluster-size 5)
"$PYTHON" -u scripts/run_pipeline.py "${args[@]}" --stages distances cluster
"$PYTHON" -u scripts/run_pipeline.py "${args[@]}" --stages validate
echo "== $(date) done"
