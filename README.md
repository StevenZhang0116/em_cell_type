# Label-free cell types from synapse anatomy

Cluster proofread neurons by the spatial distribution of their **own** synapses, without
using the dataset cell-type labels, and optionally assign every other neuron to those
clusters. Axons and dendrites are handled separately, so each side gets its own taxonomy;
the dataset labels are kept only for comparison afterwards.

Inputs are the cell and synapse tables used by the Stage-1 GMM fit in `../mixture`
(`microns_v1dd/sven/`). Code of `../mixture` is imported where needed, never modified or
copied.

## What it does

| Stage | What | Output |
|---|---|---|
| A | the synapse cloud of every proofread neuron: soma-centred offsets with border-compensation weights | `output/<dataset>/clouds_<side>.npz` |
| B | bootstrap replicates of neuron-to-neuron distances: exact W2 between resampled point clouds, or MW2 over one shared Gaussian mixture | `trials/<trial>/dist_<side>/` |
| C | consensus clustering of the replicates, choice of k, checks against the dataset labels and confounds | `trials/<trial>/cluster_<side>/` |
| D (optional) | assign unclustered neurons to the clusters (kNN on W2 and cluster templates), calibrated by cross-validation | `trials/<trial>/assign_<side>_<query>/` |

Beside the pipeline, `scripts/fit_neuron_gmms.py` fits the weighted Gaussian mixture of
`../mixture` to every neuron and measures how stable those fits are (`gmm_output/`).

## Documentation

- [docs/pipeline.md](docs/pipeline.md): Stages A to C, trial folders and options, the two
  distance methods and their cost, Stage C outputs.
- [docs/assignment.md](docs/assignment.md): Stage D, the two assignment methods,
  calibration, outputs.
- [docs/neuron_gmm.md](docs/neuron_gmm.md): per-neuron GMMs, which k to try, fitting
  stability, outputs.

## Running (on klone, inside a Slurm allocation)

`jobs/env.sh` selects the conda environment, caps BLAS at one thread (parallelism comes
from joblib over `$SLURM_CPUS_PER_TASK`) and changes into the project directory. Use an
interactive node (e.g. `salloc --account=amath --partition=cpu-g2 -c 10 --mem=32G
--time=4:00:00`) or an `sbatch` script that sources `jobs/env.sh`; Stage A needs about
96 GB of memory, everything else a few GB.

```bash
sbatch jobs/unit_tests.slurm                                       # pytest unit tests (1 CPU, about 2 min)
source jobs/env.sh                                                 # on the allocated node
$PYTHON -u scripts/run_pipeline.py --stages extract                # Stage A, once (96 GB)
$PYTHON -u scripts/run_pipeline.py --sides axon                    # all proofread axons: Stages B and C
$PYTHON -u scripts/run_pipeline.py --n-neurons 1000 --n-points 200 --stages distances cluster   # 1000 neurons per side
$PYTHON -u scripts/run_pipeline.py --sides dendrite --n-neurons 3000 --n-points 200 500 --stages assign --query rest   # Stage D
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite --n-neurons 200 --starts 3           # per-neuron GMMs
```

Every trial is named after its Stage B and C parameters, so runs with different settings
never overwrite each other; finished stages are skipped unless `--overwrite`.

## Layout

```
celltype/    library code (no command-line parsing)
  config.py      paths, dataset tables, output layout, Slurm-aware CPU count
  clouds.py      extraction and CSR reader for per-neuron synapse clouds
  distances.py   resampling and exact W2 between point clouds
  shared.py      shared-component mixture, per-neuron mixing weights, MW2
  clustering.py  clustering, stability, noise floor, label and confound checks
  templates.py   cylindrically constrained weighted GMM (cluster templates)
  assign.py      kNN and template assignment, cross-validated calibration
  plotting.py    Stage-C, Stage-D and per-neuron GMM figures
  pipeline.py    the four stages (Stage D optional)
  neuron_gmm.py  per-neuron GMMs with the weighted fitter imported from ../mixture
scripts/     run_pipeline.py (the pipeline entry point), compare_trials.py, fit_neuron_gmms.py
docs/        pipeline.md, assignment.md, neuron_gmm.md
tests/       pytest unit tests
jobs/        env.sh (job environment: conda env, 1 BLAS thread) and the unit-test Slurm script
output/      pipeline outputs per dataset (clouds, trials); ignored by git
gmm_output/  per-neuron GMM outputs, flat files overwritten by every run; ignored by git
log/         Slurm logs; ignored by git
```
