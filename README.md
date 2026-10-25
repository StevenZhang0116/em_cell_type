# Label-free cell types from synapse anatomy

Cluster proofread neurons by the spatial distribution of their **own** synapses, without
using the dataset cell-type labels. Axons and dendrites are handled separately, so each
side gets its own taxonomy; the dataset labels are kept only for comparison afterwards.

Inputs are the cell and synapse tables used by the Stage-1 GMM fit in `../mixture`
(`microns_v1dd/sven/`). Code of `../mixture` is imported where needed, never modified or
copied.

## What it does

| Stage | What | Output |
|---|---|---|
| A | the synapse cloud of every proofread neuron: soma-centred offsets with border-compensation weights | `output/<dataset>/clouds_<side>.npz` |
| validate (optional) | disjoint synapse halves; choose consensus k using A alone, refit B at that k, report agreement, coverage and held-out separation | `trials/validate_<side>/` |
| B | neuron-to-neuron distances: the exact W2 between the complete synapse clouds, plus bootstrap replicates for the noise; or MW2 over one shared Gaussian mixture | `trials/dist_<side>/` |
| C | consensus clustering of the replicates, choice of k, checks against the dataset labels and confounds | `trials/cluster_<side>/` |

Beside the pipeline, `scripts/fit_neuron_gmms.py` fits the weighted Gaussian mixture of
`../mixture` to every neuron and measures how stable those fits are (`gmm_output/`).
`scripts/analyze_split_half.py` measures within-neuron variability using disjoint
synapse halves and compares it with sample-matched between-neuron distances on a
fixed cohort (`output/<dataset>/split_half/`). These are sampling diagnostics, not
significance tests for discrete cell types.

## Documentation

- [docs/pipeline.md](docs/pipeline.md): Stages A to C, outputs and options, the two
  distance methods and their cost, Stage C outputs.
- [docs/neuron_gmm.md](docs/neuron_gmm.md): per-neuron GMMs, which k to try, fitting
  stability, outputs.
- [docs/split_half.md](docs/split_half.md): disjoint halves, matched sample sizes,
  fixed-cohort comparisons, outputs and statistical limits.
- [docs/split_cluster.md](docs/split_cluster.md): clustering validation through
  `run_pipeline.py --stages validate`, A-only selection, resumable checkpoints and a
  dendrite pilot command.

## Running (on klone, inside a Slurm allocation)

`jobs/env.sh` selects the conda environment, caps BLAS at one thread (parallelism comes
from joblib over `$SLURM_CPUS_PER_TASK`) and changes into the project directory. Use an
interactive node (e.g. `salloc --account=amath --partition=cpu-g2 -c 10 --mem=32G
--time=4:00:00`) or an `sbatch` script that sources `jobs/env.sh`; Stage A needs about
96 GB of memory, everything else a few GB.

```bash
sbatch jobs/unit_tests.slurm                                       # pytest unit tests (1 CPU, about 3 min)
source jobs/env.sh                                                 # on the allocated node
$PYTHON -u scripts/run_pipeline.py --stages extract                # Stage A, once (96 GB)
$PYTHON -u scripts/run_pipeline.py --sides axon                    # all proofread axons: Stages B and C
$PYTHON -u scripts/run_pipeline.py --sides axon --n-neurons 1000 --stages distances cluster   # 1000 axons, about 10 h
$PYTHON -u scripts/run_pipeline.py --sides dendrite --n-neurons 500 --max-synapses 2000 --n-reps 5 --stages distances cluster   # 500 dendrites, largest clouds capped
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite --n-neurons 200 --starts 3           # per-neuron GMMs
$PYTHON -u scripts/analyze_split_half.py --sides axon --n-neurons 200 --points-per-half 100 200 --n-reps 100 --between-pairs 1000  # split-half diagnostics
$PYTHON -u scripts/run_pipeline.py --sides dendrite --stages validate --n-neurons 100 --min-synapses 400 --validation-points-per-half 200 --validation-reps 10 --n-reps 5  # clustering validation
```

All outputs of a dataset live in `output/<dataset>/trials/`, one set per dataset; its
`config.json` records the Stage B and C parameters of the last run. A run with other
parameters recomputes the stages it runs and overwrites the previous files; with the same
parameters, finished stages are skipped unless `--overwrite`.

## Layout

```
celltype/    library code (no command-line parsing)
  config.py      paths, dataset tables, output layout, Slurm-aware CPU count
  clouds.py      extraction and CSR reader for per-neuron synapse clouds
  distances.py   exact W2 between complete point clouds, bootstrap replicates
  shared.py      shared-component mixture, per-neuron mixing weights, MW2
  clustering.py  clustering, stability, noise floor, label and confound checks
  plotting.py    Stage-C, per-neuron GMM, split-half and validation figures
  pipeline.py    the stages (validate optional)
  neuron_gmm.py  per-neuron GMMs with the weighted fitter imported from ../mixture
  split_half.py  disjoint, sample-matched self/between W2 diagnostics
  split_cluster.py  held-out validation of bootstrap consensus clustering
scripts/     run_pipeline.py, fit_neuron_gmms.py, analyze_split_half.py
docs/        pipeline.md, neuron_gmm.md, split_half.md, split_cluster.md
tests/       pytest unit tests
jobs/        env.sh (job environment: conda env, 1 BLAS thread), the unit-test and validation Slurm scripts
output/      pipeline outputs per dataset (clouds, trials); ignored by git
gmm_output/  per-neuron GMM outputs, flat files overwritten by every run; ignored by git
log/         Slurm logs; ignored by git
```
