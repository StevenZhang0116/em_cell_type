# Label-free cell types from synapse anatomy

Cluster proofread neurons by the spatial distribution of their **own** synapses,
without using the dataset cell-type labels. Axons (outgoing synapses, cells with
`status_axon == "extended"`) and dendrites (incoming synapses, cells with
`full_dendrite`) are always handled separately, so each side gets its own
taxonomy. The dataset labels are kept only for comparison afterwards.

Inputs are the cell and synapse tables used by the Stage-1 GMM fit in
`../mixture` (`microns_v1dd/sven/`). Nothing in `../mixture` is modified.

## Pipeline

| Stage | Function (`celltype/pipeline.py`) | What it does | Output (`output/<dataset>/`) |
|---|---|---|---|
| A | `extract_clouds` | All synapses of each proofread neuron, regardless of partner type, as soma-centred offsets `(dx, dy, dz)` with border-compensation weights; lateral cutoff 250 um | `clouds_{axon,dendrite}.npz` |
| B | `compute_distances` | Each bootstrap replicate: weighted resampling of each neuron, then pairwise 2-Wasserstein distances (exact optimal transport on the soma-relative points) plus each neuron's self-distance (noise floor) | `dist_<side>_<coords>/rep<b>.npz` |
| C | `cluster_neurons` | Average-linkage clustering of the mean distance matrix, bootstrap stability per k, AMI/ARI with dataset labels, confound checks (soma depth, synapse count, border weight, distance to border) | `cluster_<side>_<coords>/` |

All three stages run from one script, `scripts/run_pipeline.py`. Options:

- `--coords xyz` (default) compares the 3-D soma-centred offsets `(dx, dy, dz)`, so the
  in-plane direction of an arbor counts. `--coords cylindrical` uses `(r, dy)` with
  `r = hypot(dx, dz)`, which ignores rotation about the soma's vertical axis (same
  assumption as the cylindrical GMM in `../mixture`).
- `--sides` and `--coords` take several values and the script loops over every
  combination (default: both sides, `xyz`).
- `--n-neurons K` uses only K randomly chosen neurons per side (same K neurons in every
  replicate), for a quick test; its outputs go to separate `*_n<K>` directories.
- `--stages extract distances cluster` selects stages; finished outputs are skipped
  unless `--overwrite`, so an interrupted job can just be resubmitted.

## Layout

```
celltype/    library code (no command-line parsing)
  config.py      paths, dataset tables, output layout, Slurm-aware CPU count
  clouds.py      extraction and CSR reader for per-neuron synapse clouds
  distances.py   resampling and exact W2 between point clouds
  clustering.py  clustering, stability, noise floor, label and confound checks
  plotting.py    Stage-C figures
  pipeline.py    the three stages
scripts/     run_pipeline.py, the single command-line entry point
tests/       pytest unit tests
jobs/        Slurm scripts (account amath, partition cpu-g2)
```

## Running (on klone, through Slurm only)

```bash
sbatch jobs/unit_tests.slurm                               # pytest unit tests
sbatch jobs/pipeline.slurm --dataset microns               # everything: axon + dendrite, xyz
sbatch jobs/pipeline.slurm --coords xyz cylindrical        # also the cylindrical runs
sbatch jobs/pipeline.slurm --sides axon                   # one side only
sbatch jobs/pipeline.slurm --n-neurons 50 --n-reps 2      # quick test on 50 neurons per side
```

The job log is `log/pipeline_<jobid>.txt`.

## Caveat

Features use every synapse of a neuron. That is fine for exploration, but when
the new labels are later used to predict held-out connections, the clouds must
be rebuilt inside each cross-validation fold from training-side synapses only.
