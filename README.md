# Label-free cell types from synapse anatomy

Cluster proofread neurons by the spatial distribution of their **own** synapses,
without using the dataset cell-type labels. Axons (outgoing synapses, cells with
`status_axon == "extended"`) and dendrites (incoming synapses, cells with
`full_dendrite`) are always handled separately, so each side gets its own
taxonomy. The dataset labels are kept only for comparison afterwards.

Inputs are the cell and synapse tables used by the Stage-1 GMM fit in
`../mixture` (`microns_v1dd/sven/`). Nothing in `../mixture` is modified.

## Pipeline

| Stage | Script | What it does | Output (`output/<dataset>/`) |
|---|---|---|---|
| A | `scripts/01_extract_clouds.py` | All synapses of each proofread neuron, regardless of partner type, as soma-centred offsets `(dx, dy, dz)` with border-compensation weights; lateral cutoff 250 um | `clouds_{axon,dendrite}.npz` |
| B | `scripts/02_compute_distances.py` | One bootstrap replicate: weighted resampling of each neuron, then pairwise 2-Wasserstein distances (`points`: exact OT on the points; `shared_gmm`: mixture-Wasserstein on a shared GMM basis) plus each neuron's self-distance (noise floor) | `dist_<side>_<method>_<coords>_<depth>/rep<b>.npz` |
| C | `scripts/03_cluster_neurons.py` | Average-linkage clustering of the mean distance matrix, bootstrap stability per k, AMI/ARI with dataset labels, confound checks (soma depth, synapse count, border weight, distance to border) | `cluster_<side>_<method>_<coords>_<depth>/` |

Options for Stages B and C:

- `--coords xyz` (default) compares the 3-D soma-centred offsets `(dx, dy, dz)`, so the
  in-plane direction of an arbor counts. `--coords cylindrical` uses `(r, dy)` with
  `r = hypot(dx, dz)`, which ignores rotation about the soma's vertical axis (same
  assumption as the cylindrical GMM in `../mixture`).
- `--depth-mode relative` (default) compares arbor shape only; `--depth-mode absolute`
  adds the soma depth back, so cells in different layers stay apart.

## Layout

```
celltype/    library code (no command-line parsing)
  config.py      paths, dataset tables, output layout, Slurm-aware CPU count
  clouds.py      extraction and CSR reader for per-neuron synapse clouds
  distances.py   resampling, W2 between point clouds and between GMM mixtures
  clustering.py  clustering, stability, noise floor, label and confound checks
  plotting.py    Stage-C figures
scripts/     command-line entry points for each stage
tests/       pytest unit tests
jobs/        Slurm scripts (account amath, partition cpu-g2)
```

## Running (on klone, through Slurm only)

```bash
sbatch jobs/unit_tests.slurm              # pytest unit tests
bash jobs/submit_all.sh microns           # Stage A, then B + C for all 8 combinations (xyz)
bash jobs/submit_all.sh microns cylindrical
```

Single combination:

```bash
sbatch --export=ALL,DATASET=microns jobs/extract.slurm
sbatch --export=ALL,DATASET=microns,SIDE=axon,METHOD=points,DEPTH=relative jobs/distances.slurm
sbatch --export=ALL,DATASET=microns,SIDE=axon,METHOD=points,DEPTH=relative jobs/cluster.slurm
```

## Caveat

Features use every synapse of a neuron. That is fine for exploration, but when
the new labels are later used to predict held-out connections, the clouds must
be rebuilt inside each cross-validation fold from training-side synapses only.
