# The pipeline: Stages A to C

`scripts/run_pipeline.py` runs the stages below; the library code is in `celltype/`
(no command-line parsing there). Inputs are the cell and synapse tables used by the
Stage-1 GMM fit in `../mixture` (`microns_v1dd/sven/`); nothing in `../mixture` is
modified. Axons (outgoing synapses, cells with `status_axon == "extended"`) and dendrites
(incoming synapses, cells with `full_dendrite`) are always handled separately, so each
side gets its own taxonomy. The dataset cell-type labels are used only for comparison
afterwards.

## Stages

| Stage | Function (`celltype/pipeline.py`) | What it does | Output (`output/<dataset>/`) |
|---|---|---|---|
| A | `extract_clouds` | All synapses of each proofread neuron, regardless of partner type, as soma-centred offsets `(dx, dy, dz)` in um with border-compensation weights; lateral cutoff 250 um. `--all-cells` also extracts the dendrite clouds of every cell in the table | `clouds_{axon,dendrite}.npz`, `clouds_dendrite_all.npz` |
| B | `compute_distances` | Each bootstrap replicate: a neuron-to-neuron distance matrix plus each neuron's self-distance (noise floor). `--method points`: weighted resampling of `--n-points` synapses per neuron (default 200 per axon, 2000 per dendrite), then exact 2-Wasserstein distances between the 3-D soma-relative point clouds (so the in-plane direction of an arbor counts). `--method shared`: MW2 between mixing weights over one shared Gaussian mixture per side (`fit_shared_model`), see below | `trials/<trial>/dist_<side>/rep<b>.npz`, `trials/<trial>/shared_<side>.npz` |
| C | `cluster_neurons` | Each replicate is clustered (average linkage) into k major clusters of at least `--min-cluster-size` neurons, smaller clusters becoming outliers (cluster 0); the consensus labels come from clustering the co-clustering matrix. Reports per-replicate ARI against the consensus, PAC (ambiguity), AMI/ARI with dataset labels, confound checks (soma depth, synapse count, border weight, distance to border) | `trials/<trial>/cluster_<side>/` |
| D (optional) | `assign_neurons` | Only with `--stages ... assign`: turns the consensus clusters into a classifier for neurons that were not clustered, see [assignment.md](assignment.md) | `trials/<trial>/assign_<side>_<query>/` |

## Trials and options

The clouds are shared; everything else goes to a trial folder named after the Stage B and C
parameters, e.g.
`output/microns/trials/n10_rep2_pt200-2000_min100_weighted_seed20261019_average_k2-3_mcs5/`
(parameters also saved in its `config.json`), so runs with different parameters never
overwrite each other.

- `--sides` takes one or both sides (default: both).
- `--n-points` sets how many synapses are resampled per neuron in every replicate: one value
  for both sides or one per side in the order axon, dendrite (default 200 and 2000). The
  trial token is `pt200` when the sides agree and `pt200-2000` otherwise. This is the main
  cost knob, see "Cost" below.
- `--n-neurons K` clusters only K randomly chosen neurons per side (same K neurons in every
  replicate); the remaining proofread neurons can be assigned in Stage D (`--query rest`).
- `--align` rotates every cloud about the soma's vertical axis into a canonical frame before
  anything else (principal axis of the weighted lateral second moment about the soma on +x,
  lateral centroid on the positive side). Without it, two neurons of the same type whose
  arbors point in different horizontal directions are far apart in W2; with it only the
  anisotropy of the arbor counts. Replicate 0 of the points method prints an "alignment
  check": the median ratio of W2 to the smallest W2 over 36 in-plane rotations on 100 random
  pairs (stored in `rep000.npz`). Far above 1 without `--align` means orientation dominates
  the distance; near 1 with `--align` means the canonical frame recovers the rotation optimum.
- `--min-border-distance` (um, default 0) excludes neurons whose soma is closer than this to
  the volume border; their arbors are truncated and form spurious clusters. Median soma
  distance to the border is about 170 to 200 um, so useful values are 50 to 100 um. The filter
  applies to Stage D queries as well.
- `--min-cluster-size` (default 5): clusters with fewer neurons are reported as outliers
  (cluster 0, "out" in figures) and do not count towards k.
- The reported k (when `--k` is not given) is the largest k whose median per-replicate ARI is
  within 0.03 of the best and whose PAC is within 0.05 of the lowest PAC among those: on a flat
  stability curve the extra clusters are real stable splits, so resolution is preferred unless
  ambiguity rises. The rule and its candidates are recorded under `k_selection` in `summary.json`.
- `--stages` selects stages from `extract distances cluster assign`; the default is the first
  three, so Stage D runs only when `assign` is listed. Finished outputs are skipped unless
  `--overwrite`, so an interrupted job can just be resubmitted.

## Two ways to compare neurons (Stage B `--method`)

- `points` (default): nonparametric. Each replicate resamples `--n-points` synapses per
  neuron and the distance is the exact W2 between the two point clouds (`celltype/distances.py`),
  an n x n assignment problem whose time grows roughly with n^3.
- `shared`: parametric with a shared vocabulary (`celltype/shared.py`). One Gaussian mixture
  with `--shared-k` full-covariance components is fitted to the pooled synapses of the
  clustered neurons of a side (`--shared-fit-points` synapses drawn with probability
  proportional to the border weight; saved as `trials/<trial>/shared_<side>.npz`). Each
  neuron is then described by its mixing weights over these fixed components, estimated by
  EM on the weights only with a Dirichlet prior `--shared-alpha * pi` (`pi` = global
  proportions, alpha in synapses), which keeps neurons with few synapses stable by shrinking
  them towards the population instead of fitting a mixture of their own. Two neurons are
  compared with MW2: the exact discrete optimal transport between their weight vectors with
  the closed-form squared W2 between the shared Gaussians as ground cost, an upper bound of
  the W2 between the two mixture densities. Replicates bootstrap each neuron's synapses before
  estimating the weights, so the stability and noise-floor machinery is the same.

Both methods write the same replicate files, so Stages C and D are unchanged (Stage D's
kNN always uses the point-cloud W2). The trial name differs in its third token (`pt200` vs
`shared30_a10`), and `scripts/compare_trials.py` compares two trials on the same neurons:
correlation of the distance matrices, self/nearest ratio, chosen k with stability and PAC,
AMI with `cell_type`, and the ARI between the two consensus labelings.

## Cost

Measured on 1000 neurons with 10 CPUs, one replicate of 200-point clouds takes 3.5 min
(dendrites) to 4.5 min (axons), i.e. 4 to 5 ms per pair per CPU including resampling, so one
replicate of N neurons takes about 2 N^2 ms of CPU time. The exact W2 grows roughly with the
cube of the point count: expect about 16x that at 500 points, 125x at 1000 and 1000x at 2000.
The alignment check of replicate 0 (98 pairs at 37 rotations) and the Stage D query distances
(queries x references x `--assign-reps` pairs) scale the same way. Stage A needs about 96 GB
of memory, Stages B to D a few GB.

## Stage C outputs (`trials/<trial>/cluster_<side>/`)

| File | Content |
|---|---|
| `neurons.csv` | per neuron: covariates, dataset labels, consensus cluster for every reachable k (0 = outlier), self-distance, nearest-neighbour distance, co-clustering stability |
| `summary.json` | noise floor, chosen k with the selection rule, cluster sizes, per-k stability (median, quartiles, per-replicate ARI), PAC, outlier counts, AMI/ARI with dataset labels, confound statistics (`dist_border_um` is the soma distance to the volume border); only k whose consensus reaches k major clusters are listed (`k_reached`) |
| `mean_distance.npy` | replicate-mean distance matrix |
| `k_curves.png` | per-replicate stability (dots, median, IQR), PAC and AMI with dataset labels vs k |
| `co_clustering.png` | co-clustering matrix at the chosen k, sorted by consensus cluster |
| `tree_heatmap.png` | dendrogram of the mean distance matrix with cell_type and cluster barcodes |
| `noise_floor.png`, `contingency.png`, `covariates.png` | self vs nearest-neighbour distance, cell_type x cluster counts, covariates per cluster |

## Caveat

Features use every synapse of a neuron. That is fine for exploration, but when the new
labels are later used to predict held-out connections, the clouds and the templates must be
rebuilt inside each cross-validation fold from training-side synapses only.
