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
| A | `extract_clouds` | All synapses of each proofread neuron, regardless of partner type, as soma-centred offsets `(dx, dy, dz)` in um with border-compensation weights; lateral cutoff 250 um | `clouds_{axon,dendrite}.npz` |
| validate (optional) | `validate_neurons` | Disjoint synapse halves of the same neuron cohort. A selects k using its internal bootstrap consensus; B independently refits consensus at A's k. Reports ARI with coverage, per-cluster overlap, and B distances grouped by A labels only. Points method only; see [split_cluster.md](split_cluster.md) | `trials/validate_<side>/` |
| B | `compute_exact_distances`, `compute_distances` | `--method points`: the exact 2-Wasserstein distance between the complete soma-relative synapse clouds of two neurons, every synapse with its border-compensation weight as mass, solved as an optimal transport problem (so the in-plane direction of an arbor counts); written once as `exact.npz`. Then each bootstrap replicate: the same clouds with bootstrap multiplicities (n draws with replacement from a neuron's n synapses), giving a distance matrix plus each neuron's self-distance between two independent draws (noise floor). `--method shared`: MW2 between mixing weights over one shared Gaussian mixture per side (`fit_shared_model`), see below | `trials/dist_<side>/exact.npz`, `rep<b>.npz`; `trials/shared_<side>.npz` |
| C | `cluster_neurons` | Each replicate is clustered (average linkage) into k major clusters of at least `--min-cluster-size` neurons, smaller clusters becoming outliers (cluster 0); the consensus labels come from clustering the co-clustering matrix. Reports per-replicate ARI against the consensus, PAC (ambiguity), AMI/ARI with dataset labels, confound checks (soma depth, synapse count, border weight, distance to border) | `trials/cluster_<side>/` |

## Outputs and options

The clouds are shared; everything else goes to `output/<dataset>/trials/`, one set of outputs
per dataset. Its `config.json` records the Stage B and C parameters of the last run; a run
with other parameters recomputes the stages it runs and overwrites the previous files, so
files of different settings never mix. With the same parameters, finished outputs are
skipped unless `--overwrite`. `--n-reps` is the exception: a replicate depends only on the
seed and its index, so a smaller count deletes the replicate files beyond it and a larger
one adds the missing ones, and Stage C runs again on exactly the requested replicates. Stage C
is skipped only when its `summary.json` was computed with the same parameters from the
replicate files now present (`pipeline.cluster_is_current`), so pruning or adding replicates
in one run and clustering in a later one still reclusters.

- `--sides` takes one or both sides (default: both).
- `--max-synapses N` (points method; default none) caps clouds above N synapses by a uniform
  subsample without replacement, the masses staying the weights. Every synapse is used by
  default; the cap exists because a pair of clouds of n synapses costs about n^2 memory and
  a little more than n^2 time, see "Cost" below.
- `--n-neurons K` clusters only K randomly chosen neurons per side (the same K neurons in
  every replicate).
- `--min-border-distance` (um, default 0) excludes neurons whose soma is closer than this to
  the volume border; their arbors are truncated and form spurious clusters. Median soma
  distance to the border is about 170 to 200 um, so useful values are 50 to 100 um.
- `--min-cluster-size` (default 5): clusters with fewer neurons are reported as outliers
  (cluster 0, "out" in figures) and do not count towards k.
- The reported k (when `--k` is not given) is the largest k whose median per-replicate ARI is
  within 0.03 of the best and whose PAC is within 0.05 of the lowest PAC among those. This
  heuristic prefers resolution unless ambiguity rises; it does not establish discrete
  biological types. Candidates >= 3 are preferred, with a fallback to reachable candidates.
  The rule and its candidates are recorded under `k_selection` in `summary.json`.
  Full-cloud and split-half fits share `clustering.fit_consensus`. An unreachable fixed
  `--k` raises an error in Stage C and is recorded as a failure in validation.
- `--stages` selects stages from `extract validate distances cluster`; execution follows
  that order regardless of argument order. The default is `extract distances cluster`;
  validation runs only when explicitly listed. Finished outputs are skipped unless
  `--overwrite`; validation resumes after its last completed outer split.
- New distance files carry the Stage-A file's SHA-256. Stage C rejects known stale
  distance inputs and records provenance in its summary. Legacy caches still load, but
  their full fit cannot be linked to validation as verified: rerun `distances cluster`
  with `--overwrite` to establish provenance.

## Two ways to compare neurons (Stage B `--method`)

- `points` (default): nonparametric. The distance is the exact W2 between the two complete
  clouds (`celltype/distances.py`): every synapse is a point with its border weight as mass,
  and the optimal transport between the two discrete measures is solved exactly with POT's
  network simplex (`ot.emd2`), so clouds of different sizes compare directly and nothing is
  resampled. Bootstrap replicates reweight each cloud by multinomial multiplicities (n draws
  from its n synapses), so a replicate's distance matrix carries the sampling noise of the
  synapses, and the self-distance between two independent draws of the same neuron is its
  noise floor.
- `shared`: parametric with a shared vocabulary (`celltype/shared.py`). One Gaussian mixture
  with `--shared-k` full-covariance components is fitted to the pooled synapses of the
  clustered neurons of a side (`--shared-fit-points` synapses drawn with probability
  proportional to the border weight; saved as `trials/shared_<side>.npz`). Each
  neuron is then described by its mixing weights over these fixed components, estimated by
  EM on the weights only with a Dirichlet prior `--shared-alpha * pi` (`pi` = global
  proportions, alpha in synapses), which keeps neurons with few synapses stable by shrinking
  them towards the population instead of fitting a mixture of their own. Two neurons are
  compared with MW2: the exact discrete optimal transport between their weight vectors with
  the closed-form squared W2 between the shared Gaussians as ground cost, an upper bound of
  the W2 between the two mixture densities. Replicates bootstrap each neuron's synapses before
  estimating the weights, so the stability and noise-floor machinery is the same.

Both methods write the same replicate files. For the points method Stage C takes the exact
matrix as the distance (tree, noise floor, `distance.npy`) and the replicates for the
consensus and its stability; for the shared method the replicate mean stands in
(`distance_matrix` in `summary.json`).

## Cost

One exact W2 between two clouds of n synapses each (POT's network simplex, one CPU, 3-D
Gaussian test clouds):

| n | 200 | 400 | 1000 | 2000 | 3000 | 5000 | 8000 |
|---|---|---|---|---|---|---|---|
| seconds per pair | 0.01 | 0.03 | 0.23 | 1.1 | 2.9 | 9.3 | 28 |

The time grows a little faster than n^2 and the cost matrix takes 8 n^2 bytes (200 MB at
5000 synapses). A bootstrap replicate has about 63% of the synapses as distinct points and
costs about a third of the exact matrix per pair. The median proofread axon has 400 synapses
(90th percentile 2000, max 12k), the median dendrite 2400 (90th percentile 4800, max 25k).
Rough totals on 10 CPUs for the exact matrix plus 10 replicates: 1000 axons about 10 h, all
1.7k axons about 30 h; 1000 dendrites about 140 h, or about 35 h with `--max-synapses 2000
--n-reps 5`, and 500 dendrites with that cap about 9 h. So cluster the dendrite side on a
few hundred neurons or with the cap. Stage A needs about 96 GB of memory, Stages B and C a
few GB plus the cost matrices of the pairs in flight.

## Stage C outputs (`trials/cluster_<side>/`)

| File | Content |
|---|---|
| `neurons.csv` | per neuron: covariates, dataset labels, consensus cluster for every reachable k (0 = outlier), self-distance, nearest-neighbour distance, co-clustering stability |
| `summary.json` | which distance matrix was clustered (`distance_matrix`: `exact` or `replicate_mean`), noise floor, chosen k with the selection rule, cluster sizes, per-k stability (median, quartiles, per-replicate ARI), PAC, outlier counts, AMI/ARI with dataset labels, confound statistics (`dist_border_um` is the soma distance to the volume border); only k whose consensus reaches k major clusters are listed (`k_reached`) |
| `distance.npy` | the exact distance matrix of the complete clouds (points method) or the replicate mean (shared method) |
| `k_curves.png` | per-replicate stability (dots, median, IQR), PAC and AMI with dataset labels vs k |
| `co_clustering.png` | co-clustering matrix at the chosen k, sorted by consensus cluster |
| `tree_heatmap.png` | dendrogram and heatmap of the clustered distance matrix (`distance.npy`) with cell_type and cluster barcodes |
| `noise_floor.png`, `contingency.png`, `covariates.png` | self vs nearest-neighbour distance, cell_type x cluster counts, covariates per cluster |

## Caveat

Features use every synapse of a neuron. That is fine for exploration, but when the new
labels are later used to predict held-out connections, the clouds must be rebuilt inside
each cross-validation fold from training-side synapses only.
