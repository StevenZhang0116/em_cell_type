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
| C | `cluster_neurons` | Each replicate is clustered (average linkage) into k major clusters of at least `--min-cluster-size` neurons, smaller clusters becoming outliers (cluster 0); the consensus labels come from clustering the co-clustering matrix. Reports per-replicate ARI against the consensus, PAC on consensus non-outlier pairs, coverage, AMI/ARI with dataset labels, and covariate checks | `trials/cluster_<side>/` |

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
  the volume border. It can reduce boundary exposure but does not guarantee complete
  arbors. Its effect depends on the selected cohort; near-uniform compensation weights
  likewise do not establish absence of truncation.
- `--min-cluster-size` (default 5): clusters with fewer neurons are reported as outliers
  (cluster 0, "out" in figures) and do not count towards k.
- `--k-min` (default 2) and `--k-max` define both the reported and automatic-selection
  range. The reported k (when `--k` is not given) is the largest reachable k whose median per-replicate ARI is
  within 0.03 of the best and whose PAC is within 0.05 of the lowest PAC among those. This
  heuristic prefers resolution unless ambiguity rises; it does not establish discrete
  biological types. There is no additional hidden lower bound of 3. The lower bound,
  reachable candidates, tolerances and rule are recorded under `k_selection` in `summary.json`.
  PAC uses pairs whose endpoints are consensus non-outliers at that k. The old all-pair
  quantity is retained as `pac_all_pairs` for comparison only; rejection can make it
  artificially small. Coverage still changes with k, so neither PAC nor ARI alone should
  be interpreted as evidence for a true number of types. Internal stability ARI uses
  consensus non-outlier neurons and retains replicate label 0 as one group; validation's
  A/B ARI instead uses jointly accepted neurons.
  Full-cloud and split-half fits share `clustering.fit_consensus`. An unreachable fixed
  `--k` raises an error in Stage C and is recorded as a failure in validation.
- Estimator version 2 removes the hidden k>=3 restriction and changes the PAC population.
  Existing Stage-B distances remain reusable. Stage C regenerates its reports when the
  estimator version changes, even with unchanged command-line parameters. Historical
  validation is marked `estimator_mismatch` when compared with the revised full fit;
  its saved split metrics are retained, not relabelled as validation of the new estimator.
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

`jobs/pipeline_dendrite.slurm` configures a 1000-neuron exploratory run: 64 CPUs,
96 GB, a 5-day time limit, 20 full-fit bootstraps and 20 validation splits with
20 inner bootstraps per half (500 synapses/half). It keeps all observed synapses
and the existing eligibility thresholds. It first runs `distances cluster`,
then `validate`, so the full-fit report is available before validation finishes.
The 200-neuron log suggests roughly 75 hours for this configuration, assuming
pair-count and CPU scaling; this is an extrapolation, not a measured 1000-neuron
runtime. Use `TRIAL_REPS=50` via `sbatch --export=ALL,TRIAL_REPS=50` and allow
more time if retaining the previous bootstrap count (see the job comments).

At first submission, the job moves an existing different-size `trials/` run to
`trials_<old size>/` (currently `trials_200/`), refusing to overwrite an archive.
It then writes the 1000-neuron run to `trials/`. Subsequent submissions with the
same settings resume completed matrices/splits. The 1000 neurons are selected
afresh using the same seed and eligibility rules; the sample need not contain
every neuron from the 200-neuron sample. Do not run both cohort sizes concurrently
against the same `trials/` directory.

## Stage C outputs (`trials/cluster_<side>/`)

| File | Content |
|---|---|
| `neurons.csv` | per neuron: covariates, dataset labels, consensus cluster for every reachable k (0 = outlier), self-distance, nearest-neighbour distance, co-clustering stability |
| `summary.json` | estimator version, distance-matrix source, noise floor, explicit k-selection rule, cluster sizes at each reachable k, stability, both PAC populations, outlier counts, label agreement and covariate statistics |
| `k_summary.csv` | one row per requested k, including k=2 and k=3 with the default lower bound: selection/reachability, outlier count, coverage, median stability ARI, both PAC populations and cell-type AMI |
| `distance.npy` | the exact distance matrix of the complete clouds (points method) or the replicate mean (shared method) |
| `k_curves.png` | per-replicate stability, both PAC populations, label AMI and consensus outlier counts vs k |
| `co_clustering.png` | co-clustering matrix at the chosen k, sorted by consensus cluster |
| `tree_heatmap.png` | monochrome dendrogram and heatmap of `distance.npy`, with dataset and consensus-label barcodes; this distance tree is not the consensus tree, and ordinary tree cuts need not match `cut_major` labels |
| `noise_floor.png`, `contingency.png`, `covariates.png` | self vs nearest-neighbour distance, cell_type x cluster counts, varying covariates per cluster (including outlying values); the weight panel states how many neurons have mean weight exactly 1 |

For the 200-neuron MICrONS dendrite cohort (seed 20261019, >=2000 synapses), the
50-um border filter removes only 5 of 9966 count-qualified cells; 179/200 selected
neurons have mean compensation weight exactly 1. These steps have little additional
effect in this cohort. There are 174 excitatory and 26 inhibitory neurons; L5ET and
NGC have 4 cells each and BPC has 1, below the minimum cluster size of 5. Failure to
recover those as separate types is therefore not evidence against their existence.
Vertical synapse profiles and soma depth are strongly associated with the observed
distances; repeatability and label agreement do not establish discrete types or
independence from anatomical gradients. Soma-relative orientation remains unchanged.

## Caveat

Features use every synapse of a neuron. That is fine for exploration, but when the new
labels are later used to predict held-out connections, the clouds must be rebuilt inside
each cross-validation fold from training-side synapses only.
