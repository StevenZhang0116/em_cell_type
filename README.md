# Label-free cell types from synapse anatomy

Cluster proofread neurons by the spatial distribution of their **own** synapses,
without using the dataset cell-type labels; optionally (Stage D, off by default)
assign every other neuron to those clusters. Axons (outgoing synapses, cells with `status_axon == "extended"`)
and dendrites (incoming synapses, cells with `full_dendrite`) are always handled
separately, so each side gets its own taxonomy. The dataset labels are kept only
for comparison afterwards.

Inputs are the cell and synapse tables used by the Stage-1 GMM fit in
`../mixture` (`microns_v1dd/sven/`). Nothing in `../mixture` is modified.

## Pipeline

| Stage | Function (`celltype/pipeline.py`) | What it does | Output (`output/<dataset>/`) |
|---|---|---|---|
| A | `extract_clouds` | All synapses of each proofread neuron, regardless of partner type, as soma-centred offsets `(dx, dy, dz)` in um with border-compensation weights; lateral cutoff 250 um. `--all-cells` also extracts the dendrite clouds of every cell in the table | `clouds_{axon,dendrite}.npz`, `clouds_dendrite_all.npz` |
| B | `compute_distances` | Each bootstrap replicate: a neuron-to-neuron distance matrix plus each neuron's self-distance (noise floor). `--method points`: weighted resampling of `--n-points` synapses per neuron (default 200 per axon, 2000 per dendrite), then exact 2-Wasserstein distances between the 3-D soma-relative point clouds (so the in-plane direction of an arbor counts). `--method shared`: MW2 between mixing weights over one shared Gaussian mixture per side (`fit_shared_model`), see below | `trials/<trial>/dist_<side>/rep<b>.npz`, `trials/<trial>/shared_<side>.npz` |
| C | `cluster_neurons` | Each replicate is clustered (average linkage) into k major clusters of at least `--min-cluster-size` neurons, smaller clusters becoming outliers (cluster 0); the consensus labels come from clustering the co-clustering matrix. Reports per-replicate ARI against the consensus, PAC (ambiguity), AMI/ARI with dataset labels, confound checks (soma depth, synapse count, border weight, distance to border) | `trials/<trial>/cluster_<side>/` |
| D (optional) | `assign_neurons` | Only with `--stages ... assign`. Turns the consensus clusters into a classifier for neurons that were not clustered (see below): k-nearest references on W2 and cluster-template GMM likelihood, both calibrated by cross-validation on the clustered neurons | `trials/<trial>/assign_<side>_<query>/` |

All stages run from one script, `scripts/run_pipeline.py`. The clouds are shared;
everything else goes to a trial folder named after the Stage B and C parameters, e.g.
`output/microns/trials/n10_rep2_pt200-2000_min100_weighted_seed20261019_average_k2-3_mcs5/`
(parameters also saved in its `config.json`), so runs with different parameters never
overwrite each other. Options:

- `--sides` takes one or both sides (default: both).
- `--n-points` sets how many synapses are resampled per neuron in every replicate: one value
  for both sides or one per side in the order axon, dendrite (default 200 and 2000). The
  trial token is `pt200` when the sides agree and `pt200-2000` otherwise. This is the main
  cost knob, see the cost note under "Two ways to compare neurons".
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
  an n x n assignment problem whose time grows roughly with n^3. Measured on 1000 neurons
  with 10 CPUs: one replicate of 200-point clouds takes 3.5 min (dendrites) to 4.5 min
  (axons), i.e. 4 to 5 ms per pair per CPU including resampling. Expect roughly 16x that at
  500 points, 125x at 1000 and 1000x at 2000; the alignment check of replicate 0 (98 pairs
  at 37 rotations) and the Stage D query distances scale the same way.
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

## Stage D (optional): assigning neurons to the clusters

Stage D is not part of the default run; add `assign` to `--stages` to use it. The
clustering is quadratic in the number of neurons, so the dendrite side (15k proofread
neurons) can be clustered on a subset and the rest assigned. Queries are
`--query rest` (proofread neurons of the same side that were not clustered), `--query all`
(the dendrite of every cell in the table, from `clouds_dendrite_all.npz`; dendrite side
only) or `--query none` (only cross-validate and fit the templates). Two methods give a
posterior over the k major clusters plus a "none" label 0:

- **knn**: the query is resampled like a Stage-B replicate, its W2 distance to every
  clustered non-outlier neuron is averaged over `--assign-reps` resamples, and the
  `--knn` nearest references vote with weight `exp(-d^2 / tau)`.
- **template**: each cluster's synapses are pooled (soma-centred) and fitted with the
  cylindrically constrained weighted GMM of `../mixture` (`celltype/templates.py`: diagonal
  covariances, one shared lateral centre, isotropic lateral variance, K by weighted BIC up
  to `--template-k-max`). The query's score under a cluster is its weighted mean per-synapse
  log-density. Because the synapses of one neuron are not independent draws from its type's
  template, the posterior is tempered, `exp(T * score)`, with T acting as an effective
  number of independent synapses. Note that the template is rotationally symmetric about
  its centre while the W2 distance is not, so clusters that differ only in the in-plane
  direction of their arbors are seen by knn but not by the templates; the held-out
  agreement between the methods shows whether this matters.

Neither method uses the soma depth: like the clustering, the assignment depends on the
synapse cloud alone.

`tau`, `T` and the rejection thresholds (`--reject-quantile`, default 0.95, of the held-out
nearest-reference distance and of the held-out own-cluster score) are chosen by minimising
the held-out negative log-likelihood of the consensus labels in a `--cv-folds` cross-
validation of the clustered neurons (kNN pool and templates refitted without the held-out
fold). The same held-out posteriors give the reported accuracy, calibration error and
confusion matrices, and the agreement between the methods.

Caveats: axon and dendrite clusters are separate taxonomies, so a neuron gets one label
per side and the two must not be multiplied; a dendrite-only neuron has no axon label.
Non-proofread dendrites (`--query all`) may be truncated or merged.

### Stage D outputs (`trials/<trial>/assign_<side>_<query>/`)

| File | Content |
|---|---|
| `summary.json` | k, reference/query counts, calibration (`tau_um2`, `temperature`, rejection thresholds, NLL curves), held-out metrics per method (accuracy, NLL, ECE, confusion, rejected fractions), method agreement, template sizes and BIC per K, query label counts and AMI with `cell_type` |
| `reference_cv.csv` | per clustered neuron: consensus label, fold, held-out nearest-reference distance, template scores, posteriors and hard labels of every method |
| `queries.csv` | the same for every query neuron, with its covariates and dataset labels |
| `templates.json` | the fitted template of every cluster |
| `cv_confusion.png`, `cv_reliability.png` | held-out confusion matrices; accuracy vs posterior confidence |
| `templates.png`, `queries.png` | template densities over member synapses; query confidence and label counts |

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

## Per-neuron GMMs with the ../mixture fitter

`scripts/fit_neuron_gmms.py` is an analysis beside the pipeline. It fits the weighted Gaussian
mixture class of `../mixture` (`src/scratch_models/gmm_scratch.py`, `GMMWeighted`; imported,
not copied) to the soma-centred synapse cloud of every selected neuron, following the recipe
of the Stage-1 cell-type model in `../mixture/src/modeling/mixture_all.py`: axes ordered
(dx, dz, dy), synapses weighted by the border weight, at most `--max-synapses` synapses per
repetition, a train/test split per repetition, every candidate k fitted on the training
synapses and scored by the weighted BIC on the test synapses, and k chosen by
`mix_helper.best_number_component`: the smallest k whose test BICs are not significantly
different (paired Wilcoxon test) from those of the k with the lowest mean BIC. That test cannot
reject with fewer than 6 repetitions, so `--repeats` must be at least 6 (`../mixture` uses 20
or 50; default 20). Candidates are limited to k with at least `--min-synapses-per-parameter`
(20) synapses per free parameter, as in `../mixture`, so an axon with 380 synapses is usually
fitted with 2 or 3 diagonal components. The default range is `--k-min 1 --k-max 16 --k-step 3`
(1, 4, 7, 10, 13, 16; a dendrite with 2400 synapses supports up to about 17 diagonal components,
and `--k-step` keeps the cost of a wide range down); `--k-min 2 --k-max 4 --k-step 1` is what
`../mixture` tries for clouds under 10k synapses. `--method` picks the covariance: `diagonal`
(what `../mixture` uses for its relative-coordinate fits), `diagonal_constraint` (its
cylindrical variant: one lateral centre shared by all components, one lateral variance per
component) or `normal` (full covariances). Neurons are selected like a pipeline trial
(`--n-neurons`, `--min-synapses`, `--min-border-distance`, `--seed`, `--align`), so
`<side>_neurons.csv` joins a trial's `neurons.csv` on `pt_root_id`.

**Fitting stability.** A cell-type model in `../mixture` pools thousands of synapses; a single
neuron has a few hundred, so every neuron's fit is also checked for stability. The models fitted
at the chosen k during k selection (one per repetition, each on its own 80% training subsample
and EM start, so `--repeats` of them) are compared with the mixture Wasserstein distance of
`../mixture` (`distributions_handler.component_ot_distance`, imported: optimal transport between
the Gaussian components with the squared Gaussian W2 as cost, the `"ot"` metric of its
`compare_gmm`, where the same comparison across CV folds is the within-type stability reference).
Per neuron the record holds the distances between every pair of those models and from each to
the final model, the final model's total spread (root of `mixture_total_variance` summed over the
axes, the RMS distance of a synapse from the mixture mean) so that `w2_within_rel` =
median within-distance / spread is the instability relative to the cloud's size, and the median
within-distance at every candidate k. The between-neuron scale comes from `--between-pairs`
(2000) random pairs of neurons' final models. `--reruns N` additionally reruns the whole
selection N times per neuron with independent seeds (N times the fitting time) and compares the
final models the same way, recording how often they chose the same k; a cloud with at most
`--max-synapses` synapses is refitted on the same synapses each time, so the reruns isolate the
EM start and the choice of k, while the repeat models also carry the subsampling variance. Note
that the mixture Wasserstein distance charges a difference in mixing weights the full squared
distance between the components, so for a neuron with well-separated components a small weight
difference between two fits already gives a sizeable distance. `GMMWeighted` starts EM once, from
k random synapses as means and the global covariance for every component, and can stop in a poor
optimum (on a cloud with two separated depth bands, about one start in six ends with both
components wide and centred, far from the other fits and with a lower log-likelihood); the
rerun record keeps each final model's log-likelihood so such cases can be told from genuine
ambiguity, and `--starts N` (default 1, as in `../mixture`) keeps the best of N EM starts per
fit, which removes them at N times the cost.

The library API is `celltype/neuron_gmm.py`: `select_and_fit(points, weight, k_values, ...,
n_reruns=0)` for any (dx, dy, dz) cloud (a neuron, a cluster's pooled synapses, anything),
`fit_neuron(clouds, index, k_values, ...)` and `fit_neurons(clouds, indices, k_values,
n_jobs=...)` for neurons of a clouds file, the `NeuronGMM` result with `log_density`,
`responsibilities`, `summary` (mixture-averaged lateral and depth spread), `repeat_models`,
`stability` (`stability_summary`) and `to_dict` / `from_dict` (`save_fits`, `load_fits`;
`save_stability`, `load_stability` for the repeat and rerun models), and the distances `gmm_w2(a,
b, method)`, `within_distances`, `distances_to`, `total_spread` and `between_distances(fits,
n_pairs)`. Component parameters keep the fitter's (dx, dz, dy) axis order. Outputs go to
`gmm_output/`, a flat folder beside the pipeline's `output/` (both ignored by git); this is test
code, so the files carry only the side as prefix and every run overwrites the previous one (the
command and options are recorded in the summary): `<side>_neurons.csv` (covariates, labels, chosen k,
log-likelihood, BIC, spread, `w2_within_median`, `w2_within_max`, `w2_to_final_mean`,
`spread_um`, `w2_within_rel`, `n_stability_models`, `w2_rerun_median`, `rerun_k_agreement`),
`<side>_models.json` (final parameters, k-selection and stability records),
`<side>_stability.json` (the repeat and rerun models), `<side>_between.csv` (between-neuron
distances), `<side>_summary.json` (with a `stability` block: within and between medians, their
ratio, the fraction of neurons whose within-distance is below the 10th percentile of the
between-distances, rerun agreement), `<side>_selection.png` (chosen-k counts, held-out BIC
curve) and `<side>_stability.png` (within- vs between-neuron distances, within-distance vs
synapse count and vs k).

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
  plotting.py    Stage-C and Stage-D figures
  pipeline.py    the four stages (Stage D optional)
  neuron_gmm.py  per-neuron GMMs with the weighted fitter imported from ../mixture
scripts/     run_pipeline.py (the pipeline entry point), compare_trials.py, fit_neuron_gmms.py
tests/       pytest unit tests
jobs/        env.sh (job environment: conda env, 1 BLAS thread) and the unit-test Slurm script
output/      pipeline outputs per dataset (clouds, trials); ignored by git
gmm_output/  per-neuron GMM outputs, flat files overwritten by every run; ignored by git
log/         Slurm logs; ignored by git
```

## Running (on klone, through Slurm only)

`jobs/env.sh` selects the conda environment, caps BLAS at one thread (parallelism comes from
joblib over `$SLURM_CPUS_PER_TASK`) and changes into the project directory. Run the pipeline
inside a Slurm allocation, either an interactive node (e.g. `salloc --account=amath
--partition=cpu-g2 -c 10 --mem=32G --time=4:00:00`) or an `sbatch` script that sources
`jobs/env.sh`; Stage A needs about 96 GB, Stages B to D a few GB. `log/` is ignored by git.

```bash
sbatch jobs/unit_tests.slurm                                   # pytest unit tests (1 CPU)
source jobs/env.sh                                             # on the allocated node
$PYTHON -u scripts/run_pipeline.py --sides axon                # all 1.7k proofread axons: A, B, C
$PYTHON -u scripts/run_pipeline.py --n-neurons 1000 --n-points 200 --stages distances cluster   # demo, clouds already extracted
$PYTHON -u scripts/run_pipeline.py --n-neurons 1000 --n-points 200 --align --min-border-distance 100 --stages distances cluster
$PYTHON -u scripts/run_pipeline.py --n-neurons 1000 --stages distances cluster --method shared   # same neurons, MW2
$PYTHON scripts/compare_trials.py --trial-a n1000_rep10_pt200_... --trial-b n1000_rep10_pt200_aligned_...
$PYTHON -u scripts/fit_neuron_gmms.py --sides axon --n-neurons 1000 --min-border-distance 100   # per-neuron GMMs
$PYTHON -u scripts/run_pipeline.py --sides dendrite --n-neurons 3000 --n-points 200 500   # cluster 3k dendrites
$PYTHON -u scripts/run_pipeline.py --sides dendrite --n-neurons 3000 --n-points 200 500 --stages assign --query rest   # Stage D
$PYTHON -u scripts/run_pipeline.py --stages extract --all-cells    # dendrite clouds of every cell (96 GB)
```

Pairwise W2 at 200 points costs about 4 ms per pair per CPU (see the cost note above for
other counts), so one replicate of N neurons takes about 2 N^2 ms of CPU time; Stage D costs
(queries x references x `--assign-reps`) pairs.

## Caveat

Features use every synapse of a neuron. That is fine for exploration, but when
the new labels are later used to predict held-out connections, the clouds and the
templates must be rebuilt inside each cross-validation fold from training-side
synapses only.
