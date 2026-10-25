# Per-neuron GMMs with the ../mixture fitter

`scripts/fit_neuron_gmms.py` is an analysis beside the pipeline. It fits the weighted
Gaussian mixture class of `../mixture` (`src/scratch_models/gmm_scratch.py`, `GMMWeighted`;
imported, not copied) to the soma-centred synapse cloud of every selected neuron and
measures how stable those fits are. The library code is `celltype/neuron_gmm.py`.

## The fit and the choice of k

The recipe is that of the Stage-1 cell-type model in `../mixture/src/modeling/mixture_all.py`:
axes ordered (dx, dz, dy), synapses weighted by the border weight, at most `--max-synapses`
synapses per repetition, a train/test split per repetition, every candidate k fitted on the
training synapses and scored by the weighted BIC on the test synapses, and k chosen by
`mix_helper.best_number_component`: the smallest k whose test BICs are not significantly
different (paired Wilcoxon test) from those of the k with the lowest mean BIC. That test
cannot reject with fewer than 6 repetitions, so `--repeats` must be at least 6 (`../mixture`
uses 20 or 50; default 20).

`--method` picks the covariance: `diagonal` (what `../mixture` uses for its
relative-coordinate fits), `diagonal_constraint` (its cylindrical variant: one lateral centre
shared by all components, one lateral variance per component) or `normal` (full
covariances).

Neurons are selected like a pipeline trial (`--n-neurons`, `--min-synapses`,
`--min-border-distance`, `--seed`), so `<side>_neurons.csv` joins a trial's
`neurons.csv` on `pt_root_id`. Neurons are fitted in parallel over the allocated CPUs (each
neuron's own fits run one after another), and the script logs the fitting plan (candidate k
per neuron, EM fits per neuron) and then one line per finished neuron with its chosen k,
fitting time and the estimated time left.

### Which k to try

Candidates are limited to k with at least `--min-synapses-per-parameter` (20) synapses per
free parameter, as in `../mixture`; a diagonal k-component mixture has 7k - 1 parameters, so
k = 2, 3, 4 need 260, 400, 540 synapses and k = 16 needs 2220. The default range is
`--k-min 1 --k-max 16 --k-step 3` (1, 4, 7, 10, 13, 16), chosen for dendrites: the median
proofread dendrite has 2400 synapses and supports up to about k = 17, and `--k-step` keeps
the cost of the wide range down. It is wrong for axons: the median proofread axon has 400
synapses, so under the default range two thirds of the axons have k = 1 as their only
candidate (k = 4 needs 540). Run the axon side with `--k-min 1 --k-max 4 --k-step 1`
(`../mixture` tries 2 to 4 for clouds under 10k synapses). A cloud that supports none of
the candidates is fitted with the smallest one and flagged `underpowered`.

## Fitting stability

A cell-type model in `../mixture` pools thousands of synapses; a single neuron has a few
hundred, so every neuron's fit is also checked for stability. The models fitted at the
chosen k during k selection (one per repetition, each on its own 80% training subsample and
EM start, so `--repeats` of them) are compared with the mixture Wasserstein distance of
`../mixture` (`distributions_handler.component_ot_distance`, imported: optimal transport
between the Gaussian components with the squared Gaussian W2 as cost, the `"ot"` metric of
its `compare_gmm`, where the same comparison across CV folds is the within-type stability
reference). Per neuron the record holds the distances between every pair of those models
and from each to the final model, the final model's total spread (root of
`mixture_total_variance` summed over the axes, the RMS distance of a synapse from the
mixture mean) so that `w2_within_rel` = median within-distance / spread is the instability
relative to the cloud's size, and the median within-distance at every candidate k. The
between-neuron scale comes from `--between-pairs` (2000) random pairs of neurons' final
models.

`--reruns N` additionally reruns the whole selection N times per neuron with independent
seeds (N times the fitting time) and compares the final models the same way, recording how
often they chose the same k; a cloud with at most `--max-synapses` synapses is refitted on
the same synapses each time, so the reruns isolate the EM start and the choice of k, while
the repeat models also carry the subsampling variance.

Two things to keep in mind when reading the distances:

- The mixture Wasserstein distance charges a difference in mixing weights the full squared
  distance between the components, so for a neuron with well-separated components a small
  weight difference between two fits already gives a sizeable distance.
- `GMMWeighted` starts EM once, from k random synapses as means and the global covariance
  for every component, and can stop in a poor optimum (on a cloud with two separated depth
  bands, about one start in six ends with both components wide and centred, far from the
  other fits and with a lower log-likelihood). The rerun record keeps each final model's
  log-likelihood so such cases can be told from genuine ambiguity, and `--starts N`
  (default 1, as in `../mixture`) keeps the best of N EM starts per fit, which removes them
  at N times the cost.

## Library API (`celltype/neuron_gmm.py`)

- `select_and_fit(points, weight, k_values, ..., n_reruns=0)` for any (dx, dy, dz) cloud
  (a neuron, a cluster's pooled synapses, anything).
- `fit_neuron(clouds, index, k_values, ...)` and `fit_neurons(clouds, indices, k_values,
  n_jobs=..., log=print)` for neurons of a clouds file.
- The `NeuronGMM` result with `log_density`, `responsibilities`, `summary`
  (mixture-averaged lateral and depth spread), `repeat_models`, `stability`
  (`stability_summary`) and `to_dict` / `from_dict`; `save_fits`, `load_fits`;
  `save_stability`, `load_stability` for the repeat and rerun models.
- The distances `gmm_w2(a, b, method)`, `within_distances`, `distances_to`,
  `total_spread` and `between_distances(fits, n_pairs)`.

Component parameters keep the fitter's (dx, dz, dy) axis order.

## Outputs (`gmm_output/`)

Outputs go to `gmm_output/`, a flat folder beside the pipeline's `output/` (both ignored by
git). This is test code, so the files carry only the side as prefix and every run overwrites
the previous one; the command and options are recorded in the summary.

| File | Content |
|---|---|
| `<side>_neurons.csv` | per neuron: covariates, labels, chosen k, log-likelihood, BIC, spread, `w2_within_median`, `w2_within_max`, `w2_to_final_mean`, `spread_um`, `w2_within_rel`, `n_stability_models`, `w2_rerun_median`, `rerun_k_agreement` |
| `<side>_models.json` | every final mixture (weights, means, covariances) with its k-selection and stability records |
| `<side>_stability.json` | the repeat models at the chosen k and the rerun models of every neuron |
| `<side>_between.csv` | W2 between the final models of random pairs of neurons |
| `<side>_summary.json` | options, chosen-k counts, medians, a `stability` block (within and between medians, their ratio, the fraction of neurons whose within-distance is below the 10th percentile of the between-distances, rerun agreement), failures, time |
| `<side>_selection.png` | chosen-k counts and the held-out BIC curve |
| `<side>_stability.png` | within- vs between-neuron distances, within-distance vs synapse count and vs k |

## Examples

```bash
source jobs/env.sh
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite --n-neurons 200 --starts 3          # default k 1-16 by 3
$PYTHON -u scripts/fit_neuron_gmms.py --sides axon --n-neurons 1000 --k-max 4 --k-step 1 --min-border-distance 100
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite --n-neurons 50 --reruns 2           # rerun agreement
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite --n-neurons 200 --method diagonal_constraint
```
