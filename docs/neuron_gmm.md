# Per-neuron GMMs with the ../mixture fitter

`scripts/fit_neuron_gmms.py` is an analysis beside the pipeline. It fits the weighted
Gaussian mixture class of `../mixture` (`src/scratch_models/gmm_scratch.py`, `GMMWeighted`;
imported, not copied) to the soma-centred synapse cloud of every selected neuron and
measures how stable those fits are. The library code is `celltype/neuron_gmm.py`.

On this page k is the number of mixture components of one neuron's fit. It has nothing to
do with the k of the clustering (`--k-min`/`--k-max` of `run_pipeline.py`), which is why the
script's options are called `--components-min`, `--components-max` and `--components-step`.

## The fit and the choice of k

The recipe follows the Stage-1 cell-type model in `../mixture/src/modeling/mixture_all.py`:
axes ordered (dx, dz, dy), synapses weighted by the border weight, at most `--max-synapses`
synapses per repetition, a train/test split per repetition, every candidate k fitted on the
training synapses and scored on the test synapses, and k chosen by
`mix_helper.best_number_component`: the smallest k whose held-out scores are not significantly
different (paired Wilcoxon test, alpha 0.05) from those of the best k. That rule prefers
simple models; it does not say the models are equivalent.

One deliberate difference from `mixture_all`: the held-out score is, by default, the
weight-normalised log-likelihood `sum(w log q) / sum(w)` (`--select-by loglik`) rather than
the weighted BIC `p log(sum w) - 2 sum(w log q)` (`--select-by bic`). The BIC treats the
border weights as replicate counts, so multiplying every weight by a constant changes which
k it prefers, and on held-out data its parameter penalty comes on top of the overfitting that
the held-out likelihood already punishes; the normalised log-likelihood has neither problem.
Both scores are stored per candidate and repetition (`test_log_likelihood`, `test_bic`), and
`select_by` is recorded in every selection.

The paired test cannot reject with fewer than 6 pairs, so `--repeats` must be at least 6
(`../mixture` uses 20 or 50; default 20). A fit can fail (EM on an unlucky subsample); a failed fit is a
missing value, and the test runs only on candidates with at least 6 valid repetitions and
on the repetitions valid for all of them, so a candidate that failed in most repetitions
cannot win merely because its few surviving pairs cannot reject. The candidates set aside
are recorded in the selection rule (`valid_repeats` lists the valid fits per candidate); a
neuron with no usable candidate counts as a failed fit.

`--select-tolerance` replaces the paired test by an effect-size rule: the smallest k whose mean
held-out score (over the same repetitions) is within the tolerance of the best candidate's mean.
The tolerance is in the units of `--select-by`, nats per synapse for `loglik` and BIC units for
`bic`; the selection record keeps `select_tolerance`, its rule names the tolerance's choice and,
as a diagnostic, what the test alone would have chosen (the p-values are kept too), and
`selection.png` draws the tolerance as a line on the score curve. The two rules are not combined:
the test accepting a simpler model says only that the repetitions cannot tell it from the best,
not that its loss is small, so taking the simpler of the two choices would let a model whose loss
exceeds the tolerance through whenever the test lacks power. The bound is on the mean loss
estimated over the repetitions (standard error about 0.015 nats per synapse with 20 to 30
repetitions and the pilot's spread), not a bound for the population. The rule exists because
significance alone does not stop on a dendrite. In a 200-dendrite pilot
with the diagonal model (20 repetitions, 3 starts, k = 1, 4, ..., 16) the held-out log-likelihood
rose by 0.56, 0.25, 0.20, 0.16 and 0.13 nats per synapse at the successive 3-component steps
(medians over neurons, positive for every neuron) against a spread over repetitions of about
0.07, so every step was significant (p = 2e-6, the smallest value 20 pairs can give) and every
neuron chose the largest candidate its synapse count allowed; the held-out BIC did the same for
93% of them. More repetitions make smaller gains significant and push the choice further up, so
on clouds without a plateau the tolerance, not the number of repetitions, decides how many
components a neuron gets. It is a resolution choice and belongs beside the results.

`--covariance` picks the covariance form: `diagonal` (what `../mixture` uses for its
relative-coordinate fits), `diagonal_constraint` (its cylindrical variant: one lateral centre
shared by all components, one lateral variance per component) or `normal` (full
covariances).

Neurons are selected like a pipeline run (`--n-neurons`, `--min-synapses`,
`--min-border-distance`, `--seed`), so `gmm/<side>/neurons.csv` joins the pipeline's
`cluster_<side>/neurons.csv` on `pt_root_id`. Neurons are fitted in parallel over the allocated CPUs (each
neuron's own fits run one after another), and the script logs the fitting plan (candidate k
per neuron, EM fits per neuron) and then one line per finished neuron with its chosen k,
fitting time and the estimated time left.

### Which numbers of components to try

Candidates are limited to k with at least `--min-synapses-per-parameter` (20) synapses per
free parameter in the **training part** of a repetition (80% of the synapses used, so the
guard holds for every fit that enters the selection, not only for the final fit on all of
them), as in `../mixture`. A k-component 3-D mixture has 7k - 1 free parameters with
`diagonal`, 4k + 1 with `diagonal_constraint` and 10k - 1 with `normal`, so the smallest cloud
that admits a candidate is

| k | `diagonal` | `diagonal_constraint` | `normal` |
|---|---|---|---|
| 2 | 325 | 225 | 475 |
| 4 | 675 | 425 | 975 |
| 7 | 1200 | 725 | 1725 |
| 10 | 1725 | 1025 | 2475 |
| 16 | 2775 | 1625 | 3975 |
| 19 | 3300 | 1925 | 4725 |

and a cloud of 2000 synapses supports k up to 11 (`diagonal`), 19 (`diagonal_constraint`) or 8
(`normal`). The default range is `--components-min 1 --components-max 16 --components-step 3`
(1, 4, 7, 10, 13, 16), chosen for dendrites with the diagonal model: the median proofread
dendrite has 2400 synapses and supports up to about k = 13, and `--components-step` keeps the
cost of the wide range down. It is wrong for axons: the median proofread axon has 400 synapses,
so under the default range most axons have k = 1 as their only candidate (k = 4 needs 675). Run
the axon side with `--components-min 1 --components-max 4 --components-step 1` (`../mixture`
tries 2 to 4 for clouds under 10k synapses). A cloud that supports none of the candidates is
fitted with the smallest one and flagged `underpowered`.

Because the guard depends on the cloud, so does the candidate set, and when the held-out score
never plateaus the chosen k follows the cloud size rather than its structure (the pilot above:
k = 10 for 2047 to 2228 synapses, 13 for 2272 to 2651, 16 from 2850 up). For a candidate set
that is the same for every neuron, set `--components-max` at the guard's limit for
`--min-synapses`: with `diagonal_constraint` and `--min-synapses 2000` that is
`--components-max 19`.

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
between-neuron scale comes from `--between-pairs` (2000) random pairs of neurons: `w2_repeat`
compares one repeat model of each neuron, fitted exactly like the models behind the within
distances (same training-subsample size, same EM procedure), and is the reference the summary
and the figure use; `w2` compares the final models fitted on all synapses.

What the within distance is and is not: it mixes three sources of variation at once, the
training subsample, the EM start and the local optimum that start reaches, all at the chosen
k; the uncertainty of the choice of k itself is not in it (`--reruns` covers that). It is
therefore the instability of the fit, not a complete sampling uncertainty. The pairwise
distances of one neuron (190 for 20 repeat models) share models and are not independent
replicates, which is why only their median and maximum are reported. And a stable fit is not
necessarily a good one: a neuron that always comes out as one wide Gaussian is perfectly
stable and may be underfitted.

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
  for every component, and a start can stop in a poor optimum (for example both components
  wide and centred on a cloud with two separated depth bands, far from the other fits and
  with a lower log-likelihood; how often this happens depends on the data and on the fitter,
  whose `diagonal_constraint` M-step was corrected in `../mixture` on 2026-10-25). The rerun
  record keeps each final model's log-likelihood so such cases can be told from genuine
  ambiguity, and `--starts N` (default 1, as in `../mixture`) keeps the best of N EM starts
  per fit, which removes them at N times the cost.

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
- The selection pieces `supported_k`, `usable_repeats` and `choose_k(candidates, scores,
  tolerance)`.

Component parameters keep the fitter's (dx, dz, dy) axis order.

## Outputs (`output/<dataset>/gmm/<side>/`)

Outputs go to `output/<dataset>/gmm/<side>/`, beside the pipeline's `trials/`. Every run
overwrites the previous one; the command and options are recorded in the summary.

| File | Content |
|---|---|
| `neurons.csv` | per neuron: covariates, labels, chosen number of components (`n_components`), log-likelihood, BIC, spread, `w2_within_median`, `w2_within_max`, `w2_to_final_mean`, `spread_um`, `w2_within_rel`, `n_stability_models`, `w2_rerun_median`, `rerun_k_agreement` |
| `models.json` | every final mixture (weights, means, covariances) with its k-selection and stability records |
| `stability.json` | the repeat models at the chosen k and the rerun models of every neuron |
| `between.csv` | random pairs of neurons: W2 between the final models (`w2`) and between one repeat model of each (`w2_repeat`, matched to the within-neuron distances) |
| `summary.json` | options, `n_components_counts`, medians, a `stability` block (within median, between median of the repeat models and of the final models, their ratio, the fraction of neurons whose within-distance is below the 10th percentile of the between-distances, rerun agreement), failures, time |
| `selection.png` | counts of the chosen number of components and the held-out selection-score curve (the score of `--select-by`, relative to each neuron's best), with the selection tolerance as a line when one was used |
| `stability.png` | within- vs between-neuron distances, within-distance vs synapse count and vs k |

## Examples

```bash
source jobs/env.sh
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite --n-neurons 200 --starts 3          # default components 1-16 by 3
$PYTHON -u scripts/fit_neuron_gmms.py --sides axon --n-neurons 1000 --components-max 4 --components-step 1 --min-border-distance 100
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite --n-neurons 50 --reruns 2           # rerun agreement
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite --n-neurons 200 --covariance diagonal_constraint
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite --min-synapses 2000 --covariance diagonal_constraint --components-max 19 --components-step 2 --select-tolerance 0.1   # same candidates for every neuron, effect-size rule
```
