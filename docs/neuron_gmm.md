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

Because the choice moves with the tolerance, every run re-evaluates the tolerance rule at other
tolerances on the stored held-out scores (`--scan-tolerances`; default 0, 0.05, 0.1, 0.15, 0.2
and 0.3 nats per synapse for `loglik`, none for `bic`, plus the run's own tolerance when one
is used; no refitting): `tolerance_scan` in `summary.json` and `tolerance_scan.png` give the number of
neurons per chosen number of components at each tolerance. At tolerance 0 the scan takes the
lowest mean score, not the paired test. `neurons.csv` records per neuron the held-out gains
in the score's units: `gain_first_step` between the two smallest usable candidates,
`gain_total` from the smallest candidate to the best candidate's mean, `gain_beyond_chosen`
from the chosen number to that best (at most the tolerance). In a 1000-dendrite run with
`diagonal_constraint`, candidates 1, 3, ..., 19, 30 repetitions and tolerance 0.1, the median
gain from 1 to 3 components was 0.28 nats per synapse (above 0.1 for 93% of the neurons) and
the median gain beyond 3 was 0.11, so the tolerance cut the population at its median:
tolerances 0.05, 0.1, 0.15 and 0.2 left 218, 434, 558 and 596 of the 1000 neurons with 3
components. The chosen number is a resolution setting read off a smooth curve, not a property
of the neuron.

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

### The fits against the dataset labels

The labels play no part in the fits. `labels.png`, `depth_profiles.png` and `by_cell_type` in
`summary.json` (`label_summary`) group the finished per-neuron results by the dataset's
`cell_type` (the first label column present otherwise) to show what the chosen number of
components and the layout of the mixtures track. In the 1000-dendrite run above the chosen
number rose with soma depth (Spearman 0.63): 3 for L2IT and basket cells, 5 for L3IT and L4IT,
7 for L5IT, L6IT, L6CT and L5ET, 1 for neurogliaform cells, and the total held-out gain over one
component went from 0.09 (NGC) to about 1 nats per synapse (L6CT). With `diagonal_constraint`
all components share one lateral centre, so a mixture is a profile along depth with a radius per
segment, and the number of components counts the segments the cloud has along depth: deep
pyramidal cells with a basal cluster, a thin apical trunk and a tuft hundreds of micrometres
above the soma (a fifth of their mass more than 150 um above it) take more components than
superficial cells, whose arbours have no room above. The same vertical layout dominates the W2
distances of the pipeline (Stage C's distance matrix correlates 0.63 with the difference in soma
depth), so both readouts track laminar position. These unadjusted summaries alone
do not establish label-related structure beyond depth; the matched comparisons
below address that question.

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
  tolerance)`; `k_within_tolerance` (the tolerance rule alone), `selection_scores` (a
  selection record's score matrix), `selection_gains(fit)`, `tolerance_scan(fits, tolerances)`
  and `label_summary(table, column)` for the per-label medians.

Component parameters keep the fitter's (dx, dz, dy) axis order.

## Outputs (`output/<dataset>/gmm/<side>/`)

Outputs go to `output/<dataset>/gmm/<side>/`, beside the pipeline's `trials/`. Every run
overwrites the previous one; the command and options are recorded in the summary.

| File | Content |
|---|---|
| `neurons.csv` | per neuron: covariates, labels, chosen number of components (`n_components`), log-likelihood, BIC, spread, held-out gains `gain_first_step`, `gain_total`, `gain_beyond_chosen` (units of `--select-by`), `w2_within_median`, `w2_within_max`, `w2_to_final_mean`, `spread_um`, `w2_within_rel`, `n_stability_models`, `w2_rerun_median`, `rerun_k_agreement` |
| `models.json` | every final mixture (weights, means, covariances) with its k-selection and stability records |
| `stability.json` | the repeat models at the chosen k and the rerun models of every neuron |
| `between.csv` | random pairs of neurons: W2 between the final models (`w2`) and between one repeat model of each (`w2_repeat`, matched to the within-neuron distances) |
| `summary.json` | options, `n_components_counts`, medians (including `median_gain_first_step`, `median_gain_total`), `tolerance_scan` (neurons per chosen number of components at each scanned tolerance), `by_cell_type` (per-label medians of `label_summary`), a `stability` block (within median, between median of the repeat models and of the final models, their ratio, the fraction of neurons whose within-distance is below the 10th percentile of the between-distances, rerun agreement), failures, time |
| `selection.png` | counts of the chosen number of components and the held-out selection-score curve (the score of `--select-by`, relative to each neuron's best), with the selection tolerance as a line when one was used |
| `stability.png` | within- vs between-neuron distances, within-distance vs synapse count and vs k |
| `tolerance_scan.png` | fraction of neurons per chosen number of components at each scanned tolerance (the run's own marked) |
| `labels.png` | chosen number of components and total held-out gain per dataset `cell_type` (ordered by median soma depth; red: median), and the chosen number against soma depth with its Spearman correlation |
| `depth_profiles.png` | per `cell_type` with at least 5 neurons, the mean over its neurons of the fitted mixture's marginal density along the depth offset (pia upwards, dashed line at the soma) |

## Comparing labels at matched depth and sampling conditions

After fitting the mixtures, run the separate analysis inside a Slurm allocation:

```bash
source jobs/env.sh
$PYTHON -u scripts/analyze_gmm_matches.py --side dendrite
```

The job `jobs/gmm_matches.slurm` runs its tests and this command on one CPU.
The analysis joins `neurons.csv` to Stage A by exact neuron ID, checks observed
counts and soma depths, and reads only metadata and weights from the clouds.
It adds distance to the border (converted from nm to um), mean border weight,
and weight ESS / observed count, where ESS = (sum w)^2 / sum(w^2). These describe
the complete observed cloud, not the particular subsample used in a fit; ESS
does not correct for spatial dependence.

For every anchor neuron, choose up to five pairs of partners, each containing
one same-cell_type and one different-cell_type neuron within the same
classification_system. All three edges of each triplet must pass:

| Covariate | Default maximum gap |
|---|---|
| Soma depth | 25 um (`--depth-caliper`) |
| Observed and fitted counts, each | larger / smaller <= 1.25 (`--count-ratio`) |
| Soma distance to border | 50 um (`--border-caliper`) |
| Weight ESS / observed count | 0.1 (`--ess-caliper`) |

Among feasible triplets minimize `sum_edges sum_features (difference / caliper)^2`;
counts use log ratios with `log(count_ratio)` as their caliper. Ties are resolved
by neuron IDs. Selection uses only these covariates and the labels, never model
shape, component count or MW2. Mean border weight is reported as a balance
diagnostic, not used in the matching cost. After selecting a triplet, remove
both partners from that anchor's candidates and repeat, up to five times. The
first triplet is unchanged from the one-partner analysis. This greedy selection
prioritizes match quality; it does not maximize the number of feasible pairs.
Partners cannot recur within an anchor but may recur across anchors. Calipers
are enforced within each triplet, not between partners from different triplets.

Only then obtain the final-model MW2 for the two anchor edges: reuse matching
rows of `between.csv` and compute missing distances from `models.json`. At most
ten distances per matched anchor are needed, with shared edges computed once,
and no GMMs are refitted. Every run rebuilds this small analysis; there is no
separate persistent distance cache.
The primary effect is delta = MW2(anchor, different) - MW2(anchor, same), in um.
Positive values mean the same-label partner is closer. Effects are conditional
on successful matching; labels with little overlap must be read with coverage.
Repeated partners make these dependent comparisons, so no pairwise p-values or
confidence intervals are produced. The analysis uses final models only.

The robustness analysis takes the first 1, 3 or 5 selected pairs. For each anchor
it averages delta over those pairs; population medians and positive fractions
then give every anchor equal weight, regardless of its partner count. It reports
two populations at every count:

- `all_matched`: every anchor with at least one pair, using fewer than requested
  when necessary; `n_pairs` and `n_full_requested` make that shortfall explicit.
- `common_5`: the same anchors with five selected pairs at all three counts.
  This separates sensitivity to partner count from changes in the anchor cohort.

Balance also gives anchors equal weight: partner covariates and absolute gaps
are averaged within anchor before computing population means or median gaps.
Maximum gaps cover all selected pairs. The SD denominator always comes from the
full eligible cohort. These checks matter because later partners can match less
closely even though they still satisfy the calipers. More partners are not more
independent neurons, and they cannot fill gaps in covariate overlap.

Outputs go to `gmm/<side>/matched_comparison/`:

- `config.json`: calipers, fitting options and fingerprints of current inputs.
  Original GMM files lack a Stage-A fingerprint; the ID/count/depth checks do
  not retrospectively establish the original fit's full input provenance.
- `covariates.csv`: all neurons, covariates, matching/exclusion status,
  `n_partner_pairs` (0 to 5), and partner reuse across all selected pairs.
- `matches.csv`: one row per matched anchor, partner IDs/labels, covariate cost,
  both distances and delta. `distances.csv` records unique edges and their source.
- `coverage.csv`, `effects.csv`: per-label coverage/reasons and one-pair effects;
  coverage also includes counts with at least 1, 3 and 5 selected pairs.
- `balance.csv`: same/different partner means, their difference divided by the
  eligible cohort's SD, and absolute gaps for partners and anchor edges.
- `comparison.png`: coverage, paired distances, per-label deltas and balance.
- `partner_matches.csv`: all selected triplets with `partner_rank` and distances.
- `anchor_effects.csv`: one row per anchor and requested count, with actual
  `n_pairs`, mean distances/delta, and `in_common_5` membership.
- `robustness.csv`: overall and per-label effects for both populations at 1/3/5
  pairs, actual triplet counts, counts meeting the request, match cost and reuse.
- `robustness_balance.csv`: balance for both populations at each partner count.
- `robustness.png`: overall sensitivity, per-label effects in the fixed cohort,
  and coverage by partner count.
- `summary.json`: original one-pair effects, coverage, reuse and per-label effects,
  plus overall `partner_robustness`; written last. Distance-source counts cover
  all selected pairs. Original `matches.csv`, `effects.csv`, `balance.csv` and
  `comparison.png` remain one-pair results.

The tests cover removal of a synthetic depth confound, recovery of a label-related
shape difference, matching independent of outcomes and row order, all-three-edge
calipers, classification groups, large integer IDs, source joins, distance reuse,
nested partner selection, partner uniqueness, equal anchor weights, a fixed
cohort, and empty-overlap/empty-fixed-cohort cases.

## Examples

```bash
source jobs/env.sh
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite --n-neurons 200 --starts 3          # default components 1-16 by 3
$PYTHON -u scripts/fit_neuron_gmms.py --sides axon --n-neurons 1000 --components-max 4 --components-step 1 --min-border-distance 100
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite --n-neurons 50 --reruns 2           # rerun agreement
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite --n-neurons 200 --covariance diagonal_constraint
$PYTHON -u scripts/fit_neuron_gmms.py --sides dendrite --min-synapses 2000 --covariance diagonal_constraint --components-max 19 --components-step 2 --select-tolerance 0.1   # same candidates for every neuron, effect-size rule
```
