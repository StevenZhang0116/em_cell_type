# Sample-matched split-half diagnostics

`scripts/analyze_split_half.py` reads Stage-A clouds and measures how much a neuron's
spatial synapse distribution changes when it is represented by two disjoint subsets.
It is an optional analysis beside Stages B and C, implemented in `celltype/split_half.py`;
it does not require a clustering run or the per-neuron GMM fits.

## Sampling and distances

For each neuron and repetition, uniformly select **2m distinct synapses without
replacement**, then divide them into halves A and B of m synapses each. Retain the
original border weights and normalise them separately inside each half. Sampling
is uniform even when OT is weighted: weights are not applied twice. With
`--no-weights`, both halves instead have uniform masses.

The within-neuron baseline is `W2(A_i, B_i)`. The between-neuron comparison is
`W2(A_i, B_j)` for a fixed, uniformly sampled set of unordered pairs (i < j). Both
use exactly m vs m points and the existing exact weighted W2 solver. One directed
half pairing is used per unordered neuron pair; exchanging A/B is symmetric in
the randomisation, but both directions are not counted as independent replicates.
If a neuron has exactly 2m synapses, this is its complete half-half split. Otherwise
the remaining synapses are unused in that repetition; another repetition can use them.

The selected cohort must support the **largest** requested m and is fixed at every
m and repetition. Border and minimum-synapse filters are applied before optional
random selection with `--n-neurons`. This prevents changes in cohort composition
from appearing as sample-size effects. The summary reports the counts excluded
by each filter. Pair selection is fixed across m/repetitions too. Point selections
are seeded by neuron ID, m and repetition, independently of worker scheduling.

## Running

Use the same environment as the main pipeline (`jobs/env.sh` on an allocated node).
The analysis uses the existing numpy/pandas/scipy/POT/joblib/matplotlib dependencies.
First run Stage A if `clouds_<side>.npz` is not present. Then, for example:

```bash
$PYTHON -u scripts/analyze_split_half.py --sides axon --n-neurons 200 --points-per-half 100 200 --n-reps 100 --between-pairs 1000
$PYTHON -u scripts/analyze_split_half.py --sides dendrite --n-neurons 200 --points-per-half 100 200 400 --min-border-distance 50
```

Defaults: both sides, m = 100 and 200, 100 repetitions, all eligible neurons and up
to 2000 unique pairs; `--between-pairs 0` requests self distances only. A one-neuron
cohort is supported and has no between distances. If the filters leave no neurons,
the script exits with an explanation. For an inexpensive pilot, reduce neurons,
pairs and repetitions; a few repetitions give coarse split percentiles.

For L sizes, B repetitions, N neurons and P selected pairs, this computes
`L * B * (N + P)` OT problems rather than B full N-by-N matrices. Each OT problem
has an m-by-m cost matrix; the complete selected clouds and all output distances
are also held in memory. CPU parallelism follows `SLURM_CPUS_PER_TASK` (one CPU
outside Slurm), with BLAS threads capped by `jobs/env.sh` as elsewhere in this repo.
For each m, all B repetitions are dispatched in **one** joblib call: the `B * (N + P)`
problems are cut into 4 x CPUs blocks of nearly equal size and every worker draws the
halves it needs itself, so the scheduling overhead does not grow with B and the
result does not depend on the CPU count.

## Outputs

Files live in `output/<dataset>/split_half/<side>/`. `config.json` records every option and
a SHA-256 fingerprint of the input NPZ; a later call with other options or regenerated
clouds recomputes the run and overwrites these files. A completed run with the same options
is skipped unless `--overwrite`; an incomplete one is recomputed as a whole, not resumed by
repetition. `summary.json` is written last as the completion marker.

| File | Content |
|---|---|
| `config.json` | Options, schema version and input-cloud fingerprint |
| `distances.npz` | `m_values`, `eligible` (source cloud rows), `cell_ids`, `pairs` (indices into `cell_ids`), raw `self_distance` and `between_distance` with axes **(m, repetition, neuron/pair)**, complete-cloud `spread_um`, `weight_ess`, `n_synapses` |
| `neurons.csv` | One row per (m, neuron): covariates, self-distance median and 5th/95th split percentiles, self/spread ratio, weight ESS and ESS/count |
| `pairs.csv` | One row per (m, pair): between-distance median and split percentiles, plus the separate self median/percentiles of **both** endpoints |
| `summary.json` | Selection counts, descriptive medians, options and interpretation |
| `distributions.png` | Per-neuron split ranges (up to 100 evenly spaced ranks), ECDFs of per-unit split medians, and each pair's between median against both endpoints' self medians |
| `sample_size.png` | Fixed-cohort curves: median of per-unit split medians; bands show the IQR **across neurons/pairs**, not a confidence interval |
| `covariates.png` | Self variability versus complete-cloud spread, weight concentration and synapse count |

Spread is the weighted RMS distance from the complete cloud's weighted centroid.
Weight ESS is `(sum w)^2 / sum(w^2)`; it describes concentration of OT masses and
does **not** correct for synaptic dependence. Self/spread is undefined (NaN in the
CSV) for a zero-spread cloud. An empty pair set produces an empty pair table with
column headers, empty raw pair arrays and `null` between summaries.

## Interpretation and limits

These are **conditional split-variability baselines**, not calibrated p-values or
universal thresholds for merging/splitting clusters. Split percentiles are not
confidence intervals for a population parameter. Repetitions reuse the same
observed cloud, and pairs can share neurons; neither is an independent biological
replicate. Each neuron has its own shape, scale and weight distribution, so equal
sample sizes alone do not create a common null distribution.

Disjoint halves can still share branch-level dependence and the same truncation,
segmentation or detection biases. Stable individual differences can also lie on a
continuous anatomical spectrum. This diagnostic does not establish discrete cell
types or implement an exchangeability-based two-sample permutation test.

## Clustering validation

For validation of the **consensus partition itself**, use the optional
`run_pipeline.py --stages validate` stage described in [split_cluster.md](split_cluster.md).
The diagnostics above remain useful for the within-neuron sampling baseline; they
do not test the reproducibility of a fitted clustering.

## Depth/label strata and targeted partner comparisons

After a completed split-half run, use this follow-up inside Slurm:

```bash
sbatch jobs/split_half_pairs.slurm
# Or, inside an allocation after sourcing jobs/env.sh:
$PYTHON -u scripts/analyze_split_half_pairs.py --side dendrite
```

It keeps the saved cohort, m values, repetitions, seed and weights. The current
Stage-A fingerprint must match the original run; if it changed, rerun the base
analysis first. Metadata and tables are reconstructed from that verified source
and the saved raw distances. No GMM fit is used. Original outputs are preserved;
the follow-up writes to `split_half/<side>/stratified/` and reuses a completed
follow-up only when its settings and source fingerprints match (`--overwrite`
forces recomputation).

Every original random pair is annotated with its endpoints' labels and metadata,
absolute depth difference, and these strata:

- Depth difference: <=25, (25,50], (50,100], >100 um; missing depth is separate.
- Cell-type relation: same, different, or missing.
- Classification systems: E-E, I-I, E-I, or missing/unknown.

Each observed combination is summarized separately at every m, with pair and
distinct-neuron counts. No row means no observed pairs in that combination.
The descriptive split margin is `between - max(self_i, self_j)` **within each
repetition**, followed by its median across repetitions. A positive margin means
between exceeds both endpoints' self distances on this summary; it is not noise
correction, a null test, or a clustering threshold.

The targeted analysis chooses up to three same/different-cell_type partner pairs
per anchor, within the same excitatory/inhibitory class. All three edges of each
triplet must satisfy depth gap <=25 um, observed count ratio <=1.25, border-distance
gap <=50 um, and weight ESS/count gap <=0.1. Cost sums `(difference/caliper)^2`
over covariates and edges, using log counts with caliper `log(1.25)`. Choose the
lowest-cost triplet, break ties by exact neuron IDs, remove its partners for that
anchor, and repeat. Partners can recur across anchors. This greedy rule does not
maximize the number of pairs. W2 and spread are not selection features; spread
is included in the balance report. Missing labels/covariates or inadequate overlap
remain explicit exclusion statuses; calipers are never relaxed to fill a quota.

Targeted edges are deduplicated. Original edges and every self-distance array
are reused exactly; only missing edges are evaluated with the existing split
sampler and W2 solver. Pair direction always uses **original cohort indices**
i<j, comparing A_i to B_j; sorting root IDs would change the particular halves.
The anchor may therefore use half A on one edge and half B on another. No new
direction is added or treated as an independent observation.

For every matched anchor and repetition, average distances to its same-label
partners and to its different-label partners, then subtract same from different.
Take the median of these differences across splits. Population and per-type
medians/fractions give each anchor equal weight, regardless of partner count.
Split percentiles are conditional split ranges, not confidence intervals.
Random and targeted pair strata remain separate because their sampling rules differ.

| File in `stratified/` | Content |
|---|---|
| `config.json`, `summary.json` | Source fingerprints/settings; counts, exclusions, reuse and effects; summary written last |
| `distances.npz` | Original arrays plus new pairs/distances; original-order `is_random` and `is_targeted` flags |
| `pairs.csv`, `strata.csv` | Annotated union of pairs, both membership flags, split margins; separate stratified summaries |
| `cohort.csv`, `coverage.csv` | Every original neuron, match status, actual partner count and reuse; per-type coverage |
| `matches.csv` | Selected triplets, partner rank and covariate matching cost |
| `anchors.csv`, `effects.csv` | Per-anchor split summaries and equal-anchor overall/per-type effects |
| `balance.csv` | Same/different partner means with equal anchor weights, standardized differences and maximum covariate gaps |
| `comparison.png` | Random-pair depth strata at largest m (counts shown), matched effects across m and coverage |

For N anchors, the three-pair cap requests at most 6N targeted edges before
deduplication/reuse. Each new edge adds `len(m_values) * n_reps` OT evaluations.
The job requests four CPUs with single-threaded BLAS. This follow-up still
describes the observed neuron cohort; shared branches, truncation biases and
shared neurons remain limitations, and no p-values are produced.
