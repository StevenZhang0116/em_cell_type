# Split-half validation of consensus clustering

`scripts/run_pipeline.py` remains the entry point. Its optional `validate` stage
tests whether the **same consensus estimator used by Stage C** gives reproducible
partitions from disjoint synapses. It supplements the final full-cloud fit and the
sampling diagnostics in [split_half.md](split_half.md). The default stages are
unchanged; `--method shared` is currently rejected with `validate`.

## Dendrite pilot

After Stage A has generated `output/microns/clouds_dendrite.npz`, run from the
repository root in the existing environment:

```bash
python -u scripts/run_pipeline.py \
  --dataset microns --sides dendrite --method points \
  --stages validate \
  --n-neurons 100 --min-synapses 400 --min-border-distance 50 \
  --validation-points-per-half 200 --validation-reps 10 \
  --n-reps 5 --k-min 2 --k-max 10 --min-cluster-size 5
```

To include the final full-cloud fit, change only `--stages` to
`validate distances cluster`. Add `extract` first if clouds have not been built
(Stage A needs the original Feather tables and substantially more memory).
`--stages ... assign` retains the existing downstream assignment behavior.
On klone, source `jobs/env.sh` and use `$PYTHON` as in the other examples.

`--validation-reps` counts **outer disjoint splits**. `--n-reps` counts the inner
bootstrap replicates **per half**, and the replicates used by the full pipeline.
The pilot uses 10 outer splits and 5 inner bootstraps. This is a small diagnostic
run, not a calibrated inferential test. Exact OT cost still grows quadratically
in neuron count: each outer split needs two exact matrices plus two sets of inner
bootstrap matrices. Start small before increasing either count.

## Fixed cohort and preprocessing

1. Select neurons using the existing `min_synapses`, border-distance filter,
   `n_neurons` and `seed`, exactly as the full pipeline does.
2. Require **every selected neuron** to have at least `2 * max(m)` synapses.
   Multiple `--validation-points-per-half` values reuse this same cohort. If the
   requirement fails, raise an error; do not silently remove neurons. Increase
   `--min-synapses` or decrease the requested sizes and rerun both analyses with
   the same settings.
3. If `--max-synapses` is used, apply the existing deterministic uniform cap
   first, identically to the main pipeline. It must be at least `2 * max(m)`.
4. For each neuron, sample `2m` distinct synapses uniformly without replacement,
   then divide them into disjoint sets A and B of size `m`. The remaining synapses
   are unused in that outer split. When `n = 2m`, this is a half-and-half partition.
   Random streams depend on seed, cell ID, m and repeat, independently of CPU count.
5. Retain the border-compensation weights and normalize within each half; do not
   weight the outer draw by them. With `--no-weights`, OT masses are uniform.
   Inner bootstrap draws follow the existing weighted empirical bootstrap:
   multinomial counts with probabilities equal to normalized mass, followed by
   count-based masses (the border weights are not applied a second time).

The cohort and soma reference are fixed inputs; complementary splits are not
independent biological observations.

## The estimator and the held-out quantities

For a half H and inner bootstrap b, compute exact Wasserstein distances

\[
D^{H,b}_{ij}=W_2(\widehat P_i^{H,b},\widehat P_j^{H,b}).
\]

Cut each distance tree into k major clusters of at least the configured minimum
size. Smaller groups have label 0. Co-clustering frequencies across inner
replicates form a consensus dissimilarity `1 - frequency`, which is clustered
again. This is `clustering.fit_consensus`, shared by Stage C, A and B; it does not
substitute a direct cut of the exact distance matrix for consensus clustering.

**A alone selects k** with the existing internal ARI/PAC rule (or a predeclared
`--k`). B then fits the same estimator with the same inner bootstrap count at
**only A's selected k**. B never selects among candidate k values. Linkage,
minimum size, candidate k range and m values are set before looking at B results.
If A or B cannot reach k, record the failure; a smaller achieved partition does
not count as a successful fit at k. No B-driven fallback or parameter tuning is
performed. If A fails, B's fit is skipped and its fit status says so.

The exact, non-bootstrapped B matrix is also retained. Let `a_i`, `b_i` be the
two consensus labels and `S = {i: a_i > 0 and b_i > 0}`. Report:

- `ARI(a_S, b_S)`, alongside joint coverage `|S| / N` and both outlier rates.
  Failed fits and degenerate common cohorts (one group, all singletons or fewer
  than three neurons) have no evaluable ARI, stored as JSON `null` / a blank CSV
  value. A large ARI on a small retained subset is not evidence of broad recovery.
- Per-A-cluster matching to B via the Hungarian assignment maximizing total
  Jaccard similarity. For a matched pair `(C, G)`, Jaccard is
  `|C intersect G| / |C union G|`, and retained fraction is
  `|C intersect G| / |C|`. Use complete cluster memberships in these denominators,
  including neurons rejected by the other half. Unmatched A clusters get zero
  recovery. Label 0 is never an ordinary cluster. Failed-fit match rows retain
  a status and should not be pooled with successful fits.
- Held-out separation using **A labels only**:

\[
\Delta_B = \operatorname{median}_{i<j,\ a_i,a_j>0,\ a_i\ne a_j}D^B_{ij}
          -\operatorname{median}_{i<j,\ a_i=a_j>0}D^B_{ij}.
\]

  B labels do not select either set of pairs. Report both pair counts and medians.
  Missing pair groups or a failed A fit make the separation unevaluable. A failed
  B clustering does not invalidate its distances for testing A's partition.
  This pooled statistic weights larger clusters through their larger pair counts;
  it is not an equal-cluster average or a p-value.

The final full-cloud fit selects `k_full` independently using the same rule.
The validation summary records A's k frequencies and how often `k_full` was
selected. It **does not** transfer success at one k to an untested k or declare
`k_full` validated automatically. A fixed, predeclared `--k` removes this selection
difference. Choosing k or m after inspecting B is exploratory and requires a new
independent evaluation for confirmatory claims.

## Outputs and resume

Results live under
`trials/<trial>/validate_<side>/m<sizes>_rep<count>_<digest>/`. The trial remains the
same full-fit trial regardless of validation settings. The validation digest
covers algorithm version, every statistical setting, source-file SHA-256 and the
ordered cell IDs. CPU count is excluded because it does not change the result.

| File | Contents |
|---|---|
| `config.json` | Settings, full-fit parameters, input identity and cohort |
| `m<m>_rep<r>.npz` | Exact A/B distance matrices and consensus labels in cohort order; JSON `record` includes A selection, B's frozen k, inner ARI/PAC, fit status, held-out metrics and cluster matches |
| `repeats.csv` | Per-split k, status, ARI, coverage, both outlier rates, separation and pair counts |
| `cluster_matches.csv` | Per-split matched clusters, membership counts, Jaccard and retained fraction |
| `summary.json` | Per-m descriptive medians/quartiles, failure/unevaluable rates, A k frequencies and comparison with the full fit |
| `validation.png` | ARI versus coverage, held-out separation, A-only k selection frequencies |
| `representative_contingency.png` | First evaluable repeat (else first repeat), including an explicit outlier row/column when present; not selected for best agreement |

Checkpoints are written atomically after each **outer** repeat. Rerunning the same
command reuses completed repeats and regenerates tables and figures; an interrupted
repeat restarts from its beginning. `--overwrite` recomputes them. A mismatched or
invalid checkpoint raises an error rather than silently reusing different data.
Changed inputs or parameters create a separate directory.

The full fit is linked only when its recorded source hash, ordered cohort and run
parameters agree. Old distance caches without a fingerprint remain loadable but
are reported as `full_fit_provenance_unknown`; rerun `distances cluster` with
`--overwrite` and then include `validate` again to establish the comparison.
A missing full fit is expected for a validation-only pilot.

## Interpretation and tests

Repeated splits reuse the same neurons and usually reuse synapses across repeats.
Their quartiles describe split sensitivity, **not confidence intervals**. Neuron
pairs are dependent. No Gaussian z-score, pairwise significance test, FDR, or
biological-null p-value is produced. Stable partitions can also occur along
continuous anatomical gradients. This analysis does not by itself demonstrate
discrete cell types or validate assignments to new neurons. Synapse dependence
along branches is not modeled; branch-aware resampling and independent biological
replication remain separate questions.

The synthetic test suite checks recovery, A/B isolation,
shared estimator compatibility, outliers and impossible k, label permutation,
held-out pair selection, parallel reproducibility, cache identity, interruption
and resume, and the combined CLI with full-fit/assignment-reference compatibility:

```bash
python -m pytest tests -q
```
