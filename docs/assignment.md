# Stage D (optional): assigning neurons to the clusters

Stage D is not part of the default run; add `assign` to `--stages` of
`scripts/run_pipeline.py` to use it (the trial must already have its Stage C clusters, so
give the same Stage B and C options). The clustering is quadratic in the number of neurons,
so the dendrite side (15k proofread neurons) can be clustered on a subset and the rest
assigned. Queries are `--query rest` (proofread neurons of the same side that were not
clustered), `--query all` (the dendrite of every cell in the table, from
`clouds_dendrite_all.npz`; dendrite side only) or `--query none` (only cross-validate and
fit the templates). Two methods give a posterior over the k major clusters plus a "none"
label 0:

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

## Calibration and validation

`tau`, `T` and the rejection thresholds (`--reject-quantile`, default 0.95, of the held-out
nearest-reference distance and of the held-out own-cluster score) are chosen by minimising
the held-out negative log-likelihood of the consensus labels in a `--cv-folds` cross-
validation of the clustered neurons (kNN pool and templates refitted without the held-out
fold). The same held-out posteriors give the reported accuracy, calibration error and
confusion matrices, and the agreement between the methods.

## Caveats

Axon and dendrite clusters are separate taxonomies, so a neuron gets one label per side
and the two must not be multiplied; a dendrite-only neuron has no axon label. Non-proofread
dendrites (`--query all`) may be truncated or merged.

## Outputs (`trials/<trial>/assign_<side>_<query>/`)

| File | Content |
|---|---|
| `summary.json` | k, reference/query counts, calibration (`tau_um2`, `temperature`, rejection thresholds, NLL curves), held-out metrics per method (accuracy, NLL, ECE, confusion, rejected fractions), method agreement, template sizes and BIC per K, query label counts and AMI with `cell_type` |
| `reference_cv.csv` | per clustered neuron: consensus label, fold, held-out nearest-reference distance, template scores, posteriors and hard labels of every method |
| `queries.csv` | the same for every query neuron, with its covariates and dataset labels |
| `templates.json` | the fitted template of every cluster |
| `cv_confusion.png`, `cv_reliability.png` | held-out confusion matrices; accuracy vs posterior confidence |
| `templates.png`, `queries.png` | template densities over member synapses; query confidence and label counts |
