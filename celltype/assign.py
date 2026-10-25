"""
Stage D: Assigning Neurons to the Clusters
==========================================
Turns the consensus clustering of one side into a classifier for neurons
that were not clustered: the proofread neurons left out of a
``--n-neurons`` subset (query ``rest``) or the dendrite of every cell in the
table (query ``all``, needs ``clouds_dendrite_all.npz`` from ``--all-cells``).

Two methods, each giving a posterior over the k major clusters plus a "none"
option (label 0, like the consensus outliers):

knn       The exact W2 between the query's complete synapse cloud and that
          of every clustered (non-outlier) reference neuron (``distances.py``;
          the reference-to-reference matrix is Stage B's ``exact.npz``). The
          ``n_neighbours`` nearest references vote with weight
          ``exp(-d^2 / tau)``. A query farther from its nearest
          reference than the ``reject_quantile`` of held-out reference
          neurons gets label 0.
template  The synapses of each cluster's members are pooled (soma-centred)
          and fitted with the cylindrically constrained weighted GMM of
          ``../mixture`` (``templates.py``). A query's score under cluster c
          is the weighted mean per-synapse log-density. The synapses of one
          neuron are not independent draws from its type's template, so the
          score is tempered: ``posterior ∝ exp(T * score)``, where T plays
          the role of an effective number of independent synapses. Soma depth
          is not used: the posterior depends on the synapse cloud alone. A
          query whose best score is below the ``1 - reject_quantile``
          quantile of held-out own-cluster scores gets label 0.

Calibration and validation come from the same cross-validation of the
reference set: each fold is held out, the kNN pool is restricted to and the
templates are refitted on the other folds, and the held-out scores fix tau,
T and the rejection thresholds by minimising the negative
log-likelihood of the consensus labels. The held-out posteriors also give
the reported accuracy, calibration error, confusion matrices and the
agreement between the methods.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import adjusted_mutual_info_score

from .templates import fit_template

METHODS = ("knn", "template")


# ---------------------------------------------------------------------------
# Method 1: k nearest references
# ---------------------------------------------------------------------------

def nearest_references(distances, reference_labels, n_neighbours):
    """Distances (ascending) and labels of the nearest references, (Q, n) each.

    Rows are padded with NaN / 0 when fewer than n references exist.
    """
    n_query, n_reference = distances.shape
    n = min(n_neighbours, n_reference)
    order = np.argsort(distances, axis=1, kind="stable")[:, :n]
    knn_distance = np.full((n_query, n_neighbours), np.nan)
    knn_label = np.zeros((n_query, n_neighbours), dtype=int)
    knn_distance[:, :n] = np.take_along_axis(distances, order, axis=1)
    knn_label[:, :n] = reference_labels[order]
    return knn_distance, knn_label


def knn_posterior(knn_distance, knn_label, n_clusters, tau):
    """Soft vote of the nearest references with weight exp(-d^2 / tau), (Q, k)."""
    shifted = knn_distance ** 2 - np.nanmin(knn_distance, axis=1, keepdims=True) ** 2
    weight = np.where(np.isnan(knn_distance), 0.0, np.exp(-shifted / tau))
    vote = np.stack([(weight * (knn_label == c)).sum(axis=1) for c in range(1, n_clusters + 1)], axis=1)
    return vote / np.maximum(vote.sum(axis=1, keepdims=True), 1e-300)


# ---------------------------------------------------------------------------
# Method 2: cluster templates
# ---------------------------------------------------------------------------

def pooled_points(clouds, indices, max_points, rng):
    """Soma-centred synapses of all ``indices`` with weights, subsampled to ``max_points``."""
    points = np.concatenate([clouds.neuron_points(i) for i in indices]).astype(float)
    weight = np.concatenate([clouds.neuron_weights(i) for i in indices]).astype(float)
    if points.shape[0] > max_points:
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points, weight = points[keep], weight[keep]
    return points, weight


def fit_templates(clouds, indices, labels, n_clusters, *, k_values, max_points, use_weights,
                  seed, n_jobs=1):
    """One template per cluster 1..k; returns ``{c: model}`` and ``{c: {K: BIC}}``."""
    templates, selection = {}, {}
    for c in range(1, n_clusters + 1):
        members = indices[labels == c]
        if members.size == 0:
            raise ValueError(f"cluster {c} has no members to fit a template on")
        rng = np.random.default_rng([seed, 4, c])
        points, weight = pooled_points(clouds, members, max_points, rng)
        if not use_weights:
            weight = np.ones_like(weight)
        templates[c], selection[c] = fit_template(points, weight, k_values,
                                                  seed=int(rng.integers(2 ** 31)), n_jobs=n_jobs)
    return templates, selection


def template_scores(clouds, indices, templates, use_weights):
    """Weighted mean per-synapse log-density of each neuron under each template, (Q, k)."""
    order = sorted(templates)
    scores = np.empty((len(indices), len(order)))
    for row, index in enumerate(indices):
        points = clouds.neuron_points(index).astype(float)
        weight = clouds.neuron_weights(index).astype(float) if use_weights else np.ones(points.shape[0])
        weight = weight / weight.sum()
        for col, c in enumerate(order):
            scores[row, col] = float(weight @ templates[c].log_density(points))
    return scores


def softmax_posterior(scores, temperature):
    """Tempered softmax over the per-cluster template scores."""
    logits = temperature * scores
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


# ---------------------------------------------------------------------------
# Calibration and metrics
# ---------------------------------------------------------------------------

def negative_log_likelihood(prob, truth):
    """Mean -log P(true cluster); ``truth`` holds 1-based cluster labels."""
    return float(-np.mean(np.log(prob[np.arange(truth.size), truth - 1] + 1e-12)))


def calibrate(posterior_of, grid, truth):
    """Grid value whose posteriors minimise the NLL, plus the whole curve."""
    curve = {float(value): negative_log_likelihood(posterior_of(value), truth) for value in grid}
    return min(curve, key=curve.get), curve


def expected_calibration_error(prob, truth, n_bins=10):
    confidence = prob.max(axis=1)
    correct = (prob.argmax(axis=1) + 1) == truth
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def classification_metrics(prob, truth, n_clusters):
    predicted = prob.argmax(axis=1) + 1
    confusion = np.zeros((n_clusters, n_clusters), dtype=int)
    np.add.at(confusion, (truth - 1, predicted - 1), 1)
    return {"accuracy": float(np.mean(predicted == truth)),
            "nll": negative_log_likelihood(prob, truth),
            "ece": expected_calibration_error(prob, truth),
            "confusion_rows_true_cols_predicted": confusion.tolist()}


def stratified_folds(labels, n_folds, rng):
    """Fold index per neuron; every label (outliers included) is spread over the folds."""
    fold = np.empty(labels.size, dtype=int)
    for value in np.unique(labels):
        members = np.flatnonzero(labels == value)
        fold[members] = rng.permutation(members.size) % n_folds
    return fold


def cross_validate(clouds, indices, labels, n_clusters, distances, *, n_folds, n_neighbours,
                   k_values, max_points, use_weights, seed, n_jobs=1):
    """Held-out kNN neighbours and template scores for every reference."""
    n = len(indices)
    fold = stratified_folds(labels, n_folds, np.random.default_rng([seed, 5]))
    knn_distance = np.full((n, n_neighbours), np.nan)
    knn_label = np.zeros((n, n_neighbours), dtype=int)
    score = np.full((n, n_clusters), np.nan)
    for f in range(n_folds):
        held = fold == f
        train = ~held & (labels > 0)
        if not held.any():
            continue
        knn_distance[held], knn_label[held] = nearest_references(
            distances[np.ix_(held, train)], labels[train], n_neighbours)
        templates, _ = fit_templates(clouds, indices[train], labels[train], n_clusters,
                                     k_values=k_values, max_points=max_points,
                                     use_weights=use_weights, seed=seed + f, n_jobs=n_jobs)
        score[held] = template_scores(clouds, indices[held], templates, use_weights)
    return {"fold": fold, "knn_distance": knn_distance, "knn_label": knn_label, "score": score}


def calibrate_from_cv(cv, labels, n_clusters, reject_quantile):
    """tau, T and the rejection thresholds from the held-out references."""
    keep = labels > 0
    truth = labels[keep]
    nearest = cv["knn_distance"][:, 0]
    tau_grid = np.nanmedian(nearest[keep]) ** 2 * np.logspace(-1.5, 2, 29)
    tau, tau_curve = calibrate(
        lambda t: knn_posterior(cv["knn_distance"][keep], cv["knn_label"][keep], n_clusters, t),
        tau_grid, truth)
    temperature_grid = np.logspace(0, 4, 41)
    temperature, temperature_curve = calibrate(
        lambda t: softmax_posterior(cv["score"][keep], t), temperature_grid, truth)
    own_score = cv["score"][keep][np.arange(truth.size), truth - 1]
    return {
        "tau_um2": tau, "temperature": temperature,
        "reject_distance_um": float(np.nanquantile(nearest[keep], reject_quantile)),
        "reject_score": float(np.quantile(own_score, 1 - reject_quantile)),
        "reject_quantile": reject_quantile,
        "curves": {"tau_um2": tau_curve, "temperature": temperature_curve},
    }


def assignments(knn_distance, knn_label, score, n_clusters, calibration):
    """Posteriors (Q, k) and hard labels (0 = rejected) for every method."""
    prob = {
        "knn": knn_posterior(knn_distance, knn_label, n_clusters, calibration["tau_um2"]),
        "template": softmax_posterior(score, calibration["temperature"]),
    }
    rejected = {
        "knn": ~(knn_distance[:, 0] <= calibration["reject_distance_um"]),
        "template": ~(score.max(axis=1) >= calibration["reject_score"]),
    }
    hard = {m: np.where(rejected[m], 0, prob[m].argmax(axis=1) + 1) for m in METHODS}
    return prob, hard


def evaluate(prob, hard, truth, n_clusters):
    """Held-out metrics per method over the consensus non-outliers, plus method agreement."""
    keep = truth > 0
    result = {}
    for method in METHODS:
        metrics = classification_metrics(prob[method][keep], truth[keep], n_clusters)
        metrics["accuracy_with_reject"] = float(np.mean(hard[method][keep] == truth[keep]))
        metrics["rejected_fraction"] = float(np.mean(hard[method][keep] == 0))
        metrics["outliers_rejected_fraction"] = (float(np.mean(hard[method][~keep] == 0))
                                                if (~keep).any() else None)
        result[method] = metrics
    result["agreement"] = method_agreement(hard)
    return result


def method_agreement(hard):
    """Fraction of neurons given the same label by two methods, over neurons both accept."""
    result = {}
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1:]:
            both = (hard[a] > 0) & (hard[b] > 0)
            result[f"{a}_vs_{b}"] = float(np.mean(hard[a][both] == hard[b][both])) if both.any() else None
    return result


def query_summary(prob, hard, table):
    """Label counts, confidence and agreement with dataset labels for the assigned queries."""
    result = {"n_queries": int(len(table)), "agreement": method_agreement(hard)}
    for method in METHODS:
        labels = hard[method]
        entry = {"counts": {("none" if c == 0 else str(c)): int(n)
                            for c, n in zip(*np.unique(labels, return_counts=True))},
                 "mean_max_posterior": float(prob[method].max(axis=1).mean())}
        if "cell_type" in table:
            accepted = labels > 0
            if accepted.sum() >= 2:
                entry["ami_cell_type"] = float(adjusted_mutual_info_score(
                    table["cell_type"].to_numpy()[accepted], labels[accepted]))
        result[method] = entry
    return result


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

def assignment_table(base, prob, hard, knn_distance, score, n_clusters):
    """Per-neuron table: covariates/labels in ``base`` plus posteriors and hard labels."""
    table = base.copy()
    table["nearest_reference_um"] = knn_distance[:, 0]
    for c in range(1, n_clusters + 1):
        table[f"template_score_{c}"] = score[:, c - 1]
    for method in METHODS:
        for c in range(1, n_clusters + 1):
            table[f"{method}_prob_{c}"] = prob[method][:, c - 1]
        table[f"{method}_label"] = hard[method]
    return table


def serialise_templates(templates, selection, labels):
    """JSON-ready templates with their BIC curve and member count."""
    return {str(c): {**templates[c].to_dict(), "bic_per_k": selection[c],
                     "n_members": int(np.sum(labels == c))}
            for c in sorted(templates)}
