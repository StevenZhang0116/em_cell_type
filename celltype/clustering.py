"""
Hierarchical Clustering, Stability and Post-Hoc Checks
======================================================
Works on the bootstrap replicates of one (side, coords) run:

1. Noise floor -- each neuron's resampling self-distance vs its distance to
   the nearest other neuron. Self/nearest near 1 only says the nearest
   neighbour is indistinguishable from the neuron itself (expected inside a
   homogeneous type); compare self-distance with between-cluster distances.
2. Structure -- average-linkage (default) clustering of the replicate-mean
   distance matrix. Ward linkage is not offered: it assumes Euclidean
   geometry, which Wasserstein distances do not satisfy.
3. Stability -- each replicate is clustered on its own; the adjusted Rand
   index (ARI) against the consensus labels is reported for every k.
4. Old labels and confounds -- AMI/ARI with dataset labels, Kruskal-Wallis
   tests of nuisance covariates across clusters, and the Spearman correlation
   between pairwise distance and |covariate difference|.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kruskal, spearmanr
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from .config import COVARIATE_COLUMNS, LABEL_COLUMNS


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_replicates(directory):
    """Stack all replicate matrices; every replicate must share one neuron axis."""
    paths = sorted(directory.glob("rep*.npz"))
    if not paths:
        raise FileNotFoundError(f"no replicate files in {directory}; compute distances first")
    matrices, self_distances, eligible = [], [], None
    for path in paths:
        with np.load(path) as data:
            if eligible is None:
                eligible = data["eligible"]
            elif not np.array_equal(eligible, data["eligible"]):
                raise ValueError(f"{path.name} has a different neuron set; "
                                 "rerun all replicates with the same --min-synapses")
            matrices.append(data["distance"].astype(float))
            self_distances.append(data["self_distance"].astype(float))
    return np.stack(matrices), np.stack(self_distances), eligible


def neuron_table(clouds, eligible):
    """Per-neuron covariates and dataset labels for the clustered neurons."""
    meta = clouds.meta
    table = pd.DataFrame({
        "pt_root_id": clouds.cell_ids[eligible],
        "soma_depth_um": meta["soma_y"][eligible],
        "n_synapses": clouds.n_synapses[eligible],
        "mean_border_weight": [float(clouds.neuron_weights(i).mean()) for i in eligible],
    })
    if "dist_border" in meta:
        table["dist_border"] = meta["dist_border"][eligible]
    for column in LABEL_COLUMNS:
        table[column] = meta[f"label_{column}"][eligible]
    return table


def present_covariates(table):
    return [column for column in COVARIATE_COLUMNS if column in table]


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def noise_summary(mean_distance, self_distances):
    """Per-neuron nearest-neighbour and self distances plus summary numbers."""
    off_diagonal = mean_distance + np.diag(np.full(mean_distance.shape[0], np.inf))
    nearest = off_diagonal.min(axis=1)
    self_mean = self_distances.mean(axis=0)
    ratio = self_mean / nearest
    return nearest, self_mean, {
        "median_self_distance_um": float(np.median(self_mean)),
        "median_nearest_neighbour_um": float(np.median(nearest)),
        "median_pairwise_um": float(np.median(squareform(mean_distance, checks=False))),
        "median_self_over_nearest": float(np.median(ratio)),
        "fraction_self_over_nearest_gt_1": float(np.mean(ratio > 1)),
    }


def cut_tree(distance, method, k_values):
    tree = linkage(squareform(distance, checks=False), method=method)
    return tree, {k: fcluster(tree, t=k, criterion="maxclust") for k in k_values}


def stability(matrices, consensus_labels, method, k_values):
    """Mean/sd ARI of each replicate's clustering against the consensus, per k."""
    per_k = {k: [] for k in k_values}
    replicate_labels = []
    for matrix in matrices:
        _, labels = cut_tree(matrix, method, k_values)
        replicate_labels.append(labels)
        for k in k_values:
            per_k[k].append(adjusted_rand_score(consensus_labels[k], labels[k]))
    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in per_k.items()}
    return summary, replicate_labels


def most_stable_k(stab, k_min=3):
    """Suggested k: highest mean stability among k >= k_min (k=2 is trivially stable)."""
    return max((k for k in stab if k >= k_min), key=lambda k: stab[k][0])


def co_clustering(replicate_labels, k):
    """Fraction of replicates that put each pair of neurons in the same cluster."""
    labels = np.stack([rep[k] for rep in replicate_labels])
    return np.mean(labels[:, :, None] == labels[:, None, :], axis=0)


def neuron_stability(replicate_labels, labels, k):
    """Per neuron: mean co-clustering with its consensus cluster mates."""
    together = co_clustering(replicate_labels, k)
    return np.array([together[i, labels == labels[i]].mean() for i in range(labels.size)])


def label_agreement(table, consensus_labels, k_values):
    return {column: {k: {"ami": float(adjusted_mutual_info_score(table[column], consensus_labels[k])),
                         "ari": float(adjusted_rand_score(table[column], consensus_labels[k]))}
                     for k in k_values}
            for column in LABEL_COLUMNS}


def covariate_distance_correlation(mean_distance, table):
    """Spearman rho between pairwise distance and |covariate difference|."""
    upper = np.triu_indices(mean_distance.shape[0], 1)
    distances = mean_distance[upper]
    result = {}
    for column in present_covariates(table):
        values = table[column].to_numpy(dtype=float)
        if column == "n_synapses":
            values = np.log(values)
        diff = np.abs(values[:, None] - values[None, :])[upper]
        result[column] = float(spearmanr(distances, diff)[0])
    return result


def covariate_tests(table, labels):
    """Kruskal-Wallis test of each covariate across clusters of size >= 2."""
    tests = {}
    for column in present_covariates(table):
        groups = [table[column][labels == c] for c in np.unique(labels)]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            statistic, p_value = kruskal(*groups)
            tests[column] = {"H": float(statistic), "p": float(p_value)}
    return tests
