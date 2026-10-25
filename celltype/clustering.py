"""
Hierarchical Clustering, Stability and Post-Hoc Checks
======================================================
Works on the bootstrap replicates of one side:

1. Noise floor -- each neuron's bootstrap self-distance vs its distance to
   the nearest other neuron. Self/nearest near 1 only says the nearest
   neighbour is indistinguishable from the neuron itself (expected inside a
   homogeneous type); compare self-distance with between-cluster distances.
2. Replicate clusterings -- each replicate's distance matrix gets its own
   average-linkage (default) tree, cut into k *major* clusters: the tree is
   split until k clusters hold at least ``min_size`` neurons, and neurons in
   smaller clusters are labelled 0 (outliers), so a few odd neurons cannot
   use up the k clusters. Ward linkage is not offered: it assumes Euclidean
   geometry, which Wasserstein distances do not satisfy.
3. Consensus -- the co-clustering matrix (fraction of replicates that put two
   neurons in the same major cluster) is clustered again the same way; these
   are the reported labels. PAC, the fraction of neuron pairs whose
   co-clustering lies strictly between 0.1 and 0.9, measures ambiguity
   (lower is more stable).
4. Stability -- per replicate, the adjusted Rand index (ARI) against the
   consensus over the consensus non-outlier neurons; the distribution over
   replicates is reported, not just its mean.
5. Old labels and confounds -- AMI/ARI with dataset labels, Kruskal-Wallis
   tests of nuisance covariates across clusters, and the Spearman correlation
   between pairwise distance and |covariate difference| (outliers excluded
   where labels are used).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
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


def load_exact(directory, eligible):
    """The exact distance matrix of the complete clouds (``exact.npz`` of ``--method points``), or None."""
    path = directory / "exact.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        if not np.array_equal(data["eligible"], eligible):
            raise ValueError(f"{path} holds a different neuron set; rerun Stage B")
        return data["distance"].astype(float)


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
        table["dist_border_um"] = meta["dist_border"][eligible] / 1000.0   # stored in nm
    for column in LABEL_COLUMNS:
        table[column] = meta[f"label_{column}"][eligible]
    return table


def present_covariates(table):
    return [column for column in COVARIATE_COLUMNS if column in table]


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def noise_summary(distance, self_distances):
    """Per-neuron nearest-neighbour and self distances plus summary numbers."""
    off_diagonal = distance + np.diag(np.full(distance.shape[0], np.inf))
    nearest = off_diagonal.min(axis=1)
    self_mean = self_distances.mean(axis=0)
    ratio = self_mean / nearest
    return nearest, self_mean, {
        "median_self_distance_um": float(np.median(self_mean)),
        "median_nearest_neighbour_um": float(np.median(nearest)),
        "median_pairwise_um": float(np.median(squareform(distance, checks=False))),
        "median_self_over_nearest": float(np.median(ratio)),
        "fraction_self_over_nearest_gt_1": float(np.mean(ratio > 1)),
    }


def build_tree(distance, method):
    return linkage(squareform(distance, checks=False), method=method)


def leaf_order(tree):
    return leaves_list(tree)


def _leaves(tree, node, n):
    """Leaf indices under one tree node."""
    stack, leaves = [node], []
    while stack:
        current = stack.pop()
        if current < n:
            leaves.append(current)
        else:
            stack.extend(int(child) for child in tree[current - n, :2])
    return leaves


def cut_major(tree, k, min_size):
    """Labels 1..k for the k major clusters (largest first); 0 for outliers.

    Merges are undone from the top until k clusters have at least
    ``min_size`` members. With ``min_size=1`` this is the usual cut into k
    clusters. If the tree cannot supply k major clusters, the cut with the
    most major clusters (the first one reached) is returned instead.
    """
    n = tree.shape[0] + 1

    def size(node):
        return 1 if node < n else int(tree[node - n, 3])

    root = 2 * n - 2
    current = {root}
    n_major = int(size(root) >= min_size)
    best_count, best = n_major, set(current)
    for row in range(n - 2, -1, -1):
        if n_major >= k:
            break
        node = n + row
        current.remove(node)
        n_major -= size(node) >= min_size
        for child in tree[row, :2].astype(int):
            current.add(int(child))
            n_major += size(child) >= min_size
        if n_major > best_count:
            best_count, best = n_major, set(current)
    if n_major < k:
        current = best
    majors = sorted((c for c in current if size(c) >= min_size), key=lambda c: (-size(c), c))
    labels = np.zeros(n, dtype=int)
    for label, node in enumerate(majors[:k], start=1):
        labels[_leaves(tree, node, n)] = label
    return labels


def replicate_cuts(matrices, method, k_values, min_size):
    """For every replicate: {k: major-cluster labels} from its own tree."""
    cuts = []
    for matrix in matrices:
        tree = build_tree(matrix, method)
        cuts.append({k: cut_major(tree, k, min_size) for k in k_values})
    return cuts


def co_clustering(replicate_labels, k):
    """Fraction of replicates that put each pair in the same major cluster."""
    n = replicate_labels[0][k].size
    together = np.zeros((n, n), dtype=np.float32)
    for labels in replicate_labels:
        same = labels[k][:, None] == labels[k][None, :]
        together += same & (labels[k] > 0)[:, None]
    together /= len(replicate_labels)
    np.fill_diagonal(together, 1.0)
    return together


def pac(together, lower=0.1, upper=0.9):
    """Proportion of ambiguously clustered pairs."""
    values = together[np.triu_indices(together.shape[0], 1)]
    return float(np.mean((values > lower) & (values < upper)))


def consensus_clustering(replicate_labels, k_values, method, min_size):
    """Consensus labels per k from the co-clustering matrix, plus PAC per k."""
    consensus, ambiguity = {}, {}
    for k in k_values:
        together = co_clustering(replicate_labels, k)
        consensus[k] = cut_major(build_tree(1.0 - together, method), k, min_size)
        ambiguity[k] = pac(together)
    return consensus, ambiguity


def stability(replicate_labels, consensus, k_values):
    """ARI of each replicate against the consensus (consensus outliers excluded), per k.

    NaN when the consensus has fewer than k major clusters, so that k cannot
    look stable just because almost every neuron became an outlier.
    """
    result = {}
    for k in k_values:
        keep = consensus[k] > 0
        if np.unique(consensus[k][keep]).size < k:
            result[k] = np.full(len(replicate_labels), np.nan)
            continue
        result[k] = np.array([adjusted_rand_score(consensus[k][keep], labels[k][keep])
                              for labels in replicate_labels])
    return result


def stability_summary(values):
    """Median, quartiles, mean and raw values; None throughout when k was not reached."""
    if np.all(np.isnan(values)):
        return {"median": None, "q25": None, "q75": None, "mean": None, "values": None}
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {"median": float(median), "q25": float(q25), "q75": float(q75),
            "mean": float(np.mean(values)), "values": [float(v) for v in values]}


def most_stable_k(stab, ambiguity, k_min=3, ari_tolerance=0.03, pac_tolerance=0.05):
    """Suggested k and the reasoning behind it.

    Among valid k >= ``k_min`` (k=2 is trivially stable), keep those whose
    median ARI is within ``ari_tolerance`` of the best, then those whose PAC
    is within ``pac_tolerance`` of the lowest PAC in that set, and take the
    largest. This heuristic prefers resolution unless ambiguity rises; internal
    stability alone does not establish discrete biological cell types.
    """
    valid = [k for k in stab if not np.all(np.isnan(stab[k]))]
    candidates = [k for k in valid if k >= k_min] or valid
    medians = {k: float(np.median(stab[k])) for k in candidates}
    best = max(medians.values())
    near = [k for k in candidates if medians[k] >= best - ari_tolerance]
    lowest_pac = min(ambiguity[k] for k in near)
    chosen = max(k for k in near if ambiguity[k] <= lowest_pac + pac_tolerance)
    return chosen, {"rule": f"largest k with median ARI >= best - {ari_tolerance} and "
                            f"PAC <= lowest PAC among those + {pac_tolerance}",
                    "best_median_ari": best, "within_ari_tolerance": near,
                    "lowest_pac_among_those": float(lowest_pac), "chosen": chosen}


def fit_consensus(matrices, method, k_values, min_size, k=None):
    """Shared consensus fit for full clouds and either validation half.

    Automatic selection uses only these replicate matrices. With fixed k no
    selection rule runs. An unreachable k is an explicit status, not a smaller
    clustering silently presented as a successful fit. Raw consensus cuts remain
    available for diagnostics, including when no candidate k is reachable.
    """
    matrices = np.asarray(matrices)
    if (matrices.ndim != 3 or matrices.shape[0] < 1 or matrices.shape[1] < 2
            or matrices.shape[1] != matrices.shape[2] or not np.isfinite(matrices).all()):
        raise ValueError("consensus needs finite replicate matrices of shape (B, N, N), B >= 1, N >= 2")
    values = sorted(set(k_values) | ({k} if k is not None else set()))
    if (not values or any(not isinstance(kk, (int, np.integer)) or kk < 2 for kk in values)
            or not isinstance(min_size, (int, np.integer)) or min_size < 1):
        raise ValueError("k values must be integers >= 2 and min_size a positive integer")
    values = [int(kk) for kk in values]
    cuts = replicate_cuts(matrices, method, values, min_size)
    consensus, ambiguity = consensus_clustering(cuts, values, method, min_size)
    stab = stability(cuts, consensus, values)
    reached = [kk for kk in values if not np.all(np.isnan(stab[kk]))]
    for kk in set(values) - set(reached):
        ambiguity[kk] = float("nan")
    if k is not None:
        chosen, selection = int(k), {"rule": "given on the command line", "chosen": int(k)}
    elif reached:
        chosen, selection = most_stable_k(stab, ambiguity)
    else:
        chosen, selection = None, {"rule": "no reachable candidate k", "chosen": None}
    labels = np.zeros(matrices.shape[1], dtype=int) if chosen is None else consensus[chosen]
    status = "ok" if chosen in reached else "unreachable_k"
    return {"status": status, "chosen_k": chosen, "k_selection": selection, "reached": reached,
            "labels": labels, "consensus": consensus, "stability": stab, "pac": ambiguity,
            "co_clustering": None if chosen is None else co_clustering(cuts, chosen)}


def neuron_stability(together, labels):
    """Per neuron: mean co-clustering with its consensus cluster mates; NaN for outliers."""
    result = np.full(labels.size, np.nan)
    for i in np.flatnonzero(labels > 0):
        result[i] = together[i, labels == labels[i]].mean()
    return result


def label_agreement(table, consensus_labels, k_values):
    """AMI/ARI with each dataset label column over the non-outlier neurons."""
    result = {column: {} for column in LABEL_COLUMNS}
    for k in k_values:
        keep = consensus_labels[k] > 0
        for column in LABEL_COLUMNS:
            if keep.sum() < 2:
                result[column][k] = {"ami": float("nan"), "ari": float("nan")}
                continue
            truth, labels = table[column].to_numpy()[keep], consensus_labels[k][keep]
            result[column][k] = {"ami": float(adjusted_mutual_info_score(truth, labels)),
                                 "ari": float(adjusted_rand_score(truth, labels))}
    return result


def covariate_distance_correlation(distance, table):
    """Spearman rho between pairwise distance and |covariate difference|."""
    upper = np.triu_indices(distance.shape[0], 1)
    distances = distance[upper]
    result = {}
    for column in present_covariates(table):
        values = table[column].to_numpy(dtype=float)
        if column == "n_synapses":
            values = np.log(values)
        diff = np.abs(values[:, None] - values[None, :])[upper]
        result[column] = float(spearmanr(distances, diff)[0])
    return result


def covariate_tests(table, labels):
    """Kruskal-Wallis test of each covariate across clusters (outliers excluded)."""
    tests = {}
    for column in present_covariates(table):
        groups = [table[column][labels == c] for c in np.unique(labels) if c > 0]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            statistic, p_value = kruskal(*groups)
            tests[column] = {"H": float(statistic), "p": float(p_value)}
    return tests
