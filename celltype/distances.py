"""
Neuron-to-Neuron Wasserstein Distances
======================================
One bootstrap replicate draws, for every eligible neuron, two independent
resamples (A and B) of ``n_points`` synapses, with replacement and with
probability proportional to the border-compensation weight. Two methods:

``points``
    Exact 2-Wasserstein distance between resampled clouds (3-D offsets or
    cylindrical ``(r, depth)``, see ``Clouds.neuron_points``).
    With equal-size uniform samples, optimal transport reduces to an
    assignment problem, solved exactly with ``linear_sum_assignment``.

``shared_gmm``
    One GMM basis is fit on the pooled resamples (every neuron contributes
    the same number of points). Each neuron becomes its average
    responsibility vector over the shared components; two neurons are
    compared with the mixture-Wasserstein distance -- OT between the weight
    vectors with closed-form Gaussian W2^2 between components as ground cost.

Each replicate returns ``D[i, j] = W(A_i, A_j)`` and the self-distance
``W(A_i, B_i)``, the resampling noise floor of each neuron.
"""

from __future__ import annotations

import numpy as np
import ot
from joblib import Parallel, delayed
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_clouds(clouds, eligible, n_points, coords, depth_mode, use_weights, rng):
    """Two independent weighted resamples per neuron, each (N, n_points, dim)."""
    dim = 3 if coords == "xyz" else 2
    draws = np.empty((2, eligible.size, n_points, dim))
    for row, index in enumerate(eligible):
        points = clouds.neuron_points(index, coords, depth_mode).astype(float)
        prob = None
        if use_weights:
            weight = clouds.neuron_weights(index).astype(float)
            prob = weight / weight.sum()
        for draw in range(2):
            pick = rng.choice(points.shape[0], size=n_points, replace=True, p=prob)
            draws[draw, row] = points[pick]
    return draws[0], draws[1]


# ---------------------------------------------------------------------------
# Distances between two neurons
# ---------------------------------------------------------------------------

def w2_points(x, y):
    """Exact W2 between two equal-size uniform point clouds."""
    cost = cdist(x, y, metric="sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, cols].mean()))


def gaussian_w2_sq(mean_a, cov_a, mean_b, cov_b):
    """Closed-form squared W2 between two Gaussians (Bures metric)."""
    root_b = np.real(sqrtm(cov_b))
    cross = np.real(sqrtm(root_b @ cov_a @ root_b))
    bures = np.trace(cov_a + cov_b - 2.0 * cross)
    return float(np.sum((mean_a - mean_b) ** 2) + max(bures, 0.0))


def w2_mixture(pi_a, pi_b, ground_cost):
    """Mixture-Wasserstein distance between weight vectors on a shared basis."""
    return float(np.sqrt(max(ot.emd2(pi_a, pi_b, ground_cost), 0.0)))


# ---------------------------------------------------------------------------
# Shared GMM basis
# ---------------------------------------------------------------------------

def fit_shared_basis(sample, n_components, seed):
    """GMM on the pooled (N, n_points, dim) sample; every neuron weighs equally."""
    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          n_init=3, random_state=seed, reg_covar=1e-3)
    return gmm.fit(sample.reshape(-1, sample.shape[-1]))


def basis_weights(gmm, sample):
    """Average responsibility of each neuron's points over the basis components."""
    n_neurons, n_points, dim = sample.shape
    resp = gmm.predict_proba(sample.reshape(-1, dim))
    pi = resp.reshape(n_neurons, n_points, -1).mean(axis=1)
    return pi / pi.sum(axis=1, keepdims=True)


def basis_ground_cost(gmm):
    """Pairwise squared W2 between the basis components."""
    k = gmm.n_components
    return np.array([[gaussian_w2_sq(gmm.means_[a], gmm.covariances_[a],
                                     gmm.means_[b], gmm.covariances_[b])
                      for b in range(k)] for a in range(k)])


# ---------------------------------------------------------------------------
# Parallel pairwise matrices
# ---------------------------------------------------------------------------

def balanced_row_blocks(n, n_blocks):
    """Split rows so every block holds roughly the same number of upper-triangle pairs."""
    pairs_per_row = np.arange(n - 1, -1, -1)
    cumulative = np.cumsum(pairs_per_row)
    edges = np.searchsorted(cumulative, np.linspace(0, cumulative[-1], n_blocks + 1)[1:-1])
    return [block for block in np.split(np.arange(n), edges) if block.size]


def _points_block(rows, sample):
    return [(i, [w2_points(sample[i], sample[j]) for j in range(i + 1, len(sample))])
            for i in rows]


def _mixture_block(rows, pi, ground_cost):
    return [(i, [w2_mixture(pi[i], pi[j], ground_cost) for j in range(i + 1, len(pi))])
            for i in rows]


def _assemble(n, block_results):
    matrix = np.zeros((n, n), dtype=np.float32)
    for block in block_results:
        for i, values in block:
            matrix[i, i + 1:] = values
    return matrix + matrix.T


def pairwise_points(sample, n_jobs):
    blocks = balanced_row_blocks(len(sample), 4 * n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(_points_block)(b, sample) for b in blocks)
    return _assemble(len(sample), results)


def pairwise_mixture(pi, ground_cost, n_jobs):
    blocks = balanced_row_blocks(len(pi), 4 * n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(_mixture_block)(b, pi, ground_cost) for b in blocks)
    return _assemble(len(pi), results)


# ---------------------------------------------------------------------------
# One bootstrap replicate
# ---------------------------------------------------------------------------

def compute_replicate(clouds, eligible, *, method, coords, depth_mode, n_points,
                      use_weights, n_components, rng, seed, n_jobs):
    """Distance matrix, self-distances and method extras for one replicate."""
    sample_a, sample_b = resample_clouds(clouds, eligible, n_points, coords, depth_mode,
                                         use_weights, rng)
    if method == "points":
        return {
            "distance": pairwise_points(sample_a, n_jobs),
            "self_distance": np.array([w2_points(a, b) for a, b in zip(sample_a, sample_b)]),
        }
    if method == "shared_gmm":
        gmm = fit_shared_basis(sample_a, n_components, seed)
        pi_a, pi_b = basis_weights(gmm, sample_a), basis_weights(gmm, sample_b)
        ground_cost = basis_ground_cost(gmm)
        return {
            "distance": pairwise_mixture(pi_a, ground_cost, n_jobs),
            "self_distance": np.array([w2_mixture(a, b, ground_cost)
                                       for a, b in zip(pi_a, pi_b)]),
            "pi": pi_a.astype(np.float32),
            "gmm_means": gmm.means_,
            "gmm_covariances": gmm.covariances_,
            "gmm_weights": gmm.weights_,
        }
    raise ValueError(f"unknown method {method!r}")
