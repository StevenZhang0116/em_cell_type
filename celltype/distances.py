"""
Neuron-to-Neuron Wasserstein Distances
======================================
One bootstrap replicate draws, for every eligible neuron, two independent
resamples (A and B) of ``n_points`` synapses, with replacement and with
probability proportional to the border-compensation weight. Two neurons are
compared with the exact 2-Wasserstein distance between their resampled clouds
(3-D offsets or cylindrical ``(r, depth)``, see ``Clouds.neuron_points``).
With equal-size uniform samples, optimal transport reduces to an assignment
problem, solved exactly with ``linear_sum_assignment``.

Each replicate returns ``D[i, j] = W(A_i, A_j)`` and the self-distance
``W(A_i, B_i)``, the resampling noise floor of each neuron.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_clouds(clouds, eligible, n_points, coords, use_weights, rng):
    """Two independent weighted resamples per neuron, each (N, n_points, dim)."""
    dim = 3 if coords == "xyz" else 2
    draws = np.empty((2, eligible.size, n_points, dim))
    for row, index in enumerate(eligible):
        points = clouds.neuron_points(index, coords).astype(float)
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


# ---------------------------------------------------------------------------
# One bootstrap replicate
# ---------------------------------------------------------------------------

def compute_replicate(clouds, eligible, *, coords, n_points, use_weights, rng, n_jobs):
    """Distance matrix and self-distances for one replicate."""
    sample_a, sample_b = resample_clouds(clouds, eligible, n_points, coords, use_weights, rng)
    return {
        "distance": pairwise_points(sample_a, n_jobs),
        "self_distance": np.array([w2_points(a, b) for a, b in zip(sample_a, sample_b)]),
    }
