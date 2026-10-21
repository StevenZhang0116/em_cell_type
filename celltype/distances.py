"""
Neuron-to-Neuron Wasserstein Distances
======================================
One bootstrap replicate draws, for every eligible neuron, two independent
resamples (A and B) of ``n_points`` synapses, with replacement and with
probability proportional to the border-compensation weight. Two neurons are
compared with the exact 2-Wasserstein distance between their resampled clouds
of soma-centred offsets ``(dx, dy, dz)``. With equal-size uniform samples,
optimal transport reduces to an assignment problem, solved exactly with
``linear_sum_assignment``.

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

def resample_clouds(clouds, eligible, n_points, use_weights, rng):
    """Two independent weighted resamples per neuron, each (N, n_points, 3)."""
    draws = np.empty((2, eligible.size, n_points, 3))
    for row, index in enumerate(eligible):
        points = clouds.neuron_points(index).astype(float)
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


def rotation_min_w2(x, y, n_angles=36):
    """Smallest W2 over ``n_angles`` rotations of ``y`` about the vertical (dy) axis.

    The reference that canonical alignment (``Clouds(align=True)``) tries to
    reach cheaply; the grid includes angle 0, so the result never exceeds ``w2_points``.
    """
    best = np.inf
    for angle in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        cos, sin = np.cos(angle), np.sin(angle)
        rotated = np.column_stack([cos * y[:, 0] + sin * y[:, 2], y[:, 1], -sin * y[:, 0] + cos * y[:, 2]])
        best = min(best, w2_points(x, rotated))
    return best


def alignment_check(sample, rng, n_pairs=100, n_angles=36):
    """For random pairs of clouds: W2 as given vs the minimum over in-plane rotations.

    Returns the pairs, the given distances and the rotation minima. A median
    ratio well above 1 means the in-plane orientation dominates the distance.
    """
    pairs = rng.choice(len(sample), size=(n_pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    given = np.array([w2_points(sample[i], sample[j]) for i, j in pairs])
    best = np.array([rotation_min_w2(sample[i], sample[j], n_angles) for i, j in pairs])
    return pairs, given, best


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


def assemble_symmetric(n, block_results):
    """Full symmetric matrix from per-row upper-triangle values."""
    matrix = np.zeros((n, n), dtype=np.float32)
    for block in block_results:
        for i, values in block:
            matrix[i, i + 1:] = values
    return matrix + matrix.T


def pairwise_points(sample, n_jobs):
    blocks = balanced_row_blocks(len(sample), 4 * n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(_points_block)(b, sample) for b in blocks)
    return assemble_symmetric(len(sample), results)


# ---------------------------------------------------------------------------
# One bootstrap replicate
# ---------------------------------------------------------------------------

def compute_replicate(clouds, eligible, *, n_points, use_weights, rng, n_jobs):
    """Distance matrix and self-distances for one replicate."""
    sample_a, sample_b = resample_clouds(clouds, eligible, n_points, use_weights, rng)
    return {
        "distance": pairwise_points(sample_a, n_jobs),
        "self_distance": np.array([w2_points(a, b) for a, b in zip(sample_a, sample_b)]),
    }
