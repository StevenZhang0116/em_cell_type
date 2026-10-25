"""
Neuron-to-Neuron Wasserstein Distances
======================================
Two neurons are compared with the exact 2-Wasserstein distance between their
complete soma-centred synapse clouds ``(dx, dy, dz)``: every synapse of each
neuron is a point whose mass is its border-compensation weight (normalised per
neuron), and the optimal transport between the two discrete measures is solved
exactly with the network simplex of POT (``ot.emd2``). Nothing is resampled for
the distance itself. ``max_synapses`` is an optional cap for the largest clouds
only (a uniform subsample without replacement, masses kept), because the cost
of one pair grows roughly with the cube of the cloud size.

The uncertainty of the distances comes from bootstrap replicates: a replicate
draws, for every neuron, two independent multinomial resamples of its masses
(n draws with replacement from its n synapses, probability proportional to the
weight), i.e. the complete cloud with bootstrap multiplicities. Each replicate
returns ``D[i, j] = W(A_i, A_j)`` and the self-distance ``W(A_i, B_i)``, the
bootstrap noise floor of each neuron.
"""

from __future__ import annotations

import numpy as np
import ot
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

# Network-simplex pivots allowed per pair; POT's default 1e5 is too few for clouds of thousands.
NUM_ITER_MAX = 10_000_000


# ---------------------------------------------------------------------------
# Clouds as one flat array
# ---------------------------------------------------------------------------

class FlatClouds:
    """Several clouds concatenated: ``points`` (N, 3), ``mass`` (N,) summing to 1 per cloud, ``offsets`` (n + 1,)."""

    def __init__(self, points, mass, offsets):
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.mass = np.ascontiguousarray(mass, dtype=np.float64)
        self.offsets = np.asarray(offsets, dtype=np.int64)

    def __len__(self):
        return self.offsets.size - 1

    @property
    def sizes(self):
        return np.diff(self.offsets)

    def cloud(self, i):
        """``(points, mass)`` of cloud ``i``."""
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.points[a:b], self.mass[a:b]

    @classmethod
    def from_list(cls, clouds, masses=None):
        """From a list of (n_i, 3) arrays and optional (n_i,) weights (uniform when None)."""
        sizes = np.array([len(cloud) for cloud in clouds], dtype=np.int64)
        if masses is None:
            masses = [np.ones(n) for n in sizes]
        mass = np.concatenate([np.asarray(m, dtype=float) / np.sum(m) for m in masses])
        return cls(np.concatenate(clouds), mass, np.concatenate([[0], np.cumsum(sizes)]))


def gather(clouds, indices, use_weights=True, max_synapses=None, seed=0):
    """The complete clouds of ``indices`` with the border weights as masses.

    ``max_synapses`` caps the largest clouds by a uniform subsample without
    replacement (the masses stay the weights, so the measure is unbiased). The
    subsample of a neuron depends on ``seed`` and its cell id only, so the exact
    matrix and every replicate see the same points of it.
    """
    points, masses = [], []
    for index in indices:
        x = clouds.neuron_points(index).astype(float)
        w = clouds.neuron_weights(index).astype(float) if use_weights else np.ones(len(x))
        if max_synapses and len(x) > max_synapses:
            rng = np.random.default_rng([seed, int(clouds.cell_ids[index])])
            keep = np.sort(rng.choice(len(x), size=int(max_synapses), replace=False))
            x, w = x[keep], w[keep]
        points.append(x)
        masses.append(w)
    return FlatClouds.from_list(points, masses)


def bootstrap(flat, rng):
    """One bootstrap of every cloud: multinomial multiplicities over its synapses, zero-mass points dropped."""
    points, masses = [], []
    for i in range(len(flat)):
        x, m = flat.cloud(i)
        counts = rng.multinomial(len(m), m / m.sum())
        keep = counts > 0
        points.append(x[keep])
        masses.append(counts[keep].astype(float))
    return FlatClouds.from_list(points, masses)


# ---------------------------------------------------------------------------
# Distances between two clouds
# ---------------------------------------------------------------------------

def w2(x, mass_x, y, mass_y):
    """Exact W2 between two weighted point clouds; sizes may differ, masses need not be normalised."""
    cost = cdist(x, y, metric="sqeuclidean")
    a = np.asarray(mass_x, dtype=np.float64)
    b = np.asarray(mass_y, dtype=np.float64)
    value, info = ot.emd2(a / a.sum(), b / b.sum(), cost, numItermax=NUM_ITER_MAX, numThreads=1,
                          log=True)
    if info.get("warning"):
        raise RuntimeError(f"ot.emd2 stopped early on clouds of {len(x)} and {len(y)} points: "
                           f"{info['warning']}; raise distances.NUM_ITER_MAX")
    return float(np.sqrt(max(value, 0.0)))


def w2_points(x, y):
    """Exact W2 between two uniform point clouds (equal masses; sizes may differ)."""
    return w2(x, np.ones(len(x)), y, np.ones(len(y)))


# ---------------------------------------------------------------------------
# Parallel pairwise matrices
# ---------------------------------------------------------------------------

def balanced_row_blocks(n, n_blocks, sizes=None):
    """Split rows 0..n-1 into blocks of roughly equal work over the upper-triangle pairs.

    With ``sizes`` the work of pair (i, j) counts as ``sizes[i] * sizes[j]``
    (the cost matrix of the pair); without, every pair counts one.
    """
    sizes = np.ones(n) if sizes is None else np.asarray(sizes, dtype=float)
    tail = np.cumsum(sizes[::-1])[::-1]                 # tail[i] = sum of sizes[i:]
    cumulative = np.cumsum(sizes * (tail - sizes))      # row i: sizes[i] * sum of sizes[i + 1:]
    if cumulative[-1] <= 0:
        return [np.arange(n)]
    edges = np.searchsorted(cumulative, np.linspace(0, cumulative[-1], n_blocks + 1)[1:-1])
    return [block for block in np.split(np.arange(n), edges) if block.size]


def _blocks(n, n_blocks):
    return [block for block in np.array_split(np.arange(n), max(1, min(n, n_blocks))) if block.size]


def _pairs_block(rows, flat):
    return [(i, [w2(*flat.cloud(i), *flat.cloud(j)) for j in range(i + 1, len(flat))]) for i in rows]


def assemble_symmetric(n, block_results):
    """Full symmetric matrix from per-row upper-triangle values."""
    matrix = np.zeros((n, n), dtype=np.float32)
    for block in block_results:
        for i, values in block:
            matrix[i, i + 1:] = values
    return matrix + matrix.T


def pairwise(flat, n_jobs):
    """Exact W2 between every pair of clouds, (n, n)."""
    blocks = balanced_row_blocks(len(flat), 4 * n_jobs, flat.sizes)
    results = Parallel(n_jobs=n_jobs)(delayed(_pairs_block)(block, flat) for block in blocks)
    return assemble_symmetric(len(flat), results)


def _paired_block(rows, a, b):
    return [w2(*a.cloud(i), *b.cloud(i)) for i in rows]


def paired(a, b, n_jobs):
    """W2 between cloud i of ``a`` and cloud i of ``b``, for every i."""
    results = Parallel(n_jobs=n_jobs)(delayed(_paired_block)(block, a, b)
                                      for block in _blocks(len(a), 4 * n_jobs))
    return np.concatenate(results)


# ---------------------------------------------------------------------------
# Exact matrix, bootstrap replicate
# ---------------------------------------------------------------------------

def exact_distances(flat, n_jobs):
    """The exact distance matrix of the complete clouds."""
    return {"distance": pairwise(flat, n_jobs)}


def bootstrap_replicate(flat, rng, n_jobs):
    """Distance matrix and self-distances of one bootstrap replicate (two draws per cloud)."""
    a = bootstrap(flat, rng)
    b = bootstrap(flat, rng)
    return {"distance": pairwise(a, n_jobs), "self_distance": paired(a, b, n_jobs)}
