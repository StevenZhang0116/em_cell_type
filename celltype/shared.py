"""
Shared-Component Model
======================
A parametric alternative to the point-cloud W2 of ``distances.py``
(``--method shared``). One Gaussian mixture with ``n_components``
full-covariance components is fitted to the pooled soma-centred synapses of
the clustered neurons of a side (the fit sample is drawn with probability
proportional to the border weights, which makes the fit weighted). A neuron
is then described only by its mixing weights over these shared components,
estimated by EM on the weights alone with a Dirichlet prior ``alpha * pi``
(``pi`` = global proportions, ``alpha`` = pseudo-synapses). This is a convex
problem, so neurons with few synapses stay stable and are shrunk towards the
population instead of towards a noisy fit of their own.

Two neurons are compared with MW2 (Delon & Desolneux 2020): the exact
discrete optimal transport between their weight vectors with the squared W2
between the shared Gaussians as ground cost. MW2 is an upper bound of the W2
between the two mixture densities, so it stays in the same family as the
point-cloud distance and reuses its replicate machinery: a replicate
resamples every neuron's synapses with replacement (probability proportional
to the border weight) before estimating the weights, and the self-distance is
the MW2 between two such resamples of the same neuron.
"""

from __future__ import annotations

import numpy as np
import ot
from joblib import Parallel, delayed
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

from .distances import assemble_symmetric, balanced_row_blocks


# ---------------------------------------------------------------------------
# Gaussian geometry
# ---------------------------------------------------------------------------

def _sqrtm_spd(matrix):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T


def gaussian_w2_squared(mean_a, cov_a, mean_b, cov_b):
    """Closed-form squared 2-Wasserstein distance between two Gaussians."""
    root_b = _sqrtm_spd(cov_b)
    cross = _sqrtm_spd(root_b @ cov_a @ root_b)
    return float(((mean_a - mean_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * cross))


def gaussian_w2_cost(means, covariances):
    """Squared W2 between every pair of components, (K, K)."""
    k = means.shape[0]
    cost = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            cost[i, j] = cost[j, i] = gaussian_w2_squared(means[i], covariances[i],
                                                          means[j], covariances[j])
    return cost


# ---------------------------------------------------------------------------
# The shared mixture
# ---------------------------------------------------------------------------

class SharedGMM:
    """Fixed components; per-neuron fits touch only the mixing weights."""

    def __init__(self, weights, means, covariances):
        self.weights = np.asarray(weights, dtype=float)
        self.means = np.asarray(means, dtype=float)
        self.covariances = np.asarray(covariances, dtype=float)
        self.n_components, self.dim = self.means.shape
        self._cholesky = np.linalg.cholesky(self.covariances)
        log_det = 2 * np.log(np.diagonal(self._cholesky, axis1=1, axis2=2)).sum(axis=1)
        self._log_norm = -0.5 * (self.dim * np.log(2 * np.pi) + log_det)
        self.cost = gaussian_w2_cost(self.means, self.covariances)

    @classmethod
    def fit(cls, points, n_components, seed=0, reg_covar=1.0):
        """Full-covariance mixture on (unweighted) ``points``; returns (model, fit info)."""
        gmm = GaussianMixture(n_components=n_components, covariance_type="full", n_init=1,
                              max_iter=300, reg_covar=reg_covar, random_state=seed).fit(points)
        info = {"mean_log_likelihood": float(gmm.score(points)), "bic": float(gmm.bic(points)),
                "converged": bool(gmm.converged_), "n_iter": int(gmm.n_iter_),
                "n_fit_points": int(points.shape[0])}
        return cls(gmm.weights_, gmm.means_, gmm.covariances_), info

    def log_component_density(self, points):
        """log N(x | component k) for every point and component, (n, K)."""
        points = np.asarray(points, dtype=float)
        out = np.empty((points.shape[0], self.n_components))
        for k in range(self.n_components):
            z = solve_triangular(self._cholesky[k], (points - self.means[k]).T, lower=True)
            out[:, k] = self._log_norm[k] - 0.5 * (z ** 2).sum(axis=0)
        return out

    def weights_from_log_density(self, log_density, sample_weight=None, alpha=10.0, n_iter=30):
        """MAP mixing weights under Dirichlet(alpha * pi) by EM on the weights only.

        ``sample_weight`` is rescaled to sum to the number of synapses, so
        ``alpha`` is in units of synapses.
        """
        n = log_density.shape[0]
        weight = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        weight = weight * (n / weight.sum())
        theta = self.weights.copy()
        for _ in range(n_iter):
            log_resp = np.log(theta + 1e-300) + log_density
            resp = np.exp(log_resp - logsumexp(log_resp, axis=1, keepdims=True))
            theta = (alpha * self.weights + weight @ resp) / (alpha + n)
        return theta

    def mixing_weights(self, points, sample_weight=None, alpha=10.0, n_iter=30):
        return self.weights_from_log_density(self.log_component_density(points), sample_weight,
                                             alpha, n_iter)

    def save(self, path, **info):
        np.savez(path, weights=self.weights, means=self.means, covariances=self.covariances,
                 **{key: np.asarray(value) for key, value in info.items()})

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data["weights"], data["means"], data["covariances"])


# ---------------------------------------------------------------------------
# Pooled fit sample
# ---------------------------------------------------------------------------

def pooled_sample(clouds, indices, n_points, use_weights, rng):
    """``n_points`` synapses of ``indices`` drawn with probability ~ border weight."""
    points = np.concatenate([clouds.neuron_points(i) for i in indices]).astype(np.float32)
    prob = None
    if use_weights:
        weight = np.concatenate([clouds.neuron_weights(i) for i in indices]).astype(float)
        prob = weight / weight.sum()
    pick = rng.choice(points.shape[0], size=min(n_points, points.shape[0]), replace=True, p=prob)
    return points[pick].astype(float)


# ---------------------------------------------------------------------------
# MW2 between neurons
# ---------------------------------------------------------------------------

def mw2(theta_a, theta_b, cost):
    """Exact discrete OT between two weight vectors with squared-W2 ground cost."""
    a = np.asarray(theta_a, dtype=np.float64)
    b = np.asarray(theta_b, dtype=np.float64)
    return float(np.sqrt(max(ot.emd2(a / a.sum(), b / b.sum(), cost), 0.0)))


def _mw2_block(rows, theta, cost):
    return [(i, [mw2(theta[i], theta[j], cost) for j in range(i + 1, len(theta))]) for i in rows]


def pairwise_mw2(theta, cost, n_jobs):
    blocks = balanced_row_blocks(len(theta), 4 * n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(_mw2_block)(b, theta, cost) for b in blocks)
    return assemble_symmetric(len(theta), results)


# ---------------------------------------------------------------------------
# One bootstrap replicate
# ---------------------------------------------------------------------------

def compute_replicate(clouds, eligible, model, *, alpha, use_weights, rng, n_jobs):
    """Weights from two bootstrap resamples per neuron; MW2 matrix and self-distances."""
    theta = np.empty((2, eligible.size, model.n_components))
    for row, index in enumerate(eligible):
        points = clouds.neuron_points(index).astype(float)
        prob = None
        if use_weights:
            weight = clouds.neuron_weights(index).astype(float)
            prob = weight / weight.sum()
        log_density = model.log_component_density(points)
        for draw in range(2):
            pick = rng.choice(points.shape[0], size=points.shape[0], replace=True, p=prob)
            theta[draw, row] = model.weights_from_log_density(log_density[pick], alpha=alpha)
    return {
        "distance": pairwise_mw2(theta[0], model.cost, n_jobs),
        "self_distance": np.array([mw2(a, b, model.cost) for a, b in zip(theta[0], theta[1])]),
        "theta": theta[0].astype(np.float32),
    }
