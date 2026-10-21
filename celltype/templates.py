"""
Cluster Templates: Cylindrically Constrained Weighted GMM
=========================================================
The same model as the Stage-1 fit in ``../mixture`` (``gmm_scratch.GMMWeighted``
with ``method="diagonal_constraint"``): a K-component Gaussian mixture on
soma-centred ``(dx, dy, dz)`` with diagonal covariances, where every component
shares one lateral centre ``(x0, z0)`` and has an isotropic lateral variance,
and only the depth mean and variance differ between components. The mixture
is therefore rotationally symmetric about the vertical axis through the
centre. Synapses carry border-compensation weights, and K is chosen by the
weighted BIC ``p * log(sum w) - 2 * sum w log p(x)``.

Differences from ``gmm_scratch``: the EM is vectorised and the shared centre
is updated with the exact weighted M-step (precision-weighted mean of the
lateral positions) instead of the plain average of the component means.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.special import logsumexp


LATERAL = [0, 2]   # dx, dz: the cortical plane
DEPTH = 1          # dy: cortical depth offset from the soma
_LOG_2PI = float(np.log(2 * np.pi))


class CylindricalGMM:
    """Weighted EM for the constrained mixture; parameters are plain arrays."""

    def __init__(self, n_components, max_iter=200, tol=1e-6, min_var=1.0):
        self.n_components = int(n_components)
        self.max_iter, self.tol, self.min_var = max_iter, tol, min_var
        self.weights = self.centre = self.lateral_var = self.depth_mean = self.depth_var = None
        self.mean_log_likelihood_ = None

    # -- densities ---------------------------------------------------------

    def _component_log_density(self, points):
        """log N(x | component c) for every point and component, (n, K)."""
        d2 = ((points[:, LATERAL] - self.centre) ** 2).sum(axis=1)[:, None]
        depth = points[:, DEPTH][:, None]
        lateral = -_LOG_2PI - np.log(self.lateral_var) - d2 / (2 * self.lateral_var)
        vertical = (-0.5 * (_LOG_2PI + np.log(self.depth_var))
                    - (depth - self.depth_mean) ** 2 / (2 * self.depth_var))
        return lateral + vertical

    def log_density(self, points):
        """log of the mixture density at each point, (n,)."""
        points = np.asarray(points, dtype=float)
        return logsumexp(np.log(self.weights) + self._component_log_density(points), axis=1)

    # -- fitting -----------------------------------------------------------

    def _initialise(self, points, weight, rng):
        k = self.n_components
        prob = weight / weight.sum()
        lateral, depth = points[:, LATERAL], points[:, DEPTH]
        self.weights = np.full(k, 1.0 / k)
        self.centre = prob @ lateral
        pick = rng.choice(points.shape[0], size=k, replace=points.shape[0] < k, p=prob)
        self.depth_mean = depth[pick].astype(float)
        lateral_var = (prob * ((lateral - self.centre) ** 2).sum(axis=1)).sum() / 2
        depth_var = prob @ (depth - prob @ depth) ** 2
        self.lateral_var = np.full(k, max(lateral_var, self.min_var))
        self.depth_var = np.full(k, max(depth_var, self.min_var))

    def _m_step(self, points, weight, responsibility):
        effective = weight[:, None] * responsibility                 # (n, K)
        total = effective.sum(axis=0) + 1e-300
        self.weights = total / total.sum()
        depth = points[:, DEPTH][:, None]
        self.depth_mean = (effective * depth).sum(axis=0) / total
        self.depth_var = np.maximum((effective * (depth - self.depth_mean) ** 2).sum(axis=0) / total,
                                    self.min_var)
        lateral = points[:, LATERAL]
        precision = (effective / self.lateral_var).sum(axis=1)   # (n,)
        self.centre = (precision @ lateral) / precision.sum()
        d2 = ((lateral - self.centre) ** 2).sum(axis=1)[:, None]
        self.lateral_var = np.maximum((effective * d2).sum(axis=0) / (2 * total), self.min_var)

    def _run_em(self, points, weight, rng):
        self._initialise(points, weight, rng)
        previous = -np.inf
        for _ in range(self.max_iter):
            log_joint = np.log(self.weights) + self._component_log_density(points)
            log_norm = logsumexp(log_joint, axis=1)
            current = float((weight * log_norm).sum() / weight.sum())
            self._m_step(points, weight, np.exp(log_joint - log_norm[:, None]))
            if abs(current - previous) < self.tol * max(1.0, abs(current)):
                break
            previous = current
        return current

    def fit(self, points, sample_weight=None, n_init=2, seed=0):
        """Best of ``n_init`` EM runs from weighted random starts."""
        points = np.asarray(points, dtype=float)
        weight = (np.ones(points.shape[0]) if sample_weight is None
                  else np.asarray(sample_weight, dtype=float))
        rng = np.random.default_rng(seed)
        best, best_value = None, -np.inf
        for _ in range(n_init):
            value = self._run_em(points, weight, rng)
            if value > best_value:
                best, best_value = self.to_dict(), value
        self.load(best)
        self.mean_log_likelihood_ = best_value
        return self

    # -- model selection ---------------------------------------------------

    def n_parameters(self):
        """Mixing weights, shared centre, depth means, lateral and depth variances."""
        k = self.n_components
        return (k - 1) + 2 + k + 2 * k

    def bic(self, points, sample_weight=None):
        points = np.asarray(points, dtype=float)
        weight = (np.ones(points.shape[0]) if sample_weight is None
                  else np.asarray(sample_weight, dtype=float))
        return float(self.n_parameters() * np.log(weight.sum())
                     - 2 * (weight * self.log_density(points)).sum())

    # -- (de)serialisation -------------------------------------------------

    def to_dict(self):
        return {"n_components": self.n_components, "weights": self.weights.tolist(),
                "centre": self.centre.tolist(), "lateral_var": self.lateral_var.tolist(),
                "depth_mean": self.depth_mean.tolist(), "depth_var": self.depth_var.tolist()}

    def load(self, params):
        self.n_components = params["n_components"]
        self.weights = np.asarray(params["weights"], dtype=float)
        self.centre = np.asarray(params["centre"], dtype=float)
        self.lateral_var = np.asarray(params["lateral_var"], dtype=float)
        self.depth_mean = np.asarray(params["depth_mean"], dtype=float)
        self.depth_var = np.asarray(params["depth_var"], dtype=float)
        return self

    @classmethod
    def from_dict(cls, params):
        return cls(params["n_components"]).load(params)


def _fit_one(points, weight, k, seed):
    model = CylindricalGMM(k).fit(points, weight, seed=seed)
    return model.to_dict(), model.bic(points, weight)


def fit_template(points, weight, k_values, seed=0, n_jobs=1):
    """Fit every K in ``k_values`` and keep the one with the lowest weighted BIC.

    Returns the chosen model and ``{K: BIC}``.
    """
    results = Parallel(n_jobs=min(n_jobs, len(k_values)))(
        delayed(_fit_one)(points, weight, k, seed + i) for i, k in enumerate(k_values))
    bics = {int(k): float(bic) for k, (_, bic) in zip(k_values, results)}
    best = int(min(bics, key=bics.get))
    return CylindricalGMM.from_dict(results[list(k_values).index(best)][0]), bics


def density_slice(model, dx, dy):
    """Mixture density on a (dx, dy) grid through the centre (dz = z0), for figures."""
    grid_x, grid_y = np.meshgrid(dx, dy)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel(),
                              np.full(grid_x.size, model.centre[1])])
    return np.exp(model.log_density(points)).reshape(grid_x.shape)
