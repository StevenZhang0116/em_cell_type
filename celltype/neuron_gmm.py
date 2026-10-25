"""
Per-Neuron Gaussian Mixtures with the ../mixture Fitter
=======================================================
An analysis beside the pipeline stages: fit the weighted Gaussian mixture of
``../mixture`` (``src/scratch_models/gmm_scratch.py``, class ``GMMWeighted``)
to the soma-centred synapse cloud of one neuron, or of any set of neurons,
with the recipe of the Stage-1 cell-type model in
``../mixture/src/modeling/mixture_all.py`` (its "relative" coordinate fit):

1. coordinates ordered ``(dx, dz, dy)``: the two lateral axes first, then the
   depth offset, as ``GMMWeighted``'s ``diagonal_constraint`` covariance method
   expects (one shared lateral centre, one lateral variance per component,
   free depth means and variances);
2. every synapse weighted by its border-compensation weight;
3. at most ``max_synapses`` synapses per repetition, drawn without replacement;
4. per repetition a random train/test split (``test_fraction``); every
   candidate k is fitted on the training synapses and scored by the weighted
   BIC on the test synapses;
5. k chosen with ``mix_helper.best_number_component(method="ordinal")``: the
   smallest k whose test BICs are not significantly different (paired
   Wilcoxon, alpha 0.05) from those of the k with the lowest mean BIC. The
   test cannot reject with fewer than 6 repetitions, so ``n_repeats`` must be
   at least ``MIN_REPEATS``;
6. a final fit with the chosen k on all (subsampled) synapses;
7. fitting stability: the models fitted at the chosen k in step 4 (one per
   repetition, each on its own training subsample and EM start) are kept and
   compared with the mixture Wasserstein distance of ``../mixture``
   (``distributions_handler.component_ot_distance``: optimal transport between
   the Gaussian components with the pairwise Gaussian W2 as cost, the ``"ot"``
   metric of its ``compare_gmm``), between every pair of them and from each to
   the final model. Optionally (``n_reruns``) the whole selection is repeated
   with independent seeds and the final models are compared the same way.

Candidate k are capped by the synapse count, ``n >= min_synapses_per_parameter
* n_parameters(k)`` (``mixture_all`` uses 20 synapses per parameter), so a
small cloud is not fitted with more components than it can support.

``../mixture`` is imported, not copied: ``mixture_modules`` puts its source
folders on ``sys.path`` for the import only. Component parameters are kept in
the ``(dx, dz, dy)`` order of the fitter; ``NeuronGMM.log_density`` takes the
package's usual ``(dx, dy, dz)`` points and rearranges them.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import importlib
import io
import json
import sys
import time
from typing import NamedTuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import LABEL_COLUMNS, MIXTURE_ROOT

MIXTURE_SOURCE_DIRS = ("src/scratch_models", "src/modeling")
MIXTURE_MODULE_NAMES = ("gmm_scratch", "mix_helper", "distributions_handler")
COLUMNS = (0, 2, 1)                  # (dx, dy, dz) -> (dx, dz, dy)
COLUMN_NAMES = ("dx", "dz", "dy")
METHODS = ("diagonal", "diagonal_constraint", "normal")
MIN_REPEATS = 6
STABILITY_COLUMNS = ("w2_within_median", "w2_within_max", "w2_to_final_mean", "spread_um",
                     "w2_within_rel", "n_stability_models", "w2_rerun_median", "rerun_k_agreement")

_modules = None


# ---------------------------------------------------------------------------
# The ../mixture code
# ---------------------------------------------------------------------------

class MixtureModules(NamedTuple):
    gmm_scratch: object           # GMMWeighted, the weighted EM fitter
    mix_helper: object            # best_number_component, the k-selection rule
    distributions: object         # component_ot_distance, mixture_total_variance


def mixture_modules():
    """The ../mixture modules (``MixtureModules``) imported once; ``sys.path`` is restored afterwards."""
    global _modules
    if _modules is None:
        folders = [MIXTURE_ROOT / name for name in MIXTURE_SOURCE_DIRS]
        missing = [str(folder) for folder in folders if not folder.is_dir()]
        if missing:
            raise FileNotFoundError(f"../mixture source folders not found: {missing}")
        saved = list(sys.path)
        try:
            for folder in folders:
                sys.path.insert(0, str(folder))
            modules = [importlib.import_module(name) for name in MIXTURE_MODULE_NAMES]
        finally:
            sys.path[:] = saved        # the modules also edit sys.path on import
        _modules = MixtureModules(*modules)
    return _modules


def arrange(points):
    """Soma-centred ``(dx, dy, dz)`` points (n, 3) in the fitter's ``(dx, dz, dy)`` order."""
    return np.asarray(points, dtype=float)[:, list(COLUMNS)]


def n_parameters(k, method):
    """Free parameters of a k-component 3-D mixture, as ``GMMWeighted`` counts them for its BIC."""
    model = mixture_modules().gmm_scratch.GMMWeighted(k=k, method=method)
    model.m = 3
    return int(model._num_parameters())


def fit_once(points, weight, k, method, seed, max_iter=50, epsilon=1e-3, n_starts=1):
    """One weighted EM fit of ``GMMWeighted`` on already arranged points.

    ``GMMWeighted`` starts EM once, from k random synapses as means and the
    global covariance for every component, and can stop in a poor optimum (for
    example every component wide and centred, when the cloud really has
    separated bands). ``n_starts > 1`` fits that many times with independent
    seeds and keeps the fit with the highest training log-likelihood;
    ``n_starts=1`` is what ``mixture_all`` does.
    """
    gmm_scratch = mixture_modules().gmm_scratch
    seeds = [seed, *np.random.default_rng(seed).integers(2**31, size=max(int(n_starts) - 1, 0))]
    best, best_log_likelihood = None, -np.inf
    for start in seeds:
        model = gmm_scratch.GMMWeighted(k=k, method=method, max_iter=max_iter, epsilon=epsilon,
                                        random_state=int(start))
        model.fit(points, sample_weight=weight)
        log_likelihood = model.calculate_log_likelihood(points, sample_weight=weight)[0]
        if log_likelihood > best_log_likelihood:
            best, best_log_likelihood = model, log_likelihood
    return best


def _model_from_parameters(k, method, weights, means, covariances):
    """A ``GMMWeighted`` carrying given parameters (``phi``, ``mu``, full ``sigma``), for scoring."""
    model = mixture_modules().gmm_scratch.GMMWeighted(k=k, method=method)
    model.n, model.m, model.dim = 0, 3, 3
    model.phi = np.asarray(weights, dtype=float)
    model.mu = [np.asarray(mean, dtype=float) for mean in means]
    model.sigma = [np.asarray(cov, dtype=float) for cov in covariances]
    return model


# ---------------------------------------------------------------------------
# A fitted neuron
# ---------------------------------------------------------------------------

@dataclass
class NeuronGMM:
    """Mixture parameters of one fit (axes ``(dx, dz, dy)``) with the k-selection record."""
    method: str
    k: int
    weights: np.ndarray               # (k,) mixing proportions
    means: np.ndarray                 # (k, 3)
    covariances: np.ndarray           # (k, 3, 3)
    n_synapses: int                   # synapses used by the final fit
    log_likelihood: float             # weighted mean log density of those synapses
    bic: float                        # weighted BIC on those synapses
    selection: dict                   # candidates, test_bic, test_log_likelihood, p_values, failures
    model: object = field(default=None, repr=False, compare=False)
    repeat_models: dict = field(default_factory=dict, repr=False, compare=False)   # k -> parameter dicts
    rerun_models: list = field(default_factory=list, repr=False, compare=False)    # final models of reruns
    stability: dict = field(default_factory=dict, compare=False)                   # see stability_summary

    @classmethod
    def from_model(cls, model, method, points, weight, selection, repeat_models=None):
        """From a fitted ``GMMWeighted``; ``sigma`` is always a list of full (3, 3) matrices."""
        log_likelihood, _ = model.calculate_log_likelihood(points, sample_weight=weight)
        return cls(method=method, k=int(model.k), weights=np.asarray(model.phi, dtype=float),
                   means=np.asarray(model.mu, dtype=float),
                   covariances=np.asarray(model.sigma, dtype=float),
                   n_synapses=int(len(points)), log_likelihood=float(log_likelihood),
                   bic=float(model.calculate_bic(points, sample_weight=weight)),
                   selection=selection, model=model, repeat_models=repeat_models or {})

    def _ensure_model(self):
        if self.model is None:
            self.model = _model_from_parameters(self.k, self.method, self.weights, self.means,
                                                self.covariances)
        return self.model

    def log_density(self, points, weight=None):
        """Per-synapse log density (n,) of ``(dx, dy, dz)`` points under this mixture."""
        arranged = arrange(points)
        return self._ensure_model().calculate_log_likelihood(arranged, sample_weight=weight)[1]

    def responsibilities(self, points):
        """Component posteriors (n, k) of ``(dx, dy, dz)`` points."""
        return self._ensure_model().predict_proba(arrange(points))

    def summary(self):
        """Scalar descriptors: mixture-averaged lateral and depth spread and depth centre, in um."""
        lateral_var = (self.covariances[:, 0, 0] + self.covariances[:, 1, 1]) / 2
        return {"lateral_sd_um": float(np.sqrt(self.weights @ lateral_var)),
                "depth_sd_um": float(np.sqrt(self.weights @ self.covariances[:, 2, 2])),
                "depth_centre_um": float(self.weights @ self.means[:, 2])}

    def to_dict(self):
        """Parameters, selection record and stability record (not the repeat/rerun models; see ``save_stability``)."""
        return {"method": self.method, "k": self.k, "axes": list(COLUMN_NAMES),
                "weights": self.weights.tolist(), "means": self.means.tolist(),
                "covariances": self.covariances.tolist(), "n_synapses": self.n_synapses,
                "log_likelihood": self.log_likelihood, "bic": self.bic, "selection": self.selection,
                "stability": self.stability}

    @classmethod
    def from_dict(cls, params):
        return cls(method=params["method"], k=int(params["k"]),
                   weights=np.asarray(params["weights"], dtype=float),
                   means=np.asarray(params["means"], dtype=float),
                   covariances=np.asarray(params["covariances"], dtype=float),
                   n_synapses=int(params["n_synapses"]), log_likelihood=float(params["log_likelihood"]),
                   bic=float(params["bic"]), selection=params["selection"],
                   stability=params.get("stability", {}))


# ---------------------------------------------------------------------------
# Mixture Wasserstein distances (../mixture's component OT distance)
# ---------------------------------------------------------------------------

def _arrays(model):
    """``(weights, means, covariances)`` of a ``NeuronGMM``, a parameter dict or a fitted ``GMMWeighted``."""
    if isinstance(model, dict):
        return tuple(np.asarray(model[key], dtype=float) for key in ("weights", "means", "covariances"))
    if hasattr(model, "phi"):
        return tuple(np.asarray(value, dtype=float) for value in (model.phi, model.mu, model.sigma))
    return model.weights, model.means, model.covariances


def _record(model):
    """Parameter dict ``{weights, means, covariances}`` (full (k, 3, 3) covariances) of any model."""
    return dict(zip(("weights", "means", "covariances"), _arrays(model)))


def _ot_covariances(covariances, method):
    """Covariances as ``component_ot_distance`` expects them: per-axis variances (k, 3) for the
    diagonal methods (closed-form Gaussian W2), the full (k, 3, 3) matrices for ``normal``."""
    covariances = np.asarray(covariances, dtype=float)
    return covariances if method == "normal" else np.diagonal(covariances, axis1=1, axis2=2)


def gmm_w2(a, b, method):
    """Mixture Wasserstein distance (um) between two mixtures fitted with the same ``method``.

    ``component_ot_distance`` of ../mixture (``distributions_handler``): optimal
    transport between the Gaussian components of the two mixtures with the
    squared Gaussian W2 as cost, the ``"ot"`` metric of ``compare_gmm``. It is an
    upper bound on the W2 distance between the mixtures and is defined for
    mixtures with different numbers of components. ``a``, ``b`` are ``NeuronGMM``
    objects or parameter dicts.
    """
    distributions = mixture_modules().distributions
    (w1, m1, c1), (w2, m2, c2) = _arrays(a), _arrays(b)
    return float(distributions.component_ot_distance(w1 / w1.sum(), m1, _ot_covariances(c1, method),
                                                     w2 / w2.sum(), m2, _ot_covariances(c2, method)))


def within_distances(models, method):
    """Condensed vector of ``gmm_w2`` over every pair of ``models`` (empty with fewer than two)."""
    return np.array([gmm_w2(models[i], models[j], method)
                     for i in range(len(models)) for j in range(i + 1, len(models))])


def distances_to(reference, models, method):
    """``gmm_w2`` from ``reference`` to each of ``models``."""
    return np.array([gmm_w2(reference, model, method) for model in models])


def total_spread(model):
    """Root of the mixture's total variance summed over the three axes (um): the RMS distance of a
    synapse from the mixture mean, ``mixture_total_variance`` of ../mixture (law of total variance)."""
    weights, means, covariances = _arrays(model)
    variance = mixture_modules().distributions.mixture_total_variance(
        weights, means, np.diagonal(covariances, axis1=1, axis2=2))
    return float(np.sqrt(np.sum(variance)))


def _median(values):
    return float(np.median(values)) if len(values) else float("nan")


def stability_summary(fit):
    """Fitting-stability record of a ``NeuronGMM`` from its repeat (and rerun) models.

    ``within``: ``gmm_w2`` between every pair of the models fitted at the chosen
    k during k selection (one per repetition, each on its own training subsample
    and EM start); ``to_final``: from each of them to the final model;
    ``spread_um``: the final model's total spread, so ``within_rel`` =
    ``within_median / spread_um`` is the instability relative to the cloud's
    size; ``per_k``: the median within distance at every candidate k; ``rerun``:
    the same comparison between the final models of independent reruns of the
    whole selection plus the main fit (first), with their k and log-likelihood
    and the fraction of reruns that chose the main fit's k (None without
    reruns). A rerun far from the main fit with a lower log-likelihood is EM
    stuck in a worse optimum (``GMMWeighted`` starts once, from k random
    synapses), not a different reading of the data.
    """
    models = fit.repeat_models.get(fit.k, [])
    within = within_distances(models, fit.method)
    to_final = distances_to(fit, models, fit.method)
    spread = total_spread(fit)
    record = {"n_models": len(models), "within": within.tolist(), "to_final": to_final.tolist(),
              "within_median": _median(within),
              "within_mean": float(np.mean(within)) if within.size else float("nan"),
              "within_max": float(np.max(within)) if within.size else float("nan"),
              "to_final_mean": float(np.mean(to_final)) if to_final.size else float("nan"),
              "spread_um": spread,
              "within_rel": _median(within) / spread if spread > 0 else float("nan"),
              "per_k": {int(k): {"n_models": len(group),
                                 "within_median": _median(within_distances(group, fit.method))}
                        for k, group in fit.repeat_models.items()},
              "rerun": None}
    if fit.rerun_models:
        finals = [fit, *fit.rerun_models]
        rerun_within = within_distances(finals, fit.method)
        rerun_k = [int(len(_arrays(model)[0])) for model in finals]
        record["rerun"] = {"n_models": len(finals), "k": rerun_k,
                           "log_likelihood": [fit.log_likelihood] +
                                             [float(m.get("log_likelihood", np.nan)) for m in fit.rerun_models],
                           "within": rerun_within.tolist(), "within_median": _median(rerun_within),
                           "k_agreement": float(np.mean([k == fit.k for k in rerun_k[1:]]))}
    return record


def between_distances(fits, n_pairs, seed=0):
    """``(i, j, w2)`` table of ``gmm_w2`` between the final models of random pairs of distinct
    fitted neurons (``i``, ``j`` index ``fits``; None entries are skipped): the between-neuron
    reference for the within-neuron distances. All pairs when there are at most ``n_pairs``."""
    valid = np.array([i for i, fit in enumerate(fits) if fit is not None])
    rng = np.random.default_rng(seed)
    n_all = len(valid) * (len(valid) - 1) // 2
    if n_all <= n_pairs:
        pairs = [(int(valid[a]), int(valid[b])) for a in range(len(valid)) for b in range(a + 1, len(valid))]
    else:
        pairs = set()
        while len(pairs) < n_pairs:
            a, b = rng.choice(valid, size=2, replace=False)
            pairs.add((int(min(a, b)), int(max(a, b))))
        pairs = sorted(pairs)
    rows = [(i, j, gmm_w2(fits[i], fits[j], fits[i].method)) for i, j in pairs]
    return pd.DataFrame(rows, columns=["i", "j", "w2"])


# ---------------------------------------------------------------------------
# Fitting one cloud
# ---------------------------------------------------------------------------

def supported_k(k_values, n_synapses, method, min_synapses_per_parameter):
    """``(candidates, underpowered)``: the k the cloud can support, or the smallest k alone."""
    supported = [int(k) for k in k_values
                 if n_synapses >= min_synapses_per_parameter * n_parameters(k, method)]
    if supported:
        return supported, False
    return [int(min(k_values))], True


def choose_k(candidates, test_bic):
    """k by ``mix_helper.best_number_component`` on a (candidates, repeats) BIC matrix, plus p-values.

    Falls back to the lowest mean BIC if the paired test cannot be run (e.g.
    identical BICs) and marks the fallback in the returned rule.
    """
    mix_helper = mixture_modules().mix_helper
    if len(candidates) == 1:
        return candidates[0], None, "single candidate"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            chosen, p_values = mix_helper.best_number_component(list(candidates), np.asarray(test_bic),
                                                                method="ordinal")
        if int(chosen) not in candidates:      # its placeholder 3 when no index was found
            raise ValueError(f"best_number_component returned k={chosen}, not a candidate")
        return int(chosen), p_values, "mix_helper.best_number_component(ordinal)"
    except ValueError as error:
        means = np.nanmean(test_bic, axis=1)
        return int(candidates[int(np.nanargmin(means))]), None, f"lowest mean test BIC ({error})"


def _select_once(arranged, weight, k_values, rng, method, n_repeats, max_synapses, test_fraction,
                 min_synapses_per_parameter, max_iter, epsilon, n_starts):
    """One run of the selection recipe on arranged points; the ``NeuronGMM`` keeps the repeat models."""
    n = arranged.shape[0]
    size = min(n, int(max_synapses))
    candidates, underpowered = supported_k(k_values, size, method, min_synapses_per_parameter)

    test_bic = np.full((len(candidates), n_repeats), np.nan)
    test_log_likelihood = np.full((len(candidates), n_repeats), np.nan)
    repeat_models = {k: [] for k in candidates}
    failures = []
    for repeat in range(n_repeats):
        pick = rng.choice(n, size=size, replace=False)
        n_test = max(1, int(round(test_fraction * size)))
        test, train = pick[:n_test], pick[n_test:]
        for i, k in enumerate(candidates):
            try:
                model = fit_once(arranged[train], weight[train], k, method,
                                 seed=int(rng.integers(2**31)), max_iter=max_iter, epsilon=epsilon,
                                 n_starts=n_starts)
                test_log_likelihood[i, repeat] = model.calculate_log_likelihood(
                    arranged[test], sample_weight=weight[test])[0]
                test_bic[i, repeat] = model.calculate_bic(arranged[test], sample_weight=weight[test])
                repeat_models[k].append(_record(model))
            except Exception as error:       # a failed restart is a missing value, as in mixture_all
                failures.append(f"k={k} repeat={repeat}: {type(error).__name__}: {error}")
    if np.all(np.isnan(test_bic)):
        raise RuntimeError(f"every fit failed: {failures[:3]}")

    chosen, p_values, rule = choose_k(candidates, test_bic)
    selection = {"candidates": candidates, "underpowered": underpowered,
                 "test_bic": test_bic.tolist(), "test_log_likelihood": test_log_likelihood.tolist(),
                 "p_values": None if p_values is None else [float(p) for p in p_values],
                 "rule": rule, "n_repeats": n_repeats, "test_fraction": test_fraction,
                 "max_synapses": int(max_synapses), "failures": failures}
    final = rng.choice(n, size=size, replace=False)
    model = fit_once(arranged[final], weight[final], chosen, method, seed=int(rng.integers(2**31)),
                     max_iter=max_iter, epsilon=epsilon, n_starts=n_starts)
    return NeuronGMM.from_model(model, method, arranged[final], weight[final], selection, repeat_models)


def select_and_fit(points, weight, k_values, *, method="diagonal", n_repeats=20, max_synapses=10000,
                   test_fraction=0.2, min_synapses_per_parameter=20, seed=0, max_iter=50,
                   epsilon=1e-3, n_starts=1, n_reruns=0):
    """Choose k by held-out BIC over repetitions, fit the chosen k on everything, record stability.

    ``points`` are soma-centred ``(dx, dy, dz)`` in um, ``weight`` the synapse
    weights (None: uniform). ``mixture_all`` uses 20 repetitions for cell-type
    fits and 50 for single-cell fits. ``n_starts`` EM starts per fit (see
    ``fit_once``; 1 as in ``mixture_all``). ``n_reruns`` extra, independently seeded
    runs of the whole selection (each as expensive as the main one) are compared
    with the main fit in the stability record; when the cloud has at most
    ``max_synapses`` synapses their final fits see the same synapses, so they
    differ only by the chosen k and the EM start, whereas the repeat models
    also differ by their training subsample. Returns a ``NeuronGMM`` whose
    ``stability`` is ``stability_summary``; raises if every fit failed.
    """
    if n_repeats < MIN_REPEATS:
        raise ValueError(f"n_repeats must be at least {MIN_REPEATS} for the paired test to reject")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    arranged = arrange(points)
    weight = np.ones(arranged.shape[0]) if weight is None else np.asarray(weight, dtype=float)
    rng = np.random.default_rng(seed)
    settings = (method, n_repeats, max_synapses, test_fraction, min_synapses_per_parameter, max_iter,
                epsilon, n_starts)
    fit = _select_once(arranged, weight, k_values, rng, *settings)
    for _ in range(int(n_reruns)):
        rerun = _select_once(arranged, weight, k_values, rng, *settings)
        fit.rerun_models.append({**_record(rerun), "k": rerun.k, "log_likelihood": rerun.log_likelihood})
    fit.stability = stability_summary(fit)
    return fit


# ---------------------------------------------------------------------------
# Fitting neurons of a Clouds file
# ---------------------------------------------------------------------------

def fit_neuron(clouds, index, k_values, *, use_weights=True, **options):
    """``select_and_fit`` on one neuron of ``clouds``; ``options`` as for ``select_and_fit``."""
    weight = clouds.neuron_weights(index) if use_weights else None
    return select_and_fit(clouds.neuron_points(index), weight, k_values, **options)


def _fit_or_none(position, points, weight, k_values, options):
    """``(position, fit, message, seconds)``: the worker's result for one neuron."""
    start = time.time()
    try:
        fit = select_and_fit(points, weight, k_values, **options)
        fit.model = None      # only parameters travel back to the parent; _ensure_model rebuilds the object
        return position, fit, None, time.time() - start
    except Exception as error:
        return position, None, f"{type(error).__name__}: {error}", time.time() - start


def fit_plan(clouds, indices, k_values, method="diagonal", max_synapses=10000,
             min_synapses_per_parameter=20, n_repeats=20, n_starts=1, n_reruns=0, **_):
    """One line describing the candidate k of the neurons and the EM fits per neuron, for the log."""
    tops, counts = [], []
    for index in indices:
        candidates, _ = supported_k(k_values, min(int(clouds.n_synapses[index]), int(max_synapses)),
                                    method, min_synapses_per_parameter)
        tops.append(candidates[-1])
        counts.append(len(candidates))
    em_fits = (1 + n_reruns) * (n_repeats * int(np.median(counts)) + 1) * n_starts
    return (f"candidate k per neuron: median {np.median(counts):.0f} values up to k={np.median(tops):.0f} "
            f"(largest k tried {max(tops)}, smallest {min(tops)}); about {em_fits} EM fits per neuron "
            f"({n_repeats} repeats x candidates x {n_starts} starts, {n_reruns} reruns), so the first "
            f"neuron reports only after all of its fits")


def fit_neurons(clouds, indices, k_values, *, use_weights=True, n_jobs=1, seed=0, log=None, every=1,
                **options):
    """One ``NeuronGMM`` per neuron (None where the fit failed) and the failure messages.

    Neurons run in parallel; each gets its own seed derived from ``seed`` and
    its position, so the result does not depend on ``n_jobs``. ``log`` (for
    example ``print``) reports the fitting plan, then one line per finished
    neuron (every ``every`` neurons) with its k, time, and the time left.
    """
    mixture_modules()        # import in the parent too: fails early, and keeps the parent able to unpickle
    tasks = []
    for position, index in enumerate(indices):
        weight = clouds.neuron_weights(index) if use_weights else None
        tasks.append((position, clouds.neuron_points(index), weight, {**options, "seed": [seed, position]}))
    if log:
        log(f"[gmm] {len(indices)} neurons on {n_jobs} CPUs; {fit_plan(clouds, indices, k_values, **options)}")
    start = time.time()
    results = [None] * len(indices)
    done = 0
    parallel = Parallel(n_jobs=n_jobs, return_as="generator_unordered")
    for position, fit, message, seconds in parallel(
            delayed(_fit_or_none)(position, points, weight, k_values, opts)
            for position, points, weight, opts in tasks):
        results[position] = (fit, message)
        done += 1
        if log and (done % every == 0 or done == len(indices)):
            elapsed = time.time() - start
            left = elapsed / done * (len(indices) - done)
            if fit is None:
                what = f"failed ({message})"
            else:
                candidates = fit.selection["candidates"]
                what = f"k={fit.k} of {candidates[0]}-{candidates[-1]}"
            log(f"[gmm] {done}/{len(indices)} neuron {int(clouds.cell_ids[indices[position]])} "
                f"({int(clouds.n_synapses[indices[position]])} synapses): {what}, {seconds:.0f} s; "
                f"elapsed {elapsed / 60:.1f} min, about {left / 60:.0f} min left")
    fits = [fit for fit, _ in results]
    errors = {int(index): message for index, (_, message) in zip(indices, results) if message}
    return fits, errors


def fits_table(fits, clouds, indices):
    """Per-neuron table: identity, covariates, dataset labels, chosen number of components, fit quality, spread."""
    rows = []
    for index, fit in zip(indices, fits):
        row = {"pt_root_id": int(clouds.cell_ids[index]), "n_synapses": int(clouds.n_synapses[index]),
               "soma_depth_um": float(clouds.meta["soma_y"][index])}
        for column in LABEL_COLUMNS:
            key = f"label_{column}"
            if key in clouds.meta:
                row[column] = clouds.meta[key][index]
        if fit is None:
            row.update(n_components=np.nan, n_fitted=0, log_likelihood=np.nan, bic=np.nan, candidates="",
                       lateral_sd_um=np.nan, depth_sd_um=np.nan, depth_centre_um=np.nan,
                       **{**dict.fromkeys(STABILITY_COLUMNS, np.nan), "n_stability_models": 0})
        else:
            stability, rerun = fit.stability, fit.stability.get("rerun")
            row.update(n_components=fit.k, n_fitted=fit.n_synapses, log_likelihood=fit.log_likelihood, bic=fit.bic,
                       candidates="-".join(map(str, fit.selection["candidates"])), **fit.summary(),
                       w2_within_median=stability.get("within_median", np.nan),
                       w2_within_max=stability.get("within_max", np.nan),
                       w2_to_final_mean=stability.get("to_final_mean", np.nan),
                       spread_um=stability.get("spread_um", np.nan),
                       w2_within_rel=stability.get("within_rel", np.nan),
                       n_stability_models=stability.get("n_models", 0),
                       w2_rerun_median=rerun["within_median"] if rerun else np.nan,
                       rerun_k_agreement=rerun["k_agreement"] if rerun else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def save_fits(path, fits, cell_ids, options):
    """JSON with the fitting options and one entry (parameters or null) per neuron."""
    with open(path, "w") as handle:
        json.dump({"options": options, "cell_ids": [int(c) for c in cell_ids],
                   "fits": [None if fit is None else fit.to_dict() for fit in fits]}, handle)


def save_stability(path, fits, cell_ids):
    """JSON with the models behind each neuron's stability record: the repeat models fitted at the
    chosen k and the final models of the reruns (null where the neuron's fit failed)."""
    def listed(record):
        return {key: np.asarray(value).tolist() for key, value in record.items()}

    entries = [None if fit is None else
               {"k": fit.k, "repeat_models": [listed(m) for m in fit.repeat_models.get(fit.k, [])],
                "rerun_models": [listed(m) for m in fit.rerun_models]}
               for fit in fits]
    with open(path, "w") as handle:
        json.dump({"axes": list(COLUMN_NAMES), "cell_ids": [int(c) for c in cell_ids],
                   "models": entries}, handle)


def load_stability(path):
    """``(entries, cell_ids)`` written by ``save_stability``, parameter lists as arrays."""
    with open(path) as handle:
        data = json.load(handle)
    entries = []
    for entry in data["models"]:
        if entry is None:
            entries.append(None)
            continue
        entries.append({"k": int(entry["k"]),
                        "repeat_models": [_record(m) for m in entry["repeat_models"]],
                        "rerun_models": [{**m, **_record(m)} for m in entry["rerun_models"]]})
    return entries, np.asarray(data["cell_ids"], dtype=np.int64)


def load_fits(path):
    """``(fits, cell_ids, options)`` written by ``save_fits``; failed neurons are None."""
    with open(path) as handle:
        data = json.load(handle)
    fits = [None if params is None else NeuronGMM.from_dict(params) for params in data["fits"]]
    return fits, np.asarray(data["cell_ids"], dtype=np.int64), data["options"]
