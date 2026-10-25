"""
Sample-Matched Split-Half Diagnostics
====================================
Uniformly draw 2m distinct synapses per neuron and split them into two m-point
clouds, retaining (and separately normalising) their border weights. Compare
W2(A_i, B_i) with W2(A_i, B_j) on a fixed set of neuron pairs. The same neuron
cohort and pairs are used at every m and in every repetition.

This is a conditional sampling-variability baseline, not a calibrated null
test for equal distributions or discrete cell types. Repetitions reuse the
observed synapses; their percentiles are split ranges, not confidence intervals.
With Clouds(align=True), both halves inherit the full cloud's fixed rotation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .clustering import neuron_table
from .distances import FlatClouds, gather, paired, w2


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def half_sizes(values):
    """Sorted, unique positive m values (each comparison uses m vs m synapses)."""
    values = sorted({_integer(value, "points_per_half") for value in values})
    if not values:
        raise ValueError("at least one points_per_half value is required")
    return values


def split_indices(n_synapses, m, *, seed, cell_id, rep):
    """Uniform, disjoint halves; the draw is independent of neuron order and worker count."""
    m = _integer(m, "m")
    if n_synapses < 2 * m:
        raise ValueError(f"need at least {2 * m} synapses, got {n_synapses}")
    rng = np.random.default_rng([seed, 17, int(cell_id), rep, m])
    pick = rng.choice(n_synapses, size=2 * m, replace=False)
    return pick[:m], pick[m:]


def sample_pairs(n_neurons, n_pairs, seed):
    """Unique unordered pairs, uniform without replacement, without building an N x N matrix."""
    n_neurons = _integer(n_neurons, "n_neurons")
    n_pairs = _integer(n_pairs, "n_pairs", 0)
    total = n_neurons * (n_neurons - 1) // 2
    if not total or not n_pairs:
        return np.empty((0, 2), dtype=np.int64)
    ranks = np.sort(np.random.default_rng([seed, 18]).choice(total, size=min(total, n_pairs), replace=False))
    starts = np.concatenate([[0], np.cumsum(np.arange(n_neurons - 1, 0, -1, dtype=np.int64))])
    i = np.searchsorted(starts, ranks, side="right") - 1
    j = i + 1 + ranks - starts[i]
    return np.column_stack([i, j])


def split_clouds(flat, cell_ids, m, *, seed, rep):
    """Split already aligned clouds uniformly; original weights enter only as OT masses."""
    points_a, points_b, mass_a, mass_b = [], [], [], []
    for row, cell_id in enumerate(cell_ids):
        points, mass = flat.cloud(row)
        a, b = split_indices(len(points), m, seed=seed, cell_id=cell_id, rep=rep)
        points_a.append(points[a])
        points_b.append(points[b])
        mass_a.append(mass[a])
        mass_b.append(mass[b])
    return FlatClouds.from_list(points_a, mass_a), FlatClouds.from_list(points_b, mass_b)


def _pair_block(a, b, pairs):
    return [w2(*a.cloud(i), *b.cloud(j)) for i, j in pairs]


def pair_distances(a, b, pairs, n_jobs):
    """One m-vs-m W2(A_i, B_j) per selected pair; no all-pairs matrix."""
    if len(pairs) == 0:
        return np.empty(0, dtype=float)
    blocks = np.array_split(pairs, min(len(pairs), 4 * n_jobs))
    values = Parallel(n_jobs=n_jobs)(delayed(_pair_block)(a, b, block) for block in blocks)
    return np.concatenate(values)


def analyze(clouds, indices, points_per_half, *, n_reps=100, n_pairs=2000,
            use_weights=True, seed=20261019, n_jobs=1, log=None):
    """Raw split distances on a fixed cohort, with axes (m, repetition, neuron/pair).

    All selected neurons must support the largest m. The caller selects this
    cohort once, before running any m; smaller m never adds extra neurons.
    Pair indices address the returned cell_ids, not rows of the original Clouds.
    Spread and weight ESS describe the complete observed cloud; ESS is a weight
    concentration diagnostic, not a correction for spatial dependence.
    """
    sizes = half_sizes(points_per_half)
    n_reps = _integer(n_reps, "n_reps")
    n_pairs = _integer(n_pairs, "n_pairs", 0)
    seed = _integer(seed, "seed", 0)
    n_jobs = _integer(n_jobs, "n_jobs")
    indices = np.asarray(indices)
    if (indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= len(clouds))
            or np.unique(indices).size != indices.size):
        raise ValueError("indices must be a nonempty set of distinct valid neuron indices")
    if np.any(clouds.n_synapses[indices] < 2 * max(sizes)):
        raise ValueError(f"every selected neuron needs at least {2 * max(sizes)} synapses")
    cell_ids = clouds.cell_ids[indices]
    if np.unique(cell_ids).size != len(cell_ids) or np.any(cell_ids < 0):
        raise ValueError("selected cell IDs must be unique and nonnegative")
    flat = gather(clouds, indices, use_weights=use_weights)
    if (not np.isfinite(flat.points).all() or not np.isfinite(flat.mass).all()
            or np.any(flat.mass <= 0)):
        raise ValueError("clouds need finite coordinates and positive finite synapse weights")
    pairs = sample_pairs(len(indices), n_pairs, seed)
    within = np.empty((len(sizes), n_reps, len(indices)))
    between = np.empty((len(sizes), n_reps, len(pairs)))
    spread, ess = [], []
    for row in range(len(flat)):
        x, mass = flat.cloud(row)
        spread.append(np.sqrt(mass @ ((x - mass @ x) ** 2).sum(axis=1)))
        ess.append(1.0 / (mass @ mass))
    for level, m in enumerate(sizes):
        for rep in range(n_reps):
            a, b = split_clouds(flat, cell_ids, m, seed=seed, rep=rep)
            within[level, rep] = paired(a, b, n_jobs)
            between[level, rep] = pair_distances(a, b, pairs, n_jobs)
            if log is not None:
                log(f"m={m}: split {rep + 1}/{n_reps}, {len(indices)} self and {len(pairs)} between distances")
    return {"m_values": np.asarray(sizes), "eligible": indices, "cell_ids": cell_ids,
            "pairs": pairs, "self_distance": within, "between_distance": between,
            "spread_um": np.asarray(spread), "weight_ess": np.asarray(ess),
            "n_synapses": clouds.n_synapses[indices]}


def _quantiles(values):
    """Columnwise split percentiles; also supports a run with no between-neuron pairs."""
    if values.shape[1] == 0:
        return np.empty((3, 0))
    return np.percentile(values, [5, 50, 95], axis=0)


def result_tables(result, clouds):
    """Long tables per (m, neuron) and (m, pair), retaining both endpoints' own baselines."""
    neurons, pairs = [], []
    base = neuron_table(clouds, result["eligible"])
    base["spread_um"] = result["spread_um"]
    base["weight_ess"] = result["weight_ess"]
    base["weight_ess_fraction"] = result["weight_ess"] / result["n_synapses"]
    i, j = result["pairs"].T
    for level, m in enumerate(result["m_values"]):
        own = _quantiles(result["self_distance"][level])
        other = _quantiles(result["between_distance"][level])
        table = base.copy()
        table.insert(1, "m", int(m))
        for label, values in zip(("p05", "median", "p95"), own):
            table[f"self_{label}_um"] = values
        table["self_over_spread"] = np.divide(own[1], result["spread_um"],
                                              out=np.full(len(base), np.nan), where=result["spread_um"] > 0)
        neurons.append(table)
        table = pd.DataFrame({"m": np.full(len(i), m), "pt_root_id_i": result["cell_ids"][i],
                              "pt_root_id_j": result["cell_ids"][j]})
        for row, label in enumerate(("p05", "median", "p95")):
            table[f"between_{label}_um"] = other[row]
            table[f"self_i_{label}_um"] = own[row, i]
            table[f"self_j_{label}_um"] = own[row, j]
        pairs.append(table)
    return pd.concat(neurons, ignore_index=True), pd.concat(pairs, ignore_index=True)


def summary(result):
    """Descriptive summaries; no p-values or automatic cluster decisions."""
    levels = {}
    for level, m in enumerate(result["m_values"]):
        own = np.median(result["self_distance"][level], axis=0)
        other = np.median(result["between_distance"][level], axis=0)
        levels[str(m)] = {"median_neuron_self_um": float(np.median(own)),
                          "median_pair_between_um": float(np.median(other)) if other.size else None}
    return {"n_neurons": int(len(result["cell_ids"])), "n_pairs": int(len(result["pairs"])),
            "n_reps": int(result["self_distance"].shape[1]), "per_m": levels,
            "interpretation": "Conditional split variability, not a null test for cell types. "
                              "p05/p95 are repeated-split percentiles, not confidence intervals. "
                              "Repetitions and pairs are not independent biological replicates.",
            "alignment": "If enabled, both halves use the rotation estimated from the complete cloud."}
