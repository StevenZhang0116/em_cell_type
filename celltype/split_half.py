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
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .clustering import neuron_table
from .distances import FlatClouds, gather, w2


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


def _half(flat, cell_ids, row, m, *, seed, rep):
    """Halves A and B of cloud ``row`` as ``(points_a, mass_a, points_b, mass_b)``.

    The masses are the cloud's weights of the chosen synapses, not yet normalised
    per half (``w2`` and ``FlatClouds.from_list`` do that).
    """
    points, mass = flat.cloud(row)
    a, b = split_indices(len(points), m, seed=seed, cell_id=cell_ids[row], rep=int(rep))
    return points[a], mass[a], points[b], mass[b]


def split_clouds(flat, cell_ids, m, *, seed, rep):
    """Split every cloud uniformly; the original weights enter only as OT masses."""
    halves = [_half(flat, cell_ids, row, m, seed=seed, rep=rep) for row in range(len(flat))]
    return (FlatClouds.from_list([h[0] for h in halves], [h[1] for h in halves]),
            FlatClouds.from_list([h[2] for h in halves], [h[3] for h in halves]))


def _distance_block(flat, cell_ids, m, seed, units):
    """One W2 per row ``(rep, i, j)`` of ``units``: half A of neuron i against half B of neuron j
    in split ``rep`` (``j == i`` is the within-neuron distance).

    The worker draws the halves it needs itself and keeps only those of the
    current repetition (``units`` are sorted by repetition).
    """
    values = np.empty(len(units))
    halves, current = {}, None
    for k, (rep, i, j) in enumerate(units):
        if rep != current:
            halves, current = {}, rep
        for row in (i, j):
            if row not in halves:
                halves[row] = _half(flat, cell_ids, row, m, seed=seed, rep=rep)
        points_a, mass_a, _, _ = halves[i]
        _, _, points_b, mass_b = halves[j]
        values[k] = w2(points_a, mass_a, points_b, mass_b)
    return values


def split_distances(flat, cell_ids, m, pairs, *, seed, n_reps, n_jobs, log=None, include_self=True):
    """Within and between distances of every split at one m: arrays ``(n_reps, N)`` and ``(n_reps, P)``.

    All repetitions go through one joblib call: the ``n_reps * (N + P)`` OT
    problems, each m by m and hence of about the same cost, are cut into
    ``4 * n_jobs`` blocks of consecutive (repetition, comparison) units. A split
    depends only on seed, cell id, repetition and m, so the result is the same
    for every ``n_jobs``. With include_self=False, only the requested pairs are
    evaluated and the returned self array has zero columns.
    """
    n = len(flat)
    rows = np.arange(n if include_self else 0, dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    comparisons = np.concatenate([np.column_stack([rows, rows]), pairs])   # (N + P, 2): within, then between
    if not len(comparisons):
        return np.empty((n_reps, 0)), np.empty((n_reps, 0))
    units = np.concatenate([np.column_stack([np.full(len(comparisons), rep), comparisons])
                            for rep in range(n_reps)])
    blocks = np.array_split(units, min(len(units), 4 * n_jobs))
    if log is not None:
        log(f"m={m}: {n_reps} splits x ({len(rows)} self + {len(pairs)} between) distances "
            f"in {len(blocks)} blocks on {n_jobs} CPUs")
    results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_distance_block)(flat, cell_ids, m, seed, block) for block in blocks)
    values, done = [], 0
    for block, result in zip(blocks, results):
        values.append(result)
        done += len(block)
        if log is not None:
            log(f"m={m}: {done}/{len(units)} distances")
    values = np.concatenate(values).reshape(n_reps, len(comparisons))
    return values[:, :len(rows)], values[:, len(rows):]


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
        within[level], between[level] = split_distances(flat, cell_ids, m, pairs, seed=seed,
                                                         n_reps=n_reps, n_jobs=n_jobs, log=log)
    return {"m_values": np.asarray(sizes), "eligible": indices, "cell_ids": cell_ids,
            "pairs": pairs, "self_distance": within, "between_distance": between,
            "spread_um": np.asarray(spread), "weight_ess": np.asarray(ess),
            "n_synapses": clouds.n_synapses[indices]}


def _quantiles(values):
    """Columnwise split percentiles; also supports a run with no between-neuron pairs."""
    if values.shape[1] == 0:
        return np.empty((3, 0))
    return np.percentile(values, [5, 50, 95], axis=0)


def cohort_table(result, clouds):
    """One metadata row per original cohort member, in the saved cohort order."""
    base = neuron_table(clouds, result["eligible"])
    base["spread_um"] = result["spread_um"]
    base["weight_ess"] = result["weight_ess"]
    base["weight_ess_fraction"] = result["weight_ess"] / result["n_synapses"]
    return base


def result_tables(result, clouds):
    """Long tables per (m, neuron) and (m, pair), retaining both endpoints' own baselines."""
    neurons, pairs = [], []
    base = cohort_table(result, clouds)
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
                              "Repetitions and pairs are not independent biological replicates."}


# Fixed, descriptive follow-up to a saved run; no selection on W2 or spread.
DEPTH_BINS = ("<=25", "25-50", "50-100", ">100")
MATCH_SETTINGS = {"depth_um": 25.0, "count_ratio": 1.25, "border_um": 50.0,
                  "ess_fraction": 0.1, "max_partners": 3}
MATCH_FEATURES = ("soma_depth_um", "log_n_synapses", "dist_border_um", "weight_ess_fraction")


def _clean_labels(values):
    # Stage A stores labels as strings, including stringified missing values.
    return values.astype("string").str.strip().replace(["", "nan", "None", "<NA>"], pd.NA)


def annotate_pairs(pairs, neurons):
    """Join exact IDs and classify depth gaps, label relations and E/I pairs."""
    columns = ["pt_root_id", "soma_depth_um", "cell_type", "classification_system",
               "n_synapses", "dist_border_um", "weight_ess_fraction", "spread_um"]
    meta = neurons.reindex(columns=columns).copy()
    if meta.pt_root_id.duplicated().any():
        raise ValueError("pair annotation needs one metadata row per neuron")
    for column in ("cell_type", "classification_system"):
        meta[column] = _clean_labels(meta[column])
    table = pairs.copy()
    for endpoint in ("i", "j"):
        key = f"pt_root_id_{endpoint}"
        if not table[key].isin(meta.pt_root_id).all():
            raise ValueError("pair IDs are missing from neuron metadata")
        table = table.merge(meta.rename(columns={c: key if c == "pt_root_id" else f"{c}_{endpoint}"
                                                 for c in columns}), on=key, how="left", validate="many_to_one")
    table["depth_gap_um"] = (table.soma_depth_um_i - table.soma_depth_um_j).abs()
    table["depth_bin"] = pd.cut(table.depth_gap_um, [-np.inf, 25, 50, 100, np.inf],
                                labels=DEPTH_BINS).astype("string").fillna("missing")
    known = table.cell_type_i.notna() & table.cell_type_j.notna()
    same = table.cell_type_i.eq(table.cell_type_j).fillna(False)
    table["label_relation"] = np.where(known, np.where(same, "same", "different"), "missing")
    systems = table.classification_system_i.map({"excitatory": "E", "inhibitory": "I"})
    other = table.classification_system_j.map({"excitatory": "E", "inhibitory": "I"})
    table["system_pair"] = (systems + "-" + other).replace("I-E", "E-I").fillna("missing")
    return table


def stratified_summary(pairs):
    """Random and targeted pairs stay separate, including edges belonging to both."""
    groups = ["m", "system_pair", "label_relation", "depth_bin"]
    rows = []
    for sample in ("random", "targeted"):
        for key, frame in pairs.loc[pairs[f"is_{sample}"]].groupby(groups, observed=True):
            rows.append({"sample": sample, **dict(zip(groups, key)), "n_pairs": len(frame),
                         "n_neurons": pd.concat([frame.pt_root_id_i, frame.pt_root_id_j]).nunique(),
                         "median_between_um": frame.between_median_um.median(),
                         "median_margin_um": frame.margin_median_um.median(),
                         "fraction_positive_margin": (frame.margin_median_um > 0).mean()})
    return pd.DataFrame(rows, columns=["sample", *groups, "n_pairs", "n_neurons", "median_between_um",
                                       "median_margin_um", "fraction_positive_margin"])


def match_partners(neurons):
    """Up to three greedy triplets per anchor, all three edges passing fixed calipers.

    Costs sum squared caliper-scaled gaps. Ties use exact IDs. No partner repeats
    within an anchor; partners can recur across anchors. Counts describe the
    greedy selection, not a maximum-cardinality matching.
    """
    table = neurons.sort_values("pt_root_id").reset_index(drop=True).copy()
    if table.pt_root_id.duplicated().any():
        raise ValueError("matching needs unique neuron IDs")
    for column in ("cell_type", "classification_system"):
        table[column] = _clean_labels(table[column])
    if "dist_border_um" not in table:
        table["dist_border_um"] = np.nan
    table["log_n_synapses"] = np.log(table.n_synapses.where(table.n_synapses > 0))
    table["match_status"] = "eligible"
    invalid = ~np.isfinite(table[list(MATCH_FEATURES)].to_numpy(dtype=float)).all(axis=1)
    invalid |= (table.dist_border_um < 0) | (table.weight_ess_fraction <= 0) | (table.weight_ess_fraction > 1 + 1e-10)
    table.loc[invalid, "match_status"] = "invalid_covariates"
    missing = table.cell_type.isna() | ~table.classification_system.isin(["excitatory", "inhibitory"])
    table.loc[missing, "match_status"] = "missing_label"
    valid = table.loc[table.match_status == "eligible"]
    limits = np.array([MATCH_SETTINGS["depth_um"], np.log(MATCH_SETTINGS["count_ratio"]),
                       MATCH_SETTINGS["border_um"], MATCH_SETTINGS["ess_fraction"]])
    x = valid[list(MATCH_FEATURES)].to_numpy(dtype=float)
    gaps = np.abs(x[:, None, :] - x[None, :, :]) / limits
    cost = (gaps ** 2).sum(axis=2)
    ids = valid.pt_root_id.to_numpy(dtype=np.int64)
    labels = valid.cell_type.to_numpy(dtype=str)
    systems = valid.classification_system.to_numpy(dtype=str)
    close = (gaps <= 1 + 1e-12).all(axis=2) & (systems[:, None] == systems[None, :])
    np.fill_diagonal(close, False)
    records = []
    for i, index in enumerate(valid.index):
        same = np.flatnonzero(close[i] & (labels == labels[i]))
        other = np.flatnonzero(close[i] & (labels != labels[i]))
        status = "no_same_label" if not len(same) else "no_other_label" if not len(other) else "no_joint_match"
        for rank in range(1, MATCH_SETTINGS["max_partners"] + 1):
            best = None
            for j in same:
                candidates = other[close[j, other]]
                if not len(candidates):
                    continue
                scores = cost[i, j] + cost[i, candidates] + cost[j, candidates]
                pos = int(np.argmin(scores))
                proposal = (float(scores[pos]), int(j), int(candidates[pos]))
                if best is None or proposal < best:
                    best = proposal
            if best is None:
                break
            score, j, k = best
            records.append((int(ids[i]), int(ids[j]), int(ids[k]), rank, score, labels[i], labels[k], systems[i]))
            status = "matched"
            same, other = same[same != j], other[other != k]
        table.loc[index, "match_status"] = status
    matches = pd.DataFrame(records, columns=["anchor_id", "same_id", "different_id", "partner_rank", "match_cost",
                                             "cell_type", "different_cell_type", "classification_system"])
    for column in ("anchor_id", "same_id", "different_id"):
        matches[column] = matches[column].astype("int64")
    table["n_partner_pairs"] = table.pt_root_id.map(matches.anchor_id.value_counts()).fillna(0).astype(int)
    for role in ("same", "different"):
        table[f"uses_as_{role}_partner"] = table.pt_root_id.map(matches[f"{role}_id"].value_counts()).fillna(0).astype(int)
    return matches, table


def extend_pairs(result, matches, flat, *, seed, n_jobs=1, log=None):
    """Preserve original cohort-index i<j orientation; compute missing pairs only.

    W2(A_i,B_j) is not in general W2(A_j,B_i) for a particular split. Never sort
    root IDs or reverse endpoints when mapping a cached pair to its anchor.
    """
    ids = pd.Index(result["cell_ids"])
    wanted = set()
    for role in ("same_id", "different_id"):
        a, b = ids.get_indexer(matches.anchor_id), ids.get_indexer(matches[role])
        if np.any(a < 0) or np.any(b < 0):
            raise ValueError("matched IDs are outside the original split-half cohort")
        wanted.update(tuple(sorted((int(i), int(j)))) for i, j in zip(a, b))
    original = {tuple(pair) for pair in result["pairs"]}
    missing = np.asarray(sorted(wanted - original), dtype=np.int64).reshape(-1, 2)
    if log is not None:
        log(f"{len(matches)} triplets; {len(wanted)} unique targeted edges: "
            f"{len(wanted & original)} reused, {len(missing)} new")
    extra = np.empty((*result["between_distance"].shape[:2], len(missing)))
    if len(missing):
        for level, m in enumerate(result["m_values"]):
            _, extra[level] = split_distances(flat, result["cell_ids"], int(m), missing, seed=seed,
                                              n_reps=extra.shape[1], n_jobs=n_jobs, log=log, include_self=False)
    pairs = np.concatenate([result["pairs"], missing])
    return {**result, "pairs": pairs,
            "between_distance": np.concatenate([result["between_distance"], extra], axis=2),
            "is_random": np.arange(len(pairs)) < len(original),
            "is_targeted": np.array([tuple(pair) in wanted for pair in pairs], dtype=bool)}


def matched_effects(result, matches):
    """Average partners within each split, then take split medians per anchor.

    Each anchor has equal weight in overall and per-label effects. Split ranges
    and positive fractions are descriptive, not biological confidence intervals.
    """
    lookup = {tuple(result["cell_ids"][pair]): index for index, pair in enumerate(result["pairs"])}
    lookup.update({(b, a): index for (a, b), index in list(lookup.items())})
    records = []
    for anchor, frame in matches.groupby("anchor_id"):
        same = [lookup[anchor, partner] for partner in frame.same_id]
        other = [lookup[anchor, partner] for partner in frame.different_id]
        for level, m in enumerate(result["m_values"]):
            distances = result["between_distance"][level]
            own, different = distances[:, same].mean(axis=1), distances[:, other].mean(axis=1)
            delta = different - own
            records.append({"anchor_id": int(anchor), "m": int(m), "n_partner_pairs": len(frame),
                            "cell_type": frame.cell_type.iloc[0], "classification_system": frame.classification_system.iloc[0],
                            "same_median_um": np.median(own), "different_median_um": np.median(different),
                            "delta_median_um": np.median(delta), "delta_p05_um": np.percentile(delta, 5),
                            "delta_p95_um": np.percentile(delta, 95), "fraction_positive_splits": (delta > 0).mean()})
    anchors = pd.DataFrame(records, columns=["anchor_id", "m", "n_partner_pairs", "cell_type", "classification_system",
                                             "same_median_um", "different_median_um", "delta_median_um",
                                             "delta_p05_um", "delta_p95_um", "fraction_positive_splits"])
    anchors["anchor_id"] = anchors.anchor_id.astype("int64")
    summaries = []
    for m in result["m_values"]:
        frame = anchors.loc[anchors.m == m]
        groups = ["classification_system", "cell_type"]
        for key, part in [(None, frame), *list(frame.groupby(groups))]:
            summaries.append({"m": int(m), "grouping": "overall" if key is None else "cell_type",
                              **dict(zip(groups, key or (None, None))), "n_anchors": len(part),
                              "n_with_3_pairs": int((part.n_partner_pairs == 3).sum()),
                              "median_delta_um": float(part.delta_median_um.median()) if len(part) else None,
                              "fraction_same_closer": float((part.delta_median_um > 0).mean()) if len(part) else None})
    return anchors, pd.DataFrame(summaries)


def matching_diagnostics(matches, cohort):
    """Coverage/reuse and equal-anchor covariate balance, including unmatched cases."""
    groups = ["classification_system", "cell_type"]
    coverage = cohort.groupby(groups, dropna=False).match_status.value_counts().unstack(fill_value=0)
    coverage["n_neurons"] = coverage.sum(axis=1)
    coverage["n_matched"] = coverage["matched"] if "matched" in coverage else 0
    coverage["n_with_3_pairs"] = cohort.groupby(groups, dropna=False).n_partner_pairs.agg(lambda x: int((x == 3).sum()))
    coverage["coverage"] = coverage.n_matched / coverage.n_neurons
    indexed, balance = cohort.set_index("pt_root_id"), []
    eligible = cohort.loc[~cohort.match_status.isin(["invalid_covariates", "missing_label"])]
    for column in (*MATCH_FEATURES, "spread_um"):
        anchor = indexed.loc[matches.anchor_id, column].to_numpy(dtype=float)
        same = indexed.loc[matches.same_id, column].to_numpy(dtype=float)
        other = indexed.loc[matches.different_id, column].to_numpy(dtype=float)
        averaged = pd.DataFrame({"anchor_id": matches.anchor_id.to_numpy(), "same": same, "other": other}).groupby("anchor_id").mean()
        scale = eligible[column].std(ddof=0)
        balance.append({"covariate": column,
                        "mean_same_partner": float(averaged.same.mean()) if len(matches) else None,
                        "mean_different_partner": float(averaged.other.mean()) if len(matches) else None,
                        "standardized_mean_difference": float((averaged.same.mean() - averaged.other.mean()) / scale)
                        if len(matches) and scale > 0 else None,
                        "max_anchor_same_gap": float(np.max(np.abs(anchor - same))) if len(matches) else None,
                        "max_anchor_different_gap": float(np.max(np.abs(anchor - other))) if len(matches) else None,
                        "max_partner_gap": float(np.max(np.abs(same - other))) if len(matches) else None})
    return coverage.reset_index(), pd.DataFrame(balance)
