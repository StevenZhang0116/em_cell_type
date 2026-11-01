"""Covariate-matched, descriptive comparisons of saved per-neuron GMMs.

Each anchor gets pairs of same-label and different-label partners, within its
classification_system. All three edges of each triplet must pass the calipers.
Matching never uses a model parameter or a Wasserstein distance.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from . import neuron_gmm


FEATURES = ("soma_depth_um", "log_n_synapses", "log_n_fitted", "dist_border_um",
            "weight_ess_fraction")
PARTNER_COUNTS = (1, 3, 5)
MATCH_COLUMNS = ("anchor_id", "same_id", "different_id", "cell_type", "different_cell_type",
                 "classification_system", "match_cost", "partner_rank")


def load_covariates(neurons_path, clouds_path):
    """Join by exact int64 IDs; read weights and metadata, not the XYZ arrays."""
    table = pd.read_csv(neurons_path, dtype={"pt_root_id": "int64", "cell_type": "string",
                                           "classification_system": "string"})
    if table.pt_root_id.duplicated().any():
        raise ValueError("duplicate neuron IDs in neurons.csv")
    with np.load(clouds_path, allow_pickle=False) as data:
        ids, offsets = data["cell_ids"], data["offsets"]
        if pd.Index(ids).has_duplicates:
            raise ValueError("duplicate neuron IDs in Stage A")
        rows = pd.Index(ids).get_indexer(table.pt_root_id)
        if np.any(rows < 0):
            raise ValueError("neurons.csv contains IDs missing from Stage A")
        if not np.array_equal(np.diff(offsets)[rows], table.n_synapses.to_numpy()):
            raise ValueError("synapse counts disagree with Stage A")
        if not np.allclose(data["soma_y"][rows], table.soma_depth_um, atol=1e-3, rtol=0):
            raise ValueError("soma depths disagree with Stage A")
        if "dist_border" not in data:
            raise ValueError("matching requires Stage-A dist_border metadata")
        table["dist_border_um"] = data["dist_border"][rows].astype(float) / 1000.0
        weight = data["weight"]
        means, ess = [], []
        for row in rows:
            w = weight[offsets[row]:offsets[row + 1]].astype(float)
            if not len(w) or not np.isfinite(w).all() or np.any(w <= 0):
                raise ValueError("selected clouds need positive finite synapse weights")
            means.append(float(w.mean()))
            ess.append(float(w.sum() ** 2 / np.dot(w, w)))
    table["mean_border_weight"] = means
    table["weight_ess"] = ess
    table["weight_ess_fraction"] = table.weight_ess / table.n_synapses
    for name in ("n_synapses", "n_fitted"):
        table[f"log_{name}"] = np.log(table[name].where(table[name] > 0))
    return table.sort_values("pt_root_id").reset_index(drop=True)


def match_triplets(table, *, depth_um=25.0, count_ratio=1.25, border_um=50.0, ess_fraction=0.1,
                   max_partners=1):
    """Greedily choose triplets by cost then IDs, without reusing partners per anchor.

    Cost sums ((covariate difference) / caliper)**2 over features and all three
    edges. The first triplet is identical for any max_partners.
    Partners may recur across anchors; reuse is reported. This greedy procedure
    does not maximize the number of feasible triplets.
    Returns matched triplets and the full cohort with exclusion/coverage status.
    """
    settings = np.array([depth_um, count_ratio, border_um, ess_fraction], dtype=float)
    if (not np.isfinite(settings).all() or depth_um <= 0 or count_ratio <= 1
            or border_um <= 0 or not 0 < ess_fraction <= 1):
        raise ValueError("need positive depth/border calipers, count ratio > 1, and 0 < ESS caliper <= 1")
    if not isinstance(max_partners, (int, np.integer)) or max_partners < 1:
        raise ValueError("max_partners must be a positive integer")
    limits = np.array([depth_um, np.log(count_ratio), np.log(count_ratio), border_um, ess_fraction])
    cohort = table.sort_values("pt_root_id").reset_index(drop=True).copy()
    cohort["match_status"] = "eligible"
    invalid = ~np.isfinite(cohort[list(FEATURES)].to_numpy(dtype=float)).all(axis=1)
    invalid |= (cohort.n_fitted <= 0) | (cohort.n_fitted > cohort.n_synapses)
    invalid |= (cohort.weight_ess_fraction <= 0) | (cohort.weight_ess_fraction > 1 + 1e-10)
    cohort.loc[invalid, "match_status"] = "invalid_covariates"
    missing_label = cohort[["cell_type", "classification_system"]].isna().any(axis=1)
    missing_label |= cohort[["cell_type", "classification_system"]].eq("").any(axis=1)
    cohort.loc[missing_label, "match_status"] = "missing_label"
    cohort.loc[cohort.n_components.isna(), "match_status"] = "fit_failed"
    active = cohort.index[cohort.match_status == "eligible"].to_numpy()
    valid = cohort.loc[active]
    x = valid[list(FEATURES)].to_numpy(dtype=float)
    ids = valid.pt_root_id.to_numpy(dtype=np.int64)
    labels = valid.cell_type.to_numpy(dtype=str)
    classes = valid.classification_system.to_numpy(dtype=str)
    close = np.ones((len(valid), len(valid)), dtype=bool)
    cost = np.zeros(close.shape, dtype=float)
    for column, limit in enumerate(limits):
        gap = np.abs(x[:, None, column] - x[None, :, column]) / limit
        close &= gap <= 1 + 1e-12
        cost += gap ** 2
    close &= classes[:, None] == classes[None, :]
    np.fill_diagonal(close, False)
    records = []
    for i in range(len(valid)):
        same = np.flatnonzero(close[i] & (labels == labels[i]))
        other = np.flatnonzero(close[i] & (labels != labels[i]))
        status = "no_same_label" if not len(same) else "no_other_label" if not len(other) else "no_joint_match"
        for rank in range(1, max_partners + 1):
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
            status = "matched"
            records.append((int(ids[i]), int(ids[j]), int(ids[k]), labels[i], labels[k], classes[i], score, rank))
            same = same[same != j]
            other = other[other != k]
        cohort.loc[active[i], "match_status"] = status
    matches = pd.DataFrame(records, columns=MATCH_COLUMNS)
    for name in ("anchor_id", "same_id", "different_id"):
        matches[name] = matches[name].astype("int64")
    cohort["n_partner_pairs"] = cohort.pt_root_id.map(Counter(matches.anchor_id)).fillna(0).astype(int)
    for role in ("same", "different"):
        uses = Counter(matches[f"{role}_id"])
        cohort[f"uses_as_{role}_partner"] = cohort.pt_root_id.map(uses).fillna(0).astype(int)
    return matches, cohort


def attach_distances(matches, cohort, models_path, between_path):
    """Reuse final-model distances in between.csv; compute only missing selected edges.

    Distances shared across ranks or anchors are evaluated only once. Existing
    repeat distances are not mixed with newly drawn models.
    """
    fits, ids, options = neuron_gmm.load_fits(models_path)
    if len(ids) != len(fits) or len(np.unique(ids)) != len(ids):
        raise ValueError("invalid model IDs")
    models = dict(zip(map(int, ids), fits))
    if set(models) != set(cohort.pt_root_id):
        raise ValueError("models.json and neurons.csv contain different neuron sets")
    for row in cohort.itertuples(index=False):
        fit = models[row.pt_root_id]
        if fit is None:
            if pd.notna(row.n_components):
                raise ValueError("model failure disagrees with neurons.csv")
        elif fit.k != row.n_components or fit.n_synapses != row.n_fitted:
            raise ValueError("model parameters disagree with neurons.csv")
    needed = {tuple(sorted((int(i), int(j))))
              for role in ("same_id", "different_id")
              for i, j in zip(matches.anchor_id, matches[role])}
    cached = {}
    if between_path.exists():
        between = pd.read_csv(between_path, dtype={"pt_root_id_a": "int64", "pt_root_id_b": "int64"})
        for row in between.itertuples(index=False):
            pair = tuple(sorted((int(row.pt_root_id_a), int(row.pt_root_id_b))))
            if pair not in needed:
                continue
            value = float(row.w2)
            if not np.isfinite(value) or value < 0:
                raise ValueError("invalid cached distance")
            if pair in cached and not np.isclose(cached[pair], value):
                raise ValueError("conflicting duplicate distances in between.csv")
            cached[pair] = value
    records = []
    for a, b in sorted(needed):
        if (a, b) in cached:
            value, source = cached[a, b], "between.csv"
        else:
            if models[a].method != models[b].method:
                raise ValueError("cannot compare different covariance methods")
            value = neuron_gmm.gmm_w2(models[a], models[b], models[a].method)
            source = "computed"
        if not np.isfinite(value) or value < 0:
            raise ValueError("non-finite or negative GMM distance")
        records.append((a, b, value, source))
    distances = pd.DataFrame(records, columns=["pt_root_id_a", "pt_root_id_b", "w2", "source"])
    lookup = {(a, b): value for a, b, value, _ in records}
    result = matches.copy()
    for role in ("same", "different"):
        result[f"{role}_w2_um"] = [lookup[tuple(sorted((int(i), int(j))))]
                                    for i, j in zip(result.anchor_id, result[f"{role}_id"])]
    result["delta_um"] = result.different_w2_um - result.same_w2_um
    return result, distances, options


def _effect(frame):
    """One row per anchor, including when that row averages several partner pairs."""
    return {"n_matched": len(frame),
            "median_delta_um": float(frame.delta_um.median()) if len(frame) else None,
            "fraction_same_closer": float((frame.delta_um > 0).mean()) if len(frame) else None,
            "fraction_tied": float((frame.delta_um == 0).mean()) if len(frame) else None}


def _max_reuse(matches):
    uses = pd.concat([matches.same_id, matches.different_id]).value_counts()
    return int(uses.max()) if len(uses) else 0


def covariate_balance(matches, cohort):
    """Equal anchor weight: average partners/gaps within each anchor before summarizing.

    The maximum gap still covers every selected pair. The standardizing SD uses
    the full eligible cohort, so all partner counts and populations share a scale.
    """
    indexed = cohort.set_index("pt_root_id")
    balance = []
    for column in (*FEATURES, "mean_border_weight"):
        anchor = indexed.loc[matches.anchor_id, column].to_numpy(dtype=float)
        same = indexed.loc[matches.same_id, column].to_numpy(dtype=float)
        other = indexed.loc[matches.different_id, column].to_numpy(dtype=float)
        scale = cohort.loc[~cohort.match_status.isin(["fit_failed", "missing_label", "invalid_covariates"]), column].std(ddof=0)
        averaged = pd.DataFrame({"anchor_id": matches.anchor_id.to_numpy(), "same": same, "other": other,
                                 "partner_gap": np.abs(same - other),
                                 "same_gap": np.abs(anchor - same),
                                 "different_gap": np.abs(anchor - other)}).groupby("anchor_id").mean()
        balance.append({"covariate": column,
                        "mean_same_partner": float(averaged.same.mean()) if len(same) else None,
                        "mean_different_partner": float(averaged.other.mean()) if len(other) else None,
                        "standardized_mean_difference": float((averaged.same.mean() - averaged.other.mean()) / scale)
                        if len(same) and scale > 0 else None,
                        "median_partner_gap": float(averaged.partner_gap.median()) if len(same) else None,
                        "max_partner_gap": float(np.max(np.abs(same - other))) if len(same) else None,
                        "median_anchor_same_gap": float(averaged.same_gap.median()) if len(same) else None,
                        "median_anchor_different_gap": float(averaged.different_gap.median()) if len(same) else None})
    return pd.DataFrame(balance)


def reports(matches, cohort):
    """Original one-partner report, plus coverage for the larger partner counts."""
    groups = ["classification_system", "cell_type"]
    coverage = cohort.groupby(groups, dropna=False).match_status.value_counts().unstack(fill_value=0)
    coverage["n_neurons"] = coverage.sum(axis=1)
    coverage["n_matched"] = coverage["matched"] if "matched" in coverage else 0
    coverage["coverage"] = coverage.n_matched / coverage.n_neurons
    for count in PARTNER_COUNTS:
        coverage[f"n_with_{count}_pairs"] = cohort.groupby(groups, dropna=False).n_partner_pairs.agg(
            lambda values: int((values >= count).sum()))
    coverage = coverage.reset_index()
    effects = pd.DataFrame([{**dict(zip(groups, key)), **_effect(frame)}
                            for key, frame in matches.groupby(groups)],
                           columns=[*groups, "n_matched", "median_delta_um", "fraction_same_closer", "fraction_tied"])
    report = {"n_neurons": len(cohort), **_effect(matches),
              "coverage": len(matches) / len(cohort) if len(cohort) else 0.0,
              "status_counts": {str(k): int(v) for k, v in cohort.match_status.value_counts().items()},
              "max_partner_reuse": _max_reuse(matches),
              "interpretation": "Descriptive association within covariate overlap and classification_system. "
                                "Positive delta means same-label partner closer. Neurons and partners recur; "
                                "no independent-pair p-values or confidence intervals."}
    return coverage, effects, covariate_balance(matches, cohort), report


def partner_robustness(matches, cohort):
    """Compare 1/3/5 pairs with equal anchor weights, including a fixed five-pair cohort."""
    groups = ["classification_system", "cell_type"]
    common = cohort.loc[cohort.n_partner_pairs >= max(PARTNER_COUNTS), "pt_root_id"]
    anchors, summaries, balances = [], [], []
    for count in PARTNER_COUNTS:
        selected = matches.loc[matches.partner_rank <= count]
        averaged = selected.groupby(["anchor_id", *groups]).agg(
            n_pairs=("partner_rank", "size"), same_w2_um=("same_w2_um", "mean"),
            different_w2_um=("different_w2_um", "mean"), delta_um=("delta_um", "mean"),
            mean_match_cost=("match_cost", "mean")).reset_index()
        averaged["requested_pairs"] = count
        averaged["in_common_5"] = averaged.anchor_id.isin(common)
        anchors.append(averaged)
        for population in ("all_matched", "common_5"):
            frame = averaged if population == "all_matched" else averaged.loc[averaged.in_common_5]
            edges = selected.loc[selected.anchor_id.isin(frame.anchor_id)]
            tags = {"population": population, "requested_pairs": count}
            for key, part in [(None, frame), *list(frame.groupby(groups))]:
                summaries.append({**tags, "grouping": "overall" if key is None else "cell_type",
                                  **dict(zip(groups, key or (None, None))), **_effect(part),
                                  "n_triplets": int(part.n_pairs.sum()),
                                  "n_full_requested": int((part.n_pairs == count).sum()),
                                  "mean_match_cost": float(part.mean_match_cost.mean()) if len(part) else None,
                                  "max_partner_reuse": _max_reuse(edges.loc[edges.anchor_id.isin(part.anchor_id)])})
            balances.extend(covariate_balance(edges, cohort).assign(**tags).to_dict(orient="records"))
    return pd.concat(anchors, ignore_index=True), pd.DataFrame(summaries), pd.DataFrame(balances)
