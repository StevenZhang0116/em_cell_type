"""Split-half validation of the pipeline's bootstrap consensus clustering.

The neuron cohort and analysis settings are fixed before splitting synapses.
A chooses k using its own bootstrap replicates; B refits the same estimator at
that k. Repeated splits describe sensitivity to the observed synapses, not
independent biological replicates or a null model of discrete cell types.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from . import clustering, plotting
from .clouds import align_points
from .distances import FlatClouds, bootstrap, pairwise
from .split_half import split_clouds


VERSION = 1


def prepare_half(flat, *, align, use_weights):
    """Estimate the frame from this half alone, then set its transport masses.

    As in Clouds, orientation uses stored border weights even with --no-weights.
    The frame is held fixed during the inner bootstrap, as in the full pipeline.
    """
    points = align_points(flat.points, flat.mass, flat.offsets) if align else flat.points.copy()
    mass = flat.mass.copy() if use_weights else np.repeat(1.0 / flat.sizes, flat.sizes)
    return FlatClouds(points, mass, flat.offsets.copy())


def fit_half(flat, *, n_reps, seed, n_jobs, linkage, k_values, min_cluster_size, k=None):
    """Exact distances and the same bootstrap consensus estimator as Stage C."""
    distance = pairwise(flat, n_jobs)
    matrices = np.stack([pairwise(bootstrap(flat, np.random.default_rng([*seed, rep])), n_jobs)
                         for rep in range(n_reps)])
    fit = clustering.fit_consensus(matrices, linkage, k_values, min_cluster_size, k)
    return distance, fit


def fit_record(fit):
    """JSON-safe selection record; no replicate matrices are needed for a resume."""
    return {"status": fit["status"], "chosen_k": fit["chosen_k"],
            "k_selection": fit["k_selection"], "k_reached": fit["reached"],
            "stability_ari": {str(k): clustering.stability_summary(v)
                              for k, v in fit["stability"].items()},
            "pac": {str(k): float(v) if np.isfinite(v) else None for k, v in fit["pac"].items()}}


def heldout_separation(distance_b, labels_a):
    """B distances grouped ONLY by A labels; each unordered pair counted once."""
    i, j = np.triu_indices(len(labels_a), 1)
    accepted = (labels_a[i] > 0) & (labels_a[j] > 0)
    same = labels_a[i] == labels_a[j]
    within = distance_b[i[accepted & same], j[accepted & same]]
    between = distance_b[i[accepted & ~same], j[accepted & ~same]]
    evaluable = bool(within.size and between.size)
    return {"separation_status": "ok" if evaluable else "empty_pair_group",
            "n_within_pairs": int(within.size), "n_between_pairs": int(between.size),
            "within_median_um": float(np.median(within)) if within.size else None,
            "between_median_um": float(np.median(between)) if between.size else None,
            "delta_um": float(np.median(between) - np.median(within)) if evaluable else None}


def match_clusters(labels_a, labels_b):
    """One-to-one matching maximizing total Jaccard; zero is never a cluster.

    Denominators use complete A/B memberships, including neurons rejected by
    the other half. Unmatched A clusters have zero overlap and zero retention.
    """
    ca, cb = np.unique(labels_a[labels_a > 0]), np.unique(labels_b[labels_b > 0])
    overlap = np.array([[np.sum((labels_a == a) & (labels_b == b)) for b in cb] for a in ca],
                       dtype=int).reshape(len(ca), len(cb))
    na, nb = np.array([np.sum(labels_a == a) for a in ca]), np.array([np.sum(labels_b == b) for b in cb])
    union = na[:, None] + nb[None, :] - overlap
    jaccard = overlap / np.maximum(union, 1)
    rows, cols = linear_sum_assignment(-jaccard)
    matches = dict(zip(rows.tolist(), cols.tolist()))
    result = []
    for row, a in enumerate(ca):
        col = matches.get(row)
        count = int(overlap[row, col]) if col is not None else 0
        result.append({"cluster_a": int(a), "cluster_b": int(cb[col]) if col is not None else None,
                       "size_a": int(na[row]), "size_b": int(nb[col]) if col is not None else 0,
                       "overlap": count, "jaccard": float(jaccard[row, col]) if col is not None else 0.0,
                       "retained_fraction": float(count / na[row])})
    return result


def compare_halves(labels_a, labels_b, distance_b, *, status_a="ok", status_b="ok"):
    """Conditional ARI accompanied by coverage, rejection rates and cluster matches."""
    labels_a, labels_b = np.asarray(labels_a), np.asarray(labels_b)
    common = (labels_a > 0) & (labels_b > 0)
    n_common = int(common.sum())
    n_groups = [np.unique(labels[common]).size for labels in (labels_a, labels_b)]
    status = "ok"
    if status_a != "ok" or status_b != "ok":
        status = "fit_failed"
    elif n_common < 3 or min(n_groups) < 2 or max(n_groups) == n_common:
        status = "degenerate_common_cohort"
    metrics = {"status": status, "n_neurons": len(labels_a), "n_common": n_common,
               "joint_coverage": float(common.mean()),
               "outlier_rate_a": float(np.mean(labels_a == 0)), "outlier_rate_b": float(np.mean(labels_b == 0)),
               "ari": float(adjusted_rand_score(labels_a[common], labels_b[common])) if status == "ok" else None}
    # B need not fit successfully for its distances to test A's fixed partition.
    separation = heldout_separation(distance_b, labels_a)
    if status_a != "ok":
        separation.update(separation_status="fit_a_failed", delta_um=None)
    metrics.update(separation)
    return metrics, match_clusters(labels_a, labels_b)


def fit_split(a, b, *, n_reps, seed, n_jobs, linkage, k_values, min_cluster_size, k=None):
    """Freeze A's chosen k before fitting B; B cannot influence A selection."""
    settings = dict(n_reps=n_reps, n_jobs=n_jobs, linkage=linkage, min_cluster_size=min_cluster_size)
    distance_a, fit_a = fit_half(a, seed=[*seed, 0], k_values=k_values, k=k, **settings)
    if fit_a["status"] == "ok":
        chosen = fit_a["chosen_k"]
        distance_b, fit_b = fit_half(b, seed=[*seed, 1], k_values=[chosen], k=chosen, **settings)
        fit_b["k_selection"] = {"rule": "fixed from half A", "chosen": chosen}
    else:
        distance_b = pairwise(b, n_jobs)
        fit_b = {"status": "not_run_a_failed", "chosen_k": None, "labels": np.zeros(len(b), dtype=int),
                 "k_selection": {"rule": "A did not reach k", "chosen": None}, "reached": [],
                 "stability": {}, "pac": {}}
    metrics, matches = compare_halves(fit_a["labels"], fit_b["labels"], distance_b,
                                     status_a=fit_a["status"], status_b=fit_b["status"])
    return {"distance_a": distance_a, "distance_b": distance_b,
            "labels_a": fit_a["labels"], "labels_b": fit_b["labels"],
            "record": {"fit_a": fit_record(fit_a), "fit_b": fit_record(fit_b),
                       "metrics": metrics, "matches": matches}}


def write_json(path, value):
    """Atomic writes leave the previous completed result intact on interruption."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _save_checkpoint(path, result, options, m, rep):
    record = {**result["record"], "points_per_half": m, "rep": rep, "options": options}
    temporary = path.with_suffix(".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **{key: value for key, value in result.items() if key != "record"},
                            record=json.dumps(record, allow_nan=False))
    os.replace(temporary, path)
    return record


def _load_checkpoint(path, options, m, rep):
    with np.load(path, allow_pickle=False) as data:
        record = json.loads(str(data["record"]))
        if record["options"] != options or record["points_per_half"] != m or record["rep"] != rep:
            raise ValueError(f"checkpoint settings mismatch: {path}")
        n = len(options["cell_ids"])
        for key, shape in (("distance_a", (n, n)), ("distance_b", (n, n)), ("labels_a", (n,)), ("labels_b", (n,))):
            if data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError(f"invalid {key} in {path}")
    return record


def describe(values):
    """Descriptive spread over repeated splits, explicitly not a confidence interval."""
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return {"n": 0, "median": None, "q25": None, "q75": None}
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {"n": len(values), "median": float(median), "q25": float(q25), "q75": float(q75)}


def run(flat, options, outdir, *, n_jobs=1, overwrite=False):
    """Resume whole outer repeats, then regenerate tables, summaries and plots."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != options:
        raise ValueError("validation cache settings/source/cohort mismatch; use a new output directory")
    write_json(manifest, options)
    # The completion marker is rebuilt only after every requested repeat succeeds.
    (outdir / "summary.json").unlink(missing_ok=True)
    records, matches = [], []
    for m in options["points_per_half"]:
        for rep in range(options["outer_reps"]):
            path = outdir / f"m{m}_rep{rep:03d}.npz"
            if path.exists() and not overwrite:
                record = _load_checkpoint(path, options, m, rep)
            else:
                print(f"validation m={m}, split {rep + 1}/{options['outer_reps']}", flush=True)
                a, b = split_clouds(flat, np.asarray(options["cell_ids"]), m, seed=options["seed"], rep=rep)
                preparation = dict(align=options["align"], use_weights=options["use_weights"])
                a, b = prepare_half(a, **preparation), prepare_half(b, **preparation)
                result = fit_split(a, b, n_reps=options["inner_reps"], seed=[options["seed"], 29, m, rep],
                                   n_jobs=n_jobs, linkage=options["linkage"], k_values=options["k_values"],
                                   min_cluster_size=options["min_cluster_size"], k=options["k"])
                record = _save_checkpoint(path, result, options, m, rep)
            records.append(record)
            matches.extend({"points_per_half": m, "rep": rep, "fit_status": record["metrics"]["status"], **match}
                           for match in record["matches"])
    rows = [{"points_per_half": r["points_per_half"], "rep": r["rep"],
             "k_a": r["fit_a"]["chosen_k"], "status_a": r["fit_a"]["status"], "status_b": r["fit_b"]["status"],
             **r["metrics"]} for r in records]
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "repeats.csv", index=False)
    pd.DataFrame(matches, columns=["points_per_half", "rep", "fit_status", "cluster_a", "cluster_b", "size_a",
                                  "size_b", "overlap", "jaccard", "retained_fraction"]).to_csv(outdir / "cluster_matches.csv", index=False)
    summary = {"options": options, "interpretation": "Descriptive repeated splits of observed synapses; not independent biological replicates, confidence intervals, p-values, or a null model of cell types.",
               "full_fit": {"status": "not_compared", "k_full": None}, "by_m": {}}
    for m in options["points_per_half"]:
        selected = [r for r in records if r["points_per_half"] == m]
        frequency = {}
        for r in selected:
            if r["fit_a"]["status"] == "ok":
                k = str(r["fit_a"]["chosen_k"])
                frequency[k] = frequency.get(k, 0) + 1
        summary["by_m"][str(m)] = {
            "n_splits": len(selected), "k_a_frequency": frequency,
            "fit_failure_rate": float(np.mean([r["fit_a"]["status"] != "ok" or r["fit_b"]["status"] != "ok" for r in selected])),
            "ari_not_evaluable_rate": float(np.mean([r["metrics"]["ari"] is None for r in selected])),
            "separation_not_evaluable_rate": float(np.mean([r["metrics"]["delta_um"] is None for r in selected])),
            **{key: describe([r["metrics"][key] for r in selected])
               for key in ("ari", "joint_coverage", "outlier_rate_a", "outlier_rate_b", "delta_um")}}
    plotting.plot_split_cluster(table, outdir / "validation.png")
    # First evaluable repeat is reproducible and is not chosen for a high ARI.
    representative = next((r for r in records if r["metrics"]["status"] == "ok"), records[0])
    m, rep = representative["points_per_half"], representative["rep"]
    with np.load(outdir / f"m{m}_rep{rep:03d}.npz", allow_pickle=False) as data:
        plotting.plot_split_cluster_contingency(data["labels_a"], data["labels_b"],
                                                outdir / "representative_contingency.png", m, rep)
    summary["representative"] = {"points_per_half": m, "rep": rep, "rule": "first evaluable repeat, else first repeat"}
    write_json(outdir / "summary.json", summary)
    return outdir
