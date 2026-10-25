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
from .distances import bootstrap, pairwise
from .split_half import split_clouds


VERSION = 1

# Settings that make two validation runs incompatible. The half sizes and the
# number of outer splits may grow (new splits are appended to the m files);
# every other option must match for the saved splits to be reused.
IDENTITY = ("version", "dataset", "side", "source", "cell_ids", "full_params", "inner_reps",
            "seed", "use_weights", "linkage", "min_cluster_size", "k", "k_values")


def _identity(options):
    return {key: options.get(key) for key in IDENTITY}


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


ARRAYS = ("distance_a", "distance_b", "labels_a", "labels_b")


def _save_splits(path, m, options, splits):
    """Rewrite ``m<m>.npz`` atomically with every finished outer split of this m, stacked along axis 0."""
    temporary = path.with_suffix(".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, points_per_half=m, options=json.dumps(options, allow_nan=False),
                            records=json.dumps([split["record"] for split in splits], allow_nan=False),
                            **{key: np.stack([split[key] for split in splits]) for key in ARRAYS})
    os.replace(temporary, path)


def _load_splits(path, options, m):
    """The outer splits saved in ``m<m>.npz``, one dict (arrays and record) per split, in order."""
    with np.load(path, allow_pickle=False) as data:
        if int(data["points_per_half"]) != m or _identity(json.loads(str(data["options"]))) != _identity(options):
            raise ValueError(f"checkpoint settings mismatch: {path}")
        records = json.loads(str(data["records"]))
        arrays = {key: data[key] for key in ARRAYS}
    count, n = len(records), len(options["cell_ids"])
    for key, shape in (("distance_a", (count, n, n)), ("distance_b", (count, n, n)),
                       ("labels_a", (count, n)), ("labels_b", (count, n))):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"invalid {key} in {path}")
    return [{**{key: arrays[key][i] for key in ARRAYS}, "record": record} for i, record in enumerate(records)]


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
    if manifest.exists() and not overwrite:
        previous = json.loads(manifest.read_text())
        changed = sorted(key for key in IDENTITY if previous.get(key) != options.get(key))
        if changed:
            print(f"validation settings or clouds changed ({', '.join(changed)}); recomputing every split",
                  flush=True)
            overwrite = True
    if overwrite:                       # never leave splits of other settings behind
        for stale in outdir.glob("m*.npz"):
            stale.unlink()
    write_json(manifest, options)
    # The completion marker is rebuilt only after every requested split succeeds.
    (outdir / "summary.json").unlink(missing_ok=True)
    records, matches = [], []
    for m in options["points_per_half"]:
        path = outdir / f"m{m}.npz"
        splits = _load_splits(path, options, m) if path.exists() and not overwrite else []
        for rep in range(len(splits), options["outer_reps"]):
            print(f"validation m={m}, split {rep + 1}/{options['outer_reps']}", flush=True)
            a, b = split_clouds(flat, np.asarray(options["cell_ids"]), m, seed=options["seed"], rep=rep)
            result = fit_split(a, b, n_reps=options["inner_reps"], seed=[options["seed"], 29, m, rep],
                               n_jobs=n_jobs, linkage=options["linkage"], k_values=options["k_values"],
                               min_cluster_size=options["min_cluster_size"], k=options["k"])
            result["record"] = {**result["record"], "points_per_half": m, "rep": rep}
            splits.append(result)
            _save_splits(path, m, options, splits)      # after every split, so an interrupted run resumes
        for split in splits[:options["outer_reps"]]:    # a file may hold more splits than requested now
            record = split["record"]
            records.append(record)
            matches.extend({"points_per_half": m, "rep": record["rep"], "fit_status": record["metrics"]["status"],
                            **match} for match in record["matches"])
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
    with np.load(outdir / f"m{m}.npz", allow_pickle=False) as data:
        plotting.plot_split_cluster_contingency(data["labels_a"][rep], data["labels_b"][rep],
                                                outdir / "representative_contingency.png", m, rep)
    summary["representative"] = {"points_per_half": m, "rep": rep, "rule": "first evaluable repeat, else first repeat"}
    write_json(outdir / "summary.json", summary)
    return outdir
