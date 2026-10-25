"""
Pipeline Stages
===============
The stages run by ``scripts/run_pipeline.py``. Each stage reads the previous
stage's files from ``output/<dataset>/`` and writes its own:

  A  extract_clouds           clouds_{axon,dendrite}.npz
     extract_all_dendrites    clouds_dendrite_all.npz (every cell, for --query all)
     validate_neurons         trials/<trial>/validate_<side>/<run>/ (optional: --stages ... validate;
                              disjoint synapse halves of the Stage B/C cohort, see split_cluster.py)
  B  compute_exact_distances  trials/<trial>/dist_<side>/exact.npz (--method points)
     fit_shared_model         trials/<trial>/shared_<side>.npz (--method shared only)
     compute_distances        trials/<trial>/dist_<side>/rep<b>.npz (bootstrap replicates)
  C  cluster_neurons          trials/<trial>/cluster_<side>/
  D  assign_neurons           trials/<trial>/assign_<side>_<query>/ (optional: --stages ... assign)
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa

from . import assign, clustering, plotting, shared, split_cluster
from .clouds import Clouds, build_side, load_cell_table, load_synapses, proofread_cells
from .config import (SIDES, allocated_cpus, assign_dir, cluster_dir, clouds_path, distance_dir,
                     file_fingerprint, shared_model_path, validation_dir)
from .distances import bootstrap_replicate, cross, exact_distances, gather, pairwise
from .split_half import half_sizes


def extract_clouds(dataset, lateral_cutoff):
    """Stage A: per-neuron synapse clouds for both sides."""
    # Arrow otherwise sizes its thread pools from the whole node.
    pa.set_cpu_count(allocated_cpus())
    pa.set_io_thread_count(allocated_cpus())

    start = time.time()
    cells = load_cell_table(dataset)
    side_cells = {side: proofread_cells(cells, side) for side in SIDES}
    for side in SIDES:
        print(f"[{side}] proofread cells: {len(side_cells[side])}")

    synapses = load_synapses(dataset,
                             side_cells["axon"]["pt_root_id"].to_numpy(),
                             side_cells["dendrite"]["pt_root_id"].to_numpy())
    print(f"Loaded {len(synapses)} synapses ({time.time() - start:.0f}s)")

    for side in SIDES:
        output = build_side(side_cells[side], synapses, side, lateral_cutoff)
        counts = np.diff(output["offsets"])
        print(f"[{side}] synapses per neuron: median {np.median(counts):.0f}, "
              f"min {counts.min()}, max {counts.max()}")
        path = clouds_path(dataset, side)
        np.savez_compressed(path, **output)
        print(f"Saved {path}")
    print(f"Stage A time {time.time() - start:.0f}s")


def extract_all_dendrites(dataset, lateral_cutoff):
    """Stage A, optional: incoming synapses of every cell in the table, proofread or not.

    These clouds are the ``--query all`` set of Stage D. Non-proofread dendrites
    may be truncated or merged, so their assignments carry that caveat.
    """
    pa.set_cpu_count(allocated_cpus())
    pa.set_io_thread_count(allocated_cpus())
    start = time.time()
    cells = load_cell_table(dataset)
    print(f"[dendrite, all cells] {len(cells)} cells")
    synapses = load_synapses(dataset, np.array([], dtype=np.int64), cells["pt_root_id"].to_numpy())
    print(f"Loaded {len(synapses)} synapses ({time.time() - start:.0f}s)")
    output = build_side(cells, synapses, "dendrite", lateral_cutoff)
    counts = np.diff(output["offsets"])
    print(f"[dendrite, all cells] synapses per cell: median {np.median(counts):.0f}, "
          f"min {counts.min()}, max {counts.max()}; {int(np.sum(counts == 0))} cells without synapses")
    path = clouds_path(dataset, "dendrite", "all")
    np.savez_compressed(path, **output)
    print(f"Saved {path} ({time.time() - start:.0f}s)")


def eligible_neurons(clouds, min_synapses, min_border_um=0.0):
    """Neurons with enough synapses whose soma is at least ``min_border_um`` from the volume border.

    Border truncation cuts arbors and produces spurious clusters; ``dist_border``
    is stored in nm, the threshold is given in um.
    """
    keep = clouds.n_synapses >= min_synapses
    if min_border_um > 0:
        if "dist_border" not in clouds.meta:
            raise ValueError("this dataset has no dist_border column; drop --min-border-distance")
        keep &= clouds.meta["dist_border"] >= min_border_um * 1000.0
    return np.flatnonzero(keep)


def select_neurons(clouds, min_synapses, n_neurons, seed, min_border_um=0.0):
    """Eligible neurons; optionally a random subset of ``n_neurons``.

    The subset depends only on ``seed``, so every replicate uses the same neurons.
    """
    eligible = eligible_neurons(clouds, min_synapses, min_border_um)
    if n_neurons is None or n_neurons >= eligible.size:
        return eligible
    return np.sort(np.random.default_rng(seed).choice(eligible, size=n_neurons, replace=False))


def validate_neurons(dataset, trial, side, *, points_per_half, outer_reps, params, overwrite=False):
    """Optional validation stage, on exactly the full pipeline's selected cohort."""
    if params["method"] != "points":
        raise ValueError("validate currently supports --method points only")
    sizes = half_sizes(points_per_half)
    if outer_reps < 1 or params["n_reps"] < 1:
        raise ValueError("outer and inner replicate counts must be positive")
    clouds = Clouds(clouds_path(dataset, side))
    eligible = select_neurons(clouds, params["min_synapses"], params["n_neurons"], params["seed"], params["min_border"])
    if eligible.size < 2:
        raise ValueError("validation needs at least two selected neurons")
    required = 2 * max(sizes)
    if np.any(clouds.n_synapses[eligible] < required):
        raise ValueError(f"selected cohort cannot supply two disjoint halves of {max(sizes)} synapses; "
                         f"raise --min-synapses to at least {required} or lower --validation-points-per-half; "
                         "validation never silently changes the full pipeline's cohort")
    if params["max_synapses"] is not None and params["max_synapses"] < required:
        raise ValueError(f"--max-synapses must be at least {required} for these validation half sizes")
    flat = gather(clouds, eligible, params["use_weights"], params["max_synapses"], params["seed"])
    options = {"version": split_cluster.VERSION, "dataset": dataset, "side": side,
               "source": file_fingerprint(clouds.path), "cell_ids": clouds.cell_ids[eligible].tolist(),
               "full_params": params, "points_per_half": sizes, "outer_reps": int(outer_reps),
               "inner_reps": params["n_reps"], "seed": params["seed"],
               "use_weights": params["use_weights"], "linkage": params["linkage"],
               "min_cluster_size": params["min_cluster_size"], "k": params["k"],
               "k_values": list(range(params["k_min"], params["k_max"] + 1))}
    outdir = validation_dir(dataset, trial, side, options)
    split_cluster.run(flat, options, outdir, n_jobs=allocated_cpus(), overwrite=overwrite)
    print(f"Validation outputs in {outdir}")
    return outdir


def compare_validation_to_full(outdir, dataset, trial, side):
    """Report k_full coverage without using the held-out results to choose k."""
    path = outdir / "summary.json"
    summary = json.loads(path.read_text())
    full_path = cluster_dir(dataset, trial, side) / "summary.json"
    comparison = {"status": "full_fit_missing", "k_full": None}
    if full_path.exists():
        full = json.loads(full_path.read_text())
        options = summary["options"]
        comparison["k_full"] = full["chosen_k"]
        if full.get("source_sha256") is None:
            comparison["status"] = "full_fit_provenance_unknown"
        elif (full["source_sha256"] != options["source"]["sha256"]
              or full.get("cell_ids") != options["cell_ids"]
              or any(full["run"].get(key) != value for key, value in options["full_params"].items())):
            comparison["status"] = "full_fit_mismatch"
        else:
            comparison.update(status="compatible", note="k coverage is descriptive, not a validation pass/fail decision")
            comparison["k_full_selection_frequency_by_m"] = {
                m: {"count": stats["k_a_frequency"].get(str(full["chosen_k"]), 0), "n_splits": stats["n_splits"]}
                for m, stats in summary["by_m"].items()}
    summary["full_fit"] = comparison
    split_cluster.write_json(path, summary)


def _source_sha256(clouds):
    """Hash once per loaded cloud object, not once per bootstrap replicate."""
    if not hasattr(clouds, "_source_sha256"):
        clouds._source_sha256 = file_fingerprint(clouds.path)["sha256"]
    return clouds._source_sha256


def _distance_provenance(directory, clouds, eligible):
    """Reject known stale inputs; mark legacy inputs without fingerprints unknown."""
    current = _source_sha256(clouds)
    known = True
    paths = sorted(directory.glob("rep*.npz"))
    if (directory / "exact.npz").exists():
        paths.append(directory / "exact.npz")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if "source_sha256" not in data:
                known = False
            elif str(data["source_sha256"]) != current:
                raise ValueError(f"{path} was computed from different clouds; rerun distances with --overwrite")
            if "cell_ids" in data and not np.array_equal(data["cell_ids"], clouds.cell_ids[eligible]):
                raise ValueError(f"{path} has different cell IDs; rerun distances with --overwrite")
    return current if known else None


def fit_shared_model(clouds, dataset, trial, side, *, min_synapses, n_neurons, seed, use_weights,
                     n_components, fit_points, min_border_um=0.0):
    """Stage B (``--method shared``): the side's shared mixture, fitted once per trial."""
    path = shared_model_path(dataset, trial, side)
    if path.exists():
        print(f"[{side}] loading shared mixture {path}")
        return shared.SharedGMM.load(path)
    start = time.time()
    eligible = select_neurons(clouds, min_synapses, n_neurons, seed, min_border_um)
    sample = shared.pooled_sample(clouds, eligible, fit_points, use_weights,
                                  np.random.default_rng([seed, 6]))
    print(f"[{side}] fitting {n_components} shared components on {sample.shape[0]} synapses "
          f"pooled from {eligible.size} neurons")
    model, info = shared.SharedGMM.fit(sample, n_components, seed=seed)
    model.save(path, **info, n_neurons=eligible.size)
    print(f"[{side}] shared mixture: mean log-likelihood {info['mean_log_likelihood']:.3f}, "
          f"converged {info['converged']} in {info['n_iter']} iterations; "
          f"component W2 range {np.sqrt(model.cost[model.cost > 0]).min():.0f} - "
          f"{np.sqrt(model.cost.max()):.0f} um ({time.time() - start:.0f}s)")
    return model


def _report_eligible(clouds, side, min_synapses, n_neurons, seed, min_border_um, what, n_jobs):
    """The neurons of a Stage B computation, with the selection printed."""
    eligible = select_neurons(clouds, min_synapses, n_neurons, seed, min_border_um)
    n_enough = int(np.sum(clouds.n_synapses >= min_synapses))
    n_border = n_enough - eligible_neurons(clouds, min_synapses, min_border_um).size
    print(f"[{side}] {n_enough} / {len(clouds)} neurons have >= {min_synapses} synapses, "
          f"{n_border} of them within {min_border_um:g} um of the border; using {eligible.size}; "
          f"{what}; {n_jobs} CPUs")
    return eligible


def compute_exact_distances(clouds, dataset, trial, side, *, min_synapses, use_weights, seed,
                            n_neurons=None, min_border_um=0.0, max_synapses=None):
    """Stage B (``--method points``): the exact W2 matrix of the complete clouds, ``exact.npz``.

    Every synapse enters with its border weight as mass; nothing is resampled
    (``distances.py``).
    """
    start = time.time()
    n_jobs = allocated_cpus()
    eligible = _report_eligible(clouds, side, min_synapses, n_neurons, seed, min_border_um,
                                f"exact W2 of the complete clouds (cap {max_synapses or 'none'})",
                                n_jobs)
    flat = gather(clouds, eligible, use_weights, max_synapses, seed)
    sizes = flat.sizes
    print(f"synapses per cloud: median {np.median(sizes):.0f}, max {sizes.max()}; "
          f"{eligible.size * (eligible.size - 1) // 2} pairs")
    result = exact_distances(flat, n_jobs)
    path = distance_dir(dataset, trial, side) / "exact.npz"
    np.savez_compressed(path, eligible=eligible, cell_ids=clouds.cell_ids[eligible], method="points",
                        source_sha256=_source_sha256(clouds),
                        max_synapses=max_synapses or 0, n_synapses=sizes, min_synapses=min_synapses,
                        min_border_um=min_border_um, use_weights=use_weights, **result)
    upper = result["distance"][np.triu_indices(eligible.size, 1)]
    print(f"median pairwise distance {np.median(upper):.2f} um")
    print(f"Saved {path} ({time.time() - start:.0f}s)")


def compute_distances(clouds, dataset, trial, side, rep, *, min_synapses, use_weights, seed,
                      n_neurons=None, min_border_um=0.0, method="points", shared_model=None,
                      shared_alpha=10.0, max_synapses=None):
    """Stage B: one bootstrap replicate of the neuron-to-neuron distance matrix, ``rep<b>.npz``.

    ``method="points"``: W2 between the complete clouds with bootstrap
    multiplicities (``distances.bootstrap_replicate``); ``method="shared"``:
    MW2 between mixing weights estimated from bootstrapped synapses over
    ``shared_model`` (``shared.py``).
    """
    start = time.time()
    n_jobs = allocated_cpus()
    eligible = _report_eligible(clouds, side, min_synapses, n_neurons, seed, min_border_um,
                                f"replicate {rep}; method {method}", n_jobs)
    rng = np.random.default_rng([seed, rep])
    if method == "points":
        flat = gather(clouds, eligible, use_weights, max_synapses, seed)
        result = bootstrap_replicate(flat, rng, n_jobs)
    else:
        result = shared.compute_replicate(clouds, eligible, shared_model, alpha=shared_alpha,
                                          use_weights=use_weights, rng=rng, n_jobs=n_jobs)

    path = distance_dir(dataset, trial, side) / f"rep{rep:03d}.npz"
    np.savez_compressed(path, eligible=eligible, cell_ids=clouds.cell_ids[eligible],
                        source_sha256=_source_sha256(clouds),
                        method=method, max_synapses=max_synapses or 0, min_synapses=min_synapses,
                        min_border_um=min_border_um, use_weights=use_weights, **result)
    upper = result["distance"][np.triu_indices(eligible.size, 1)]
    print(f"median self-distance {np.median(result['self_distance']):.2f} um; "
          f"median pairwise distance {np.median(upper):.2f} um")
    print(f"Saved {path} ({time.time() - start:.0f}s)")


def cluster_neurons(dataset, trial, side, *, linkage, k_values, min_cluster_size,
                    k=None, run_info=None):
    """Stage C: consensus clustering over replicates; write tables and figures."""
    directory = distance_dir(dataset, trial, side)
    matrices, self_distances, eligible = clustering.load_replicates(directory)
    exact = clustering.load_exact(directory, eligible)
    print(f"Loaded {matrices.shape[0]} replicates over {eligible.size} neurons"
          + ("; exact distance matrix" if exact is not None else ""))
    clouds = Clouds(clouds_path(dataset, side))
    source_sha256 = _distance_provenance(directory, clouds, eligible)
    table = clustering.neuron_table(clouds, eligible)
    # The exact matrix of the complete clouds is the distance; the bootstrap
    # replicates give the consensus, its stability and the noise floor. Without
    # an exact matrix (--method shared) the replicate mean stands in.
    distance = matrices.mean(axis=0) if exact is None else exact

    nearest, self_mean, noise = clustering.noise_summary(distance, self_distances)
    fit = clustering.fit_consensus(matrices, linkage, k_values, min_cluster_size, k)
    if fit["status"] != "ok":
        raise ValueError(f"consensus did not reach the requested k; reachable k: {fit['reached']}. "
                         "Lower k or --min-cluster-size.")
    consensus, ambiguity, stab = fit["consensus"], fit["pac"], fit["stability"]
    k_values = list(consensus)
    agreement = clustering.label_agreement(table, consensus, k_values)
    # k whose consensus has fewer than k major clusters repeats a smaller cut; blank it out
    # (fit_consensus already blanked its PAC).
    reached = fit["reached"]
    for kk in set(k_values) - set(reached):
        for column in agreement:
            agreement[column][kk] = {"ami": float("nan"), "ari": float("nan")}
    chosen_k, k_selection = fit["chosen_k"], fit["k_selection"]
    labels, together = fit["labels"], fit["co_clustering"]
    tree = clustering.build_tree(distance, linkage)

    outdir = cluster_dir(dataset, trial, side)
    for kk in reached:
        table[f"cluster_k{kk}"] = consensus[kk]
    table["self_distance_um"] = self_mean
    table["nearest_neighbour_um"] = nearest
    table["co_clustering_stability"] = clustering.neuron_stability(together, labels)
    table.to_csv(outdir / "neurons.csv", index=False)
    np.save(outdir / "distance.npy", distance.astype(np.float32))

    summary = {
        "run": {"dataset": dataset, "trial": trial, "side": side, "linkage": linkage,
                **(run_info or {})},
        "n_neurons": int(eligible.size),
        "cell_ids": clouds.cell_ids[eligible].tolist(),
        "source_sha256": source_sha256,
        "n_replicates": int(matrices.shape[0]),
        "distance_matrix": "replicate_mean" if exact is None else "exact",
        "noise": noise,
        "chosen_k": chosen_k,
        "k_selection": k_selection,
        "cluster_sizes": {("outliers" if c == 0 else str(c)): int(n)
                          for c, n in zip(*np.unique(labels, return_counts=True))},
        "k_reached": reached,
        "n_outliers": {str(kk): int(np.sum(consensus[kk] == 0)) for kk in reached},
        "stability_ari": {str(kk): clustering.stability_summary(stab[kk]) for kk in reached},
        "pac": {str(kk): ambiguity[kk] for kk in reached},
        "label_agreement": {column: {str(kk): per_k[kk] for kk in reached}
                            for column, per_k in agreement.items()},
        "distance_vs_covariate_spearman": clustering.covariate_distance_correlation(distance, table),
        "covariate_kruskal": clustering.covariate_tests(table, labels),
    }
    with open(outdir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    title = f"{dataset} {side} | {linkage} | min cluster {min_cluster_size}"
    plotting.plot_tree_heatmap(tree, distance, table, labels, chosen_k,
                               outdir / "tree_heatmap.png", title)
    plotting.plot_k_curves(stab, ambiguity, agreement, chosen_k, outdir / "k_curves.png")
    plotting.plot_co_clustering(together, labels, clustering.leaf_order(tree), chosen_k,
                                outdir / "co_clustering.png")
    plotting.plot_noise(nearest, self_mean, outdir / "noise_floor.png")
    plotting.plot_contingency(table, labels, outdir / "contingency.png")
    plotting.plot_covariates(table, labels, outdir / "covariates.png")

    print(json.dumps({key: summary[key] for key in
                      ("n_neurons", "noise", "chosen_k", "cluster_sizes",
                       "distance_vs_covariate_spearman")}, indent=2))
    print(f"Outputs in {outdir}")


# ---------------------------------------------------------------------------
# Stage D
# ---------------------------------------------------------------------------

def load_reference(dataset, trial, side, clouds, k=None):
    """Clustered neurons (indices into ``clouds``) and their consensus labels at k."""
    outdir = cluster_dir(dataset, trial, side)
    with open(outdir / "summary.json") as handle:
        summary = json.load(handle)
    k = int(k or summary["chosen_k"])
    table = pd.read_csv(outdir / "neurons.csv")
    column = f"cluster_k{k}"
    if column not in table:
        raise ValueError(f"k={k} was not reached by the consensus; choose from {summary['k_reached']}")
    index_of = pd.Series(np.arange(len(clouds)), index=clouds.cell_ids)
    indices = index_of.loc[table["pt_root_id"].to_numpy()].to_numpy()
    return indices, table[column].to_numpy(dtype=int), k


def query_set(dataset, side, query, clouds, reference, min_synapses, min_border_um=0.0):
    """Clouds and indices of the neurons to assign (never the reference neurons).

    ``query="none"`` assigns nobody: Stage D then only calibrates and validates.
    The border filter applies to queries too: a truncated cloud would be misassigned.
    """
    if query == "none":
        return clouds, np.array([], dtype=int)
    if query == "rest":
        eligible = eligible_neurons(clouds, min_synapses, min_border_um)
        return clouds, np.setdiff1d(eligible, reference)
    if side != "dendrite":
        raise ValueError("--query all is only available for the dendrite side")
    path = clouds_path(dataset, side, "all")
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run the extract stage with --all-cells")
    all_clouds = Clouds(path)
    eligible = eligible_neurons(all_clouds, min_synapses, min_border_um)
    known = np.isin(all_clouds.cell_ids[eligible], clouds.cell_ids[reference])
    return all_clouds, eligible[~known]


def assign_neurons(dataset, trial, side, *, query, min_synapses, use_weights, seed, n_neighbours,
                   template_k_max, template_points, cv_folds, reject_quantile, min_border_um=0.0,
                   k=None, run_info=None, max_synapses=None):
    """Stage D: calibrate both assignment methods on the clustered neurons, then assign the queries.

    ``min_border_um`` must match the Stage B/C run being assigned to.
    """
    start = time.time()
    n_jobs = allocated_cpus()
    clouds = Clouds(clouds_path(dataset, side))
    reference, labels, k = load_reference(dataset, trial, side, clouds, k)
    n_clusters = int(labels.max())
    query_clouds, queries = query_set(dataset, side, query, clouds, reference, min_synapses,
                                      min_border_um)
    k_values = list(range(1, template_k_max + 1))
    templating = dict(k_values=k_values, max_points=template_points, use_weights=use_weights,
                      n_jobs=n_jobs)
    print(f"[{side}] {reference.size} reference neurons in {n_clusters} clusters (k={k}, "
          f"{int(np.sum(labels == 0))} outliers); {queries.size} query neurons ({query}); "
          f"{n_jobs} CPUs")

    # 1. Cross-validation on the reference set: calibration and held-out metrics.
    distances = clustering.load_exact(distance_dir(dataset, trial, side), reference)
    if distances is None:        # e.g. a --method shared trial: the point-cloud W2 is computed here
        distances = pairwise(gather(clouds, reference, use_weights, max_synapses, seed), n_jobs)
    print(f"reference distances ready ({time.time() - start:.0f}s)")
    cv = assign.cross_validate(clouds, reference, labels, n_clusters, distances, n_folds=cv_folds,
                               n_neighbours=n_neighbours, seed=seed, **templating)
    calibration = assign.calibrate_from_cv(cv, labels, n_clusters, reject_quantile)
    cv_prob, cv_hard = assign.assignments(cv["knn_distance"], cv["knn_label"], cv["score"],
                                          n_clusters, calibration)
    cv_metrics = assign.evaluate(cv_prob, cv_hard, labels, n_clusters)
    print(f"cross-validation done ({time.time() - start:.0f}s)")

    # 2. Final templates from all reference neurons.
    templates, selection = assign.fit_templates(clouds, reference, labels, n_clusters, seed=seed,
                                                **templating)

    # 3. Queries.
    outdir = assign_dir(dataset, trial, side, query)
    queries_summary = None
    if queries.size:
        pool = reference[labels > 0]
        q_distances = cross(gather(query_clouds, queries, use_weights, max_synapses, seed),
                            gather(clouds, pool, use_weights, max_synapses, seed), n_jobs)
        knn_distance, knn_label = assign.nearest_references(q_distances, labels[labels > 0], n_neighbours)
        score = assign.template_scores(query_clouds, queries, templates, use_weights)
        q_prob, q_hard = assign.assignments(knn_distance, knn_label, score, n_clusters, calibration)
        q_table = assign.assignment_table(clustering.neuron_table(query_clouds, queries), q_prob,
                                          q_hard, knn_distance, score, n_clusters)
        q_table.to_csv(outdir / "queries.csv", index=False)
        queries_summary = assign.query_summary(q_prob, q_hard, q_table)
        plotting.plot_queries(q_prob, q_hard, outdir / "queries.png")
        print(f"queries done ({time.time() - start:.0f}s)")

    # 4. Reference table, templates, summary and figures.
    base = clustering.neuron_table(clouds, reference)
    base.insert(1, "consensus", labels)
    base.insert(2, "fold", cv["fold"])
    assign.assignment_table(base, cv_prob, cv_hard, cv["knn_distance"], cv["score"],
                            n_clusters).to_csv(outdir / "reference_cv.csv", index=False)
    with open(outdir / "templates.json", "w") as handle:
        json.dump(assign.serialise_templates(templates, selection, labels), handle, indent=2)
    summary = {
        "run": {"dataset": dataset, "trial": trial, "side": side, "query": query,
                "min_border_um": min_border_um,
                "max_synapses": max_synapses, "n_neighbours": n_neighbours,
                "template_k_max": template_k_max, "template_points": template_points,
                "cv_folds": cv_folds, **(run_info or {})},
        "k": k,
        "n_clusters": n_clusters,
        "n_reference": int(reference.size),
        "n_reference_outliers": int(np.sum(labels == 0)),
        "n_queries": int(queries.size),
        "calibration": calibration,
        "cross_validation": cv_metrics,
        "templates": {str(c): {"n_components": templates[c].n_components, "bic_per_k": selection[c]}
                      for c in sorted(templates)},
        "queries": queries_summary,
    }
    with open(outdir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    plotting.plot_assign_cv(cv_metrics, cv_prob, labels, outdir / "cv_confusion.png",
                            outdir / "cv_reliability.png")
    plotting.plot_templates(templates, clouds, reference, labels, outdir / "templates.png")

    brief = {method: {key: cv_metrics[method][key] for key in
                      ("accuracy", "nll", "ece", "rejected_fraction", "outliers_rejected_fraction")}
             for method in assign.METHODS}
    print(json.dumps({"calibration": {key: calibration[key] for key in
                                      ("tau_um2", "temperature", "reject_distance_um", "reject_score")},
                      "cross_validation": brief, "agreement": cv_metrics["agreement"],
                      "queries": queries_summary}, indent=2))
    print(f"Outputs in {outdir} ({time.time() - start:.0f}s)")
