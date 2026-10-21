"""
Pipeline Stages
===============
The stages run by ``scripts/run_pipeline.py``. Each stage reads the previous
stage's files from ``output/<dataset>/`` and writes its own:

  A  extract_clouds         clouds_{axon,dendrite}.npz
     extract_all_dendrites  clouds_dendrite_all.npz (every cell, for --query all)
  B  fit_shared_model       trials/<trial>/shared_<side>.npz (--method shared only)
     compute_distances      trials/<trial>/dist_<side>/rep<b>.npz
  C  cluster_neurons        trials/<trial>/cluster_<side>/
  D  assign_neurons         trials/<trial>/assign_<side>_<query>/
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa

from . import assign, clustering, plotting, shared
from .clouds import Clouds, build_side, load_cell_table, load_synapses, proofread_cells
from .config import (SIDES, allocated_cpus, assign_dir, cluster_dir, clouds_path, distance_dir,
                     shared_model_path)
from .distances import alignment_check, compute_replicate, resample_clouds


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


def compute_distances(clouds, dataset, trial, side, rep, *, n_points, min_synapses, use_weights,
                      seed, n_neurons=None, min_border_um=0.0, method="points", shared_model=None,
                      shared_alpha=10.0):
    """Stage B: one bootstrap replicate of the neuron-to-neuron distance matrix.

    ``method="points"``: W2 between resampled point clouds (``distances.py``);
    ``method="shared"``: MW2 between mixing weights over ``shared_model`` (``shared.py``).
    Replicate 0 of the points method also reports, on 100 random pairs, how
    much smaller the W2 would be at the best in-plane rotation (``alignment_check``).
    """
    start = time.time()
    n_jobs = allocated_cpus()
    eligible = select_neurons(clouds, min_synapses, n_neurons, seed, min_border_um)
    n_enough = int(np.sum(clouds.n_synapses >= min_synapses))
    n_border = n_enough - eligible_neurons(clouds, min_synapses, min_border_um).size
    print(f"[{side}] {n_enough} / {len(clouds)} neurons have >= {min_synapses} synapses, "
          f"{n_border} of them within {min_border_um:g} um of the border; using {eligible.size}; "
          f"replicate {rep}; method {method}; aligned {clouds.align}; {n_jobs} CPUs")

    rng = np.random.default_rng([seed, rep])
    if method == "points":
        result = compute_replicate(clouds, eligible, n_points=n_points, use_weights=use_weights,
                                   rng=rng, n_jobs=n_jobs)
        if rep == 0 and eligible.size >= 2:
            check_rng = np.random.default_rng([seed, 99])
            subset = check_rng.choice(eligible, size=min(100, eligible.size), replace=False)
            sample, _ = resample_clouds(clouds, subset, n_points, use_weights, check_rng)
            pairs, given, best = alignment_check(sample, check_rng)
            ratio = given / np.maximum(best, 1e-9)
            print(f"alignment check ({pairs.shape[0]} pairs): W2 / W2 at best rotation: "
                  f"median {np.median(ratio):.3f}, 90th percentile {np.percentile(ratio, 90):.3f}")
            result.update(alignment_check_pairs=subset[pairs], alignment_check_w2=given,
                          alignment_check_rotation_min_w2=best)
    else:
        result = shared.compute_replicate(clouds, eligible, shared_model, alpha=shared_alpha,
                                          use_weights=use_weights, rng=rng, n_jobs=n_jobs)

    path = distance_dir(dataset, trial, side) / f"rep{rep:03d}.npz"
    np.savez_compressed(path, eligible=eligible, cell_ids=clouds.cell_ids[eligible],
                        method=method, n_points=n_points, min_synapses=min_synapses,
                        min_border_um=min_border_um, aligned=clouds.align,
                        use_weights=use_weights, **result)
    upper = result["distance"][np.triu_indices(eligible.size, 1)]
    print(f"median self-distance {np.median(result['self_distance']):.2f} um; "
          f"median pairwise distance {np.median(upper):.2f} um")
    print(f"Saved {path} ({time.time() - start:.0f}s)")


def cluster_neurons(dataset, trial, side, *, linkage, k_values, min_cluster_size,
                    depth_weight=0.0, k=None, run_info=None):
    """Stage C: consensus clustering over replicates; write tables and figures.

    ``depth_weight`` (lambda) adds the soma depth as an extra coordinate:
    ``D'^2 = D^2 + lambda * (depth_i - depth_j)^2``, exact for W2 and applied
    to the stored replicates, so Stage B is not rerun.
    """
    matrices, self_distances, eligible = clustering.load_replicates(distance_dir(dataset, trial, side))
    print(f"Loaded {matrices.shape[0]} replicates over {eligible.size} neurons; "
          f"depth weight {depth_weight:g}")
    table = clustering.neuron_table(Clouds(clouds_path(dataset, side)), eligible)
    depths = table["soma_depth_um"].to_numpy(dtype=float)
    matrices = clustering.add_depth_term(matrices, depths, depths, depth_weight)
    mean_distance = matrices.mean(axis=0)

    nearest, self_mean, noise = clustering.noise_summary(mean_distance, self_distances)
    replicate_labels = clustering.replicate_cuts(matrices, linkage, k_values, min_cluster_size)
    consensus, ambiguity = clustering.consensus_clustering(replicate_labels, k_values, linkage,
                                                           min_cluster_size)
    stab = clustering.stability(replicate_labels, consensus, k_values)
    agreement = clustering.label_agreement(table, consensus, k_values)
    # k whose consensus has fewer than k major clusters repeats a smaller cut; blank it out.
    reached = [kk for kk in k_values if not np.all(np.isnan(stab[kk]))]
    for kk in set(k_values) - set(reached):
        ambiguity[kk] = float("nan")
        for column in agreement:
            agreement[column][kk] = {"ami": float("nan"), "ari": float("nan")}
    if k:
        chosen_k, k_selection = k, {"rule": "given on the command line", "chosen": k}
    else:
        chosen_k, k_selection = clustering.most_stable_k(stab, ambiguity)
    labels = consensus[chosen_k]
    together = clustering.co_clustering(replicate_labels, chosen_k)
    tree = clustering.build_tree(mean_distance, linkage)

    outdir = cluster_dir(dataset, trial, side, depth_weight)
    for kk in reached:
        table[f"cluster_k{kk}"] = consensus[kk]
    table["self_distance_um"] = self_mean
    table["nearest_neighbour_um"] = nearest
    table["co_clustering_stability"] = clustering.neuron_stability(together, labels)
    table.to_csv(outdir / "neurons.csv", index=False)
    np.save(outdir / "mean_distance.npy", mean_distance.astype(np.float32))

    summary = {
        "run": {"dataset": dataset, "trial": trial, "side": side, "linkage": linkage,
                "depth_weight": depth_weight, **(run_info or {})},
        "n_neurons": int(eligible.size),
        "n_replicates": int(matrices.shape[0]),
        "noise": noise,
        "depth_weight": depth_weight,
        "depth_share_of_squared_distance": clustering.depth_share(mean_distance, depths, depth_weight),
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
        "cluster_soma_depth": clustering.cluster_depth_summary(depths, labels),
        "distance_vs_covariate_spearman": clustering.covariate_distance_correlation(mean_distance, table),
        "covariate_kruskal": clustering.covariate_tests(table, labels),
    }
    with open(outdir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    title = (f"{dataset} {side} | {linkage} | min cluster {min_cluster_size} | "
             f"depth weight {depth_weight:g}")
    plotting.plot_tree_heatmap(tree, mean_distance, table, labels, chosen_k,
                               outdir / "tree_heatmap.png", title)
    plotting.plot_k_curves(stab, ambiguity, agreement, chosen_k, outdir / "k_curves.png")
    plotting.plot_co_clustering(together, labels, clustering.leaf_order(tree), chosen_k,
                                outdir / "co_clustering.png")
    plotting.plot_noise(nearest, self_mean, outdir / "noise_floor.png")
    plotting.plot_contingency(table, labels, outdir / "contingency.png")
    note = (f"depth weight {depth_weight:g}: depth term is "
            f"{100 * summary['depth_share_of_squared_distance']:.0f}% of the squared distance (median pair)"
            if depth_weight else None)
    plotting.plot_covariates(table, labels, outdir / "covariates.png", note=note)

    print(json.dumps({key: summary[key] for key in
                      ("n_neurons", "noise", "depth_weight", "depth_share_of_squared_distance",
                       "chosen_k", "cluster_sizes", "cluster_soma_depth",
                       "distance_vs_covariate_spearman")}, indent=2))
    print(f"Outputs in {outdir}")


# ---------------------------------------------------------------------------
# Stage D
# ---------------------------------------------------------------------------

def load_reference(dataset, trial, side, clouds, k=None, depth_weight=0.0):
    """Clustered neurons (indices into ``clouds``) and their consensus labels at k."""
    outdir = cluster_dir(dataset, trial, side, depth_weight)
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
    all_clouds = Clouds(path, align=clouds.align)
    eligible = eligible_neurons(all_clouds, min_synapses, min_border_um)
    known = np.isin(all_clouds.cell_ids[eligible], clouds.cell_ids[reference])
    return all_clouds, eligible[~known]


def assign_neurons(dataset, trial, side, *, query, n_points, min_synapses, use_weights,
                   seed, n_reps, n_neighbours, template_k_max, template_points, cv_folds,
                   reject_quantile, depth_weight=0.0, min_border_um=0.0, align=False, k=None,
                   run_info=None):
    """Stage D: calibrate both assignment methods on the clustered neurons, then assign the queries.

    ``depth_weight``, ``min_border_um`` and ``align`` must match the Stage B/C
    run being assigned to; the kNN distances get the same soma-depth term (the
    template method has its depth prior instead).
    """
    start = time.time()
    n_jobs = allocated_cpus()
    clouds = Clouds(clouds_path(dataset, side), align=align)
    reference, labels, k = load_reference(dataset, trial, side, clouds, k, depth_weight)
    reference_depths = clouds.meta["soma_y"][reference].astype(float)
    n_clusters = int(labels.max())
    query_clouds, queries = query_set(dataset, side, query, clouds, reference, min_synapses,
                                      min_border_um)
    k_values = list(range(1, template_k_max + 1))
    sampling = dict(n_reps=n_reps, n_points=n_points, use_weights=use_weights, seed=seed,
                    n_jobs=n_jobs)
    templating = dict(k_values=k_values, max_points=template_points, use_weights=use_weights,
                      n_jobs=n_jobs)
    print(f"[{side}] {reference.size} reference neurons in {n_clusters} clusters (k={k}, "
          f"{int(np.sum(labels == 0))} outliers); {queries.size} query neurons ({query}); "
          f"depth weight {depth_weight:g}; {n_jobs} CPUs")

    # 1. Cross-validation on the reference set: calibration and held-out metrics.
    distances = clustering.add_depth_term(assign.reference_distances(clouds, reference, **sampling),
                                          reference_depths, reference_depths, depth_weight)
    print(f"reference distances done ({time.time() - start:.0f}s)")
    cv = assign.cross_validate(clouds, reference, labels, n_clusters, distances, n_folds=cv_folds,
                               n_neighbours=n_neighbours, seed=seed, **templating)
    calibration = assign.calibrate_from_cv(cv, labels, n_clusters, reject_quantile)
    cv_prob, cv_hard = assign.assignments(cv["knn_distance"], cv["knn_label"], cv["score"],
                                          cv["log_prior"], n_clusters, calibration)
    cv_metrics = assign.evaluate(cv_prob, cv_hard, labels, n_clusters)
    print(f"cross-validation done ({time.time() - start:.0f}s)")

    # 2. Final templates and depth prior from all reference neurons.
    templates, selection = assign.fit_templates(clouds, reference, labels, n_clusters, seed=seed,
                                                **templating)
    prior = assign.DepthPrior(clouds.meta["soma_y"][reference], labels, n_clusters)

    # 3. Queries.
    outdir = assign_dir(dataset, trial, side, query, depth_weight)
    queries_summary = None
    if queries.size:
        pool = reference[labels > 0]
        q_distances = clustering.add_depth_term(
            assign.query_reference_distances(query_clouds, queries, clouds, pool, **sampling),
            query_clouds.meta["soma_y"][queries].astype(float), reference_depths[labels > 0],
            depth_weight)
        knn_distance, knn_label = assign.nearest_references(q_distances, labels[labels > 0], n_neighbours)
        score = assign.template_scores(query_clouds, queries, templates, use_weights)
        log_prior = prior.log_prior(query_clouds.meta["soma_y"][queries])
        q_prob, q_hard = assign.assignments(knn_distance, knn_label, score, log_prior, n_clusters,
                                            calibration)
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
                "depth_weight": depth_weight, "min_border_um": min_border_um, "aligned": align,
                "n_reps": n_reps, "n_neighbours": n_neighbours,
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
                                      ("tau_um2", "temperature", "temperature_with_prior",
                                       "reject_distance_um", "reject_score")},
                      "cross_validation": brief, "agreement": cv_metrics["agreement"],
                      "queries": queries_summary}, indent=2))
    print(f"Outputs in {outdir} ({time.time() - start:.0f}s)")
