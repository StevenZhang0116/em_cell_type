"""
Pipeline Stages
===============
The three stages run by ``scripts/run_pipeline.py``. Each stage reads the
previous stage's files from ``output/<dataset>/`` and writes its own:

  A  extract_clouds     clouds_{axon,dendrite}.npz
  B  compute_replicate  dist_<side>_<coords>/rep<b>.npz
  C  cluster_neurons    cluster_<side>_<coords>/
"""

from __future__ import annotations

import json
import time

import numpy as np
import pyarrow as pa

from . import clustering, plotting
from .clouds import Clouds, build_side, load_cell_table, load_synapses, proofread_cells
from .config import SIDES, allocated_cpus, cluster_dir, clouds_path, distance_dir
from .distances import compute_replicate


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


def select_neurons(clouds, min_synapses, n_neurons, seed):
    """Neurons with enough synapses; optionally a random subset of ``n_neurons``.

    The subset depends only on ``seed``, so every replicate uses the same neurons.
    """
    eligible = np.flatnonzero(clouds.n_synapses >= min_synapses)
    if n_neurons is None or n_neurons >= eligible.size:
        return eligible
    return np.sort(np.random.default_rng(seed).choice(eligible, size=n_neurons, replace=False))


def compute_distances(clouds, dataset, side, coords, rep, *,
                      n_points, min_synapses, use_weights, seed, n_neurons=None):
    """Stage B: one bootstrap replicate of the neuron-to-neuron distance matrix."""
    start = time.time()
    n_jobs = allocated_cpus()
    eligible = select_neurons(clouds, min_synapses, n_neurons, seed)
    n_enough = int(np.sum(clouds.n_synapses >= min_synapses))
    print(f"[{side}] {n_enough} / {len(clouds)} neurons have >= {min_synapses} synapses; "
          f"using {eligible.size}; replicate {rep}; {n_jobs} CPUs")

    result = compute_replicate(
        clouds, eligible, coords=coords, n_points=n_points, use_weights=use_weights,
        rng=np.random.default_rng([seed, rep]), n_jobs=n_jobs)

    path = distance_dir(dataset, side, coords, n_neurons) / f"rep{rep:03d}.npz"
    np.savez_compressed(path, eligible=eligible, cell_ids=clouds.cell_ids[eligible],
                        n_points=n_points, min_synapses=min_synapses,
                        use_weights=use_weights, **result)
    upper = result["distance"][np.triu_indices(eligible.size, 1)]
    print(f"median self-distance {np.median(result['self_distance']):.2f} um; "
          f"median pairwise distance {np.median(upper):.2f} um")
    print(f"Saved {path} ({time.time() - start:.0f}s)")


def cluster_neurons(dataset, side, coords, *, linkage, k_values, k=None, n_neurons=None,
                    run_info=None):
    """Stage C: cluster the mean distance matrix and write tables and figures."""
    matrices, self_distances, eligible = clustering.load_replicates(
        distance_dir(dataset, side, coords, n_neurons))
    print(f"Loaded {matrices.shape[0]} replicates over {eligible.size} neurons")
    table = clustering.neuron_table(Clouds(clouds_path(dataset, side)), eligible)
    mean_distance = matrices.mean(axis=0)

    nearest, self_mean, noise = clustering.noise_summary(mean_distance, self_distances)
    tree, consensus = clustering.cut_tree(mean_distance, linkage, k_values)
    stab, replicate_labels = clustering.stability(matrices, consensus, linkage, k_values)
    agreement = clustering.label_agreement(table, consensus, k_values)
    chosen_k = k or clustering.most_stable_k(stab)
    labels = consensus[chosen_k]

    outdir = cluster_dir(dataset, side, coords, n_neurons)
    for kk in k_values:
        table[f"cluster_k{kk}"] = consensus[kk]
    table["self_distance_um"] = self_mean
    table["nearest_neighbour_um"] = nearest
    table["co_clustering_stability"] = clustering.neuron_stability(replicate_labels, labels, chosen_k)
    table.to_csv(outdir / "neurons.csv", index=False)
    np.save(outdir / "mean_distance.npy", mean_distance.astype(np.float32))

    summary = {
        "run": {"dataset": dataset, "side": side, "coords": coords, "linkage": linkage,
                **(run_info or {})},
        "n_neurons": int(eligible.size),
        "n_replicates": int(matrices.shape[0]),
        "noise": noise,
        "chosen_k": chosen_k,
        "cluster_sizes": {str(c): int(n) for c, n in zip(*np.unique(labels, return_counts=True))},
        "stability_ari": {str(kk): stab[kk] for kk in k_values},
        "label_agreement": {column: {str(kk): v for kk, v in per_k.items()}
                            for column, per_k in agreement.items()},
        "distance_vs_covariate_spearman": clustering.covariate_distance_correlation(mean_distance, table),
        "covariate_kruskal": clustering.covariate_tests(table, labels),
    }
    with open(outdir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    title = f"{dataset} {side} | {coords} | {linkage}"
    plotting.plot_tree_heatmap(tree, mean_distance, table, labels, chosen_k,
                               outdir / "tree_heatmap.png", title)
    plotting.plot_k_curves(stab, agreement, chosen_k, outdir / "k_curves.png")
    plotting.plot_noise(nearest, self_mean, outdir / "noise_floor.png")
    plotting.plot_contingency(table, labels, outdir / "contingency.png")
    plotting.plot_covariates(table, labels, outdir / "covariates.png")

    print(json.dumps({key: summary[key] for key in
                      ("n_neurons", "noise", "chosen_k", "cluster_sizes",
                       "distance_vs_covariate_spearman")}, indent=2))
    print(f"Outputs in {outdir}")
