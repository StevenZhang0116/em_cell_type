"""
Stage C: Cluster Neurons and Check the Clusters
===============================================
Combines all replicates of one (side, method, depth mode) run, clusters the
neurons and writes ``neurons.csv``, ``summary.json``, ``mean_distance.npy``
and figures to ``output/<dataset>/cluster_<side>_<method>_<coords>_<depth>/`` (see
celltype/clustering.py and celltype/plotting.py).

Example::

    python -u scripts/03_cluster_neurons.py --dataset microns --side axon \
        --method points --coords xyz --depth-mode relative
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from celltype import clustering, plotting
from celltype.clouds import Clouds
from celltype.config import (COORDS, DEPTH_MODES, METHODS, SIDES, cluster_dir,
                             clouds_path, distance_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="microns")
    parser.add_argument("--side", choices=SIDES, required=True)
    parser.add_argument("--method", choices=METHODS, default="points")
    parser.add_argument("--coords", choices=COORDS, default="xyz",
                        help="xyz: 3-D soma-centred offsets; cylindrical: (r, depth)")
    parser.add_argument("--depth-mode", choices=DEPTH_MODES, default="relative")
    parser.add_argument("--linkage", choices=("average", "complete"), default="average")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--k", type=int, default=None,
                        help="k for the detailed figures (default: most stable k >= 3)")
    args = parser.parse_args()

    k_values = list(range(args.k_min, args.k_max + 1))
    matrices, self_distances, eligible = clustering.load_replicates(
        distance_dir(args.dataset, args.side, args.method, args.coords, args.depth_mode))
    print(f"Loaded {matrices.shape[0]} replicates over {eligible.size} neurons")
    table = clustering.neuron_table(Clouds(clouds_path(args.dataset, args.side)), eligible)
    mean_distance = matrices.mean(axis=0)

    nearest, self_mean, noise = clustering.noise_summary(mean_distance, self_distances)
    tree, consensus = clustering.cut_tree(mean_distance, args.linkage, k_values)
    stab, replicate_labels = clustering.stability(matrices, consensus, args.linkage, k_values)
    agreement = clustering.label_agreement(table, consensus, k_values)
    chosen_k = args.k or clustering.most_stable_k(stab)
    labels = consensus[chosen_k]

    outdir = cluster_dir(args.dataset, args.side, args.method, args.coords, args.depth_mode)
    for k in k_values:
        table[f"cluster_k{k}"] = consensus[k]
    table["self_distance_um"] = self_mean
    table["nearest_neighbour_um"] = nearest
    table["co_clustering_stability"] = clustering.neuron_stability(replicate_labels, labels, chosen_k)
    table.to_csv(outdir / "neurons.csv", index=False)
    np.save(outdir / "mean_distance.npy", mean_distance.astype(np.float32))

    summary = {
        "run": vars(args),
        "n_neurons": int(eligible.size),
        "n_replicates": int(matrices.shape[0]),
        "noise": noise,
        "chosen_k": chosen_k,
        "cluster_sizes": {str(c): int(n) for c, n in zip(*np.unique(labels, return_counts=True))},
        "stability_ari": {str(k): stab[k] for k in k_values},
        "label_agreement": {column: {str(k): v for k, v in per_k.items()}
                            for column, per_k in agreement.items()},
        "distance_vs_covariate_spearman": clustering.covariate_distance_correlation(mean_distance, table),
        "covariate_kruskal": clustering.covariate_tests(table, labels),
    }
    with open(outdir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    title = f"{args.dataset} {args.side} | {args.method} | {args.coords} | {args.depth_mode} depth | {args.linkage}"
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


if __name__ == "__main__":
    main()
