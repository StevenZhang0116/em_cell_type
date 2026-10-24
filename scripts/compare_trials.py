"""
Compare Two Clusterings
=======================
Side by side comparison of two Stage C runs on the same neurons: two trials
with different Stage B settings (typically ``--method points`` vs ``--method
shared``, or with and without ``--align``; same ``--n-neurons`` and ``--seed``
give the same neurons). Prints, per side: correlation of the two mean distance
matrices, self/nearest-neighbour ratio, chosen k with its stability and PAC,
AMI with cell_type, and the ARI between the two consensus labelings. Saves a
scatter of the two distance matrices. Outputs are labelled by the trial-name
tokens in which the two trials differ.

Example::

    python scripts/compare_trials.py --sides axon \\
        --trial-a n1000_rep10_pt200_min100_weighted_seed20261019_average_k2-20_mcs5 \\
        --trial-b n1000_rep10_shared30_a10_min100_weighted_seed20261019_average_k2-20_mcs5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from celltype.config import SIDES, cluster_dir, dataset_dir


def load_clustering(dataset, trial, side):
    outdir = cluster_dir(dataset, trial, side)
    with open(outdir / "summary.json") as handle:
        summary = json.load(handle)
    table = pd.read_csv(outdir / "neurons.csv")
    return summary, table, np.load(outdir / "mean_distance.npy")


def describe(summary):
    k = str(summary["chosen_k"])
    return {
        "n_neurons": summary["n_neurons"],
        "self_over_nearest": round(summary["noise"]["median_self_over_nearest"], 3),
        "chosen_k": summary["chosen_k"],
        "stability_ari_median": round(summary["stability_ari"][k]["median"], 3),
        "pac": round(summary["pac"][k], 3),
        "ami_cell_type": round(summary["label_agreement"]["cell_type"][k]["ami"], 3),
        "outliers": summary["n_outliers"][k],
    }


def trial_tokens(trial_a, trial_b):
    """Short labels for the two trials: the name tokens each has and the other lacks."""
    parts_a, parts_b = trial_a.split("_"), trial_b.split("_")
    only_a = "_".join(p for p in parts_a if p not in parts_b) or "base"
    only_b = "_".join(p for p in parts_b if p not in parts_a) or "base"
    return only_a, only_b


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="microns")
    parser.add_argument("--trial-a", required=True)
    parser.add_argument("--trial-b", required=True)
    parser.add_argument("--sides", nargs="+", choices=SIDES, default=list(SIDES))
    args = parser.parse_args()
    if args.trial_b == args.trial_a:
        parser.error("the two trials are identical")
    outdir = dataset_dir(args.dataset) / "comparisons"
    outdir.mkdir(exist_ok=True)
    token_a, token_b = trial_tokens(args.trial_a, args.trial_b)

    for side in args.sides:
        summary_a, table_a, dist_a = load_clustering(args.dataset, args.trial_a, side)
        summary_b, table_b, dist_b = load_clustering(args.dataset, args.trial_b, side)
        common, rows_a, rows_b = np.intersect1d(table_a["pt_root_id"], table_b["pt_root_id"],
                                               return_indices=True)
        if common.size < len(table_a) or common.size < len(table_b):
            print(f"[{side}] warning: trials share only {common.size} neurons")
        upper = np.triu_indices(common.size, 1)
        values_a = dist_a[np.ix_(rows_a, rows_a)][upper]
        values_b = dist_b[np.ix_(rows_b, rows_b)][upper]

        labels_a = table_a[f"cluster_k{summary_a['chosen_k']}"].to_numpy()[rows_a]
        labels_b = table_b[f"cluster_k{summary_b['chosen_k']}"].to_numpy()[rows_b]
        both = (labels_a > 0) & (labels_b > 0)
        report = {
            "side": side,
            "n_common_neurons": int(common.size),
            "distance_pearson": round(float(pearsonr(values_a, values_b)[0]), 3),
            "distance_spearman": round(float(spearmanr(values_a, values_b)[0]), 3),
            "ari_between_consensus_labels": round(float(adjusted_rand_score(labels_a[both], labels_b[both])), 3),
            token_a: describe(summary_a),
            token_b: describe(summary_b),
        }
        print(json.dumps(report, indent=2))
        with open(outdir / f"{side}_{token_a}_vs_{token_b}.json", "w") as handle:
            json.dump(report, handle, indent=2)

        fig, ax = plt.subplots(figsize=(5, 5))
        pick = np.random.default_rng(0).choice(values_a.size, size=min(20000, values_a.size),
                                               replace=False)
        ax.scatter(values_a[pick], values_b[pick], s=2, alpha=0.3)
        ax.set_xlabel(f"{token_a} distance (um)")
        ax.set_ylabel(f"{token_b} distance (um)")
        ax.set_title(f"{side}: Spearman {report['distance_spearman']:.2f}")
        fig.tight_layout()
        fig.savefig(outdir / f"{side}_{token_a}_vs_{token_b}.png", dpi=200)
        plt.close(fig)
    print(f"Saved comparisons in {outdir}")


if __name__ == "__main__":
    main()
