"""
Figures for the Clustering Stage
================================
  tree_heatmap   dendrogram, cell_type / cluster barcodes, sorted distance matrix
  k_curves       per-replicate stability (median, IQR), PAC and AMI with dataset labels vs k
  co_clustering  co-clustering matrix at the chosen k, sorted by consensus cluster
  noise_floor    self-distance vs nearest-neighbour distance per neuron
  contingency    dataset cell_type x new cluster counts
  covariates     nuisance covariates per new cluster
Cluster 0 (outliers: neurons left in clusters below the minimum size) is shown as "out".

Stage D figures:
  cv_confusion   held-out confusion matrix of each assignment method
  cv_reliability reliability diagram (accuracy vs posterior confidence) per method
  templates      each cluster's template density (slice through its centre) over member synapses
  queries        posterior confidence and label counts of the assigned queries

Per-neuron GMM figures (celltype/neuron_gmm.py):
  gmm_selection  chosen-k counts and the held-out BIC curve (relative to each neuron's best k)
  gmm_stability  within-neuron vs between-neuron mixture Wasserstein distances, vs synapse count and k

Split-half diagnostics (celltype/split_half.py):
  split_half_distributions  split ranges per neuron, distributions of medians, endpoint baselines
  split_half_curves         matched-distance medians vs points per half, with a fixed cohort
  split_half_covariates     self-distance vs cloud spread and weight concentration
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from .assign import METHODS
from .clustering import present_covariates
from .config import LABEL_COLUMNS
from .templates import density_slice


def _category_colors(values):
    unique = sorted(set(values))
    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    colors = {value: cmap(i) for i, value in enumerate(unique)}
    if "out" in colors:
        colors["out"] = (0.85, 0.85, 0.85, 1.0)
    return colors


def _cluster_names(labels):
    return np.array(["out" if label == 0 else str(label) for label in labels])


def _save(fig, path):
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_split_cluster(table, path):
    """Outer-split values without error bars implying independent replicates."""
    sizes = sorted(table["points_per_half"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for index, m in enumerate(sizes):
        rows = table[table["points_per_half"] == m]
        color = plt.get_cmap("tab10")(index % 10)
        axes[0].scatter(rows["joint_coverage"], rows["ari"], color=color, alpha=0.65, label=f"m={m}")
        # Deterministic jitter is cosmetic, never part of an analysis.
        x = index + np.linspace(-0.12, 0.12, len(rows))
        axes[1].scatter(x, rows["delta_um"], color=color, alpha=0.65)
        counts = rows.loc[rows["status_a"] == "ok", "k_a"].value_counts().sort_index()
        axes[2].plot(counts.index, counts / len(rows), "o-", color=color, label=f"m={m}")
    axes[0].set(xlabel="Fraction accepted in both halves", ylabel="ARI on jointly accepted neurons",
                xlim=(-0.03, 1.03), ylim=(-1.03, 1.03), title="Agreement and coverage")
    axes[0].legend(loc="lower left")
    axes[1].axhline(0, color="0.6", linewidth=1)
    axes[1].set(xticks=range(len(sizes)), xticklabels=[str(m) for m in sizes], xlabel="Synapses per half",
                ylabel="Median between - median within (um)", title="B distances, A labels only")
    axes[2].set(xlabel="k chosen using A only", ylabel="Fraction of all outer splits", ylim=(-0.03, 1.03),
                title="Selection frequency (failures omitted)")
    observed_k = sorted(table["k_a"].dropna().astype(int).unique())
    if observed_k:
        axes[2].set(xticks=observed_k, xlim=(min(observed_k) - 0.5, max(observed_k) + 0.5))
    failures = int((table["status"] != "ok").sum())
    fig.suptitle(f"Split-half consensus validation | {failures}/{len(table)} ARIs not evaluable | descriptive splits")
    _save(fig, path)


def plot_split_cluster_contingency(labels_a, labels_b, path, m, rep):
    """Include rejection as 'out', while the numerical matching excludes it."""
    ca, cb = np.unique(labels_a), np.unique(labels_b)
    counts = np.array([[np.sum((labels_a == a) & (labels_b == b)) for b in cb] for a in ca])
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(cb) + 2), max(4, 0.4 * len(ca) + 2)), constrained_layout=True)
    shown = ax.imshow(counts, cmap="Blues", aspect="auto")
    fig.colorbar(shown, ax=ax, label="Neurons")
    ax.set(xticks=range(len(cb)), yticks=range(len(ca)), xticklabels=_cluster_names(cb),
           yticklabels=_cluster_names(ca), xlabel="B consensus label", ylabel="A consensus label",
           title=f"Representative split: m={m}, repeat={rep}\nLabels are not matched by numeric ID")
    if max(len(ca), len(cb)) <= 20:
        for (i, j), value in np.ndenumerate(counts):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > counts.max() / 2 else "black")
    _save(fig, path)


def plot_split_half_distributions(neurons, pairs, path):
    """Percentiles across splits are ranges, never confidence intervals or testing thresholds."""
    sizes = sorted(neurons["m"].unique())
    fig, axes = plt.subplots(len(sizes), 3, figsize=(16, 4 * len(sizes)), squeeze=False,
                             constrained_layout=True)
    for row, m in enumerate(sizes):
        own = neurons[neurons["m"] == m].sort_values("self_median_um")
        other = pairs[pairs["m"] == m]
        ax_range, ax_cdf, ax_pair = axes[row]
        # Keep dense runs readable; sample evenly through the sorted ranks, not just the extremes.
        ranks = np.unique(np.linspace(0, len(own) - 1, min(len(own), 100)).astype(int))
        shown = own.iloc[ranks]
        ax_range.vlines(ranks + 1, shown["self_p05_um"], shown["self_p95_um"], color="C0", alpha=0.5)
        ax_range.scatter(ranks + 1, shown["self_median_um"], s=9, color="C0")
        ax_range.set(xlabel="Neuron rank by median self-distance", ylabel="Self W2 (um)",
                     title=f"m={m}: median and 5-95% split range\nUp to 100 evenly spaced ranks")
        for values, label, color in ((own["self_median_um"], "Self (per neuron)", "C0"),
                                      (other["between_median_um"], "Between (per pair)", "C1")):
            if len(values):
                ordered = np.sort(values)
                ax_cdf.step(np.r_[ordered[0], ordered, ordered[-1]],
                             np.r_[0.0, np.arange(1, len(values) + 1) / len(values), 1.0],
                             where="post", label=label, color=color)
        ax_cdf.set(xlabel="Median across splits, W2 (um)", ylabel="Empirical cumulative fraction",
                   title=f"m={m}: descriptive distributions")
        ax_cdf.legend(fontsize=8)
        if len(other):
            # Both endpoints retain their own baseline; no universal pooled null is assumed.
            for endpoint, color in (("i", "C0"), ("j", "C2")):
                ax_pair.scatter(other["between_median_um"], other[f"self_{endpoint}_median_um"],
                                 s=8, alpha=0.4, color=color, label=f"Endpoint {endpoint}")
            limit = max(other["between_median_um"].max(), other["self_i_median_um"].max(),
                        other["self_j_median_um"].max(), 1.0)
            ax_pair.plot([0, limit], [0, limit], ls="--", color="0.5", lw=1)
            ax_pair.legend(fontsize=8)
        else:
            ax_pair.text(0.5, 0.5, "No between-neuron pairs", ha="center", transform=ax_pair.transAxes)
        ax_pair.set(xlabel="Pair's median between W2 (um)", ylabel="Endpoint's median self W2 (um)",
                    title=f"m={m}: each pair vs its endpoints\nDashed line: equal distances, not a test")
    _save(fig, path)


def plot_split_half_curves(neurons, pairs, path):
    """Population medians and IQRs of per-unit split medians, on the same cohort at every m."""
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for table, column, label, color in ((neurons, "self_median_um", "Self (neurons)", "C0"),
                                         (pairs, "between_median_um", "Between (pairs)", "C1")):
        if table.empty:
            continue
        sizes = sorted(table["m"].unique())
        stats = np.array([np.percentile(table.loc[table["m"] == m, column], [25, 50, 75]) for m in sizes])
        ax.plot(sizes, stats[:, 1], "o-", label=label, color=color)
        ax.fill_between(sizes, stats[:, 0], stats[:, 2], color=color, alpha=0.2)
    ax.set(xlabel="Synapses per half (m)", ylabel="W2 (um)",
           title="Fixed neurons and pairs at every m\nMedians across splits; bands: IQR across neurons/pairs, not CI")
    ax.legend(fontsize=9)
    _save(fig, path)


def plot_split_half_covariates(neurons, path):
    """Raw and spread-normalised self variability; weight ESS does not account for dependence."""
    sizes = sorted(neurons["m"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for index, m in enumerate(sizes):
        table = neurons[neurons["m"] == m]
        color = f"C{index % 10}"
        axes[0].scatter(table["spread_um"], table["self_median_um"], s=10, alpha=0.6, color=color, label=f"m={m}")
        axes[1].scatter(table["weight_ess_fraction"], table["self_over_spread"], s=10, alpha=0.6, color=color)
        axes[2].scatter(table["n_synapses"], table["self_median_um"], s=10, alpha=0.6, color=color)
    axes[0].set(xlabel="Complete-cloud RMS spread (um)", ylabel="Median self W2 (um)")
    axes[1].set(xlabel="Weight ESS / observed synapse count", ylabel="Median self W2 / spread",
                title="Weight concentration only; not independence")
    axes[2].set(xlabel="Complete-cloud synapse count", ylabel="Median self W2 (um)")
    axes[0].legend(fontsize=8)
    _save(fig, path)


def plot_tree_heatmap(tree, distance, table, labels, k, path, title):
    fig = plt.figure(figsize=(11, 12))
    ax_tree = fig.add_axes([0.14, 0.80, 0.74, 0.15])
    leaves = dendrogram(tree, ax=ax_tree, no_labels=True, color_threshold=None,
                        above_threshold_color="0.3")["leaves"]
    ax_tree.set_axis_off()
    ax_tree.set_title(title)

    barcodes = (("dataset cell_type", table["cell_type"].to_numpy()),
                (f"cluster (k={k})", _cluster_names(labels)))
    for row, (name, values) in enumerate(barcodes):
        colors = _category_colors(values)
        ax_bar = fig.add_axes([0.14, 0.775 - 0.025 * row, 0.74, 0.02])
        ax_bar.imshow([[colors[v] for v in values[leaves]]], aspect="auto")
        ax_bar.set_xticks([])
        ax_bar.set_yticks([0])
        ax_bar.set_yticklabels([name], fontsize=8)
        if row == 0:
            handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=t)
                       for t, c in colors.items()]
            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.89, 0.95),
                       fontsize=7, title="cell_type", frameon=False)

    ax_heat = fig.add_axes([0.14, 0.05, 0.74, 0.69])
    image = ax_heat.imshow(distance[np.ix_(leaves, leaves)], cmap="viridis",
                           interpolation="nearest", aspect="auto")
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    fig.colorbar(image, cax=fig.add_axes([0.90, 0.05, 0.02, 0.4]), label="W2 distance (um)")
    _save(fig, path)


def plot_k_curves(stab, ambiguity, agreement, chosen_k, path):
    k_values = sorted(stab)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    rng = np.random.default_rng(0)
    for k in k_values:
        axes[0].scatter(k + rng.uniform(-0.15, 0.15, stab[k].size), stab[k],
                        s=8, color="0.6", alpha=0.7, zorder=1)
    # k whose consensus has fewer than k major clusters is NaN and leaves a gap.
    q25, median, q75 = np.percentile([stab[k] for k in k_values], [25, 50, 75], axis=1)
    axes[0].fill_between(k_values, q25, q75, color="C0", alpha=0.25, zorder=0)
    axes[0].plot(k_values, median, marker="o", color="C0", zorder=2, label="median (band: IQR)")
    axes[0].set_ylim(min(-0.05, axes[0].get_ylim()[0]), 1.05)
    axes[0].set_ylabel("ARI(replicate, consensus)")
    axes[0].set_title("Bootstrap stability (dots: replicates)")
    axes[0].legend(fontsize=8, loc="lower left")

    axes[1].plot(k_values, [ambiguity[k] for k in k_values], marker="o", color="C3")
    axes[1].set_ylabel("PAC (pairs with 0.1 < co-clustering < 0.9)")
    axes[1].set_title("Ambiguity (lower = more stable)")

    for column in LABEL_COLUMNS:
        axes[2].plot(k_values, [agreement[column][k]["ami"] for k in k_values],
                     marker="o", label=column)
    axes[2].set_ylabel("AMI with dataset labels")
    axes[2].set_title("Agreement with existing labels")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axvline(chosen_k, color="0.6", ls="--")
        ax.set_xlabel("number of clusters k")
    fig.tight_layout()
    _save(fig, path)


def plot_co_clustering(together, labels, leaves, k, path):
    """Co-clustering matrix, neurons grouped by consensus cluster (outliers last)."""
    rank = np.empty(labels.size, dtype=int)
    rank[leaves] = np.arange(labels.size)
    group = np.where(labels == 0, labels.max() + 1, labels)
    order = np.lexsort((rank, group))
    fig, ax = plt.subplots(figsize=(7, 6.5))
    image = ax.imshow(together[np.ix_(order, order)], cmap="magma", vmin=0, vmax=1,
                      interpolation="nearest")
    edges = np.flatnonzero(np.diff(group[order])) + 0.5
    for edge in edges:
        ax.axhline(edge, color="w", lw=0.6)
        ax.axvline(edge, color="w", lw=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Co-clustering across replicates (k={k}; outliers last)")
    fig.colorbar(image, ax=ax, label="fraction of replicates in the same cluster")
    fig.tight_layout()
    _save(fig, path)


def plot_noise(nearest, self_mean, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(nearest, self_mean, s=6, alpha=0.6)
    limit = max(nearest.max(), self_mean.max())
    ax.plot([0, limit], [0, limit], color="0.5", ls="--", lw=1)
    ax.set_xlabel("distance to nearest other neuron (um)")
    ax.set_ylabel("bootstrap self-distance (um)")
    ax.set_title("Noise floor (below the diagonal = resolvable)")
    fig.tight_layout()
    _save(fig, path)


def plot_contingency(table, labels, path):
    counts = pd.crosstab(table["cell_type"], labels)
    fig, ax = plt.subplots(figsize=(1 + 0.5 * counts.shape[1], 1 + 0.35 * counts.shape[0]))
    ax.imshow(counts.to_numpy(), cmap="Blues", aspect="auto")
    for (row, col), value in np.ndenumerate(counts.to_numpy()):
        if value:
            ax.text(col, row, str(value), ha="center", va="center", fontsize=7)
    ax.set_xticks(range(counts.shape[1]))
    ax.set_xticklabels(_cluster_names(counts.columns))
    ax.set_yticks(range(counts.shape[0]))
    ax.set_yticklabels(counts.index, fontsize=8)
    ax.set_xlabel("new cluster")
    ax.set_ylabel("dataset cell_type")
    fig.tight_layout()
    _save(fig, path)


def plot_covariates(table, labels, path):
    """Covariates per cluster."""
    columns = present_covariates(table)
    clusters = np.unique(labels)
    fig, axes = plt.subplots(len(columns), 1, sharex=True,
                             figsize=(max(6, 0.6 * clusters.size), 2.6 * len(columns)))
    axes = np.atleast_1d(axes)
    for ax, column in zip(axes, columns):
        ax.boxplot([table[column][labels == c] for c in clusters],
                   tick_labels=_cluster_names(clusters), showfliers=False)
        ax.set_ylabel(column)
        if column == "n_synapses":
            ax.set_yscale("log")
        if column == "soma_depth_um":
            ax.invert_yaxis()
    axes[-1].set_xlabel("new cluster")
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------------------
# Stage D
# ---------------------------------------------------------------------------

def plot_assign_cv(cv_metrics, prob, truth, path_confusion, path_reliability):
    """Held-out confusion matrices (counts, shaded by row fraction) and reliability curves."""
    n_clusters = prob[METHODS[0]].shape[1]
    fig, axes = plt.subplots(1, len(METHODS), figsize=(1.5 + 3.8 * len(METHODS), 4.2))
    for ax, method in zip(axes, METHODS):
        metrics = cv_metrics[method]
        confusion = np.array(metrics["confusion_rows_true_cols_predicted"])
        fraction = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
        ax.imshow(fraction, cmap="Blues", vmin=0, vmax=1)
        for (row, col), value in np.ndenumerate(confusion):
            if value:
                ax.text(col, row, str(value), ha="center", va="center", fontsize=7,
                        color="w" if fraction[row, col] > 0.6 else "k")
        ax.set_xticks(range(n_clusters))
        ax.set_xticklabels(range(1, n_clusters + 1))
        ax.set_yticks(range(n_clusters))
        ax.set_yticklabels(range(1, n_clusters + 1))
        ax.set_xlabel("assigned cluster")
        ax.set_ylabel("consensus cluster")
        ax.set_title(f"{method}: acc {metrics['accuracy']:.2f}, NLL {metrics['nll']:.2f}, "
                     f"ECE {metrics['ece']:.2f}", fontsize=9)
    fig.tight_layout()
    _save(fig, path_confusion)

    keep = truth > 0
    fig, ax = plt.subplots(figsize=(5, 5))
    edges = np.linspace(0, 1, 11)
    for method in METHODS:
        confidence = prob[method][keep].max(axis=1)
        correct = (prob[method][keep].argmax(axis=1) + 1) == truth[keep]
        centres, accuracy = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
            if mask.sum() >= 5:
                centres.append(confidence[mask].mean())
                accuracy.append(correct[mask].mean())
        ax.plot(centres, accuracy, marker="o", label=method)
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("posterior of the assigned cluster")
    ax.set_ylabel("fraction correct (held-out references)")
    ax.set_title("Reliability (bins with >= 5 neurons)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path_reliability)


def plot_templates(templates, clouds, indices, labels, path, slab=30.0, max_points=4000):
    """Template density on the (dx, dy) plane through its centre, over member synapses
    within ``slab`` um of that plane."""
    clusters = sorted(templates)
    rng = np.random.default_rng(0)
    member_points = {}
    for c in clusters:
        points = np.concatenate([clouds.neuron_points(i) for i in indices[labels == c]])
        near = np.abs(points[:, 2] - templates[c].centre[1]) < slab
        points = points[near]
        if points.shape[0] > max_points:
            points = points[rng.choice(points.shape[0], size=max_points, replace=False)]
        member_points[c] = points
    pooled = np.concatenate(list(member_points.values()))
    x_low, x_high = np.percentile(pooled[:, 0], [0.5, 99.5])
    y_low, y_high = np.percentile(pooled[:, 1], [0.5, 99.5])
    dx, dy = np.linspace(x_low, x_high, 101), np.linspace(y_low, y_high, 201)

    fig, axes = plt.subplots(1, len(clusters), figsize=(3.4 * len(clusters), 5.5), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), clusters):
        ax.contourf(dx, dy, density_slice(templates[c], dx, dy), levels=12, cmap="Blues")
        ax.scatter(member_points[c][:, 0], member_points[c][:, 1], s=1, color="k", alpha=0.15)
        ax.set_title(f"cluster {c}: {int(np.sum(labels == c))} neurons, "
                     f"K={templates[c].n_components}", fontsize=9)
        ax.set_xlabel("dx (um)")
    axes = np.atleast_1d(axes)
    axes[0].set_ylabel("depth offset dy (um)")
    axes[0].invert_yaxis()
    fig.tight_layout()
    _save(fig, path)


def plot_gmm_selection(fits, path):
    """Chosen k per neuron and, per candidate k, the mean held-out BIC minus the neuron's lowest.

    ``fits`` are ``NeuronGMM`` objects (None entries skipped). The right panel
    shows the median and interquartile range over neurons; a flat curve means
    the extra components buy little held-out evidence.
    """
    fits = [fit for fit in fits if fit is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    chosen = np.array([fit.k for fit in fits])
    values = np.arange(chosen.min(), chosen.max() + 1) if chosen.size else np.arange(1)
    axes[0].bar(values, [int(np.sum(chosen == v)) for v in values], color="0.4")
    axes[0].set_xticks(values)
    axes[0].set_xlabel("chosen number of components")
    axes[0].set_ylabel("neurons")

    relative = {}
    for fit in fits:
        mean_bic = np.nanmean(np.asarray(fit.selection["test_bic"]), axis=1)
        if np.all(np.isnan(mean_bic)):
            continue
        for k, value in zip(fit.selection["candidates"], mean_bic - np.nanmin(mean_bic)):
            relative.setdefault(k, []).append(value)
    ks = sorted(relative)
    if ks:
        low, mid, high = (np.array([np.nanpercentile(relative[k], q) for k in ks]) for q in (25, 50, 75))
        axes[1].plot(ks, mid, marker="o", color="k")
        axes[1].fill_between(ks, low, high, alpha=0.25, color="0.5")
        axes[1].set_xticks(ks)
    axes[1].set_xlabel("number of components")
    axes[1].set_ylabel("held-out BIC minus the neuron's lowest")
    axes[1].set_title("median and IQR over neurons", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def plot_gmm_stability(fits, between, path):
    """Fitting stability of the per-neuron mixtures.

    Left: per neuron, the median mixture Wasserstein distance between its repeat
    fits (and, when reruns were made, between the reruns' final models) against
    the distance between the final models of random pairs of neurons
    (``between``, a sequence of distances). Middle: the within-neuron distance
    against the number of synapses in the fit. Right: the within-neuron distance
    at every candidate k (median and IQR over neurons).
    """
    fits = [fit for fit in fits if fit is not None and fit.stability]
    within = np.array([fit.stability["within_median"] for fit in fits], dtype=float)
    rerun = np.array([fit.stability["rerun"]["within_median"] for fit in fits if fit.stability["rerun"]],
                     dtype=float)
    between = np.asarray(between, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    series = [(within, "within neuron: repeat fits"), (rerun, "within neuron: reruns"),
              (between, "between neurons")]
    finite = [values[np.isfinite(values)] for values, _ in series]
    top = max((values.max() for values in finite if values.size), default=1.0)
    bins = np.linspace(0, top, 31)
    for values, (_, label) in zip(finite, series):
        if values.size:
            axes[0].hist(values, bins=bins, histtype="step", density=True, label=label)
    axes[0].set_xlabel("mixture Wasserstein distance (um)")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)

    n_synapses = np.array([fit.n_synapses for fit in fits])
    axes[1].scatter(n_synapses, within, s=8, alpha=0.4, color="k")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("synapses in the fit")
    axes[1].set_ylabel("within-neuron median distance (um)")

    per_k = {}
    for fit in fits:
        for k, record in fit.stability["per_k"].items():
            if np.isfinite(record["within_median"]):
                per_k.setdefault(int(k), []).append(record["within_median"])
    ks = sorted(per_k)
    if ks:
        low, mid, high = (np.array([np.percentile(per_k[k], q) for k in ks]) for q in (25, 50, 75))
        axes[2].plot(ks, mid, marker="o", color="k")
        axes[2].fill_between(ks, low, high, alpha=0.25, color="0.5")
        axes[2].set_xticks(ks)
    axes[2].set_xlabel("number of components")
    axes[2].set_ylabel("within-neuron median distance (um)")
    axes[2].set_title("median and IQR over neurons", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def plot_queries(prob, hard, path):
    """Posterior confidence per method and the number of queries per assigned label."""
    n_clusters = prob[METHODS[0]].shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in METHODS:
        axes[0].hist(prob[method].max(axis=1), bins=np.linspace(0, 1, 21), histtype="step",
                     label=method)
    axes[0].set_xlabel("posterior of the assigned cluster")
    axes[0].set_ylabel("queries")
    axes[0].legend(fontsize=8)

    values = np.arange(n_clusters + 1)
    width = 0.8 / len(METHODS)
    for i, method in enumerate(METHODS):
        counts = [int(np.sum(hard[method] == v)) for v in values]
        axes[1].bar(values + (i - (len(METHODS) - 1) / 2) * width, counts, width, label=method)
    axes[1].set_xticks(values)
    axes[1].set_xticklabels(["none"] + [str(v) for v in values[1:]])
    axes[1].set_xlabel("assigned cluster")
    axes[1].set_ylabel("queries")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)
