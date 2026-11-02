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

Per-neuron GMM figures (celltype/neuron_gmm.py):
  gmm_selection       chosen number of components per neuron and the held-out selection-score curve (relative to each neuron's best)
  gmm_stability       within-neuron vs between-neuron mixture Wasserstein distances, vs synapse count and k
  gmm_tolerance_scan  what the tolerance rule chooses at other tolerances, from the stored scores
  gmm_labels          chosen number of components and total held-out gain by dataset label, and vs soma depth
  gmm_depth_profiles  mean marginal density of the fitted mixtures along the depth offset, per dataset label

Matched GMM comparisons (celltype/gmm_matches.py):
  gmm_matches             one-pair baseline: coverage, distances, label effects and balance
  gmm_partner_robustness  1/3/5 partner pairs, fixed-cohort sensitivity and coverage

Split-half diagnostics (celltype/split_half.py):
  split_half_distributions  split ranges per neuron, distributions of medians, endpoint baselines
  split_half_curves         matched-distance medians vs points per half, with a fixed cohort
  split_half_covariates     self-distance vs cloud spread and weight concentration
  split_half_pairs         depth/label strata and anatomically matched partner comparisons

Clustering validation (celltype/split_cluster.py):
  split_cluster              ARI vs joint coverage, held-out separation, k chosen from half A, per m
  split_cluster_contingency  A labels x B labels of one representative split
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import spearmanr

from .clustering import present_covariates
from .config import LABEL_COLUMNS


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


def plot_split_half_pairs(strata, effects, coverage, path):
    """Random-pair depth strata at largest m, matched effects across m, and coverage."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    bins = ["<=25", "25-50", "50-100", ">100"]
    largest = int(effects.m.max())
    random = strata.loc[(strata["sample"] == "random") & (strata.m == largest)]
    for (system, relation), frame in random.groupby(["system_pair", "label_relation"]):
        if system == "missing" or relation == "missing":
            continue
        frame = frame.set_index("depth_bin").reindex(bins)
        label = f"{system}, {relation}; n=" + "/".join(str(int(n)) if pd.notna(n) else "0" for n in frame.n_pairs)
        for ax, column in ((axes[0, 0], "median_between_um"), (axes[0, 1], "median_margin_um")):
            ax.plot(range(4), frame[column], "o-", label=label)
    for ax in axes[0]:
        ax.set(xticks=range(4), xticklabels=bins, xlabel="Soma depth gap (um)")
    if len(random):
        axes[0, 0].legend(fontsize=7)
    else:
        axes[0, 0].text(0.5, 0.5, "No random pairs", ha="center", transform=axes[0, 0].transAxes)
    axes[0, 0].set(ylabel="Median between W2 (um)", title=f"Original random pairs, m={largest}; counts by bin")
    axes[0, 1].axhline(0, color="0.5", ls="--")
    axes[0, 1].set(ylabel="Median split margin (um)", title="Between - max(endpoint self), within split")
    for (system, label), frame in effects.loc[effects.grouping == "cell_type"].groupby(["classification_system", "cell_type"]):
        frame = frame.sort_values("m")
        axes[1, 0].plot(frame.m, frame.median_delta_um, "o-", label=f"{label} ({system}); n={int(frame.n_anchors.iloc[0])}")
    overall = effects.loc[effects.grouping == "overall"].sort_values("m")
    axes[1, 0].plot(overall.m, overall.median_delta_um, "o--", color="black", label=f"All; n={int(overall.n_anchors.iloc[0])}")
    axes[1, 0].axhline(0, color="0.5", ls="--")
    axes[1, 0].set(xticks=overall.m, xlabel="Synapses per half (m)", ylabel="Median anchor delta (um)",
                   title="Matched partners: different - same; equal anchor weights")
    axes[1, 0].legend(fontsize=7)
    x = np.arange(len(coverage))
    axes[1, 1].bar(x - 0.18, coverage.coverage, width=0.36, label=">=1 partner pair")
    axes[1, 1].bar(x + 0.18, coverage.n_with_3_pairs / coverage.n_neurons, width=0.36, label="3 partner pairs")
    names = [f"{label}\n{matched}/{total}" for label, matched, total in
             zip(coverage.cell_type.fillna("missing"), coverage.n_matched, coverage.n_neurons)]
    axes[1, 1].set(xticks=x, xticklabels=names, ylim=(0, 1.05), ylabel="Fraction of original cohort",
                   title="Matching coverage; labels show matched / total")
    axes[1, 1].tick_params(axis="x", labelsize=7)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Split-half follow-up | fixed cohort and splits | descriptive comparisons")
    _save(fig, path)


def plot_tree_heatmap(tree, distance, table, labels, k, path, title):
    fig = plt.figure(figsize=(11, 12))
    ax_tree = fig.add_axes([0.14, 0.80, 0.74, 0.15])
    leaves = dendrogram(tree, ax=ax_tree, no_labels=True, color_threshold=0,
                        above_threshold_color="0.3")["leaves"]
    ax_tree.set_axis_off()
    ax_tree.set_title(title)

    barcodes = (("dataset cell_type", table["cell_type"].to_numpy()),
                (f"consensus (k={k})", _cluster_names(labels)))
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


def plot_k_curves(stab, ambiguity, agreement, chosen_k, path, *, n_outliers, pac_all_pairs):
    k_values = sorted(stab)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
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
    axes[0].set_title("ARI on consensus inliers\n(replicate label 0 retained)")
    axes[0].legend(fontsize=8, loc="lower left")

    axes[1].plot(k_values, [ambiguity[k] for k in k_values], marker="o", color="C3",
                 label="Consensus inlier pairs (selection)")
    axes[1].plot(k_values, [pac_all_pairs[k] for k in k_values], ls="--", color="0.5",
                 label="All pairs (diagnostic)")
    axes[1].set_ylabel("PAC (0.1 < co-clustering < 0.9)")
    axes[1].set_title("Ambiguity; compare with coverage")
    axes[1].legend(fontsize=7)

    for column in LABEL_COLUMNS:
        axes[2].plot(k_values, [agreement[column][k]["ami"] for k in k_values],
                     marker="o", label=column)
    axes[2].set_ylabel("AMI with dataset labels")
    axes[2].set_title("Agreement with existing labels")
    axes[2].legend(fontsize=8)
    axes[3].plot(k_values, [n_outliers[k] for k in k_values], marker="o", color="C4")
    axes[3].set(ylabel="Number of consensus outliers", title="Neurons excluded at each k")
    axes[3].set_ylim(bottom=0)
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
    if not columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No varying covariates", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, path)
        return
    clusters = np.unique(labels)
    fig, axes = plt.subplots(len(columns), 1, sharex=True,
                             figsize=(max(6, 0.6 * clusters.size), 2.6 * len(columns)))
    axes = np.atleast_1d(axes)
    for ax, column in zip(axes, columns):
        ax.boxplot([table[column][labels == c] for c in clusters],
                   tick_labels=_cluster_names(clusters), showfliers=True)
        ax.set_ylabel(column)
        if column == "mean_border_weight":
            ax.set_title(f"{int((table[column] == 1).sum())}/{len(table)} neurons have mean weight exactly 1")
        if column == "n_synapses":
            ax.set_yscale("log")
        if column == "soma_depth_um":
            ax.invert_yaxis()
    axes[-1].set_xlabel("new cluster")
    fig.tight_layout()
    _save(fig, path)


def plot_gmm_selection(fits, path):
    """Chosen number of components per neuron and, per candidate number, the mean held-out selection
    score minus the neuron's lowest (lower is better: minus the normalised log-likelihood, or the BIC).

    ``fits`` are ``NeuronGMM`` objects (None entries skipped). The right panel
    shows the median and interquartile range over neurons; a flat curve means
    the extra components buy little held-out evidence. The selection tolerance
    (``select_tolerance`` of the selection records, the largest when they
    differ) is drawn as a line when one was used.
    """
    fits = [fit for fit in fits if fit is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    chosen = np.array([fit.k for fit in fits])
    values = np.arange(chosen.min(), chosen.max() + 1) if chosen.size else np.arange(1)
    axes[0].bar(values, [int(np.sum(chosen == v)) for v in values], color="0.4")
    axes[0].set_xticks(values)
    axes[0].set_xlabel("chosen number of components")
    axes[0].set_ylabel("neurons")

    relative, scores, tolerances = {}, set(), set()
    for fit in fits:
        select_by = fit.selection.get("select_by", "bic")
        score = np.asarray(fit.selection["test_bic" if select_by == "bic" else "test_log_likelihood"], dtype=float)
        mean_score = np.nanmean(score if select_by == "bic" else -score, axis=1)
        if np.all(np.isnan(mean_score)):
            continue
        scores.add(select_by)
        tolerances.add(float(fit.selection.get("select_tolerance", 0.0)))
        for k, value in zip(fit.selection["candidates"], mean_score - np.nanmin(mean_score)):
            relative.setdefault(k, []).append(value)
    ks = sorted(relative)
    if ks:
        low, mid, high = (np.array([np.nanpercentile(relative[k], q) for k in ks]) for q in (25, 50, 75))
        axes[1].plot(ks, mid, marker="o", color="k")
        axes[1].fill_between(ks, low, high, alpha=0.25, color="0.5")
        axes[1].set_xticks(ks)
    if tolerances and max(tolerances) > 0:
        tolerance = max(tolerances)
        axes[1].axhline(tolerance, color="C3", linestyle="--", linewidth=1,
                        label=f"selection tolerance {tolerance:g}")
        axes[1].legend(fontsize=8, loc="upper right")
    what = "held-out BIC" if scores == {"bic"} else "minus held-out log-likelihood per synapse"
    axes[1].set_xlabel("number of components")
    axes[1].set_ylabel(f"{what}\nminus the neuron's lowest")
    axes[1].set_title("median and IQR over neurons", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def plot_gmm_stability(fits, between, path):
    """Fitting stability of the per-neuron mixtures.

    Left: per neuron, the median mixture Wasserstein distance between its repeat
    fits (and, when reruns were made, between the reruns' final models) against
    the distance between one repeat fit of each neuron for random pairs of
    neurons (``between``, a sequence of distances). Middle: the within-neuron distance
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
              (between, "between neurons: repeat fits")]
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


def plot_gmm_tolerance_scan(scan, path, used=None):
    """Stacked fractions of neurons per chosen number of components at each tolerance of a
    ``neuron_gmm.tolerance_scan`` (``{tolerance: {k: neurons}}``); ``used`` marks the run's tolerance."""
    scan = {float(t): {int(k): int(n) for k, n in counts.items()} for t, counts in scan.items()}
    tolerances = sorted(scan)
    ks = sorted({k for counts in scan.values() for k in counts})
    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * len(tolerances) + 2.5), 4))
    bottom = np.zeros(len(tolerances))
    cmap = plt.get_cmap("viridis", max(len(ks), 1))
    totals = np.array([sum(scan[t].values()) for t in tolerances], dtype=float)
    for j, k in enumerate(ks):
        count = np.array([scan[t].get(k, 0) for t in tolerances], dtype=float)
        fraction = np.divide(count, totals, out=np.zeros_like(count), where=totals > 0)
        ax.bar(range(len(tolerances)), fraction, bottom=bottom, width=0.7, color=cmap(j), label=f"k={k}")
        bottom += fraction
    ax.set_xticks(range(len(tolerances)))
    ax.set_xticklabels([f"{t:g}" + ("\n(used)" if used is not None and np.isclose(t, used) else "")
                        for t in tolerances])
    ax.set_xlabel("selection tolerance (units of the selection score)")
    ax.set_ylabel("fraction of neurons")
    ax.set_title("number of components the tolerance rule would choose", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    _save(fig, path)


def _label_order(table, column):
    """Label values ordered by their neurons' median soma depth (pia-side first)."""
    return table.groupby(column)["soma_depth_um"].median().sort_values().index.tolist()


def plot_gmm_matches(matches, coverage, balance, path):
    """Coverage, paired distances, per-label effects and covariate balance; no inferential bars."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    names = coverage.cell_type.fillna("missing").astype(str).tolist()
    axes[0, 0].bar(range(len(names)), coverage.coverage, color="0.4")
    axes[0, 0].set(xticks=range(len(names)),
                   xticklabels=[f"{name}\n{m}/{n}" for name, m, n in
                                zip(names, coverage.n_matched, coverage.n_neurons)],
                   ylim=(0, 1.05), ylabel="Matched fraction", title="Coverage by anchor label")
    axes[0, 0].tick_params(axis="x", labelsize=7)
    if len(matches):
        axes[0, 1].scatter(matches.same_w2_um, matches.different_w2_um, s=10, alpha=0.5)
        top = max(matches.same_w2_um.max(), matches.different_w2_um.max(), 1)
        axes[0, 1].plot([0, top], [0, top], "--", color="0.5")
        groups = list(matches.groupby("cell_type", sort=True))
        for i, (name, frame) in enumerate(groups):
            axes[1, 0].scatter(np.full(len(frame), i), frame.delta_um, s=9, alpha=0.35, color="0.3")
            axes[1, 0].plot([i - 0.25, i + 0.25], [frame.delta_um.median()] * 2, color="C3", lw=2)
        axes[1, 0].set(xticks=range(len(groups)),
                       xticklabels=[f"{name}\n{len(frame)}" for name, frame in groups])
        axes[1, 0].tick_params(axis="x", labelsize=7)
    else:
        axes[0, 1].text(0.5, 0.5, "No matched triplets", ha="center", transform=axes[0, 1].transAxes)
    axes[0, 1].set(xlabel="MW2 to same-label partner (um)", ylabel="MW2 to different-label partner (um)",
                   title="One point per anchor; above line = same closer")
    axes[1, 0].axhline(0, color="0.5", ls="--")
    axes[1, 0].set(ylabel="Different - same MW2 (um)", title="Conditional differences; red = median")
    finite_balance = balance.dropna(subset=["standardized_mean_difference"])
    axes[1, 1].barh(finite_balance.covariate, finite_balance.standardized_mean_difference, color="0.4")
    axes[1, 1].axvline(0, color="0.5", lw=1)
    axes[1, 1].set(xlabel="Same - different partner mean / cohort SD",
                   title="Matched covariate balance (undefined omitted)")
    fig.suptitle("1 partner pair per anchor | within classification_system | descriptive comparison")
    _save(fig, path)


def plot_gmm_partner_robustness(summary, coverage, path):
    """Partner-count sensitivity; each distance difference is averaged within anchor."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    counts = sorted(summary.requested_pairs.unique())
    overall = summary.loc[summary.grouping == "overall"]
    for population, name in (("all_matched", "All matched (up to k pairs)"),
                             ("common_5", "Fixed cohort (5 pairs available)")):
        frame = overall.loc[overall.population == population].sort_values("requested_pairs")
        label = f"{name}; n={int(frame.n_matched.iloc[0])}"
        axes[0, 0].plot(frame.requested_pairs, frame.median_delta_um, "o-", label=label)
        axes[0, 1].plot(frame.requested_pairs, frame.fraction_same_closer, "o-", label=label)
    axes[0, 0].axhline(0, color="0.6", ls="--")
    axes[0, 0].set(ylabel="Median anchor mean difference (um)", title="Overall distance difference")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(ylabel="Fraction of anchors with positive mean difference", ylim=(0, 1),
                   title="Same-label partners closer on average")
    common = summary.loc[(summary.population == "common_5") & (summary.grouping == "cell_type")]
    for (system, label), frame in common.groupby(["classification_system", "cell_type"]):
        frame = frame.sort_values("requested_pairs")
        axes[1, 0].plot(frame.requested_pairs, frame.median_delta_um, "o-",
                        label=f"{label} ({system}); n={int(frame.n_matched.iloc[0])}")
    if len(common):
        axes[1, 0].legend(fontsize=7)
    else:
        axes[1, 0].text(0.5, 0.5, "No anchors with 5 pairs", ha="center", transform=axes[1, 0].transAxes)
    axes[1, 0].axhline(0, color="0.6", ls="--")
    axes[1, 0].set(ylabel="Median anchor mean difference (um)", title="Per label, fixed cohort")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set(xticks=counts, xlabel="Requested partner pairs (k)")
    x = np.arange(len(coverage))
    width = 0.25
    for index, count in enumerate(counts):
        axes[1, 1].bar(x + (index - 1) * width, coverage[f"n_with_{count}_pairs"] / coverage.n_neurons,
                       width=width, label=f">= {count} pairs")
    axes[1, 1].set(xticks=x, xticklabels=coverage.cell_type.fillna("missing"), ylim=(0, 1.05),
                   ylabel="Fraction of original neurons", title="Coverage by available partner count")
    axes[1, 1].tick_params(axis="x", labelsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("1 / 3 / 5 partner pairs | equal anchor weights | descriptive comparisons")
    _save(fig, path)


def plot_gmm_labels(table, column, path):
    """Per dataset label (a column of ``fits_table``, ordered by median soma depth): the chosen number
    of components and the total held-out gain over the smallest candidate (dots: neurons, red: median),
    and the chosen number against soma depth with its Spearman correlation. False when the column is
    missing or empty. Descriptive: the labels play no part in the fits."""
    if column not in table:
        return False
    fitted = table[table["n_components"].notna() & table[column].notna()]
    if fitted.empty:
        return False
    order = _label_order(fitted, column)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    panels = ((axes[0], "n_components", "chosen number of components"),
              (axes[1], "gain_total", "held-out gain of the best candidate\nover the smallest (score units)"))
    for ax, value, label in panels:
        for i, name in enumerate(order):
            y = fitted.loc[fitted[column] == name, value].to_numpy(dtype=float)
            ax.scatter(i + rng.uniform(-0.25, 0.25, y.size), y, s=6, alpha=0.3, color="k")
            if np.isfinite(y).any():
                ax.plot([i - 0.3, i + 0.3], [np.nanmedian(y)] * 2, color="C3", linewidth=2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{name}\n{int((fitted[column] == name).sum())}" for name in order], fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(f"by dataset {column}, ordered by median soma depth", fontsize=9)
    depth = fitted["soma_depth_um"].to_numpy(dtype=float)
    k = fitted["n_components"].to_numpy(dtype=float)
    axes[2].scatter(depth, k, s=6, alpha=0.3, color="k")
    if len(fitted) > 2 and np.std(depth) > 0 and np.std(k) > 0:
        axes[2].set_title(f"Spearman correlation {spearmanr(depth, k)[0]:.2f}", fontsize=9)
    axes[2].set_xlabel("soma depth (um)")
    axes[2].set_ylabel("chosen number of components")
    fig.tight_layout()
    _save(fig, path)
    return True


def plot_gmm_depth_profiles(fits, table, column, path, min_count=5):
    """One panel per dataset label with at least ``min_count`` fitted neurons (ordered by median soma
    depth): the mean over its neurons of the fitted mixture's marginal density along the depth offset
    (negative = towards the pia, drawn upwards; dashed line: the soma). Every profile integrates to
    one and the panels share the density axis, so a compact cloud shows as a higher peak. ``fits``
    and ``table`` are in the same order. False when nothing can be drawn."""
    if column not in table:
        return False
    rows = [(fit, label) for fit, label in zip(fits, table[column]) if fit is not None and pd.notna(label)]
    counts = pd.Series([label for _, label in rows]).value_counts() if rows else pd.Series(dtype=int)
    order = [name for name in _label_order(table[table[column].notna()], column) if counts.get(name, 0) >= min_count]
    if not order:
        return False
    centres = np.concatenate([fit.means[:, 2] for fit, _ in rows])
    sds = np.sqrt(np.concatenate([fit.covariances[:, 2, 2] for fit, _ in rows]))
    grid = np.linspace(np.percentile(centres - 3 * sds, 2), np.percentile(centres + 3 * sds, 98), 400)
    fig, axes = plt.subplots(1, len(order), figsize=(1.6 * len(order) + 1.5, 5), sharex=True, sharey=True,
                             squeeze=False)
    for ax, name in zip(axes[0], order):
        profile = np.zeros_like(grid)
        members = [fit for fit, label in rows if label == name]
        for fit in members:
            mu, sd = fit.means[:, 2][:, None], np.sqrt(fit.covariances[:, 2, 2])[:, None]
            profile += fit.weights @ (np.exp(-0.5 * ((grid[None, :] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)))
        ax.plot(profile / len(members), grid, color="k")
        ax.axhline(0, color="0.5", linestyle="--", linewidth=1)
        ax.set_title(f"{name}\n{len(members)} neurons", fontsize=8)
        ax.set_xticks([])
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylabel("depth offset from the soma (um); pia upwards")
    fig.suptitle(f"mean marginal density of the fitted mixtures along depth, by dataset {column}", fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    return True
