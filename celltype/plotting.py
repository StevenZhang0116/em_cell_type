"""
Figures for the Clustering Stage
================================
  tree_heatmap   dendrogram, cell_type / cluster barcodes, sorted distance matrix
  k_curves       bootstrap stability and AMI with dataset labels vs k
  noise_floor    self-distance vs nearest-neighbour distance per neuron
  contingency    dataset cell_type x new cluster counts
  covariates     nuisance covariates per new cluster
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from .clustering import present_covariates
from .config import LABEL_COLUMNS


def _category_colors(values):
    unique = sorted(set(values))
    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    return {value: cmap(i) for i, value in enumerate(unique)}


def _save(fig, path):
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tree_heatmap(tree, mean_distance, table, labels, k, path, title):
    fig = plt.figure(figsize=(11, 12))
    ax_tree = fig.add_axes([0.14, 0.80, 0.74, 0.15])
    leaves = dendrogram(tree, ax=ax_tree, no_labels=True, color_threshold=None,
                        above_threshold_color="0.3")["leaves"]
    ax_tree.set_axis_off()
    ax_tree.set_title(title)

    barcodes = (("dataset cell_type", table["cell_type"].to_numpy()),
                (f"cluster (k={k})", labels.astype(str)))
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
    image = ax_heat.imshow(mean_distance[np.ix_(leaves, leaves)], cmap="viridis",
                           interpolation="nearest", aspect="auto")
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    fig.colorbar(image, cax=fig.add_axes([0.90, 0.05, 0.02, 0.4]), label="W2 distance (um)")
    _save(fig, path)


def plot_k_curves(stab, agreement, chosen_k, path):
    k_values = sorted(stab)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].errorbar(k_values, [stab[k][0] for k in k_values],
                     yerr=[stab[k][1] for k in k_values], marker="o", capsize=3)
    axes[0].set_ylabel("ARI(replicate, consensus)")
    axes[0].set_title("Bootstrap stability")
    for column in LABEL_COLUMNS:
        axes[1].plot(k_values, [agreement[column][k]["ami"] for k in k_values],
                     marker="o", label=column)
    axes[1].set_ylabel("AMI with dataset labels")
    axes[1].set_title("Agreement with existing labels")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axvline(chosen_k, color="0.6", ls="--")
        ax.set_xlabel("number of clusters k")
    fig.tight_layout()
    _save(fig, path)


def plot_noise(nearest, self_mean, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(nearest, self_mean, s=6, alpha=0.6)
    limit = max(nearest.max(), self_mean.max())
    ax.plot([0, limit], [0, limit], color="0.5", ls="--", lw=1)
    ax.set_xlabel("distance to nearest other neuron (um)")
    ax.set_ylabel("resampling self-distance (um)")
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
    ax.set_xticklabels(counts.columns)
    ax.set_yticks(range(counts.shape[0]))
    ax.set_yticklabels(counts.index, fontsize=8)
    ax.set_xlabel("new cluster")
    ax.set_ylabel("dataset cell_type")
    fig.tight_layout()
    _save(fig, path)


def plot_covariates(table, labels, path):
    columns = present_covariates(table)
    clusters = np.unique(labels)
    fig, axes = plt.subplots(len(columns), 1, sharex=True,
                             figsize=(max(6, 0.6 * clusters.size), 2.6 * len(columns)))
    axes = np.atleast_1d(axes)
    for ax, column in zip(axes, columns):
        ax.boxplot([table[column][labels == c] for c in clusters],
                   tick_labels=clusters.astype(str), showfliers=False)
        ax.set_ylabel(column)
        if column == "n_synapses":
            ax.set_yscale("log")
        if column == "soma_depth_um":
            ax.invert_yaxis()
    axes[-1].set_xlabel("new cluster")
    fig.tight_layout()
    _save(fig, path)
