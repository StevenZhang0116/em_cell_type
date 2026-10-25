"""
Paths, Dataset Tables and Output Layout
=======================================
Input tables are the ones used by the Stage-1 GMM fit in
``../mixture/src/modeling/mixture_all.py``. Stage outputs live under
``output/<dataset>/``:

  clouds_<side>.npz                              (Stage A, shared)
  clouds_dendrite_all.npz                        (Stage A with --all-cells)
  trials/<trial>/config.json                     (run parameters)
  trials/<trial>/shared_<side>.npz               (Stage B, --method shared only)
  trials/<trial>/dist_<side>/exact.npz           (Stage B, --method points: complete clouds)
  trials/<trial>/dist_<side>/rep<b>.npz          (Stage B, bootstrap replicates)
  trials/<trial>/cluster_<side>/...              (Stage C)
  trials/<trial>/assign_<side>_<query>/          (Stage D, optional)

``<trial>`` is built from the Stage B and C parameters (``trial_name``), so
runs with different parameters never share or overwrite each other's files.

The per-neuron GMM analysis (scripts/fit_neuron_gmms.py, test code) writes
flat files gmm_output/<side>_*.{csv,json,png}; every run overwrites the last.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIXTURE_ROOT = PROJECT_ROOT.parent / "mixture"
DATA_DIR = MIXTURE_ROOT / "microns_v1dd" / "sven"
OUTPUT_ROOT = PROJECT_ROOT / "output"
GMM_OUTPUT_ROOT = PROJECT_ROOT / "gmm_output"       # per-neuron GMM analysis, apart from the pipeline

NM_TO_UM = 1.0 / 1000.0
SIDES = ("axon", "dendrite")

DATASETS = {
    "microns": {
        "cell_table": "microns_cell_annos_CV_250904.feather",
        "synapse_table": "synapses_minnie65_phase3_v1_1507_combined_filtered_incl_trafo_250904_weight.feather",
        "has_weights": True,
    },
    "v1dd": {
        "cell_table": "v1dd_cell_annos_CV_woob_min200_250904.feather",
        "synapse_table": "synapses_v1dd_1196_combined_incl_trafo.feather",
        "has_weights": False,
    },
}

# Dataset labels kept only for post-hoc comparison; never used for clustering.
LABEL_COLUMNS = ("classification_system", "cell_type", "cell_type_v2")

# Per-neuron nuisance variables checked against the clusters.
COVARIATE_COLUMNS = ("soma_depth_um", "n_synapses", "mean_border_weight", "dist_border_um")


def allocated_cpus():
    """CPUs granted by Slurm; 1 outside a job (never the node's core count)."""
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


# ``--method points`` compares the complete synapse clouds; the exact W2 between
# clouds of n and m points costs about n * m memory and a little more than n^2 time (see
# docs/pipeline.md). Dendrites have a median of 2400 synapses (max 25k), axons 400
# (max 12k); ``max_synapses`` caps only the largest clouds.


def trial_name(*, n_neurons, n_reps, min_synapses, use_weights, seed,
               linkage, k_min, k_max, min_cluster_size, k=None, max_synapses=None,
               method="points", shared_k=30, shared_alpha=10.0, align=False, min_border=0.0):
    """Readable folder name that encodes every Stage B and C parameter.

    The third token says how neurons are compared: ``full`` for the exact W2
    between the complete synapse clouds (``max<n>`` when clouds above n
    synapses are capped), ``shared<K>_a<alpha>`` for MW2 between mixing
    weights over K shared components with Dirichlet strength alpha. ``aligned``
    marks canonically rotated clouds, ``border<um>`` a minimum soma distance
    from the volume border.
    """
    if method == "points":
        sampling = f"max{max_synapses}" if max_synapses else "full"
    else:
        sampling = f"shared{shared_k}_a{shared_alpha:g}"
    parts = [f"n{n_neurons or 'all'}", f"rep{n_reps}", sampling]
    if align:
        parts.append("aligned")
    parts.append(f"min{min_synapses}")
    if min_border:
        parts.append(f"border{min_border:g}")
    parts += ["weighted" if use_weights else "unweighted", f"seed{seed}", linkage,
              f"k{k_min}-{k_max}", f"mcs{min_cluster_size}"]
    if k is not None:
        parts.append(f"show{k}")
    return "_".join(parts)


def _ensure(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir(dataset):
    return _ensure(OUTPUT_ROOT / dataset)


def clouds_path(dataset, side, scope="proofread"):
    """``scope="all"``: the clouds of every cell in the table, not only proofread ones."""
    suffix = "" if scope == "proofread" else "_all"
    return dataset_dir(dataset) / f"clouds_{side}{suffix}.npz"


def trial_dir(dataset, trial):
    return _ensure(dataset_dir(dataset) / "trials" / trial)


def shared_model_path(dataset, trial, side):
    """The fitted shared-component mixture of ``--method shared``."""
    return trial_dir(dataset, trial) / f"shared_{side}.npz"


def distance_dir(dataset, trial, side):
    return _ensure(trial_dir(dataset, trial) / f"dist_{side}")


def cluster_dir(dataset, trial, side):
    return _ensure(trial_dir(dataset, trial) / f"cluster_{side}")


def assign_dir(dataset, trial, side, query):
    return _ensure(trial_dir(dataset, trial) / f"assign_{side}_{query}")


def gmm_dir():
    """The flat folder of the per-neuron GMM analysis (``celltype/neuron_gmm.py``); runs overwrite each other."""
    return _ensure(GMM_OUTPUT_ROOT)


def split_half_name(options):
    """Readable diagnostic name plus a digest of all options and the input-cloud fingerprint."""
    digest = hashlib.sha256(json.dumps(options, sort_keys=True, allow_nan=False).encode()).hexdigest()[:12]
    sizes = "-".join(str(m) for m in options["points_per_half"])
    return f"m{sizes}_rep{options['n_reps']}_n{options['n_neurons'] or 'all'}_{digest}"


def split_half_dir(dataset, run, side):
    """Independent diagnostics beside the main pipeline trials."""
    return _ensure(dataset_dir(dataset) / "split_half" / run / side)
