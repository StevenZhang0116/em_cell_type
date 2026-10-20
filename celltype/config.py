"""
Paths, Dataset Tables and Output Layout
=======================================
Input tables are the ones used by the Stage-1 GMM fit in
``../mixture/src/modeling/mixture_all.py``. Stage outputs live under
``output/<dataset>/``:

  clouds_<side>.npz                               (scripts/01_extract_clouds.py)
  dist_<side>_<method>_<coords>_<depth>/rep<b>.npz   (scripts/02_compute_distances.py)
  cluster_<side>_<method>_<coords>_<depth>/...       (scripts/03_cluster_neurons.py)
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIXTURE_ROOT = PROJECT_ROOT.parent / "mixture"
DATA_DIR = MIXTURE_ROOT / "microns_v1dd" / "sven"
OUTPUT_ROOT = PROJECT_ROOT / "output"

NM_TO_UM = 1.0 / 1000.0
SIDES = ("axon", "dendrite")
# xyz: soma-centred (dx, dy, dz); cylindrical: (r, dy), invariant to rotation about the soma axis.
COORDS = ("xyz", "cylindrical")
DEPTH_MODES = ("relative", "absolute")
METHODS = ("points", "shared_gmm")

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
COVARIATE_COLUMNS = ("soma_depth_um", "n_synapses", "mean_border_weight", "dist_border")


def allocated_cpus():
    """CPUs granted by Slurm; 1 outside a job (never the node's core count)."""
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


def run_name(side, method, coords, depth_mode):
    return f"{side}_{method}_{coords}_{depth_mode}"


def _ensure(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir(dataset):
    return _ensure(OUTPUT_ROOT / dataset)


def clouds_path(dataset, side):
    return dataset_dir(dataset) / f"clouds_{side}.npz"


def distance_dir(dataset, side, method, coords, depth_mode):
    return _ensure(dataset_dir(dataset) / f"dist_{run_name(side, method, coords, depth_mode)}")


def cluster_dir(dataset, side, method, coords, depth_mode):
    return _ensure(dataset_dir(dataset) / f"cluster_{run_name(side, method, coords, depth_mode)}")
