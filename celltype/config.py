"""
Paths, Dataset Tables and Output Layout
=======================================
Input tables are the ones used by the Stage-1 GMM fit in
``../mixture/src/modeling/mixture_all.py``. Stage outputs live under
``output/<dataset>/``:

  clouds_<side>.npz                              (Stage A, shared)
  trials/config.json                             (Stage B/C parameters of the last run)
  trials/validate_<side>/                        (optional split-half clustering validation)
  trials/shared_<side>.npz                       (Stage B, --method shared only)
  trials/dist_<side>/exact.npz                   (Stage B, --method points: complete clouds)
  trials/dist_<side>/rep<b>.npz                  (Stage B, bootstrap replicates)
  trials/cluster_<side>/...                      (Stage C)
  split_half/<side>/                             (scripts/analyze_split_half.py)

One set of outputs per dataset. ``trials/config.json`` and the validation and
split-half ``config.json`` files record the options that produced them; a
later call with other options (or regenerated clouds) recomputes and
overwrites the files, so outputs of different settings never mix.

The per-neuron GMM analysis (scripts/fit_neuron_gmms.py, test code) writes
flat files gmm_output/<side>_*.{csv,json,png}; every run overwrites the last.
"""

from __future__ import annotations

import hashlib
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


def _ensure(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir(dataset):
    return _ensure(OUTPUT_ROOT / dataset)


def clouds_path(dataset, side):
    """The Stage A clouds of the proofread neurons of one side."""
    return dataset_dir(dataset) / f"clouds_{side}.npz"


def trial_dir(dataset):
    """Every Stage B, Stage C and validation output of the dataset."""
    return _ensure(dataset_dir(dataset) / "trials")


def shared_model_path(dataset, side):
    """The fitted shared-component mixture of ``--method shared``."""
    return trial_dir(dataset) / f"shared_{side}.npz"


def distance_dir(dataset, side):
    return _ensure(trial_dir(dataset) / f"dist_{side}")


def cluster_dir(dataset, side):
    return _ensure(trial_dir(dataset) / f"cluster_{side}")


def gmm_dir():
    """The flat folder of the per-neuron GMM analysis (``celltype/neuron_gmm.py``); runs overwrite each other."""
    return _ensure(GMM_OUTPUT_ROOT)


def split_half_dir(dataset, side):
    """Outputs of ``scripts/analyze_split_half.py`` for one side."""
    return _ensure(dataset_dir(dataset) / "split_half" / side)


def file_fingerprint(path):
    """Content identity used to bind validation caches to their Stage-A source."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"filename": Path(path).name, "size_bytes": Path(path).stat().st_size,
            "sha256": digest.hexdigest()}


def validation_dir(dataset, side):
    """The split-half clustering validation of one side."""
    return _ensure(trial_dir(dataset) / f"validate_{side}")
