"""Unit tests for the celltype package. Run: python -m pytest tests -q"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from celltype import clustering, distances
from celltype.clouds import Clouds, build_side


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def test_w2_points_identical_is_zero():
    x = np.random.default_rng(0).normal(size=(50, 2))
    assert distances.w2_points(x, x) == pytest.approx(0.0)


def test_w2_points_translation_equals_shift():
    x = np.random.default_rng(1).normal(size=(80, 2))
    shift = np.array([3.0, -4.0])
    assert distances.w2_points(x, x + shift) == pytest.approx(5.0)


def test_w2_points_is_permutation_invariant():
    x = np.random.default_rng(2).normal(size=(40, 2))
    y = np.random.default_rng(3).normal(size=(40, 2))
    assert distances.w2_points(x, y) == pytest.approx(distances.w2_points(x[::-1], y))


@pytest.mark.parametrize("n, n_blocks", [(1, 4), (7, 3), (100, 16)])
def test_balanced_row_blocks_cover_every_row_once(n, n_blocks):
    blocks = distances.balanced_row_blocks(n, n_blocks)
    assert np.array_equal(np.concatenate(blocks), np.arange(n))


def test_pairwise_points_is_symmetric_with_zero_diagonal():
    sample = np.random.default_rng(4).normal(size=(6, 30, 2))
    matrix = distances.pairwise_points(sample, n_jobs=1)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix[1, 4] == pytest.approx(distances.w2_points(sample[1], sample[4]), rel=1e-6)


# ---------------------------------------------------------------------------
# Clouds
# ---------------------------------------------------------------------------

def _toy_tables():
    cells = pd.DataFrame({
        "pt_root_id": [20, 10],
        "pt_position_x_trafo": [0.0, 100.0],
        "pt_position_y_trafo": [50.0, 300.0],
        "pt_position_z_trafo": [0.0, 0.0],
        "classification_system": ["E", "I"],
        "cell_type": ["L23", "BC"],
        "cell_type_v2": ["L23", "BC"],
    })
    synapses = pd.DataFrame({
        "pre_pt_root_id": [10, 20, 10, 20, 20],
        "post_pt_root_id": [99, 99, 99, 99, 99],
        "pt_position_x_trafo": [103.0, 3.0, 100.0, 0.0, 400.0],
        "pt_position_y_trafo": [310.0, 40.0, 290.0, 50.0, 50.0],
        "pt_position_z_trafo": [4.0, 4.0, 0.0, 0.0, 0.0],
        "weights_pre": [1.0, 2.0, 1.0, 1.0, 1.0],
        "weights_post": [1.0] * 5,
    })
    return cells, synapses


def test_build_side_csr_follows_cell_order_and_cutoff(tmp_path):
    cells, synapses = _toy_tables()
    output = build_side(cells, synapses, "axon", lateral_cutoff=250.0)
    # Cell 20 keeps 2 synapses (the one at x=400 is beyond the cutoff); cell 10 keeps 2.
    assert output["offsets"].tolist() == [0, 2, 4]
    np.savez(tmp_path / "clouds.npz", **output)
    clouds = Clouds(tmp_path / "clouds.npz")

    # Cell 20 (soma at 0, 50, 0): offsets (3, -10, 4) and (0, 0, 0).
    first = clouds.neuron_points(0, "xyz")
    assert sorted(map(tuple, first.round(6).tolist())) == [(0.0, 0.0, 0.0), (3.0, -10.0, 4.0)]
    cylindrical = clouds.neuron_points(0, "cylindrical")
    assert sorted(map(tuple, cylindrical.round(6).tolist())) == [(0.0, 0.0), (5.0, -10.0)]
    assert sorted(clouds.neuron_weights(0).tolist()) == [1.0, 2.0]

    # Cell 10 (soma depth 300): depth is relative to the soma.
    second = clouds.neuron_points(1, "xyz")
    assert sorted(second[:, 1].tolist()) == [-10.0, 10.0]


def test_rotation_about_soma_axis_matters_only_for_xyz(tmp_path):
    """A 90-degree in-plane rotation of the same arbor: cylindrical sees no change."""
    rng = np.random.default_rng(5)
    n = 300
    dx, dy, dz = rng.normal(80, 20, n), rng.normal(0, 30, n), rng.normal(0, 20, n)
    np.savez(tmp_path / "clouds.npz", cell_ids=np.array([1, 2]), offsets=np.array([0, n, 2 * n]),
             dx=np.concatenate([dx, -dz]), dy=np.concatenate([dy, dy]),
             dz=np.concatenate([dz, dx]), weight=np.ones(2 * n),
             soma_y=np.zeros(2), label_cell_type=np.array(["a", "a"]))
    clouds = Clouds(tmp_path / "clouds.npz")

    def distance(coords):
        return distances.w2_points(clouds.neuron_points(0, coords),
                                   clouds.neuron_points(1, coords))

    assert distance("cylindrical") == pytest.approx(0.0, abs=1e-6)
    assert distance("xyz") > 50.0



# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _toy_clouds(tmp_path, monkeypatch):
    """Three groups of three neurons with distinct arbors, saved as dataset "toy"."""
    from celltype import config

    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path)
    rng = np.random.default_rng(6)
    n, centres = 150, [(0.0, -100.0), (100.0, 0.0), (0.0, 100.0)]
    dx, dy, dz = [], [], []
    for cx, cy in np.repeat(centres, 3, axis=0):
        dx.append(rng.normal(cx, 15, n))
        dy.append(rng.normal(cy, 15, n))
        dz.append(rng.normal(0, 15, n))
    groups = np.repeat(["a", "b", "c"], 3)
    np.savez(config.clouds_path("toy", "axon"), cell_ids=np.arange(9), offsets=np.arange(10) * n,
             dx=np.concatenate(dx), dy=np.concatenate(dy), dz=np.concatenate(dz),
             weight=np.ones(9 * n), soma_y=rng.uniform(0, 500, 9),
             **{f"label_{column}": groups for column in config.LABEL_COLUMNS})
    return Clouds(config.clouds_path("toy", "axon"))


def test_pipeline_stages_b_and_c_end_to_end(tmp_path, monkeypatch):
    """Stage C recovers the three toy groups at k = 3."""
    from sklearn.metrics import adjusted_rand_score

    from celltype import config, pipeline

    clouds = _toy_clouds(tmp_path, monkeypatch)
    for rep in range(2):
        pipeline.compute_distances(clouds, "toy", "t", "axon", "xyz", rep, n_points=40,
                                   min_synapses=100, use_weights=True, seed=0)
    pipeline.cluster_neurons("toy", "t", "axon", "xyz", linkage="average", k_values=[2, 3, 4],
                             min_cluster_size=2, k=3)

    outdir = config.cluster_dir("toy", "t", "axon", "xyz")
    table = pd.read_csv(outdir / "neurons.csv")
    assert adjusted_rand_score(table["cell_type"], table["cluster_k3"]) == pytest.approx(1.0)
    assert (outdir / "co_clustering.png").exists()


def test_pipeline_random_subset_is_shared_across_replicates(tmp_path, monkeypatch):
    from celltype import config, pipeline

    clouds = _toy_clouds(tmp_path, monkeypatch)
    for rep in range(2):
        pipeline.compute_distances(clouds, "toy", "t", "axon", "xyz", rep, n_points=20,
                                   min_synapses=100, use_weights=True, seed=0, n_neurons=6)
    matrices, _, eligible = clustering.load_replicates(config.distance_dir("toy", "t", "axon", "xyz"))
    assert matrices.shape == (2, 6, 6) and eligible.size == 6
    # K at least the number of eligible neurons means all of them.
    assert pipeline.select_neurons(clouds, 100, 50, seed=0).size == 9


def test_trial_name_encodes_every_parameter():
    from celltype import config

    base = dict(n_neurons=10, n_reps=2, n_points=200, min_synapses=100, use_weights=True,
                seed=1, linkage="average", k_min=2, k_max=3, min_cluster_size=5)
    assert config.trial_name(**base) == "n10_rep2_pt200_min100_weighted_seed1_average_k2-3_mcs5"
    assert config.trial_name(**{**base, "n_neurons": None}).startswith("nall_")
    names = {config.trial_name(**{**base, key: value}) for key, value in
             [("n_reps", 3), ("n_points", 100), ("min_synapses", 50), ("use_weights", False),
              ("seed", 2), ("linkage", "complete"), ("k_max", 5), ("min_cluster_size", 2)]}
    assert len(names) == 8 and config.trial_name(**base) not in names


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _blobs_with_outlier():
    """Two groups of 5 points on a line plus one far-away point (index 10)."""
    x = np.concatenate([np.arange(5) * 0.1, 10 + np.arange(5) * 0.1, [100.0]])
    return np.abs(x[:, None] - x[None, :])


def test_cut_major_skips_small_clusters():
    tree = clustering.build_tree(_blobs_with_outlier(), "average")
    # A plain cut into 2 clusters isolates the outlier; the major cut finds the two groups.
    labels = clustering.cut_major(tree, 2, min_size=3)
    assert labels[10] == 0
    assert len(set(labels[:5])) == 1 and len(set(labels[5:10])) == 1
    assert labels[0] != labels[5] and set(labels[:10]) == {1, 2}
    # min_size=1 is the ordinary cut: the outlier alone is one of the two clusters.
    plain = clustering.cut_major(tree, 2, min_size=1)
    assert sorted(np.bincount(plain)[1:]) == [1, 10]
    # Only two groups reach min_size: asking for 3 keeps the best cut, not all outliers.
    fallback = clustering.cut_major(tree, 3, min_size=3)
    assert set(fallback) == {0, 1, 2} and np.sum(fallback == 0) == 1


def test_stability_is_nan_when_consensus_misses_k():
    distance = _blobs_with_outlier()
    cuts = clustering.replicate_cuts(np.stack([distance, distance]), "average", [2, 3], 3)
    consensus, _ = clustering.consensus_clustering(cuts, [2, 3], "average", 3)
    stab = clustering.stability(cuts, consensus, [2, 3])
    assert np.all(np.isnan(stab[3])) and np.allclose(stab[2], 1.0)
    assert clustering.most_stable_k(stab, k_min=3) == 2


def test_consensus_and_stability_on_agreeing_replicates():
    distance = _blobs_with_outlier()
    cuts = clustering.replicate_cuts(np.stack([distance, distance]), "average", [2], 3)
    consensus, ambiguity = clustering.consensus_clustering(cuts, [2], "average", 3)
    assert consensus[2][10] == 0
    assert clustering.adjusted_rand_score(consensus[2], cuts[0][2]) == pytest.approx(1.0)
    assert ambiguity[2] == 0.0
    assert np.allclose(clustering.stability(cuts, consensus, [2])[2], 1.0)
    together = clustering.co_clustering(cuts, 2)
    stab = clustering.neuron_stability(together, consensus[2])
    assert np.isnan(stab[10]) and np.allclose(stab[:10], 1.0)
