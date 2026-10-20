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
from celltype.synthetic import synthetic_side


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


def test_gaussian_w2_matches_closed_form_for_isotropic():
    # W2^2 between N(m1, s1^2 I) and N(m2, s2^2 I) in 2-D: |m1-m2|^2 + 2 (s1-s2)^2
    value = distances.gaussian_w2_sq(np.zeros(2), 4.0 * np.eye(2), np.array([3.0, 0.0]), np.eye(2))
    assert value == pytest.approx(9.0 + 2.0)


def test_w2_mixture_identical_weights_is_zero():
    cost = np.array([[0.0, 4.0], [4.0, 0.0]])
    pi = np.array([0.3, 0.7])
    assert distances.w2_mixture(pi, pi, cost) == pytest.approx(0.0)
    assert distances.w2_mixture(np.array([1.0, 0.0]), np.array([0.0, 1.0]), cost) == pytest.approx(2.0)


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
    first = clouds.neuron_points(0, "xyz", "relative")
    assert sorted(map(tuple, first.round(6).tolist())) == [(0.0, 0.0, 0.0), (3.0, -10.0, 4.0)]
    cylindrical = clouds.neuron_points(0, "cylindrical", "relative")
    assert sorted(map(tuple, cylindrical.round(6).tolist())) == [(0.0, 0.0), (5.0, -10.0)]
    assert sorted(clouds.neuron_weights(0).tolist()) == [1.0, 2.0]

    # Cell 10 (soma depth 300): absolute depth adds the soma depth back.
    second = clouds.neuron_points(1, "xyz", "absolute")
    assert sorted(second[:, 1].tolist()) == [290.0, 310.0]


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
        return distances.w2_points(clouds.neuron_points(0, coords, "relative"),
                                   clouds.neuron_points(1, coords, "relative"))

    assert distance("cylindrical") == pytest.approx(0.0, abs=1e-6)
    assert distance("xyz") > 50.0


# ---------------------------------------------------------------------------
# End to end on synthetic shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coords", ["xyz", "cylindrical"])
@pytest.mark.parametrize("method", ["points", "shared_gmm"])
def test_synthetic_shapes_are_recovered(tmp_path, method, coords):
    np.savez(tmp_path / "clouds.npz", **synthetic_side(n_per_shape=8, rng=np.random.default_rng(0)))
    clouds = Clouds(tmp_path / "clouds.npz")
    eligible = np.arange(len(clouds))
    matrices = np.stack([
        distances.compute_replicate(
            clouds, eligible, method=method, coords=coords, depth_mode="relative", n_points=80,
            use_weights=True, n_components=6, rng=np.random.default_rng(rep), seed=rep,
            n_jobs=1)["distance"]
        for rep in range(2)])
    _, consensus = clustering.cut_tree(matrices.mean(axis=0), "average", [3])
    truth = clouds.meta["label_cell_type"]
    agreement = clustering.label_agreement(pd.DataFrame({c: truth for c in
                                                         ("classification_system", "cell_type", "cell_type_v2")}),
                                           consensus, [3])
    assert agreement["cell_type"][3]["ami"] > 0.9
