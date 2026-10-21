"""Unit tests for the celltype package. Run: python -m pytest tests -q"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd
import pytest

from celltype import assign, clustering, distances, shared, templates
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
    first = clouds.neuron_points(0)
    assert sorted(map(tuple, first.round(6).tolist())) == [(0.0, 0.0, 0.0), (3.0, -10.0, 4.0)]
    assert sorted(clouds.neuron_weights(0).tolist()) == [1.0, 2.0]

    # Cell 10 (soma depth 300): depth is relative to the soma.
    second = clouds.neuron_points(1)
    assert sorted(second[:, 1].tolist()) == [-10.0, 10.0]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _toy_clouds(tmp_path, monkeypatch, per_group=3):
    """Three groups of ``per_group`` neurons with distinct arbors, saved as dataset "toy"."""
    from celltype import config

    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path)
    rng = np.random.default_rng(6)
    n, centres = 150, [(0.0, -100.0), (100.0, 0.0), (0.0, 100.0)]
    dx, dy, dz = [], [], []
    for cx, cy in np.repeat(centres, per_group, axis=0):
        dx.append(rng.normal(cx, 15, n))
        dy.append(rng.normal(cy, 15, n))
        dz.append(rng.normal(0, 15, n))
    total = 3 * per_group
    groups = np.repeat(["a", "b", "c"], per_group)
    np.savez(config.clouds_path("toy", "axon"), cell_ids=np.arange(total),
             offsets=np.arange(total + 1) * n,
             dx=np.concatenate(dx), dy=np.concatenate(dy), dz=np.concatenate(dz),
             weight=np.ones(total * n), soma_y=rng.uniform(0, 500, total),
             **{f"label_{column}": groups for column in config.LABEL_COLUMNS})
    return Clouds(config.clouds_path("toy", "axon"))


def test_pipeline_stages_b_and_c_end_to_end(tmp_path, monkeypatch):
    """Stage C recovers the three toy groups at k = 3."""
    from sklearn.metrics import adjusted_rand_score

    from celltype import config, pipeline

    clouds = _toy_clouds(tmp_path, monkeypatch)
    for rep in range(2):
        pipeline.compute_distances(clouds, "toy", "t", "axon", rep, n_points=40,
                                   min_synapses=100, use_weights=True, seed=0)
    pipeline.cluster_neurons("toy", "t", "axon", linkage="average", k_values=[2, 3, 4],
                             min_cluster_size=2, k=3)

    outdir = config.cluster_dir("toy", "t", "axon")
    table = pd.read_csv(outdir / "neurons.csv")
    assert adjusted_rand_score(table["cell_type"], table["cluster_k3"]) == pytest.approx(1.0)
    assert (outdir / "co_clustering.png").exists()


def test_pipeline_random_subset_is_shared_across_replicates(tmp_path, monkeypatch):
    from celltype import config, pipeline

    clouds = _toy_clouds(tmp_path, monkeypatch)
    for rep in range(2):
        pipeline.compute_distances(clouds, "toy", "t", "axon", rep, n_points=20,
                                   min_synapses=100, use_weights=True, seed=0, n_neurons=6)
    matrices, _, eligible = clustering.load_replicates(config.distance_dir("toy", "t", "axon"))
    assert matrices.shape == (2, 6, 6) and eligible.size == 6
    # K at least the number of eligible neurons means all of them.
    assert pipeline.select_neurons(clouds, 100, 50, seed=0).size == 9


def test_assign_stage_recovers_toy_groups(tmp_path, monkeypatch):
    """Stage D: held-out references and the three unclustered neurons get their group's cluster."""
    from celltype import config, pipeline

    clouds = _toy_clouds(tmp_path, monkeypatch, per_group=5)
    for rep in range(2):
        pipeline.compute_distances(clouds, "toy", "t", "axon", rep, n_points=40,
                                   min_synapses=100, use_weights=True, seed=0, n_neurons=12)
    pipeline.cluster_neurons("toy", "t", "axon", linkage="average", k_values=[2, 3, 4],
                             min_cluster_size=2, k=3)
    pipeline.assign_neurons("toy", "t", "axon", query="rest", n_points=40, min_synapses=100,
                            use_weights=True, seed=0, n_reps=2, n_neighbours=3, template_k_max=2,
                            template_points=2000, cv_folds=2, reject_quantile=1.0, k=3)

    outdir = config.assign_dir("toy", "t", "axon", "rest")
    with open(outdir / "summary.json") as handle:
        summary = json.load(handle)
    assert summary["n_reference"] == 12 and summary["n_queries"] == 3
    for method in assign.METHODS:
        assert summary["cross_validation"][method]["accuracy"] == pytest.approx(1.0)
        assert summary["cross_validation"][method]["accuracy_with_reject"] == pytest.approx(1.0)
    assert summary["calibration"]["temperature"] >= 1.0

    reference = pd.read_csv(outdir / "reference_cv.csv")
    queries = pd.read_csv(outdir / "queries.csv")
    assert len(queries) == 3 and (reference["consensus"] > 0).all()
    group_of = reference.groupby("consensus")["cell_type"].agg(lambda s: s.mode()[0])
    for method in assign.METHODS:
        prob = queries[[f"{method}_prob_{c}" for c in (1, 2, 3)]].to_numpy()
        assert np.allclose(prob.sum(axis=1), 1.0)
        assert [group_of[c] for c in prob.argmax(axis=1) + 1] == queries["cell_type"].tolist()
    for name in ("templates.json", "cv_confusion.png", "cv_reliability.png", "templates.png",
                 "queries.png"):
        assert (outdir / name).exists()

    # query="none": calibration and validation only.
    pipeline.assign_neurons("toy", "t", "axon", query="none", n_points=40, min_synapses=100,
                            use_weights=True, seed=0, n_reps=2, n_neighbours=3, template_k_max=2,
                            template_points=2000, cv_folds=2, reject_quantile=1.0, k=3)
    outdir = config.assign_dir("toy", "t", "axon", "none")
    with open(outdir / "summary.json") as handle:
        summary = json.load(handle)
    assert summary["n_queries"] == 0 and summary["queries"] is None
    assert not (outdir / "queries.csv").exists() and (outdir / "reference_cv.csv").exists()


def test_trial_name_encodes_every_parameter():
    from celltype import config

    base = dict(n_neurons=10, n_reps=2, n_points=200, min_synapses=100, use_weights=True,
                seed=1, linkage="average", k_min=2, k_max=3, min_cluster_size=5)
    assert config.trial_name(**base) == "n10_rep2_pt200_min100_weighted_seed1_average_k2-3_mcs5"
    assert config.trial_name(**base, method="shared", shared_k=30, shared_alpha=10.0) \
        == "n10_rep2_shared30_a10_min100_weighted_seed1_average_k2-3_mcs5"
    assert config.trial_name(**{**base, "n_neurons": None}).startswith("nall_")
    names = {config.trial_name(**{**base, key: value}) for key, value in
             [("n_reps", 3), ("n_points", 100), ("min_synapses", 50), ("use_weights", False),
              ("seed", 2), ("linkage", "complete"), ("k_max", 5), ("min_cluster_size", 2),
              ("method", "shared")]}
    assert len(names) == 9 and config.trial_name(**base) not in names


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


def test_add_depth_term_is_exact_and_identity_at_zero():
    distance = np.array([[0.0, 3.0], [3.0, 0.0]])
    depths = np.array([0.0, 4.0])
    assert clustering.add_depth_term(distance, depths, depths, 0.0) is distance
    assert np.allclose(clustering.add_depth_term(distance, depths, depths, 1.0), [[0, 5], [5, 0]])
    stacked = clustering.add_depth_term(np.stack([distance, distance]), depths, depths, 0.25)
    assert stacked.shape == (2, 2, 2) and stacked[1, 0, 1] == pytest.approx(np.sqrt(9 + 4))
    # Query x reference matrix with its own depth vectors.
    query = clustering.add_depth_term(np.array([[1.0, 1.0]]), [10.0], depths, 1.0)
    assert np.allclose(query, [[np.sqrt(101), np.sqrt(37)]])
    assert clustering.depth_share(np.array([[0.0, 5.0], [5.0, 0.0]]), depths, 1.0) == pytest.approx(16 / 25)
    summary = clustering.cluster_depth_summary([100, 110, 120, 500, 510, 520], np.array([1, 1, 1, 2, 2, 2]))
    assert summary["fraction_of_cluster_pairs_with_overlapping_iqr"] == 0.0
    assert summary["quartiles_um"][1][1] == 110.0


def _same_shape_clouds(tmp_path, monkeypatch):
    """Six neurons with the same arbor shape: three shallow somata, three deep."""
    from celltype import config

    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path)
    rng = np.random.default_rng(9)
    n = 150
    groups = np.array(["shallow"] * 3 + ["deep"] * 3)
    np.savez(config.clouds_path("toy", "axon"), cell_ids=np.arange(6), offsets=np.arange(7) * n,
             dx=rng.normal(0, 15, 6 * n), dy=rng.normal(50, 15, 6 * n), dz=rng.normal(0, 15, 6 * n),
             weight=np.ones(6 * n), soma_y=np.array([100.0, 110.0, 120.0, 500.0, 510.0, 520.0]),
             **{f"label_{column}": groups for column in config.LABEL_COLUMNS})
    return Clouds(config.clouds_path("toy", "axon"))


def test_depth_weight_separates_same_shape_groups(tmp_path, monkeypatch):
    """Identical arbors at two soma depths: only the depth term can tell them apart."""
    from sklearn.metrics import adjusted_rand_score

    from celltype import config, pipeline

    clouds = _same_shape_clouds(tmp_path, monkeypatch)
    for rep in range(2):
        pipeline.compute_distances(clouds, "toy", "d", "axon", rep, n_points=40, min_synapses=100,
                                   use_weights=True, seed=0)
    pipeline.cluster_neurons("toy", "d", "axon", linkage="average", k_values=[2], min_cluster_size=2,
                             depth_weight=1.0, k=2)
    outdir = config.cluster_dir("toy", "d", "axon", 1.0)
    assert outdir.name == "cluster_axon_dw1"
    table = pd.read_csv(outdir / "neurons.csv")
    assert adjusted_rand_score(table["cell_type"], table["cluster_k2"]) == pytest.approx(1.0)
    with open(outdir / "summary.json") as handle:
        summary = json.load(handle)
    assert summary["depth_weight"] == 1.0 and summary["depth_share_of_squared_distance"] > 0.9
    assert summary["cluster_soma_depth"]["fraction_of_cluster_pairs_with_overlapping_iqr"] == 0.0
    # The default directory (weight 0) is untouched by the weighted run.
    assert not (config.cluster_dir("toy", "d", "axon") / "summary.json").exists()


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


# ---------------------------------------------------------------------------
# Templates and assignment
# ---------------------------------------------------------------------------

def _two_layer_cloud(rng, n=4000, centre=(20.0, -10.0)):
    """Two depth components sharing one lateral centre, with different lateral spreads."""
    half = n // 2
    depth = np.concatenate([rng.normal(-150, 20, half), rng.normal(100, 40, n - half)])
    lateral_sd = np.concatenate([np.full(half, 30.0), np.full(n - half, 60.0)])
    dx, dz = rng.normal(centre[0], lateral_sd), rng.normal(centre[1], lateral_sd)
    return np.column_stack([dx, depth, dz])


def test_cylindrical_gmm_recovers_components_and_selects_k():
    points = _two_layer_cloud(np.random.default_rng(7))
    model, bics = templates.fit_template(points, np.ones(len(points)), [1, 2, 3, 4], seed=0)
    assert model.n_components == 2 and bics[2] < bics[1]
    assert np.allclose(model.centre, [20.0, -10.0], atol=5)
    order = np.argsort(model.depth_mean)
    assert np.allclose(model.depth_mean[order], [-150, 100], atol=8)
    assert np.allclose(np.sqrt(model.lateral_var[order]), [30, 60], rtol=0.15)
    assert np.allclose(np.sqrt(model.depth_var[order]), [20, 40], rtol=0.2)
    assert np.allclose(model.weights, 0.5, atol=0.05)


def test_cylindrical_gmm_log_density_matches_closed_form():
    from scipy.stats import norm

    model = templates.CylindricalGMM.from_dict({
        "n_components": 1, "weights": [1.0], "centre": [0.0, 0.0],
        "lateral_var": [25.0], "depth_mean": [10.0], "depth_var": [100.0]})
    point = np.array([[3.0, 4.0, 4.0]])   # dx=3, dy=4, dz=4
    expected = norm.logpdf(3, 0, 5) + norm.logpdf(4, 0, 5) + norm.logpdf(4, 10, 10)
    assert model.log_density(point)[0] == pytest.approx(expected)
    assert model.n_parameters() == 5
    assert templates.CylindricalGMM.from_dict(model.to_dict()).to_dict() == model.to_dict()


def test_knn_posterior_votes_by_distance_and_pads_missing_neighbours():
    distances_qr = np.array([[1.0, 5.0, 1.2], [5.0, 1.0, 5.0]])
    labels = np.array([1, 2, 1])
    d, lab = assign.nearest_references(distances_qr, labels, n_neighbours=5)
    assert d.shape == (2, 5) and np.isnan(d[0, 3:]).all()
    assert lab[0].tolist() == [1, 1, 2, 0, 0]
    prob = assign.knn_posterior(d, lab, n_clusters=2, tau=1.0)
    assert np.allclose(prob.sum(axis=1), 1.0)
    assert prob[0, 0] > 0.9 and prob[1, 1] > 0.9
    # Very large tau: plain majority vote over the available neighbours.
    assert np.allclose(assign.knn_posterior(d, lab, 2, tau=1e12)[0], [2 / 3, 1 / 3])


def test_calibrate_picks_the_nll_minimiser():
    scores = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    grid = np.array([0.1, 1.0, 10.0])
    posterior = lambda t: assign.softmax_posterior(scores, t)  # noqa: E731
    best, curve = assign.calibrate(posterior, grid, np.array([1, 2, 1]))
    assert best == 10.0 and curve[0.1] > curve[1.0] > curve[10.0]
    assert assign.calibrate(posterior, grid, np.array([2, 1, 2]))[0] == 0.1
    assert assign.expected_calibration_error(posterior(10.0), np.array([1, 2, 1])) < 0.01


def test_stratified_folds_spread_every_label():
    labels = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    fold = assign.stratified_folds(labels, 2, np.random.default_rng(0))
    for value in (0, 1, 2):
        assert set(fold[labels == value]) == {0, 1}


# ---------------------------------------------------------------------------
# Shared-component model
# ---------------------------------------------------------------------------

def _two_component_model():
    means = np.array([[0.0, -100.0, 0.0], [0.0, 100.0, 0.0]])
    covariance = np.diag([100.0, 400.0, 100.0])
    return shared.SharedGMM([0.5, 0.5], means, [covariance, covariance])


def test_shared_gmm_density_and_cost_match_closed_forms():
    from scipy.stats import multivariate_normal

    model = _two_component_model()
    points = np.random.default_rng(0).normal(0, 50, (20, 3))
    expected = np.column_stack([multivariate_normal(model.means[k], model.covariances[k]).logpdf(points)
                                for k in range(2)])
    assert np.allclose(model.log_component_density(points), expected)
    # Equal covariances: the W2 between the components is the distance between their means.
    assert model.cost[0, 1] == pytest.approx(200.0 ** 2)
    assert shared.gaussian_w2_squared(np.zeros(3), np.eye(3), np.zeros(3), 4 * np.eye(3)) \
        == pytest.approx(3.0)


def test_shared_mixing_weights_shrink_towards_the_prior():
    model = _two_component_model()
    rng = np.random.default_rng(1)
    points = rng.normal(model.means[0], np.sqrt(np.diag(model.covariances[0])), (300, 3))
    sharp = model.mixing_weights(points, alpha=0.0)
    assert sharp[0] > 0.99 and sharp.sum() == pytest.approx(1.0)
    # As many pseudo-synapses as synapses: halfway between the data (1, 0) and the prior (.5, .5).
    assert model.mixing_weights(points, alpha=300.0)[0] == pytest.approx(0.75, abs=0.01)
    assert np.allclose(model.mixing_weights(points, alpha=1e9), model.weights, atol=1e-3)


def test_mw2_matches_gaussian_w2_for_one_hot_weights():
    model = _two_component_model()
    a, b, half = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])
    assert shared.mw2(a, a, model.cost) == pytest.approx(0.0)
    assert shared.mw2(a, b, model.cost) == pytest.approx(200.0)
    assert shared.mw2(a, half, model.cost) == pytest.approx(200.0 / np.sqrt(2))
    matrix = shared.pairwise_mw2(np.array([a, b, half]), model.cost, n_jobs=1)
    assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 0.0)
    assert matrix[0, 1] == pytest.approx(200.0) and matrix[0, 2] == pytest.approx(matrix[1, 2])


def test_shared_pipeline_recovers_toy_groups(tmp_path, monkeypatch):
    """Stage B with --method shared, then Stage C, finds the three toy groups."""
    from sklearn.metrics import adjusted_rand_score

    from celltype import config, pipeline

    clouds = _toy_clouds(tmp_path, monkeypatch)
    model = pipeline.fit_shared_model(clouds, "toy", "s", "axon", min_synapses=100, n_neurons=None,
                                      seed=0, use_weights=True, n_components=3, fit_points=1000)
    assert config.shared_model_path("toy", "s", "axon").exists() and model.n_components == 3
    for rep in range(2):
        pipeline.compute_distances(clouds, "toy", "s", "axon", rep, n_points=40, min_synapses=100,
                                   use_weights=True, seed=0, method="shared", shared_model=model,
                                   shared_alpha=5.0)
    pipeline.cluster_neurons("toy", "s", "axon", linkage="average", k_values=[2, 3, 4],
                             min_cluster_size=2, k=3)
    table = pd.read_csv(config.cluster_dir("toy", "s", "axon") / "neurons.csv")
    assert adjusted_rand_score(table["cell_type"], table["cluster_k3"]) == pytest.approx(1.0)
    with np.load(config.distance_dir("toy", "s", "axon") / "rep000.npz") as data:
        assert str(data["method"]) == "shared" and data["theta"].shape == (9, 3)
    # Reloading gives the same model.
    reloaded = shared.SharedGMM.load(config.shared_model_path("toy", "s", "axon"))
    assert np.allclose(reloaded.cost, model.cost)
