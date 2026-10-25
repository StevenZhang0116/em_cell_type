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

def test_w2_identical_is_zero():
    x = np.random.default_rng(0).normal(size=(50, 3))
    assert distances.w2_points(x, x) == pytest.approx(0.0, abs=1e-9)


def test_w2_translation_equals_shift():
    x = np.random.default_rng(1).normal(size=(80, 3))
    shift = np.array([3.0, -4.0, 0.0])
    assert distances.w2_points(x, x + shift) == pytest.approx(5.0)


def test_w2_is_a_distance_between_weighted_measures():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 3))
    w = rng.uniform(0.5, 2.0, 40)
    assert distances.w2(x, w, x + np.array([0.0, 3.0, 4.0]), w) == pytest.approx(5.0)
    # Splitting a point's mass over two copies describes the same measure.
    split = np.concatenate([w, w[:10] / 2])
    split[:10] /= 2
    assert distances.w2(x, w, np.concatenate([x, x[:10]]), split) == pytest.approx(0.0, abs=1e-9)
    # Masses need not be normalised, sizes may differ, and the distance is symmetric.
    y = rng.normal(size=(25, 3))
    assert distances.w2(x, w, y, np.ones(25)) == pytest.approx(distances.w2(x, 3 * w, y, 7 * np.ones(25)))
    assert distances.w2(x, w, y, np.ones(25)) == pytest.approx(distances.w2(y, np.ones(25), x, w))


def test_flat_clouds_and_bootstrap():
    rng = np.random.default_rng(3)
    clouds = [rng.normal(size=(n, 3)) for n in (30, 50, 20)]
    masses = [rng.uniform(0.5, 2.0, n) for n in (30, 50, 20)]
    flat = distances.FlatClouds.from_list(clouds, masses)
    assert len(flat) == 3 and list(flat.sizes) == [30, 50, 20]
    x, m = flat.cloud(1)
    assert np.allclose(x, clouds[1]) and np.allclose(m, masses[1] / masses[1].sum())
    boot = distances.bootstrap(flat, np.random.default_rng(0))
    assert len(boot) == 3
    for i in range(3):
        bx, bm = boot.cloud(i)
        # Bootstrap points are original points with multiplicities in steps of 1/n.
        assert bm.sum() == pytest.approx(1.0) and 0 < bm.size <= flat.sizes[i]
        assert set(map(tuple, bx)).issubset(set(map(tuple, flat.cloud(i)[0])))
        assert np.allclose(bm * flat.sizes[i], np.round(bm * flat.sizes[i]))


@pytest.mark.parametrize("n, n_blocks", [(1, 4), (7, 3), (100, 16)])
def test_balanced_row_blocks_cover_every_row_once(n, n_blocks):
    blocks = distances.balanced_row_blocks(n, n_blocks)
    assert np.array_equal(np.concatenate(blocks), np.arange(n))
    sized = distances.balanced_row_blocks(n, n_blocks, sizes=np.arange(1, n + 1))
    assert np.array_equal(np.concatenate(sized), np.arange(n))


def test_pairwise_cross_and_paired_agree_with_w2():
    rng = np.random.default_rng(4)
    flat = distances.FlatClouds.from_list([rng.normal(size=(n, 3)) for n in (30, 20, 25, 40, 35, 30)])
    matrix = distances.pairwise(flat, n_jobs=1)
    assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 0.0)
    assert matrix[1, 4] == pytest.approx(distances.w2(*flat.cloud(1), *flat.cloud(4)), rel=1e-6)
    other = distances.FlatClouds.from_list([rng.normal(size=(n, 3)) for n in (10, 15)])
    across = distances.cross(other, flat, n_jobs=2)
    assert across.shape == (2, 6)
    assert across[1, 3] == pytest.approx(distances.w2(*other.cloud(1), *flat.cloud(3)), rel=1e-6)
    assert np.allclose(distances.paired(flat, flat, n_jobs=1), 0.0, atol=1e-9)


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
    pipeline.compute_exact_distances(clouds, "toy", "t", "axon", min_synapses=100, use_weights=True,
                                     seed=0)
    for rep in range(2):
        pipeline.compute_distances(clouds, "toy", "t", "axon", rep, min_synapses=100,
                                   use_weights=True, seed=0)
    pipeline.cluster_neurons("toy", "t", "axon", linkage="average", k_values=[2, 3, 4],
                             min_cluster_size=2, k=3)

    outdir = config.cluster_dir("toy", "t", "axon")
    table = pd.read_csv(outdir / "neurons.csv")
    assert adjusted_rand_score(table["cell_type"], table["cluster_k3"]) == pytest.approx(1.0)
    assert (outdir / "co_clustering.png").exists()
    # The clustered distance is the exact matrix of the complete clouds, not the replicate mean.
    with open(outdir / "summary.json") as handle:
        assert json.load(handle)["distance_matrix"] == "exact"
    with np.load(config.distance_dir("toy", "t", "axon") / "exact.npz") as data:
        exact = data["distance"]
        assert list(data["n_synapses"]) == [150] * 9
    assert np.allclose(np.load(outdir / "distance.npy"), exact)
    # Bootstrap noise floor (about 10 um on 150-point clouds) far below the group separation.
    assert (table["self_distance_um"] > 0).all() and (table["self_distance_um"] < 30).all()


def test_pipeline_random_subset_is_shared_across_replicates(tmp_path, monkeypatch):
    from celltype import config, pipeline

    clouds = _toy_clouds(tmp_path, monkeypatch)
    pipeline.compute_exact_distances(clouds, "toy", "t", "axon", min_synapses=100, use_weights=True,
                                     seed=0, n_neurons=6, max_synapses=100)
    for rep in range(2):
        pipeline.compute_distances(clouds, "toy", "t", "axon", rep, min_synapses=100,
                                   use_weights=True, seed=0, n_neurons=6, max_synapses=100)
    directory = config.distance_dir("toy", "t", "axon")
    matrices, _, eligible = clustering.load_replicates(directory)
    assert matrices.shape == (2, 6, 6) and eligible.size == 6
    assert clustering.load_exact(directory, eligible).shape == (6, 6)
    with pytest.raises(ValueError):
        clustering.load_exact(directory, eligible[:5])
    # The cap subsamples the clouds above it, the same way for every call.
    with np.load(directory / "exact.npz") as data:
        assert int(data["max_synapses"]) == 100 and list(data["n_synapses"]) == [100] * 6
    capped = distances.gather(clouds, eligible, max_synapses=100, seed=0)
    again = distances.gather(clouds, eligible[::-1], max_synapses=100, seed=0)
    assert np.allclose(capped.cloud(0)[0], again.cloud(5)[0])
    assert list(distances.gather(clouds, eligible).sizes) == [150] * 6
    # K at least the number of eligible neurons means all of them.
    assert pipeline.select_neurons(clouds, 100, 50, seed=0).size == 9


def test_assign_stage_recovers_toy_groups(tmp_path, monkeypatch):
    """Stage D: held-out references and the three unclustered neurons get their group's cluster."""
    from celltype import config, pipeline

    clouds = _toy_clouds(tmp_path, monkeypatch, per_group=5)
    pipeline.compute_exact_distances(clouds, "toy", "t", "axon", min_synapses=100, use_weights=True,
                                     seed=0, n_neurons=12)
    for rep in range(2):
        pipeline.compute_distances(clouds, "toy", "t", "axon", rep, min_synapses=100,
                                   use_weights=True, seed=0, n_neurons=12)
    pipeline.cluster_neurons("toy", "t", "axon", linkage="average", k_values=[2, 3, 4],
                             min_cluster_size=2, k=3)
    pipeline.assign_neurons("toy", "t", "axon", query="rest", min_synapses=100, use_weights=True,
                            seed=0, n_neighbours=3, template_k_max=2, template_points=2000,
                            cv_folds=2, reject_quantile=1.0, k=3)

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
    pipeline.assign_neurons("toy", "t", "axon", query="none", min_synapses=100, use_weights=True,
                            seed=0, n_neighbours=3, template_k_max=2, template_points=2000,
                            cv_folds=2, reject_quantile=1.0, k=3)
    outdir = config.assign_dir("toy", "t", "axon", "none")
    with open(outdir / "summary.json") as handle:
        summary = json.load(handle)
    assert summary["n_queries"] == 0 and summary["queries"] is None
    assert not (outdir / "queries.csv").exists() and (outdir / "reference_cv.csv").exists()


def test_trial_name_encodes_every_parameter():
    from celltype import config

    base = dict(n_neurons=10, n_reps=2, min_synapses=100, use_weights=True,
                seed=1, linkage="average", k_min=2, k_max=3, min_cluster_size=5)
    assert config.trial_name(**base) == "n10_rep2_full_min100_weighted_seed1_average_k2-3_mcs5"
    assert config.trial_name(**base, max_synapses=3000).split("_")[2] == "max3000"
    assert config.trial_name(**base, method="shared", shared_k=30, shared_alpha=10.0) \
        == "n10_rep2_shared30_a10_min100_weighted_seed1_average_k2-3_mcs5"
    assert config.trial_name(**{**base, "n_neurons": None}).startswith("nall_")
    assert config.trial_name(**base, min_border=50.0) \
        == "n10_rep2_full_min100_border50_weighted_seed1_average_k2-3_mcs5"
    names = {config.trial_name(**{**base, key: value}) for key, value in
             [("n_reps", 3), ("max_synapses", 1000), ("min_synapses", 50), ("use_weights", False),
              ("seed", 2), ("linkage", "complete"), ("k_max", 5), ("min_cluster_size", 2),
              ("method", "shared"), ("min_border", 50.0)]}
    assert len(names) == 10 and config.trial_name(**base) not in names


def test_border_filter_uses_um_against_stored_nm(tmp_path):
    n = 120
    np.savez(tmp_path / "clouds.npz", cell_ids=np.arange(3), offsets=np.arange(4) * n,
             dx=np.zeros(3 * n), dy=np.zeros(3 * n), dz=np.zeros(3 * n), weight=np.ones(3 * n),
             soma_y=np.zeros(3), dist_border=np.array([20_000.0, 200_000.0, 60_000.0]))
    from celltype import pipeline

    clouds = Clouds(tmp_path / "clouds.npz")
    assert pipeline.eligible_neurons(clouds, 100).tolist() == [0, 1, 2]
    assert pipeline.eligible_neurons(clouds, 100, min_border_um=50).tolist() == [1, 2]
    assert pipeline.eligible_neurons(clouds, 100, min_border_um=100).tolist() == [1]
    np.savez(tmp_path / "no_border.npz", cell_ids=np.arange(1), offsets=np.array([0, n]),
             dx=np.zeros(n), dy=np.zeros(n), dz=np.zeros(n), weight=np.ones(n), soma_y=np.zeros(1))
    with pytest.raises(ValueError):
        pipeline.eligible_neurons(Clouds(tmp_path / "no_border.npz"), 100, min_border_um=50)


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
    consensus, ambiguity = clustering.consensus_clustering(cuts, [2, 3], "average", 3)
    stab = clustering.stability(cuts, consensus, [2, 3])
    assert np.all(np.isnan(stab[3])) and np.allclose(stab[2], 1.0)
    assert clustering.most_stable_k(stab, ambiguity, k_min=3)[0] == 2


def test_most_stable_k_prefers_the_largest_k_within_tolerance():
    stab = {2: np.full(5, 0.9), 3: np.full(5, 0.80), 4: np.full(5, 0.79), 5: np.full(5, 0.78),
            6: np.full(5, 0.60), 7: np.full(5, np.nan)}
    pac = {2: 0.1, 3: 0.20, 4: 0.21, 5: 0.40, 6: 0.2, 7: float("nan")}
    k, rationale = clustering.most_stable_k(stab, pac)
    # 3, 4, 5 are within 0.03 of the best ARI; 5 is too ambiguous; the largest of the rest is 4.
    assert k == 4 and rationale["within_ari_tolerance"] == [3, 4, 5]
    assert rationale["best_median_ari"] == pytest.approx(0.80)


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
        pipeline.compute_distances(clouds, "toy", "s", "axon", rep, min_synapses=100,
                                   use_weights=True, seed=0, method="shared", shared_model=model,
                                   shared_alpha=5.0)
    pipeline.cluster_neurons("toy", "s", "axon", linkage="average", k_values=[2, 3, 4],
                             min_cluster_size=2, k=3)
    table = pd.read_csv(config.cluster_dir("toy", "s", "axon") / "neurons.csv")
    assert adjusted_rand_score(table["cell_type"], table["cluster_k3"]) == pytest.approx(1.0)
    with open(config.cluster_dir("toy", "s", "axon") / "summary.json") as handle:
        assert json.load(handle)["distance_matrix"] == "replicate_mean"   # no exact.npz for shared
    with np.load(config.distance_dir("toy", "s", "axon") / "rep000.npz") as data:
        assert str(data["method"]) == "shared" and data["theta"].shape == (9, 3)
    # Reloading gives the same model.
    reloaded = shared.SharedGMM.load(config.shared_model_path("toy", "s", "axon"))
    assert np.allclose(reloaded.cost, model.cost)


# ---------------------------------------------------------------------------
# Per-neuron GMMs with the ../mixture fitter
# ---------------------------------------------------------------------------

def _mixture_available():
    from celltype import config

    return (config.MIXTURE_ROOT / "src" / "scratch_models" / "gmm_scratch.py").exists()


needs_mixture = pytest.mark.skipif(not _mixture_available(), reason="../mixture is not checked out")


@needs_mixture
def test_mixture_gmm_is_imported_not_copied():
    from celltype import neuron_gmm

    modules = neuron_gmm.mixture_modules()
    root = neuron_gmm.MIXTURE_ROOT.resolve()
    assert all(Path(module.__file__).resolve().is_relative_to(root) for module in modules)
    assert hasattr(modules.mix_helper, "best_number_component")
    assert hasattr(modules.distributions, "component_ot_distance")
    # GMMWeighted's free-parameter counts in 3-D: 7k-1 (diagonal), 4k+1 (constraint), 10k-1 (full).
    assert [neuron_gmm.n_parameters(2, method) for method in neuron_gmm.METHODS] == [13, 9, 19]
    assert neuron_gmm.supported_k([1, 2, 3], 300, "diagonal", 20) == ([1, 2], False)
    assert neuron_gmm.supported_k([2, 3], 100, "diagonal", 20) == ([2], True)
    with pytest.raises(ValueError):
        neuron_gmm.select_and_fit(np.zeros((50, 3)), None, [1], n_repeats=2)
    # Non-consecutive candidates (a k step) are fine: 4 has the lowest BIC in every repetition.
    bic = np.array([[100.0] * 8, [50.0 + 0.1 * i for i in range(8)], [60.0 + 0.1 * i for i in range(8)]])
    chosen, p_values, rule = neuron_gmm.choose_k([1, 4, 7], bic)
    assert chosen == 4 and rule.startswith("mix_helper")


@needs_mixture
def test_gmm_w2_is_the_component_ot_distance():
    from celltype import neuron_gmm

    def mixture(weights, means, sds):
        return {"weights": np.asarray(weights, float), "means": np.asarray(means, float),
                "covariances": np.array([np.diag(np.square(sd)) for sd in sds], float)}

    a = mixture([0.5, 0.5], [[0, 0, -50], [0, 0, 50]], [[10, 10, 20], [15, 15, 20]])
    assert neuron_gmm.gmm_w2(a, a, "diagonal") == pytest.approx(0.0, abs=1e-6)
    # Shifting every component by the same vector moves the mixture by exactly its length ...
    shifted = {**a, "means": a["means"] + np.array([3.0, 4.0, 0.0])}
    assert neuron_gmm.gmm_w2(a, shifted, "diagonal") == pytest.approx(5.0, abs=1e-6)
    # ... the full-covariance formula agrees, and the metric is defined across different k.
    assert neuron_gmm.gmm_w2(a, shifted, "normal") == pytest.approx(5.0, abs=1e-4)
    single = mixture([1.0], [[0, 0, 0]], [[12, 12, 50]])
    assert np.isfinite(neuron_gmm.gmm_w2(a, single, "diagonal")) and neuron_gmm.gmm_w2(a, single, "diagonal") > 0
    assert neuron_gmm.within_distances([a, shifted, a], "diagonal").tolist() == pytest.approx([5, 0, 5], abs=1e-6)
    assert neuron_gmm.distances_to(a, [shifted, a], "diagonal").tolist() == pytest.approx([5, 0], abs=1e-6)
    # Total spread: within-component variance plus the between-component spread along depth.
    expected = np.sqrt(0.5 * (100 + 100 + 400) + 0.5 * (225 + 225 + 400) + 50**2)
    assert neuron_gmm.total_spread(a) == pytest.approx(expected)


@needs_mixture
def test_select_and_fit_recovers_two_depth_components():
    from celltype import neuron_gmm

    rng = np.random.default_rng(3)
    # (dx, dy, dz): two depth bands sharing the lateral centre, 300 synapses each.
    shallow = np.column_stack([rng.normal(0, 20, 300), rng.normal(-100, 15, 300), rng.normal(0, 20, 300)])
    deep = np.column_stack([rng.normal(0, 30, 300), rng.normal(80, 25, 300), rng.normal(0, 30, 300)])
    points = np.vstack([shallow, deep])
    fit = neuron_gmm.select_and_fit(points, None, [1, 2, 3], method="diagonal_constraint", n_repeats=6,
                                    min_synapses_per_parameter=10, seed=0)
    assert fit.k == 2 and fit.means.shape == (2, 3) and fit.covariances.shape == (2, 3, 3)
    assert np.allclose(sorted(fit.means[:, 2]), [-100, 80], atol=10)      # depth is the last axis
    assert np.allclose(fit.means[0, :2], fit.means[1, :2])                # one shared lateral centre
    assert fit.selection["rule"].startswith("mix_helper") and fit.selection["candidates"] == [1, 2, 3]
    assert not fit.selection["underpowered"] and np.asarray(fit.selection["test_bic"]).shape == (3, 6)
    density = fit.log_density(points)
    assert density.shape == (600,) and np.all(np.isfinite(density))
    assert np.allclose(fit.responsibilities(points).sum(axis=1), 1.0)
    # Stability: the six repeat fits at k=2 are close to each other and to the final model, well
    # below the cloud's spread (about 100 um, dominated by the two depth bands). The mixture
    # Wasserstein distance charges a mixing-weight difference the full squared separation of the
    # components (180 um here), so a 1% weight difference between subsamples already costs ~18 um.
    stability = fit.stability
    assert sorted(fit.repeat_models) == [1, 2, 3] and len(fit.repeat_models[2]) == 6
    assert stability["n_models"] == 6 and len(stability["within"]) == 15 and len(stability["to_final"]) == 6
    assert stability["spread_um"] == pytest.approx(100, abs=15)
    assert 0 < stability["within_median"] < 40 and stability["within_rel"] < 0.4
    assert stability["to_final_mean"] < 40 and stability["rerun"] is None
    assert set(stability["per_k"]) == {1, 2, 3} and 0 < stability["per_k"][1]["within_median"] < 20
    # The JSON parameters score identically without the fitted object.
    reloaded = neuron_gmm.NeuronGMM.from_dict(json.loads(json.dumps(fit.to_dict())))
    assert reloaded.model is None and np.allclose(reloaded.log_density(points), density)
    assert reloaded.summary()["depth_sd_um"] == pytest.approx(fit.summary()["depth_sd_um"])
    assert reloaded.stability["within_median"] == pytest.approx(stability["within_median"])
    # Reruns of the whole selection choose k=2 again. GMMWeighted starts EM once, from two random
    # synapses with the global covariance, and with this seed one rerun (and one of the six repeat
    # fits above) stops in the degenerate optimum where both components are wide and centred: it
    # sits ~114 um from the proper fits with a lower log-likelihood, while the other rerun lands on
    # the main fit exactly (same 600 synapses, only the EM start differs).
    rerun = neuron_gmm.select_and_fit(points, None, [1, 2, 3], method="diagonal_constraint", n_repeats=6,
                                      min_synapses_per_parameter=10, seed=0, n_reruns=2)
    record = rerun.stability["rerun"]
    assert len(rerun.rerun_models) == 2 and rerun.k == fit.k and np.allclose(rerun.means, fit.means)
    assert record["k"] == [2, 2, 2] and record["k_agreement"] == 1.0 and len(record["within"]) == 3
    assert record["within_median"] == pytest.approx(np.median(record["within"]))
    assert min(record["within"]) < 1 and max(record["within"]) > 50 and stability["within_max"] > 50
    assert max(record["log_likelihood"]) - min(record["log_likelihood"]) > 0.5
    # Keeping the best of three EM starts per fit removes that failure mode.
    starts = neuron_gmm.select_and_fit(points, None, [1, 2, 3], method="diagonal_constraint", n_repeats=6,
                                       min_synapses_per_parameter=10, seed=0, n_starts=3, n_reruns=2)
    assert starts.k == 2 and starts.stability["within_max"] < 40
    assert max(starts.stability["rerun"]["within"]) < 1 and starts.stability["rerun"]["k"] == [2, 2, 2]


@needs_mixture
def test_fit_neurons_table_and_files(tmp_path, monkeypatch):
    from celltype import neuron_gmm

    clouds = _toy_clouds(tmp_path, monkeypatch)        # 9 neurons, one 150-synapse blob each (sd 15)
    indices = np.arange(len(clouds))
    # Two worker processes: the results must come back without the fitter object (whose class
    # lives in ../mixture, not importable by name in the parent).
    fits, errors = neuron_gmm.fit_neurons(clouds, indices, [1, 2], n_jobs=2, seed=0, n_repeats=6,
                                          min_synapses_per_parameter=20)
    # 150 synapses support k=1 (6 parameters) but not k=2 (13 parameters x 20).
    assert not errors and all(fit.k == 1 and fit.selection["candidates"] == [1] for fit in fits)
    assert fits[0].model is None and np.isfinite(fits[0].log_density(clouds.neuron_points(0))).all()
    table = neuron_gmm.fits_table(fits, clouds, indices)
    assert len(table) == 9 and {"pt_root_id", "cell_type", "k", "bic", "lateral_sd_um"} <= set(table)
    assert set(neuron_gmm.STABILITY_COLUMNS) <= set(table)
    assert np.allclose(table["lateral_sd_um"], 15.0, atol=4)
    # One Gaussian refitted on 120-synapse subsamples moves by a few um, far below its 26 um spread.
    assert (table["n_stability_models"] == 6).all() and table["w2_within_median"].between(0.5, 10).all()
    assert table["spread_um"].between(20, 32).all() and table["w2_rerun_median"].isna().all()
    neuron_gmm.save_fits(tmp_path / "models.json", fits, clouds.cell_ids[indices], {"k_values": [1, 2]})
    loaded, cell_ids, options = neuron_gmm.load_fits(tmp_path / "models.json")
    assert cell_ids.tolist() == list(range(9)) and options == {"k_values": [1, 2]}
    assert loaded[0].k == 1 and np.allclose(loaded[0].means, fits[0].means)
    assert loaded[0].stability["within_median"] == pytest.approx(fits[0].stability["within_median"])
    neuron_gmm.save_stability(tmp_path / "stability.json", fits, clouds.cell_ids[indices])
    entries, cell_ids = neuron_gmm.load_stability(tmp_path / "stability.json")
    assert cell_ids.tolist() == list(range(9)) and entries[0]["k"] == 1 and entries[0]["rerun_models"] == []
    assert len(entries[0]["repeat_models"]) == 6 and entries[0]["repeat_models"][0]["covariances"].shape == (1, 3, 3)
    # Between-neuron reference: the three toy groups sit 100 um or more apart, so the 27 pairs
    # across groups are far apart while the 9 pairs within a group look like repeat fits.
    between = neuron_gmm.between_distances(fits, 100, seed=0)
    same_group = (between["i"] // 3 == between["j"] // 3).to_numpy()
    assert len(between) == 36 and same_group.sum() == 9
    assert (between["w2"][~same_group] > 50).all() and (between["w2"][same_group] < 15).all()
    assert len(neuron_gmm.between_distances(fits, 10, seed=0)) == 10
    from celltype import plotting

    plotting.plot_gmm_selection(fits, tmp_path / "selection.png")
    plotting.plot_gmm_stability(fits, between["w2"], tmp_path / "stability.png")
    assert (tmp_path / "selection.png").exists() and (tmp_path / "stability.png").exists()
