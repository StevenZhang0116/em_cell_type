"""Held-out clustering validation, leakage checks, resume and CLI regressions."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from celltype import clustering, config, pipeline, split_cluster
from celltype.clouds import Clouds
from celltype.distances import FlatClouds, bootstrap, gather, pairwise
from celltype.split_half import split_clouds
from scripts import run_pipeline


@pytest.fixture
def clouds(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    rng = np.random.default_rng(14)
    sizes = np.arange(40, 52)
    groups = np.repeat(np.arange(3), 4)
    points = np.concatenate([rng.normal([20, group * 80, 5], [3, 2, 1], (n, 3))
                             for n, group in zip(sizes, groups)])
    path = config.clouds_path("microns", "dendrite")
    np.savez(path, cell_ids=720575940000000001 + np.arange(12, dtype=np.int64),
             offsets=np.r_[0, sizes.cumsum()], dx=points[:, 0], dy=points[:, 1], dz=points[:, 2],
             weight=rng.uniform(1, 4, len(points)), soma_y=rng.uniform(0, 300, 12),
             dist_border=np.linspace(30_000, 100_000, 12),
             **{f"label_{column}": groups.astype(str) for column in config.LABEL_COLUMNS})
    return Clouds(path)


def _params(**changes):
    params = dict(method="points", n_neurons=None, n_reps=2, max_synapses=None,
                  shared_k=30, shared_alpha=10.0, min_synapses=40,
                  min_border=0.0, use_weights=True, seed=17, linkage="average",
                  k_min=2, k_max=3, min_cluster_size=2, k=None)
    return {**params, **changes}


def _halves(clouds, m=12):
    return split_clouds(gather(clouds, np.arange(len(clouds))), clouds.cell_ids, m, seed=17, rep=0)


def _fit(a, b, **changes):
    params = dict(n_reps=2, seed=[17, 29, 12, 0], n_jobs=1, linkage="average",
                  k_values=[2, 3], min_cluster_size=2)
    return split_cluster.fit_split(a, b, **{**params, **changes})


def test_common_consensus_preserves_existing_estimator():
    rng = np.random.default_rng(8)
    matrices = np.stack([squareform(pdist(np.repeat([[0, 0], [40, 0], [0, 40]], 4, axis=0)
                                         + rng.normal(size=(12, 2)))) for _ in range(3)])
    values = [2, 3, 4]
    cuts = clustering.replicate_cuts(matrices, "average", values, 2)
    labels, pac = clustering.consensus_clustering(cuts, values, "average", 2)
    stability = clustering.stability(cuts, labels, values)
    chosen, selection = clustering.most_stable_k(stability, pac)
    fit = clustering.fit_consensus(matrices, "average", values, 2)
    assert fit["chosen_k"] == chosen and fit["k_selection"] == selection
    np.testing.assert_array_equal(fit["labels"], labels[chosen])
    np.testing.assert_array_equal(fit["co_clustering"], clustering.co_clustering(cuts, chosen))


def test_unreachable_k_is_explicit_and_fixed_k_can_extend_candidates():
    distance = squareform(pdist(np.arange(6)[:, None]))
    failed = clustering.fit_consensus(np.array([distance]), "average", [2, 3], 4, k=3)
    assert failed["status"] == "unreachable_k" and failed["chosen_k"] == 3
    assert failed["reached"] == []
    auto = clustering.fit_consensus(np.array([distance]), "average", [2, 3], 4)
    assert auto["status"] == "unreachable_k" and auto["chosen_k"] is None
    fit = clustering.fit_consensus(np.array([distance]), "average", [2], 1, k=3)
    assert fit["status"] == "ok" and fit["chosen_k"] == 3
    numpy_candidates = clustering.fit_consensus(np.array([distance]), "average", np.array([2, 3]), 1)
    json.dumps(split_cluster.fit_record(numpy_candidates), allow_nan=False)


def test_a_selection_cannot_see_b_and_b_uses_frozen_k(clouds):
    a, b = _halves(clouds)
    first = _fit(a, b)
    rng = np.random.default_rng(200)
    altered = FlatClouds(rng.normal(size=b.points.shape), b.mass, b.offsets)
    second = _fit(a, altered)
    assert first["record"]["fit_a"] == second["record"]["fit_a"]
    np.testing.assert_array_equal(first["labels_a"], second["labels_a"])
    np.testing.assert_array_equal(first["distance_a"], second["distance_a"])
    assert first["record"]["fit_a"]["chosen_k"] == 3
    assert first["record"]["fit_b"]["k_selection"] == {"rule": "fixed from half A", "chosen": 3}
    assert list(first["record"]["fit_b"]["stability_ari"]) == ["3"]
    assert first["record"]["metrics"]["ari"] == pytest.approx(1)
    assert first["record"]["metrics"]["delta_um"] > 50


def test_inner_bootstrap_matches_main_order(clouds, monkeypatch):
    a, _ = _halves(clouds)
    expected = np.stack([pairwise(bootstrap(a, np.random.default_rng([7, rep])), 1) for rep in range(2)])
    original = clustering.fit_consensus
    def inspect(matrices, *args, **kwargs):
        np.testing.assert_array_equal(matrices, expected)
        return original(matrices, *args, **kwargs)
    monkeypatch.setattr(clustering, "fit_consensus", inspect)
    split_cluster.fit_half(a, n_reps=2, seed=[7], n_jobs=1, linkage="average", k_values=[3], min_cluster_size=2)


def test_metrics_are_label_invariant_and_separation_never_uses_b_labels():
    a = np.array([1, 1, 2, 2, 0, 3])
    b = np.array([7, 7, 9, 9, 4, 0])
    distance = squareform(pdist(np.array([0, 1, 10, 11, 20, 30])[:, None]))
    metrics, matches = split_cluster.compare_halves(a, b, distance)
    assert metrics["ari"] == 1 and metrics["joint_coverage"] == pytest.approx(4 / 6)
    assert metrics["outlier_rate_a"] == metrics["outlier_rate_b"] == pytest.approx(1 / 6)
    assert metrics["n_within_pairs"] == 2 and metrics["n_between_pairs"] == 8
    assert matches[0]["jaccard"] == 1 and matches[2]["retained_fraction"] == 0
    shuffled = np.array([2, 2, 1, 1, 3, 0])
    other, _ = split_cluster.compare_halves(a, shuffled, distance)
    assert other == metrics
    altered, _ = split_cluster.compare_halves(a, np.array([1, 2, 1, 2, 1, 2]), distance)
    assert altered["delta_um"] == metrics["delta_um"]
    assert altered["ari"] != metrics["ari"]


def test_outliers_degenerate_and_empty_pair_groups_are_not_successes():
    distance = np.ones((4, 4)) - np.eye(4)
    metrics, matches = split_cluster.compare_halves(np.zeros(4, int), np.zeros(4, int), distance)
    assert metrics["ari"] is None and metrics["joint_coverage"] == 0 and matches == []
    assert metrics["delta_um"] is None and metrics["separation_status"] == "empty_pair_group"
    metrics, _ = split_cluster.compare_halves(np.array([1, 1, 2, 2]), np.array([1, 1, 0, 0]), distance)
    assert metrics["status"] == "degenerate_common_cohort" and metrics["ari"] is None
    failed, _ = split_cluster.compare_halves(np.array([1, 1, 2, 2]), np.array([1, 1, 2, 2]), distance,
                                             status_b="unreachable_k")
    assert failed["status"] == "fit_failed" and failed["ari"] is None
    assert failed["delta_um"] == 0  # A partition still testable on B distances
    singletons, _ = split_cluster.compare_halves(np.arange(1, 5), np.arange(1, 5), distance)
    assert singletons["ari"] is None and singletons["status"] == "degenerate_common_cohort"


def test_parallel_reproducibility(clouds):
    a, b = _halves(clouds, m=8)
    serial, parallel = _fit(a, b), _fit(a, b, n_jobs=2)
    assert serial["record"] == parallel["record"]
    for key in ("distance_a", "distance_b", "labels_a", "labels_b"):
        np.testing.assert_array_equal(serial[key], parallel[key])


def test_resume_partial_run_reuses_completed_splits(clouds, monkeypatch):
    params = _params()
    original = split_cluster.fit_split
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)
    monkeypatch.setattr(split_cluster, "fit_split", counted)
    with pytest.raises(RuntimeError, match="interruption"):
        pipeline.validate_neurons("microns", "dendrite", points_per_half=[8, 12], outer_reps=2, params=params)
    directory = config.validation_dir("microns", "dendrite")
    assert [p.name for p in directory.glob("*.npz")] == ["m8.npz"] and not (directory / "summary.json").exists()
    with np.load(directory / "m8.npz", allow_pickle=False) as data:
        first = data["distance_a"].copy()
        assert first.shape == (1, 12, 12) and data["labels_b"].shape == (1, 12)
    # The resumed run computes only the three missing splits and keeps the saved one.
    result = pipeline.validate_neurons("microns", "dendrite", points_per_half=[8, 12], outer_reps=2, params=params)
    assert result == directory and len(calls) == 5
    assert sorted(p.name for p in directory.glob("*.npz")) == ["m12.npz", "m8.npz"]
    with np.load(directory / "m8.npz", allow_pickle=False) as data:
        assert data["distance_a"].shape == (2, 12, 12) and np.array_equal(data["distance_a"][:1], first)
        assert [r["rep"] for r in json.loads(str(data["records"]))] == [0, 1]
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["by_m"]["8"]["ari"]["median"] == 1
    assert summary["options"]["cell_ids"] == clouds.cell_ids.tolist()
    monkeypatch.setattr(split_cluster, "fit_split", lambda *a, **k: pytest.fail("cache was recomputed"))
    pipeline.validate_neurons("microns", "dendrite", points_per_half=[12, 8], outer_reps=2, params=params)


def test_validation_recomputes_on_other_settings_or_clouds_but_extends_splits(clouds):
    def saved(name):
        with np.load(out / name, allow_pickle=False) as data:
            return data["distance_a"].copy()

    out = pipeline.validate_neurons("microns", "dendrite", points_per_half=[8], outer_reps=1, params=_params())
    assert out == config.validation_dir("microns", "dendrite")
    first = saved("m8.npz")
    assert first.shape == (1, 12, 12)
    # More half sizes or outer splits extend the run, reusing the saved split.
    pipeline.validate_neurons("microns", "dendrite", points_per_half=[8, 12], outer_reps=2, params=_params())
    assert sorted(p.name for p in out.glob("m*.npz")) == ["m12.npz", "m8.npz"]
    assert saved("m8.npz").shape == (2, 12, 12) and np.array_equal(saved("m8.npz")[:1], first)
    # Fewer splits than saved: the extra ones stay in the file and are left out of the summary.
    pipeline.validate_neurons("microns", "dendrite", points_per_half=[8], outer_reps=1, params=_params())
    assert saved("m8.npz").shape == (2, 12, 12)
    assert json.loads((out / "summary.json").read_text())["by_m"]["8"]["n_splits"] == 1
    # Other settings recompute every split; files of the old settings do not survive.
    pipeline.validate_neurons("microns", "dendrite", points_per_half=[8], outer_reps=1, params=_params(seed=900))
    assert [p.name for p in out.glob("m*.npz")] == ["m8.npz"]
    assert saved("m8.npz").shape == (1, 12, 12) and not np.array_equal(saved("m8.npz"), first)
    assert json.loads((out / "config.json").read_text())["seed"] == 900
    # Regenerated clouds recompute as well.
    with np.load(clouds.path) as data:
        values = dict(data)
    values["dy"] = values["dy"] + 1
    np.savez(clouds.path, **values)
    pipeline.validate_neurons("microns", "dendrite", points_per_half=[8], outer_reps=1, params=_params(seed=900))
    assert json.loads((out / "config.json").read_text())["source"] == config.file_fingerprint(clouds.path)


def test_validation_uses_main_selection_and_cap(clouds, monkeypatch):
    params = _params(n_neurons=6, min_border=50, max_synapses=24)
    selected = pipeline.select_neurons(clouds, 40, 6, 17, 50)
    expected = gather(clouds, selected, True, 24, 17)
    def inspect(flat, options, outdir, **kwargs):
        assert options["cell_ids"] == clouds.cell_ids[selected].tolist()
        assert options["points_per_half"] == [8, 12]
        np.testing.assert_array_equal(flat.points, expected.points)
        np.testing.assert_array_equal(flat.mass, expected.mass)
        assert (flat.sizes == 24).all()
    monkeypatch.setattr(split_cluster, "run", inspect)
    pipeline.validate_neurons("microns", "dendrite", points_per_half=[12, 8], outer_reps=1, params=params)


def test_known_stale_distance_source_is_rejected(clouds):
    directory = config.distance_dir("microns", "dendrite")
    np.savez(directory / "rep000.npz", source_sha256="old-source", cell_ids=clouds.cell_ids)
    with pytest.raises(ValueError, match="different clouds"):
        pipeline._distance_provenance(directory, clouds, np.arange(len(clouds)))
    np.savez(directory / "rep000.npz", cell_ids=clouds.cell_ids)
    assert pipeline._distance_provenance(directory, clouds, np.arange(len(clouds))) is None


def test_fixed_cohort_fails_instead_of_silently_filtering(clouds):
    with pytest.raises(ValueError, match="never silently changes"):
        pipeline.validate_neurons("microns", "dendrite", points_per_half=[8, 25], outer_reps=1, params=_params())
    with pytest.raises(ValueError, match="max-synapses"):
        pipeline.validate_neurons("microns", "dendrite", points_per_half=[12], outer_reps=1,
                                  params=_params(max_synapses=20))
    with pytest.raises(ValueError, match="points only"):
        pipeline.validate_neurons("microns", "dendrite", points_per_half=[8], outer_reps=1,
                                  params=_params(method="shared"))


def test_failed_fits_are_saved_and_plotted(clouds):
    out = pipeline.validate_neurons("microns", "dendrite", points_per_half=[8], outer_reps=1,
                                    params=_params(k=8, min_cluster_size=6))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["by_m"]["8"]["fit_failure_rate"] == 1
    assert summary["by_m"]["8"]["ari"]["median"] is None
    assert summary["by_m"]["8"]["delta_um"]["median"] is None
    assert (out / "validation.png").exists()


def test_cli_validate_then_full_fit(clouds):
    args = ["--dataset", "microns", "--sides", "dendrite", "--stages", "cluster", "validate", "distances",
            "--min-synapses", "40", "--n-reps", "2", "--k-max", "3", "--min-cluster-size", "2",
            "--seed", "17", "--validation-points-per-half", "8", "12", "--validation-reps", "1"]
    run_pipeline.main(args)
    assert json.loads((config.trial_dir("microns") / "config.json").read_text()) == _params()
    full = json.loads((config.cluster_dir("microns", "dendrite") / "summary.json").read_text())
    path = config.validation_dir("microns", "dendrite") / "summary.json"
    summary = json.loads(path.read_text())
    assert summary["full_fit"]["status"] == "compatible" and summary["full_fit"]["k_full"] == full["chosen_k"] == 3
    assert full["cell_ids"] == summary["options"]["cell_ids"]
    assert summary["full_fit"]["k_full_selection_frequency_by_m"]["8"] == {"count": 1, "n_splits": 1}
    # A legacy full fit cannot silently inherit validation provenance.
    full.pop("source_sha256")
    (config.cluster_dir("microns", "dendrite") / "summary.json").write_text(json.dumps(full))
    pipeline.compare_validation_to_full(path.parent, "microns", "dendrite")
    assert json.loads(path.read_text())["full_fit"]["status"] == "full_fit_provenance_unknown"
    full["source_sha256"] = "stale"
    (config.cluster_dir("microns", "dendrite") / "summary.json").write_text(json.dumps(full))
    pipeline.compare_validation_to_full(path.parent, "microns", "dendrite")
    assert json.loads(path.read_text())["full_fit"]["status"] == "full_fit_mismatch"
    # Another replicate count adds or removes replicate files and reclusters; the exact
    # matrix stays. Stage C then sees exactly the requested replicates.
    distances = config.distance_dir("microns", "dendrite")
    exact = (distances / "exact.npz").stat().st_mtime_ns
    run_pipeline.main(args + ["--n-reps", "3"])
    assert sorted(p.name for p in distances.glob("rep*.npz")) == ["rep000.npz", "rep001.npz", "rep002.npz"]
    assert json.loads((config.trial_dir("microns") / "config.json").read_text())["n_reps"] == 3
    assert json.loads(path.read_text())["full_fit"]["status"] == "compatible"
    run_pipeline.main(args + ["--n-reps", "1"])
    assert [p.name for p in distances.glob("rep*.npz")] == ["rep000.npz"]
    assert (distances / "exact.npz").stat().st_mtime_ns == exact
    assert json.loads((config.cluster_dir("microns", "dendrite") / "summary.json").read_text())["n_replicates"] == 1
    # Any other parameter clears the distance files and recomputes them.
    run_pipeline.main(args + ["--n-reps", "1", "--seed", "18"])
    assert (distances / "exact.npz").stat().st_mtime_ns != exact
    assert [p.name for p in distances.glob("rep*.npz")] == ["rep000.npz"]


@pytest.mark.parametrize("extra", [["--method", "shared"], ["--validation-reps", "0"],
                                 ["--validation-points-per-half", "0"], ["--max-synapses", "100"]])
def test_cli_rejects_invalid_validation_before_extract(extra, monkeypatch):
    monkeypatch.setattr(pipeline, "extract_clouds", lambda *a: pytest.fail("extraction should not run"))
    with pytest.raises(SystemExit) as exc:
        run_pipeline.main(["--stages", "extract", "validate", *extra])
    assert exc.value.code == 2


def test_default_stages_preserved():
    assert run_pipeline.DEFAULT_STAGES == ("extract", "distances", "cluster")
