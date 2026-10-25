"""Split-half statistical invariants and the diagnostic entry point (no real data required)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from celltype import config, distances, split_half
from celltype.clouds import Clouds


def _clouds(path, sizes=(80, 80, 80, 80), *, constant=False):
    rng = np.random.default_rng(17)
    # Distinct IDs also exercise seeds/CSV output with connectome-sized integers.
    ids = np.arange(len(sizes), dtype=np.int64) + 720575940000000000
    points = [np.zeros((n, 3)) if constant else rng.uniform(-1, 1, (n, 3)) for n in sizes]
    for row, x in enumerate(points):
        x[:, 1] += 100 * (row // 2)
    all_points = np.concatenate(points)
    np.savez(path, cell_ids=ids, offsets=np.concatenate([[0], np.cumsum(sizes)]),
             dx=all_points[:, 0], dy=all_points[:, 1], dz=all_points[:, 2],
             weight=np.linspace(1, 5, sum(sizes)), soma_y=np.arange(len(sizes)) * 100,
             dist_border=np.array([100000] * (len(sizes) - 1) + [20000]), lateral_cutoff=250.0,
             **{f"label_{c}": np.array(["unused"] * len(sizes)) for c in config.LABEL_COLUMNS})
    return Clouds(path)


@pytest.mark.parametrize("n,m", [(40, 20), (41, 20), (80, 10)])
def test_disjoint_splits_and_reproducibility(n, m):
    a, b = split_half.split_indices(n, m, seed=1, cell_id=720575940000000001, rep=0)
    assert len(a) == len(b) == m
    assert len(np.unique(np.concatenate([a, b]))) == 2 * m
    assert np.all((a >= 0) & (a < n)) and np.all((b >= 0) & (b < n))
    aa, bb = split_half.split_indices(n, m, seed=1, cell_id=720575940000000001, rep=0)
    assert np.array_equal(a, aa) and np.array_equal(b, bb)
    other, _ = split_half.split_indices(n, m, seed=1, cell_id=720575940000000001, rep=1)
    assert not np.array_equal(a, other)
    if n == 2 * m:
        assert np.array_equal(np.sort(np.concatenate([a, b])), np.arange(n))


def test_split_weights_are_retained_once(tmp_path):
    clouds = _clouds(tmp_path / "clouds.npz")
    indices = np.arange(len(clouds))
    weighted = distances.gather(clouds, indices)
    uniform = distances.gather(clouds, indices, use_weights=False)
    a, b = split_half.split_clouds(weighted, clouds.cell_ids, 20, seed=3, rep=0)
    ua, ub = split_half.split_clouds(uniform, clouds.cell_ids, 20, seed=3, rep=0)
    for row in indices:
        left, right = split_half.split_indices(80, 20, seed=3, cell_id=clouds.cell_ids[row], rep=0)
        for half, unweighted, pick in ((a, ua, left), (b, ub, right)):
            x, mass = half.cloud(row)
            assert np.array_equal(x, clouds.neuron_points(row)[pick])
            assert np.array_equal(x, unweighted.cloud(row)[0])  # weights do not change the split
            weight = clouds.neuron_weights(row)[pick]
            assert np.allclose(mass, weight / weight.sum())
            assert np.allclose(unweighted.cloud(row)[1], np.full(20, 1 / 20))


def test_pairs_cover_upper_triangle_and_allow_zero_pairs():
    expected = np.column_stack(np.triu_indices(6, 1))
    assert np.array_equal(split_half.sample_pairs(6, 100, 2), expected)
    few = split_half.sample_pairs(10000, 30, 2)
    assert few.shape == (30, 2) and len(np.unique(few, axis=0)) == 30
    assert np.all(few[:, 0] < few[:, 1]) and np.all(few < 10000)
    assert np.array_equal(few, split_half.sample_pairs(10000, 30, 2))
    assert split_half.sample_pairs(1, 100, 2).shape == (0, 2)
    assert split_half.sample_pairs(6, 0, 2).shape == (0, 2)


def test_batched_split_distances_match_one_split_at_a_time(tmp_path):
    clouds = _clouds(tmp_path / "clouds.npz")
    flat = distances.gather(clouds, np.arange(4))
    pairs = split_half.sample_pairs(4, 3, 9)
    within, between = split_half.split_distances(flat, clouds.cell_ids, 10, pairs, seed=9, n_reps=3, n_jobs=2)
    assert within.shape == (3, 4) and between.shape == (3, 3)
    for rep in range(3):
        a, b = split_half.split_clouds(flat, clouds.cell_ids, 10, seed=9, rep=rep)
        assert np.allclose(within[rep], [distances.w2(*a.cloud(i), *b.cloud(i)) for i in range(4)])
        assert np.allclose(between[rep], [distances.w2(*a.cloud(i), *b.cloud(j)) for i, j in pairs])
    empty = np.empty((0, 2))   # a float array, as np.empty gives it, must not break the integer indexing
    assert split_half.split_distances(flat, clouds.cell_ids, 10, empty, seed=9, n_reps=2, n_jobs=1)[1].shape == (2, 0)


def test_matched_distances_detect_known_spatial_separation(tmp_path):
    clouds = _clouds(tmp_path / "clouds.npz")
    result = split_half.analyze(clouds, np.arange(4), [20, 10], n_reps=5, n_pairs=100, seed=4)
    assert result["self_distance"].shape == (2, 5, 4)
    assert result["between_distance"].shape == (2, 5, 6)
    assert result["m_values"].tolist() == [10, 20]
    assert np.all(result["self_distance"] < 3.5)  # both halves lie in the same unit cube
    i, j = result["pairs"].T
    same = i // 2 == j // 2
    assert np.all(result["between_distance"][:, :, same] < 3.5)
    assert np.all(result["between_distance"][:, :, ~same] > 96.5)
    neurons, pairs = split_half.result_tables(result, clouds)
    assert neurons.shape[0] == 8 and pairs.shape[0] == 12
    for level, m in enumerate(result["m_values"]):
        table = pairs[pairs["m"] == m]
        self_median = np.median(result["self_distance"][level], axis=0)
        assert np.allclose(table["self_i_median_um"], self_median[i])
        assert np.allclose(table["self_j_median_um"], self_median[j])
        assert np.allclose(table["between_p95_um"], np.percentile(result["between_distance"][level], 95, axis=0))
    assert (neurons["weight_ess_fraction"] < 1).all()
    assert "not a null test" in split_half.summary(result)["interpretation"]


def test_draws_do_not_depend_on_workers_m_order_or_other_neurons(tmp_path):
    clouds = _clouds(tmp_path / "clouds.npz")
    options = dict(n_reps=2, n_pairs=4, seed=5)
    a = split_half.analyze(clouds, np.arange(4), [10, 20], n_jobs=1, **options)
    b = split_half.analyze(clouds, np.arange(4), [20, 10, 20], n_jobs=2, **options)
    for key in ("pairs", "cell_ids", "self_distance", "between_distance"):
        assert np.array_equal(a[key], b[key])
    reordered = split_half.analyze(clouds, np.array([2, 0]), [20], **options)
    assert np.array_equal(reordered["self_distance"][0], a["self_distance"][1][:, [2, 0]])


def test_zero_spread_and_single_neuron_are_well_defined(tmp_path):
    clouds = _clouds(tmp_path / "clouds.npz", constant=True)
    result = split_half.analyze(clouds, np.arange(4), [10], n_reps=2, n_pairs=100)
    assert np.allclose(result["self_distance"], 0)
    i, j = result["pairs"].T
    expected = 100 * np.abs(i // 2 - j // 2)
    assert np.allclose(result["between_distance"], expected)
    one = split_half.analyze(clouds, np.array([0]), [10], n_reps=2, n_pairs=100)
    neurons, pairs = split_half.result_tables(one, clouds)
    assert pairs.empty and neurons["self_over_spread"].isna().all()
    assert one["between_distance"].shape == (1, 2, 0)
    assert split_half.summary(one)["per_m"]["10"]["median_pair_between_um"] is None
    json.dumps(split_half.summary(one), allow_nan=False)


@pytest.mark.parametrize("options", [{"n_reps": 0}, {"n_pairs": -1}, {"n_jobs": 0}, {"seed": -1}])
def test_invalid_analysis_options_fail_early(tmp_path, options):
    clouds = _clouds(tmp_path / "clouds.npz")
    with pytest.raises(ValueError):
        split_half.analyze(clouds, np.arange(4), [10], **options)


def test_insufficient_clouds_and_invalid_m_are_rejected(tmp_path):
    clouds = _clouds(tmp_path / "clouds.npz")
    for sizes in ([], [0], [-1], [1.5], [True], [41]):
        with pytest.raises(ValueError):
            split_half.analyze(clouds, np.arange(4), sizes)
    for indices in ([], [0, 0], [-1], [4], [0.0]):
        with pytest.raises(ValueError):
            split_half.analyze(clouds, indices, [10])


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_split_half.py"
    spec = importlib.util.spec_from_file_location("analyze_split_half", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entry_point_fixed_cohort_outputs_and_safe_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    path = config.clouds_path("microns", "axon")
    clouds = _clouds(path, sizes=(80, 45, 25, 80))
    script = _script()
    args = ["--sides", "axon", "--points-per-half", "10", "20", "--n-reps", "2",
            "--min-synapses", "1", "--min-border-distance", "50", "--between-pairs", "10"]
    script.main(args)
    summaries = list(tmp_path.glob("microns/split_half/*/axon/summary.json"))
    assert len(summaries) == 1
    outdir = summaries[0].parent
    report = json.loads(summaries[0].read_text())
    assert report["n_neurons"] == 2 and report["n_pairs"] == 1
    assert report["selection"] == {"n_input": 4, "n_below_min_synapses": 0, "n_excluded_by_border": 1,
                                    "n_insufficient_for_largest_m": 1, "n_eligible_at_all_m": 2,
                                    "n_not_selected": 0}
    table = pd.read_csv(outdir / "neurons.csv")
    for _, group in table.groupby("m"):
        assert group["pt_root_id"].tolist() == clouds.cell_ids[:2].tolist()
    with np.load(outdir / "distances.npz", allow_pickle=False) as result:
        assert result["self_distance"].shape == (2, 2, 2)
        assert result["between_distance"].shape == (2, 2, 1)
    for name in ("config.json", "pairs.csv", "distributions.png", "sample_size.png", "covariates.png"):
        assert (outdir / name).stat().st_size > 0
    timestamp = (outdir / "distances.npz").stat().st_mtime_ns
    script.main(args)
    assert (outdir / "distances.npz").stat().st_mtime_ns == timestamp
    # Regenerating Stage A in place must produce a different run, not stale reuse.
    old = report["options"]
    changed = {**old, "source": {**old["source"], "sha256": "different"}}
    assert config.split_half_name(old) != config.split_half_name(changed)
    assert config.split_half_name(old) != config.split_half_name({**old, "use_weights": False})
    assert config.split_half_name(old) != config.split_half_name({**old, "n_pairs": 0})
    assert config.split_half_name(old) != config.split_half_name({**old, "min_border_um": 60.0})
    with np.load(path, allow_pickle=False) as archive:
        contents = dict(archive)
    contents["dx"] = contents["dx"] * 2
    np.savez(path, **contents)
    script.main(args)
    summaries = list(tmp_path.glob("microns/split_half/*/axon/summary.json"))
    assert len(summaries) == 2
    assert len({json.loads(p.read_text())["options"]["source"]["sha256"] for p in summaries}) == 2
    assert (outdir / "distances.npz").stat().st_mtime_ns == timestamp


def test_entry_point_self_only_and_incomplete_run_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    _clouds(config.clouds_path("v1dd", "dendrite"), sizes=(40,), constant=True)
    script = _script()
    args = ["--dataset", "v1dd", "--sides", "dendrite", "--points-per-half", "20",
            "--n-reps", "1", "--min-synapses", "1", "--between-pairs", "0", "--no-weights"]
    script.main(args)
    outdir = next(tmp_path.glob("v1dd/split_half/*/dendrite/summary.json")).parent
    assert pd.read_csv(outdir / "pairs.csv").empty
    assert json.loads((outdir / "summary.json").read_text())["n_pairs"] == 0
    assert pd.read_csv(outdir / "neurons.csv")["weight_ess"].tolist() == pytest.approx([40])
    (outdir / "distributions.png").unlink()
    script.main(args)
    assert (outdir / "distributions.png").is_file()
    assert (outdir / "summary.json").is_file()


@pytest.mark.parametrize("args", [["--points-per-half", "0"], ["--n-reps", "0"],
                                  ["--between-pairs", "-1"], ["--min-border-distance", "nan"]])
def test_entry_point_rejects_invalid_arguments(args):
    with pytest.raises(SystemExit) as error:
        _script().main(args)
    assert error.value.code == 2
