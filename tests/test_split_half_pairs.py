"""Strata, fixed matching, exact cached splits, and equal-anchor summaries."""

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


def fixture(path, *, single_label=False):
    rng = np.random.default_rng(31)
    n, size = 16, 20
    # Original cohort order intentionally differs from numeric root-ID order.
    ids = 864691135564542999 + rng.permutation(n).astype(np.int64)
    labels = np.array(["A"] * 8 + ["B"] * 8)
    points = rng.normal(size=(n, size, 3))
    points[8:, :, 1] += 40
    if single_label:
        labels[:] = "A"
    np.savez(path, cell_ids=ids, offsets=np.arange(n + 1) * size,
             dx=points[:, :, 0].ravel(), dy=points[:, :, 1].ravel(), dz=points[:, :, 2].ravel(),
             weight=np.tile(np.linspace(1, 2, size), n), soma_y=np.zeros(n),
             dist_border=np.full(n, 100000), label_cell_type=labels, label_cell_type_v2=labels,
             label_classification_system=np.full(n, "excitatory"))
    clouds = Clouds(path)
    result = split_half.analyze(clouds, np.arange(n), [4, 6], n_reps=3, n_pairs=4, seed=9)
    return clouds, result, split_half.cohort_table(result, clouds)


def test_strata_boundaries_missing_labels_and_exact_ids(tmp_path):
    _, _, neurons = fixture(tmp_path / "clouds.npz")
    neurons = neurons.iloc[:6].copy()
    neurons["soma_depth_um"] = [0, 25, 50, 100, 101, np.nan]
    neurons["cell_type"] = ["A", "A", "B", "B", "nan", "A"]
    neurons["classification_system"] = ["excitatory"] * 3 + ["inhibitory"] * 2 + [None]
    pairs = pd.DataFrame({"pt_root_id_i": [neurons.pt_root_id.iloc[0]] * 5,
                          "pt_root_id_j": neurons.pt_root_id.iloc[1:].to_numpy(), "m": 4,
                          "between_median_um": 20.0, "margin_median_um": 10.0,
                          "is_random": True, "is_targeted": [True, False, False, False, False]})
    annotated = split_half.annotate_pairs(pairs, neurons.sample(frac=1, random_state=2))
    assert annotated.depth_bin.tolist() == ["<=25", "25-50", "50-100", ">100", "missing"]
    assert annotated.label_relation.tolist() == ["same", "different", "different", "missing", "same"]
    assert annotated.system_pair.tolist() == ["E-E", "E-E", "E-I", "E-I", "missing"]
    assert annotated.pt_root_id_j.tolist() == pairs.pt_root_id_j.tolist()
    assert annotated.pt_root_id_j.dtype == np.int64
    stats = split_half.stratified_summary(annotated)
    assert stats.loc[stats["sample"] == "random", "n_pairs"].sum() == 5
    assert stats.loc[stats["sample"] == "targeted", "n_pairs"].sum() == 1
    with pytest.raises(ValueError, match="missing from"):
        split_half.annotate_pairs(pairs, neurons.iloc[:2])


def test_matching_uses_covariates_without_partner_reuse_or_outcome_selection(tmp_path):
    _, _, neurons = fixture(tmp_path / "clouds.npz")
    matches, cohort = split_half.match_partners(neurons)
    assert (cohort.n_partner_pairs == 3).all()
    without_border = split_half.match_partners(neurons.drop(columns=["dist_border_um"]))
    assert without_border[0].empty
    assert (without_border[1].match_status == "invalid_covariates").all()
    changed = neurons.sample(frac=1, random_state=2).copy()
    changed["spread_um"] = np.arange(len(changed)) * 1000
    changed["self_median_um"] = np.arange(len(changed))[::-1]
    actual, _ = split_half.match_partners(changed)
    pd.testing.assert_frame_equal(actual, matches)
    indexed = cohort.set_index("pt_root_id")
    limits = np.array([25, np.log(1.25), 50, 0.1])
    for anchor, frame in matches.groupby("anchor_id"):
        partners = [*frame.same_id, *frame.different_id]
        assert len(set(partners)) == 6 and anchor not in partners
        for row in frame.itertuples(index=False):
            triple = indexed.loc[[anchor, row.same_id, row.different_id]]
            assert np.all(np.ptp(triple[list(split_half.MATCH_FEATURES)].to_numpy(), axis=0) <= limits + 1e-12)
            assert triple.classification_system.nunique() == 1
            assert triple.cell_type.tolist() in (["A", "A", "B"], ["B", "B", "A"])
    # Both candidates match the anchor, but not each other.
    three = neurons.iloc[[0, 1, 8]].copy()
    three["soma_depth_um"] = [0, -20, 20]
    assert split_half.match_partners(three)[0].empty
    neurons.loc[neurons.cell_type == "B", "classification_system"] = "inhibitory"
    assert split_half.match_partners(neurons)[0].empty
    neurons["cell_type"] = "nan"
    assert (split_half.match_partners(neurons)[1].match_status == "missing_label").all()


def test_only_new_edges_are_computed_and_original_orientation_is_preserved(tmp_path, monkeypatch):
    clouds, saved, neurons = fixture(tmp_path / "clouds.npz")
    matches, _ = split_half.match_partners(neurons)
    flat = distances.gather(clouds, saved["eligible"])
    original, calls = split_half.w2, []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(split_half, "w2", counted)
    result = split_half.extend_pairs(saved, matches, flat, seed=9)
    n_new = int((~result["is_random"]).sum())
    assert n_new > 0 and len(calls) == len(saved["m_values"]) * 3 * n_new
    assert np.array_equal(result["self_distance"], saved["self_distance"])
    assert np.array_equal(result["between_distance"][:, :, :len(saved["pairs"])], saved["between_distance"])
    assert len(np.unique(result["pairs"], axis=0)) == len(result["pairs"])
    # Check a pair whose root-ID order is the reverse of its required cohort-index order.
    reversed_ids = saved["cell_ids"][result["pairs"][:, 0]] > saved["cell_ids"][result["pairs"][:, 1]]
    pos = np.flatnonzero(reversed_ids)[0]
    i, j = result["pairs"][pos]
    a, b = split_half.split_clouds(flat, saved["cell_ids"], 4, seed=9, rep=0)
    assert result["between_distance"][0, 0, pos] == pytest.approx(distances.w2(*a.cloud(i), *b.cloud(j)))
    assert not np.isclose(result["between_distance"][0, 0, pos], distances.w2(*a.cloud(j), *b.cloud(i)))
    calls.clear()
    reused = split_half.extend_pairs(result, matches, flat, seed=9)
    assert not calls
    assert np.array_equal(reused["between_distance"], result["between_distance"])


def test_effects_average_partners_then_splits_with_equal_anchor_weights():
    ids = 864691135564542999 + np.arange(10, dtype=np.int64)
    pairs = np.array([[0, 1], [0, 4], [0, 2], [0, 5], [0, 3], [0, 6], [7, 8], [7, 9]])
    raw = np.zeros((1, 3, len(pairs)))
    for rank in range(3):
        raw[0, :, rank * 2] = [0, 100, 200]
        raw[0, :, rank * 2 + 1] = np.array([10, 90, 210]) + 3 * (rank - 1)
    raw[0, :, 6], raw[0, :, 7] = 10, 6
    matches = pd.DataFrame({"anchor_id": ids[[0, 0, 0, 7]], "same_id": ids[[1, 2, 3, 8]],
                            "different_id": ids[[4, 5, 6, 9]], "cell_type": "A", "classification_system": "excitatory"})
    anchors, effects = split_half.matched_effects({"cell_ids": ids, "pairs": pairs, "m_values": [4],
                                                  "between_distance": raw}, matches)
    assert anchors.delta_median_um.tolist() == [10, -4]
    assert anchors.n_partner_pairs.tolist() == [3, 1]
    assert effects.iloc[0].median_delta_um == 3
    assert effects.iloc[0].fraction_same_closer == 0.5
    assert anchors.iloc[0].fraction_positive_splits == pytest.approx(2 / 3)


@pytest.mark.parametrize("empty", [False, True])
def test_entry_point_outputs_reuse_and_rejects_changed_source(tmp_path, monkeypatch, empty):
    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    root = config.split_half_dir("microns", "dendrite")
    source = config.clouds_path("microns", "dendrite")
    clouds, result, _ = fixture(source, single_label=empty)
    if empty:
        result["pairs"] = np.empty((0, 2), dtype=np.int64)
        result["between_distance"] = np.empty((2, 3, 0))
    options = {"schema_version": 1, "dataset": "microns", "side": "dendrite", "points_per_half": [4, 6],
               "n_reps": 3, "seed": 9, "use_weights": True, "source": config.file_fingerprint(source)}
    (root / "config.json").write_text(json.dumps(options))
    (root / "summary.json").write_text(json.dumps({"options": options}))
    np.savez_compressed(root / "distances.npz", **result)
    baseline = config.file_fingerprint(root / "distances.npz")
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_split_half_pairs.py"
    spec = importlib.util.spec_from_file_location("analyze_split_half_pairs", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = script.main([])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_matched_anchors"] == (0 if empty else 16)
    assert summary["n_random_pairs"] == (0 if empty else 4)
    assert config.file_fingerprint(root / "distances.npz") == baseline
    with np.load(out / "distances.npz") as data:
        assert np.array_equal(data["self_distance"], result["self_distance"])
        assert np.array_equal(data["between_distance"][:, :, :len(result["pairs"])], result["between_distance"])
    assert (out / "comparison.png").stat().st_size > 0
    assert pd.read_csv(out / "anchors.csv").empty == empty
    timestamp = (out / "distances.npz").stat().st_mtime_ns
    script.main([])
    assert (out / "distances.npz").stat().st_mtime_ns == timestamp
    with source.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(SystemExit):
        script.main([])
