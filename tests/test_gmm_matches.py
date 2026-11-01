"""Matched comparison: confounding, label signal, coverage and exact neuron IDs."""

import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from celltype import config, gmm_matches, neuron_gmm

pytestmark = pytest.mark.filterwarnings("error:The behavior of DataFrame concatenation.*:FutureWarning")


def cohort():
    # Both labels occur at each depth, but A is mostly superficial and B mostly deep.
    depth = np.repeat([0.0, 100.0], 6)
    labels = ["A"] * 4 + ["B"] * 2 + ["A"] * 2 + ["B"] * 4
    n = len(depth)
    return pd.DataFrame({"pt_root_id": 864691135564542999 + np.arange(n, dtype=np.int64),
                         "cell_type": labels, "classification_system": ["excitatory"] * n,
                         "soma_depth_um": depth, "n_synapses": 20, "n_fitted": 20,
                         "n_components": 1, "dist_border_um": 100.0,
                         "weight_ess_fraction": 1.0, "mean_border_weight": 1.0,
                         "log_n_synapses": np.log(20), "log_n_fitted": np.log(20)})


def multi_partner_cohort():
    # Three disjoint depth bands permit respectively 5, 1, and 3 partner pairs.
    table = cohort().iloc[np.zeros(24, dtype=int)].copy().reset_index(drop=True)
    table["pt_root_id"] = 864691135564542999 + np.arange(24, dtype=np.int64)
    table["soma_depth_um"] = np.repeat([0.0, 100.0, 200.0], [12, 4, 8])
    table["cell_type"] = ["A"] * 6 + ["B"] * 6 + ["A"] * 2 + ["B"] * 2 + ["A"] * 4 + ["B"] * 4
    return table


def save_inputs(root, table, shift=0.0):
    root.mkdir(parents=True, exist_ok=True)
    table.to_csv(root / "neurons.csv", index=False)
    fits = [neuron_gmm.NeuronGMM(
        method="diagonal_constraint", k=1, weights=np.ones(1),
        means=np.array([[0, 0, row.soma_depth_um + shift * (row.cell_type == "B")]]),
        covariances=np.array([np.eye(3)]), n_synapses=20,
        log_likelihood=-1.0, bic=1.0, selection={}) for row in table.itertuples(index=False)]
    neuron_gmm.save_fits(root / "models.json", fits, table.pt_root_id, {"method": "diagonal_constraint"})
    return fits


@pytest.mark.parametrize("shift", [0.0, 40.0])
def test_matching_removes_depth_confound_and_retains_shape_signal(tmp_path, shift):
    table = cohort()
    save_inputs(tmp_path, table, shift)
    matched, annotated = gmm_matches.match_triplets(table)
    result, distances, _ = gmm_matches.attach_distances(
        matched, annotated, tmp_path / "models.json", tmp_path / "between.csv")
    assert len(result) == len(table)
    assert np.allclose(result.delta_um, shift)
    assert result.anchor_id.dtype == np.int64
    assert set(result.anchor_id) == set(table.pt_root_id)
    coverage, effects, balance, summary = gmm_matches.reports(result, annotated)
    assert (coverage.coverage == 1).all()
    assert np.allclose(effects.median_delta_um, shift)
    assert (balance.max_partner_gap == 0).all()
    assert summary["fraction_same_closer"] == (1.0 if shift else 0.0)
    json.dumps(summary, allow_nan=False)
    _, robustness, _ = gmm_matches.partner_robustness(result, annotated)
    common = robustness.loc[robustness.population == "common_5"]
    assert (common.n_matched == 0).all() and common.median_delta_um.isna().all()


@pytest.mark.parametrize("max_partners", [1, 5])
def test_matching_is_order_invariant_and_blind_to_outcomes(max_partners):
    table = cohort()
    expected, _ = gmm_matches.match_triplets(table, max_partners=max_partners)
    changed = table.sample(frac=1, random_state=4).copy()
    changed["n_components"] = 9
    changed["log_likelihood"] = np.arange(len(changed)) * 100
    changed["w2"] = np.arange(len(changed))[::-1]
    actual, _ = gmm_matches.match_triplets(changed, max_partners=max_partners)
    pd.testing.assert_frame_equal(actual, expected)


def test_partner_counts_are_nested_unique_and_keep_calipers():
    table = multi_partner_cohort()
    many, annotated = gmm_matches.match_triplets(table, max_partners=5)
    for count in (1, 3):
        fewer, _ = gmm_matches.match_triplets(table, max_partners=count)
        pd.testing.assert_frame_equal(many.loc[many.partner_rank <= count].reset_index(drop=True), fewer)
    assert annotated.n_partner_pairs.tolist() == [5] * 12 + [1] * 4 + [3] * 8
    indexed = table.set_index("pt_root_id")
    limits = np.array([25, np.log(1.25), np.log(1.25), 50, 0.1])
    for anchor, frame in many.groupby("anchor_id"):
        partners = [*frame.same_id, *frame.different_id]
        assert anchor not in partners and len(set(partners)) == len(partners)
        assert frame.partner_rank.tolist() == list(range(1, len(frame) + 1))
        for row in frame.itertuples(index=False):
            features = indexed.loc[[anchor, row.same_id, row.different_id], list(gmm_matches.FEATURES)].to_numpy()
            assert np.all(np.ptp(features, axis=0) <= limits + 1e-12)
            assert indexed.loc[anchor, "cell_type"] == indexed.loc[row.same_id, "cell_type"]
            assert indexed.loc[anchor, "cell_type"] != indexed.loc[row.different_id, "cell_type"]


def test_robustness_uses_anchor_means_equal_weights_and_fixed_cohort():
    table = multi_partner_cohort()
    table["mean_border_weight"] = np.where(table.soma_depth_um == 0, 100.0, 10.0)
    matches, annotated = gmm_matches.match_triplets(table, max_partners=5)
    dense = matches.anchor_id.isin(table.loc[table.soma_depth_um == 0, "pt_root_id"])
    matches["delta_um"] = np.where(dense, 9.0 + matches.partner_rank, -4.0)
    matches["same_w2_um"] = 10.0
    matches["different_w2_um"] = 10.0 + matches.delta_um
    anchors, summary, balance = gmm_matches.partner_robustness(matches, annotated)
    overall = summary.loc[summary.grouping == "overall"]
    all_anchors = overall.loc[overall.population == "all_matched"]
    common = overall.loc[overall.population == "common_5"]
    assert all_anchors.n_matched.tolist() == [24, 24, 24]
    assert all_anchors.n_full_requested.tolist() == [24, 20, 12]
    assert all_anchors.n_triplets.tolist() == [24, 64, 88]
    assert all_anchors.fraction_same_closer.tolist() == [0.5] * 3
    assert all_anchors.median_delta_um.tolist() == [3.0, 3.5, 4.0]
    assert common.n_matched.tolist() == [12] * 3
    assert common.median_delta_um.tolist() == [10.0, 11.0, 12.0]
    assert anchors.loc[anchors.in_common_5].groupby("requested_pairs").anchor_id.apply(set).tolist() == [
        set(table.pt_root_id.iloc[:12])] * 3
    assert np.allclose(anchors.delta_um, anchors.different_w2_um - anchors.same_w2_um)
    weights = balance.loc[(balance.population == "all_matched") & (balance.covariate == "mean_border_weight")]
    assert weights.mean_same_partner.tolist() == [55.0] * 3
    coverage, _, _, _ = gmm_matches.reports(matches.loc[matches.partner_rank == 1], annotated)
    assert coverage[["n_with_1_pairs", "n_with_3_pairs", "n_with_5_pairs"]].sum().tolist() == [24, 20, 12]


def test_all_three_edges_must_match_and_classes_cannot_mix():
    table = cohort().iloc[:3].copy()
    table["cell_type"] = ["A", "A", "B"]
    table["soma_depth_um"] = [0, -20, 20]
    matched, annotated = gmm_matches.match_triplets(table)
    assert matched.empty
    assert annotated.iloc[0].match_status == "no_joint_match"
    table["soma_depth_um"] = 0
    table.loc[table.index[2], "classification_system"] = "inhibitory"
    assert gmm_matches.match_triplets(table)[0].empty
    for option in ({"depth_um": 0}, {"count_ratio": 1}, {"ess_fraction": np.nan},
                   {"max_partners": 0}, {"max_partners": 1.5}):
        with pytest.raises(ValueError):
            gmm_matches.match_triplets(table, **option)


def test_cached_pairs_are_unordered_and_only_missing_edges_are_computed(tmp_path, monkeypatch):
    table = cohort()
    save_inputs(tmp_path, table, 40)
    matched, annotated = gmm_matches.match_triplets(table, max_partners=5)
    first = matched.iloc[0]
    a, b = int(first.anchor_id), int(first.same_id)
    pd.DataFrame([(b, a, 0.0), (a, b, 0.0)], columns=["pt_root_id_a", "pt_root_id_b", "w2"]).to_csv(
        tmp_path / "between.csv", index=False)
    original, calls = neuron_gmm.gmm_w2, []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(neuron_gmm, "gmm_w2", counted)
    _, distances, _ = gmm_matches.attach_distances(matched, annotated, tmp_path / "models.json", tmp_path / "between.csv")
    assert (distances.source == "between.csv").sum() == 1
    assert len(calls) == len(distances) - 1
    assert not distances.duplicated(["pt_root_id_a", "pt_root_id_b"]).any()


def test_stage_a_join_weights_and_stale_counts(tmp_path):
    table = cohort()
    table.to_csv(tmp_path / "neurons.csv", index=False)
    np.savez(tmp_path / "clouds.npz", cell_ids=table.pt_root_id.to_numpy()[::-1],
             offsets=np.arange(len(table) + 1) * 20, soma_y=table.soma_depth_um.to_numpy()[::-1],
             dist_border=np.full(len(table), 100000), weight=np.tile([1.0, 3.0], 10 * len(table)))
    actual = gmm_matches.load_covariates(tmp_path / "neurons.csv", tmp_path / "clouds.npz")
    assert actual.pt_root_id.tolist() == table.pt_root_id.tolist()
    assert np.allclose(actual.weight_ess_fraction, 0.8)
    assert np.allclose(actual.mean_border_weight, 2)
    assert np.allclose(actual.dist_border_um, 100)
    table.loc[0, "n_synapses"] = 19
    table.to_csv(tmp_path / "neurons.csv", index=False)
    with pytest.raises(ValueError, match="synapse counts"):
        gmm_matches.load_covariates(tmp_path / "neurons.csv", tmp_path / "clouds.npz")


def test_cli_and_no_overlap_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path)
    root = config.gmm_dir("microns", "dendrite")
    table = multi_partner_cohort()
    save_inputs(root, table, 40)
    cloud_path = config.clouds_path("microns", "dendrite")
    np.savez(cloud_path, cell_ids=table.pt_root_id.to_numpy(), offsets=np.arange(len(table) + 1) * 20,
             soma_y=table.soma_depth_um.to_numpy(), dist_border=np.full(len(table), 100000),
             weight=np.ones(20 * len(table)))
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_gmm_matches.py"
    spec = importlib.util.spec_from_file_location("analyze_gmm_matches", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = script.main([])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_matched"] == 24 and summary["median_delta_um"] == pytest.approx(40)
    assert [row["n_matched"] for row in summary["partner_robustness"]] == [24, 12] * 3
    assert len(pd.read_csv(out / "partner_matches.csv")) == 88
    assert len(pd.read_csv(out / "anchor_effects.csv")) == 72
    assert (out / "robustness.png").stat().st_size > 0
    assert (out / "comparison.png").stat().st_size > 0
    # No overlapping labels must produce an explicit empty result, including plots/JSON.
    table["cell_type"] = "A"
    save_inputs(root, table)
    script.main([])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_matched"] == 0 and summary["median_delta_um"] is None
    assert pd.read_csv(out / "matches.csv").empty
    assert pd.read_csv(out / "anchor_effects.csv").empty
    assert all(row["n_matched"] == 0 for row in summary["partner_robustness"])
