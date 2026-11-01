"""Stratify a saved split-half run and add anatomically matched label comparisons.

Run in Slurm after analyze_split_half.py. Keep the original cohort, splits and
pair orientation; only missing targeted distances are computed. Outputs live
in split_half/<side>/stratified/. This is a descriptive follow-up, not a null test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from celltype import plotting, split_half
from celltype.clouds import Clouds
from celltype.config import DATASETS, SIDES, allocated_cpus, clouds_path, file_fingerprint, split_half_dir
from celltype.distances import gather


def load_saved(root, clouds, options):
    """Reject mixed or incomplete saved arrays before reusing distances."""
    with np.load(root / "distances.npz", allow_pickle=False) as data:
        result = dict(data)
    ids, rows, pairs = result["cell_ids"], result["eligible"], result["pairs"]
    shape = (len(result["m_values"]), options["n_reps"])
    if (not np.array_equal(result["m_values"], options["points_per_half"])
            or len(np.unique(ids)) != len(ids) or not np.array_equal(ids, clouds.cell_ids[rows])
            or not np.array_equal(result["n_synapses"], clouds.n_synapses[rows])
            or result["self_distance"].shape != (*shape, len(ids))
            or result["between_distance"].shape != (*shape, len(pairs))):
        raise ValueError("saved arrays disagree with the source or split configuration")
    if (pairs.shape != (len(pairs), 2) or np.any(pairs < 0) or np.any(pairs >= len(ids))
            or np.any(pairs[:, 0] >= pairs[:, 1]) or len(np.unique(pairs, axis=0)) != len(pairs)):
        raise ValueError("saved pairs must be unique original-cohort indices i < j")
    for name in ("self_distance", "between_distance"):
        if not np.isfinite(result[name]).all() or np.any(result[name] < 0):
            raise ValueError(f"invalid saved {name}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="microns")
    parser.add_argument("--side", choices=SIDES, default="dendrite")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    root = split_half_dir(args.dataset, args.side)
    for name in ("config.json", "summary.json", "distances.npz"):
        if not (root / name).exists():
            parser.error(f"missing {root / name}; complete analyze_split_half.py first")
    original_config = json.loads((root / "config.json").read_text())
    original_summary = json.loads((root / "summary.json").read_text())
    if (original_summary["options"] != original_config or original_config["dataset"] != args.dataset
            or original_config["side"] != args.side or original_config["schema_version"] != 1):
        parser.error("base configuration and completion summary disagree")
    source = clouds_path(args.dataset, args.side)
    if file_fingerprint(source) != original_config["source"]:
        parser.error("Stage-A clouds changed; rerun the base split-half analysis before this follow-up")
    options = {"schema_version": 1, "base_config": original_config,
               "base_distances": file_fingerprint(root / "distances.npz"),
               "matching": split_half.MATCH_SETTINGS, "depth_bins_um": list(split_half.DEPTH_BINS),
               "pair_orientation": "A of lower original cohort index, B of higher original cohort index"}
    outdir = root / "stratified"
    names = ("config.json", "summary.json", "distances.npz", "pairs.csv", "strata.csv", "cohort.csv",
             "matches.csv", "anchors.csv", "effects.csv", "coverage.csv", "balance.csv", "comparison.png")
    if not args.overwrite and all((outdir / name).exists() for name in names):
        if json.loads((outdir / "config.json").read_text()) == options:
            print(f"Reusing completed analysis: {outdir}")
            return outdir
    n_jobs = allocated_cpus()
    if n_jobs < 1:
        parser.error("SLURM_CPUS_PER_TASK must be positive")
    clouds = Clouds(source)
    saved = load_saved(root, clouds, original_config)
    neurons = split_half.cohort_table(saved, clouds)
    matches, cohort = split_half.match_partners(neurons)
    flat = gather(clouds, saved["eligible"], use_weights=original_config["use_weights"])
    print(f"{len(neurons)} original neurons; {matches.anchor_id.nunique()} matched anchors; {n_jobs} CPUs", flush=True)
    result = split_half.extend_pairs(saved, matches, flat, seed=original_config["seed"], n_jobs=n_jobs, log=print)
    _, pairs = split_half.result_tables(result, clouds)
    i, j = result["pairs"].T
    for level, m in enumerate(result["m_values"]):
        mask = pairs.m == m
        margin = result["between_distance"][level] - np.maximum(
            result["self_distance"][level][:, i], result["self_distance"][level][:, j])
        pairs.loc[mask, "margin_median_um"] = np.median(margin, axis=0)
    pairs["is_random"] = np.tile(result["is_random"], len(result["m_values"]))
    pairs["is_targeted"] = np.tile(result["is_targeted"], len(result["m_values"]))
    pairs = split_half.annotate_pairs(pairs, neurons)
    strata = split_half.stratified_summary(pairs)
    anchors, effects = split_half.matched_effects(result, matches)
    coverage, balance = split_half.matching_diagnostics(matches, cohort)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").unlink(missing_ok=True)
    (outdir / "config.json").write_text(json.dumps(options, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(outdir / "distances.npz", **result)
    for name, frame in (("pairs", pairs), ("strata", strata), ("cohort", cohort), ("matches", matches),
                        ("anchors", anchors), ("effects", effects), ("coverage", coverage), ("balance", balance)):
        frame.to_csv(outdir / f"{name}.csv", index=False)
    plotting.plot_split_half_pairs(strata, effects, coverage, outdir / "comparison.png")
    report = {"n_neurons": len(neurons), "n_random_pairs": int(result["is_random"].sum()),
              "n_targeted_pairs": int(result["is_targeted"].sum()),
              "n_reused_targeted_pairs": int((result["is_random"] & result["is_targeted"]).sum()),
              "n_new_pairs": int((~result["is_random"]).sum()), "n_matched_anchors": matches.anchor_id.nunique(),
              "n_triplets": len(matches), "max_partner_reuse": int((cohort.uses_as_same_partner + cohort.uses_as_different_partner).max()),
              "status_counts": {str(k): int(v) for k, v in cohort.match_status.value_counts().items()},
              "effects": json.loads(effects.to_json(orient="records")),
              "interpretation": "Descriptive comparisons, conditional on the observed cohort and match coverage. "
                                "Margin is between minus the larger endpoint self distance in each split, then its split median; "
                                "it is not noise correction or a test. Matched delta averages partner distances within each split, "
                                "then takes split medians per anchor, with equal anchor weight. "
                                "Canonical A/B orientation is preserved; the anchor half can differ between its edges. "
                                "Splits and shared-neuron comparisons are not independent biological replicates."}
    (outdir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved {outdir}")
    return outdir


if __name__ == "__main__":
    main()
