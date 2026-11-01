"""Compare saved GMMs using 1, 3 and 5 pairs of covariate-matched label partners.

Run inside a Slurm allocation, after fit_neuron_gmms.py. Only missing distances
of the selected triplets are computed; no mixtures are refitted. Report both
all matched anchors and the fixed cohort with five pairs, with equal anchor weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from celltype import gmm_matches, plotting
from celltype.config import DATASETS, SIDES, clouds_path, file_fingerprint, gmm_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="microns")
    parser.add_argument("--side", choices=SIDES, default="dendrite")
    parser.add_argument("--depth-caliper", type=float, default=25.0, help="maximum depth gap in um")
    parser.add_argument("--count-ratio", type=float, default=1.25, help="maximum ratio for both observed and fitted counts")
    parser.add_argument("--border-caliper", type=float, default=50.0, help="maximum border-distance gap in um")
    parser.add_argument("--ess-caliper", type=float, default=0.1, help="maximum gap in weight ESS / observed count")
    args = parser.parse_args(argv)
    root = gmm_dir(args.dataset, args.side)
    paths = {"neurons": root / "neurons.csv", "models": root / "models.json",
             "clouds": clouds_path(args.dataset, args.side), "between": root / "between.csv"}
    for key in ("neurons", "models", "clouds"):
        if not paths[key].exists():
            parser.error(f"missing input: {paths[key]}")
    settings = {"depth_um": args.depth_caliper, "count_ratio": args.count_ratio,
                "border_um": args.border_caliper, "ess_fraction": args.ess_caliper}
    print("Reading covariates and Stage-A weights (no XYZ arrays)", flush=True)
    table = gmm_matches.load_covariates(paths["neurons"], paths["clouds"])
    partners, cohort = gmm_matches.match_triplets(table, max_partners=max(gmm_matches.PARTNER_COUNTS), **settings)
    print(f"Matched {partners.anchor_id.nunique()}/{len(cohort)} anchors in {len(partners)} triplets; "
          "filling selected distances", flush=True)
    partners, distances, fit_options = gmm_matches.attach_distances(
        partners, cohort, paths["models"], paths["between"])
    matches = partners.loc[partners.partner_rank == 1].copy()
    coverage, effects, balance, report = gmm_matches.reports(matches, cohort)
    anchors, robustness, robustness_balance = gmm_matches.partner_robustness(partners, cohort)
    config = {"dataset": args.dataset, "side": args.side, "calipers": settings,
              "label": "cell_type", "within": "classification_system", "distance": "final_model_MW2",
              "selection": "greedy minimum total squared caliper-scaled gap over all three edges; "
                           "ties by IDs; no partner reuse within an anchor",
              "partner_counts": gmm_matches.PARTNER_COUNTS,
              "aggregation": "mean distance difference within anchor, then equal weight per anchor",
              "populations": {"all_matched": "up to requested pairs, using all available selected pairs",
                              "common_5": "same anchors with five selected pairs at every requested count"},
              "fit_options": fit_options,
              "inputs": {key: file_fingerprint(path) for key, path in paths.items() if path.exists()}}
    outdir = root / "matched_comparison"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").unlink(missing_ok=True)
    (outdir / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    for name, frame in (("covariates", cohort), ("matches", matches), ("distances", distances),
                        ("coverage", coverage), ("effects", effects), ("balance", balance),
                        ("partner_matches", partners), ("anchor_effects", anchors),
                        ("robustness", robustness), ("robustness_balance", robustness_balance)):
        frame.to_csv(outdir / f"{name}.csv", index=False)
    plotting.plot_gmm_matches(matches, coverage, balance, outdir / "comparison.png")
    plotting.plot_gmm_partner_robustness(robustness, coverage, outdir / "robustness.png")
    report["distance_sources"] = {str(k): int(v) for k, v in distances.source.value_counts().items()}
    report["by_cell_type"] = json.loads(effects.to_json(orient="records"))
    report["partner_robustness"] = json.loads(robustness.loc[robustness.grouping == "overall"].to_json(orient="records"))
    (outdir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved {outdir}")
    return outdir


if __name__ == "__main__":
    main()
