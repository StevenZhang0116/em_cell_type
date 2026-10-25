"""
Fit a Gaussian Mixture to Every Selected Neuron, with Fitting Stability
========================================================================
An analysis beside the pipeline: the weighted GMM of ../mixture
(``GMMWeighted``, imported by celltype/neuron_gmm.py) is fitted to the
soma-centred synapse cloud of each selected neuron, with the number of
components chosen by held-out BIC as in ../mixture's cell-type model.

Because a cell-type model pools thousands of synapses while a single neuron
has a few hundred, the stability of each neuron's fit is measured too: the
models fitted at the chosen k during k selection (one per repetition, each on
its own training subsample and EM start) are compared pairwise with the
mixture Wasserstein distance of ../mixture (``component_ot_distance``), and
the distance between the final models of random pairs of neurons gives the
between-neuron scale to read them against. With --reruns N the whole
selection is additionally rerun N times per neuron and the final models
compared the same way (N times the fitting time).

Neurons are selected exactly like a pipeline trial (same --n-neurons,
--min-synapses, --min-border-distance, --seed, --align), so the outputs join
the trial's cluster tables on pt_root_id. This is test code, so the outputs
are flat files in gmm_output/ with the side as prefix, and every run
overwrites the previous one (the options and command are in the summary):

  <side>_neurons.csv    per neuron: covariates, labels, chosen k, log-likelihood, BIC, spread,
                        within-neuron W2 (median, max, relative to the spread), rerun agreement
  <side>_models.json    every final mixture (weights, means, covariances in (dx, dz, dy) order)
                        with its k-selection and stability records
  <side>_stability.json the repeat models at the chosen k (and rerun models) of every neuron
  <side>_between.csv    W2 between the final models of random pairs of neurons
  <side>_summary.json   options, chosen-k counts, medians, stability summary, failures, time
  <side>_selection.png  chosen-k histogram and the held-out BIC curve
  <side>_stability.png  within- vs between-neuron W2, W2 vs synapse count, W2 vs k

Example::

    python -u scripts/fit_neuron_gmms.py --sides axon --n-neurons 1000 --k-max 4 --k-step 1 --min-border-distance 100
    python -u scripts/fit_neuron_gmms.py --sides dendrite --n-neurons 200 --method diagonal_constraint
    python -u scripts/fit_neuron_gmms.py --sides axon --n-neurons 50 --k-min 2 --k-max 4 --k-step 1 --repeats 6
    python -u scripts/fit_neuron_gmms.py --sides axon --n-neurons 100 --reruns 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from celltype import neuron_gmm, pipeline, plotting
from celltype.clouds import Clouds
from celltype.config import DATASETS, SIDES, allocated_cpus, clouds_path, gmm_dir


def _median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def stability_report(table, between):
    """Summary numbers of the within-neuron distances against the between-neuron reference."""
    fitted = table[table["k"].notna()]
    within = fitted["w2_within_median"].to_numpy(dtype=float)
    within = within[np.isfinite(within)]
    report = {"n_neurons_with_stability": int(within.size),
              "w2_within_median_um": _median(within),
              "w2_within_rel_median": _median(fitted["w2_within_rel"]),
              "w2_to_final_mean_median_um": _median(fitted["w2_to_final_mean"]),
              "n_between_pairs": int(len(between)),
              "w2_between_median_um": _median(between["w2"]) if len(between) else None}
    if within.size and len(between):
        p10 = float(np.percentile(between["w2"], 10))
        report["within_between_ratio"] = report["w2_within_median_um"] / report["w2_between_median_um"]
        report["w2_between_p10_um"] = p10
        report["frac_neurons_within_below_between_p10"] = float(np.mean(within < p10))
    rerun = fitted["w2_rerun_median"].to_numpy(dtype=float)
    if np.isfinite(rerun).any():
        report["w2_rerun_median_um"] = _median(rerun)
        report["rerun_k_agreement_mean"] = float(np.nanmean(fitted["rerun_k_agreement"]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="microns")
    parser.add_argument("--sides", nargs="+", choices=SIDES, default=list(SIDES))
    # Neuron selection (same meaning as in run_pipeline.py)
    parser.add_argument("--n-neurons", type=int, default=None,
                        help="fit only this many randomly chosen neurons per side (default: all)")
    parser.add_argument("--min-synapses", type=int, default=100)
    parser.add_argument("--min-border-distance", type=float, default=0.0,
                        help="exclude neurons whose soma is closer than this (um) to the volume border")
    parser.add_argument("--align", action="store_true",
                        help="canonically rotate every cloud first (as --align in run_pipeline.py)")
    parser.add_argument("--no-weights", action="store_true", help="ignore border-compensation weights")
    parser.add_argument("--seed", type=int, default=20261019)
    # Fit options (see celltype/neuron_gmm.py)
    parser.add_argument("--method", choices=neuron_gmm.METHODS, default="diagonal",
                        help="GMMWeighted covariance method; diagonal is what ../mixture uses for its "
                             "relative-coordinate cell-type fits, diagonal_constraint its cylindrical variant")
    parser.add_argument("--k-min", type=int, default=1,
                        help="smallest number of components tried (../mixture tries 2 to 4 for clouds "
                             "under 10k synapses; the default 1-16 by 3 tries 1, 4, 7, 10, 13, 16, which "
                             "suits dendrites; axons with a few hundred synapses support only k up to "
                             "2-4 at 20 synapses per parameter, so use --k-max 4 --k-step 1 for them)")
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--k-step", type=int, default=3,
                        help="try every k-step-th value from --k-min to --k-max")
    parser.add_argument("--repeats", type=int, default=20,
                        help="train/test repetitions per neuron for choosing k (at least "
                             f"{neuron_gmm.MIN_REPEATS}; ../mixture uses 20 or 50); the models fitted "
                             "at the chosen k in these repetitions are the stability sample")
    parser.add_argument("--max-synapses", type=int, default=10000,
                        help="synapses per neuron per repetition (subsampled without replacement)")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-synapses-per-parameter", type=int, default=20,
                        help="a k is tried only if the cloud has this many synapses per free parameter")
    parser.add_argument("--starts", type=int, default=1,
                        help="EM starts per fit, keeping the one with the highest training "
                             "log-likelihood (1 as in ../mixture, whose single start can stop in a "
                             "poor optimum; see neuron_gmm.fit_once)")
    # Stability options
    parser.add_argument("--reruns", type=int, default=0,
                        help="independent reruns of the whole k selection per neuron, whose final "
                             "models are compared like the repeat fits (multiplies the fitting time)")
    parser.add_argument("--between-pairs", type=int, default=2000,
                        help="random pairs of neurons whose final models give the between-neuron "
                             "distance reference")
    args = parser.parse_args()
    start = time.time()

    outdir = gmm_dir()
    k_values = list(range(args.k_min, args.k_max + 1, max(args.k_step, 1)))
    options = dict(method=args.method, n_repeats=args.repeats, max_synapses=args.max_synapses,
                   test_fraction=args.test_fraction,
                   min_synapses_per_parameter=args.min_synapses_per_parameter, n_starts=args.starts,
                   n_reruns=args.reruns)
    n_jobs = allocated_cpus()
    print(f"== Per-neuron GMMs: {outdir}; {n_jobs} CPUs")

    for side in args.sides:
        path = clouds_path(args.dataset, side)
        if not path.exists():
            sys.exit(f"{path} not found: extract the clouds first with "
                     f"'scripts/run_pipeline.py --dataset {args.dataset} --stages extract' "
                     "(Stage A, about 96 GB of memory for microns)")
        clouds = Clouds(path, align=args.align)
        indices = pipeline.select_neurons(clouds, args.min_synapses, args.n_neurons, args.seed,
                                          args.min_border_distance)
        print(f"== {side}: fitting {indices.size} neurons, k in {k_values}, method {args.method}, "
              f"{args.repeats} repeats, {args.starts} EM starts per fit, {args.reruns} reruns")
        fits, errors = neuron_gmm.fit_neurons(clouds, indices, k_values, use_weights=not args.no_weights,
                                              n_jobs=n_jobs, seed=args.seed, log=print, **options)
        cell_ids = clouds.cell_ids[indices]
        table = neuron_gmm.fits_table(fits, clouds, indices)
        table.to_csv(outdir / f"{side}_neurons.csv", index=False)
        neuron_gmm.save_fits(outdir / f"{side}_models.json", fits, cell_ids,
                             {**options, "k_values": k_values, "use_weights": not args.no_weights,
                              "align": args.align, "seed": args.seed})
        neuron_gmm.save_stability(outdir / f"{side}_stability.json", fits, cell_ids)
        plotting.plot_gmm_selection(fits, outdir / f"{side}_selection.png")

        between = neuron_gmm.between_distances(fits, args.between_pairs, seed=args.seed)
        between.insert(0, "pt_root_id_b", cell_ids[between.pop("j").to_numpy(dtype=int)])
        between.insert(0, "pt_root_id_a", cell_ids[between.pop("i").to_numpy(dtype=int)])
        between.to_csv(outdir / f"{side}_between.csv", index=False)
        plotting.plot_gmm_stability(fits, between["w2"], outdir / f"{side}_stability.png")

        fitted = table[table["k"].notna()]
        summary = {
            "side": side, "command": " ".join(sys.argv), "dataset": args.dataset,
            "options": options, "k_values": k_values, "seed": args.seed,
            "selection": {"n_neurons": args.n_neurons, "min_synapses": args.min_synapses,
                          "min_border_distance": args.min_border_distance, "align": args.align,
                          "use_weights": not args.no_weights},
            "n_neurons": int(indices.size), "n_fitted": int(len(fitted)), "n_failed": len(errors),
            "chosen_k_counts": {str(int(k)): int(n) for k, n in fitted["k"].value_counts().sort_index().items()},
            "n_underpowered": int(sum(fit is not None and fit.selection["underpowered"] for fit in fits)),
            "median_log_likelihood": _median(fitted["log_likelihood"]),
            "median_lateral_sd_um": _median(fitted["lateral_sd_um"]),
            "median_depth_sd_um": _median(fitted["depth_sd_um"]),
            "stability": stability_report(table, between),
            "failures": errors,
            "elapsed_s": round(time.time() - start),
        }
        with open(outdir / f"{side}_summary.json", "w") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps({key: summary[key] for key in
                          ("n_neurons", "n_fitted", "n_failed", "chosen_k_counts", "n_underpowered",
                           "median_log_likelihood", "median_lateral_sd_um", "median_depth_sd_um",
                           "stability")}, indent=2))
        print(f"   {side} done in {(time.time() - start) / 60:.1f} min; outputs in {outdir}")

    print(f"All done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
