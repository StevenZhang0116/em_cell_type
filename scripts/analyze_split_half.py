"""
Measure Split-Half Variability at Matched Synapse Counts
=======================================================
Read Stage-A clouds; uniformly draw two disjoint m-synapse halves per neuron.
Compare within-neuron W2 with m-vs-m between-neuron W2 on fixed random pairs.
The same cohort (eligible at the largest m) and pairs are used at every m.
See docs/split_half.md for interpretation, outputs and computational cost.

Example::

    python -u scripts/analyze_split_half.py --sides axon --n-neurons 200 --points-per-half 100 200 --n-reps 100 --between-pairs 1000
    python -u scripts/analyze_split_half.py --sides dendrite --n-neurons 200 --align --min-border-distance 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from celltype import pipeline, plotting, split_half
from celltype.clouds import Clouds
from celltype.config import (DATASETS, SIDES, allocated_cpus, clouds_path,
                             split_half_dir, split_half_name)


def _positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def _source_info(path):
    """Content fingerprint prevents stale reuse when Stage A was regenerated in place."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="microns")
    parser.add_argument("--sides", nargs="+", choices=SIDES, default=list(SIDES))
    parser.add_argument("--points-per-half", nargs="+", type=_positive, default=[100, 200],
                        help="m values; every selected neuron must have >= 2 * max(m) synapses")
    parser.add_argument("--n-reps", type=_positive, default=100)
    parser.add_argument("--between-pairs", type=_nonnegative, default=2000,
                        help="fixed random unordered pairs (0 for self distances only)")
    parser.add_argument("--n-neurons", type=_positive, default=None)
    parser.add_argument("--min-synapses", type=_positive, default=100)
    parser.add_argument("--min-border-distance", type=float, default=0.0)
    parser.add_argument("--align", action="store_true",
                        help="both halves inherit the canonical rotation of the complete cloud")
    parser.add_argument("--no-weights", action="store_true")
    parser.add_argument("--seed", type=_nonnegative, default=20261019)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite(args.min_border_distance) or args.min_border_distance < 0:
        parser.error("--min-border-distance must be finite and nonnegative")
    sizes = split_half.half_sizes(args.points_per_half)
    n_jobs = allocated_cpus()
    if n_jobs < 1:
        parser.error("SLURM_CPUS_PER_TASK must be positive")

    for side in dict.fromkeys(args.sides):
        start = time.time()
        path = clouds_path(args.dataset, side)
        if not path.exists():
            parser.error(f"{path} not found; run scripts/run_pipeline.py --dataset {args.dataset} --stages extract first")
        clouds = Clouds(path, align=args.align)
        options = {"schema_version": 1, "dataset": args.dataset, "side": side,
                   "points_per_half": sizes, "n_reps": args.n_reps, "n_pairs": args.between_pairs,
                   "n_neurons": args.n_neurons, "min_synapses": args.min_synapses,
                   "min_border_um": args.min_border_distance, "align": args.align,
                   "use_weights": not args.no_weights, "seed": args.seed, "source": _source_info(path)}
        outdir = split_half_dir(args.dataset, split_half_name(options), side)
        outputs = ("distances.npz", "neurons.csv", "pairs.csv", "config.json", "summary.json",
                   "distributions.png", "sample_size.png", "covariates.png")
        if not args.overwrite and all((outdir / name).exists() for name in outputs):
            print(f"[{side}] reusing completed analysis: {outdir}")
            continue
        # The summary is the completion marker; remove it before any partial rerun.
        (outdir / "summary.json").unlink(missing_ok=True)
        eligible = pipeline.eligible_neurons(clouds, args.min_synapses, args.min_border_distance)
        required = max(args.min_synapses, 2 * max(sizes))
        supported = pipeline.eligible_neurons(clouds, required, args.min_border_distance)
        indices = pipeline.select_neurons(clouds, required, args.n_neurons, args.seed, args.min_border_distance)
        if not len(indices):
            parser.error(f"[{side}] no neurons pass the filters and >= {required} synapses; lower m or relax filters")
        counts = {"n_input": len(clouds), "n_below_min_synapses": int(np.sum(clouds.n_synapses < args.min_synapses)),
                  "n_excluded_by_border": int(np.sum(clouds.n_synapses >= args.min_synapses) - len(eligible)),
                  "n_insufficient_for_largest_m": len(eligible) - len(supported),
                  "n_eligible_at_all_m": len(supported), "n_not_selected": len(supported) - len(indices)}
        print(f"[{side}] {len(indices)} neurons, m={sizes}, {args.n_reps} splits, "
              f"up to {args.between_pairs} pairs; {n_jobs} CPUs\nOutput: {outdir}")
        with open(outdir / "config.json", "w") as handle:
            json.dump(options, handle, indent=2, allow_nan=False)
        result = split_half.analyze(clouds, indices, sizes, n_reps=args.n_reps, n_pairs=args.between_pairs,
                                    use_weights=not args.no_weights, seed=args.seed, n_jobs=n_jobs, log=print)
        neurons, pairs = split_half.result_tables(result, clouds)
        np.savez_compressed(outdir / "distances.npz", **result)
        neurons.to_csv(outdir / "neurons.csv", index=False)
        pairs.to_csv(outdir / "pairs.csv", index=False)
        plotting.plot_split_half_distributions(neurons, pairs, outdir / "distributions.png")
        plotting.plot_split_half_curves(neurons, pairs, outdir / "sample_size.png")
        plotting.plot_split_half_covariates(neurons, outdir / "covariates.png")
        report = {**split_half.summary(result), "options": options, "selection": counts,
                  "elapsed_seconds": time.time() - start}
        with open(outdir / "summary.json", "w") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        print(f"[{side}] finished in {time.time() - start:.1f}s: {outdir}")


if __name__ == "__main__":
    main()
