"""
Run the Whole Pipeline
======================
Stage A (extract clouds), then for every side: Stage B (bootstrap distance
replicates), Stage C (clustering and checks) and Stage D (assigning the
neurons that were not clustered). See celltype/pipeline.py for the stages.

Everything except the clouds is written to output/<dataset>/trials/<trial>/,
where <trial> encodes the Stage B and C parameters (e.g.
n10_rep2_pt200_min100_weighted_seed20261019_average_k2-3_mcs5), so different
parameter settings never overwrite each other. Within a trial, finished
outputs are skipped, so a job that hits its time limit can simply be
resubmitted; pass --overwrite to recompute everything.

Example::

    python -u scripts/run_pipeline.py --dataset microns
    python -u scripts/run_pipeline.py --sides axon
    python -u scripts/run_pipeline.py --n-neurons 50 --n-reps 2   # quick test
    python -u scripts/run_pipeline.py --sides dendrite --n-neurons 3000 --query rest
    python -u scripts/run_pipeline.py --n-neurons 1000 --method shared --stages distances cluster
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from celltype import pipeline
from celltype.clouds import Clouds
from celltype.config import (DATASETS, SIDES, assign_dir, cluster_dir, clouds_path, distance_dir,
                             trial_dir, trial_name)


STAGES = ("extract", "distances", "cluster", "assign")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="microns")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--sides", nargs="+", choices=SIDES, default=list(SIDES))
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute outputs that already exist")
    # Stage A
    parser.add_argument("--lateral-cutoff", type=float, default=250.0,
                        help="max lateral synapse-to-soma distance in um")
    parser.add_argument("--all-cells", action="store_true",
                        help="also extract the dendrite clouds of every cell in the table "
                             "(clouds_dendrite_all.npz), needed for --query all")
    # Stage B
    parser.add_argument("--method", choices=("points", "shared"), default="points",
                        help="points: W2 between resampled synapse clouds; shared: MW2 between "
                             "mixing weights over one shared Gaussian mixture per side")
    parser.add_argument("--n-reps", type=int, default=10, help="bootstrap replicates")
    parser.add_argument("--n-points", type=int, default=200,
                        help="[points] synapses resampled per neuron per replicate")
    parser.add_argument("--shared-k", type=int, default=30,
                        help="[shared] components of the shared mixture")
    parser.add_argument("--shared-alpha", type=float, default=10.0,
                        help="[shared] Dirichlet strength in synapses that shrinks a neuron's "
                             "weights towards the global proportions")
    parser.add_argument("--shared-fit-points", type=int, default=300_000,
                        help="[shared] pooled synapses used to fit the shared mixture")
    parser.add_argument("--min-synapses", type=int, default=100,
                        help="neurons with fewer synapses are excluded")
    parser.add_argument("--n-neurons", type=int, default=None,
                        help="cluster only this many randomly chosen neurons per side "
                             "(default: all); the rest can be assigned in Stage D")
    parser.add_argument("--no-weights", action="store_true",
                        help="ignore border-compensation weights")
    parser.add_argument("--seed", type=int, default=20261019)
    # Stage C
    parser.add_argument("--linkage", choices=("average", "complete"), default="average")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="clusters smaller than this are labelled outliers (cluster 0) "
                             "and do not count towards k")
    parser.add_argument("--k", type=int, default=None,
                        help="k for the detailed figures and for Stage D (default: most stable k >= 3)")
    # Stage D
    parser.add_argument("--query", choices=("rest", "all", "none"), default="rest",
                        help="neurons to assign: the proofread neurons not clustered (rest), "
                             "every cell's dendrite (all; dendrite side only, needs --all-cells) "
                             "or nobody (none: only cross-validate and fit the templates)")
    parser.add_argument("--assign-reps", type=int, default=3,
                        help="resamples averaged for each query-to-reference distance")
    parser.add_argument("--knn", type=int, default=10, help="nearest references that vote")
    parser.add_argument("--template-k-max", type=int, default=8,
                        help="largest number of GMM components tried per cluster template")
    parser.add_argument("--template-points", type=int, default=100_000,
                        help="synapses per cluster used to fit its template")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="folds of the reference set for calibration and validation")
    parser.add_argument("--reject-quantile", type=float, default=0.95,
                        help="queries beyond this quantile of held-out reference distances "
                             "(or below 1 - it in template score) get label 0")
    args = parser.parse_args()
    start = time.time()

    if "extract" in args.stages:
        if not args.overwrite and all(clouds_path(args.dataset, s).exists() for s in SIDES):
            print("== Stage A: reusing existing clouds")
        else:
            print("== Stage A: extracting clouds")
            pipeline.extract_clouds(args.dataset, args.lateral_cutoff)
        if args.all_cells:
            if not args.overwrite and clouds_path(args.dataset, "dendrite", "all").exists():
                print("== Stage A: reusing existing all-cell dendrite clouds")
            else:
                print("== Stage A: extracting the dendrite clouds of every cell")
                pipeline.extract_all_dendrites(args.dataset, args.lateral_cutoff)

    params = {"method": args.method, "n_neurons": args.n_neurons, "n_reps": args.n_reps,
              "n_points": args.n_points, "shared_k": args.shared_k, "shared_alpha": args.shared_alpha,
              "min_synapses": args.min_synapses, "use_weights": not args.no_weights,
              "seed": args.seed, "linkage": args.linkage, "k_min": args.k_min,
              "k_max": args.k_max, "min_cluster_size": args.min_cluster_size, "k": args.k}
    trial = trial_name(**params)
    with open(trial_dir(args.dataset, trial) / "config.json", "w") as handle:
        json.dump(params, handle, indent=2)
    print(f"== Trial: output/{args.dataset}/trials/{trial}")

    for side in args.sides:
        fresh = False   # an upstream output was recomputed in this run
        if "distances" in args.stages:
            clouds = Clouds(clouds_path(args.dataset, side))
            shared_model = None
            if args.method == "shared":
                print(f"== Stage B: {side} | shared mixture")
                shared_model = pipeline.fit_shared_model(
                    clouds, args.dataset, trial, side, min_synapses=args.min_synapses,
                    n_neurons=args.n_neurons, seed=args.seed, use_weights=not args.no_weights,
                    n_components=args.shared_k, fit_points=args.shared_fit_points)
            for rep in range(args.n_reps):
                path = distance_dir(args.dataset, trial, side) / f"rep{rep:03d}.npz"
                if path.exists() and not args.overwrite:
                    continue
                print(f"== Stage B: {side} | replicate {rep}")
                pipeline.compute_distances(
                    clouds, args.dataset, trial, side, rep,
                    n_points=args.n_points, min_synapses=args.min_synapses,
                    use_weights=not args.no_weights, seed=args.seed,
                    n_neurons=args.n_neurons, method=args.method, shared_model=shared_model,
                    shared_alpha=args.shared_alpha)
                fresh = True

        if "cluster" in args.stages:
            summary = cluster_dir(args.dataset, trial, side) / "summary.json"
            if summary.exists() and not (fresh or args.overwrite):
                print(f"== Stage C: {side} already clustered")
            else:
                print(f"== Stage C: {side}")
                pipeline.cluster_neurons(
                    args.dataset, trial, side, linkage=args.linkage,
                    k_values=list(range(args.k_min, args.k_max + 1)),
                    min_cluster_size=args.min_cluster_size, k=args.k, run_info=params)
                fresh = True

        if "assign" in args.stages:
            summary = assign_dir(args.dataset, trial, side, args.query) / "summary.json"
            if summary.exists() and not (fresh or args.overwrite):
                print(f"== Stage D: {side} | {args.query} already assigned")
            else:
                print(f"== Stage D: {side} | query {args.query}")
                pipeline.assign_neurons(
                    args.dataset, trial, side, query=args.query,
                    n_points=args.n_points, min_synapses=args.min_synapses,
                    use_weights=not args.no_weights, seed=args.seed, n_reps=args.assign_reps,
                    n_neighbours=args.knn, template_k_max=args.template_k_max,
                    template_points=args.template_points, cv_folds=args.cv_folds,
                    reject_quantile=args.reject_quantile, k=args.k, run_info=params)
        print(f"   elapsed {(time.time() - start) / 3600:.2f} h")

    print(f"All done in {(time.time() - start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
