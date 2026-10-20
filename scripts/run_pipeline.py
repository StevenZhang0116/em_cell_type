"""
Run the Whole Pipeline
======================
Stage A (extract clouds), then for every side x coords:
Stage B (bootstrap distance replicates) and Stage C (clustering and checks).
See celltype/pipeline.py for the stages.

Finished outputs are skipped, so a job that hits its time limit can simply be
resubmitted; pass --overwrite to recompute everything.

Example::

    python -u scripts/run_pipeline.py --dataset microns
    python -u scripts/run_pipeline.py --sides axon --coords xyz cylindrical
    python -u scripts/run_pipeline.py --n-neurons 50 --n-reps 2   # quick test
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from celltype import pipeline
from celltype.clouds import Clouds
from celltype.config import COORDS, DATASETS, SIDES, cluster_dir, clouds_path, distance_dir


STAGES = ("extract", "distances", "cluster")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="microns")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--sides", nargs="+", choices=SIDES, default=list(SIDES))
    parser.add_argument("--coords", nargs="+", choices=COORDS, default=["xyz"],
                        help="xyz: 3-D soma-centred offsets; cylindrical: (r, depth)")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute outputs that already exist")
    # Stage A
    parser.add_argument("--lateral-cutoff", type=float, default=250.0,
                        help="max lateral synapse-to-soma distance in um")
    # Stage B
    parser.add_argument("--n-reps", type=int, default=10, help="bootstrap replicates")
    parser.add_argument("--n-points", type=int, default=200,
                        help="synapses resampled per neuron per replicate")
    parser.add_argument("--min-synapses", type=int, default=100,
                        help="neurons with fewer synapses are excluded")
    parser.add_argument("--n-neurons", type=int, default=None,
                        help="use only this many randomly chosen neurons per side, for a quick "
                             "test; outputs go to *_n<K> directories (default: all neurons)")
    parser.add_argument("--no-weights", action="store_true",
                        help="ignore border-compensation weights")
    parser.add_argument("--seed", type=int, default=20261019)
    # Stage C
    parser.add_argument("--linkage", choices=("average", "complete"), default="average")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--k", type=int, default=None,
                        help="k for the detailed figures (default: most stable k >= 3)")
    args = parser.parse_args()
    start = time.time()

    if "extract" in args.stages:
        if not args.overwrite and all(clouds_path(args.dataset, s).exists() for s in SIDES):
            print("== Stage A: reusing existing clouds")
        else:
            print("== Stage A: extracting clouds")
            pipeline.extract_clouds(args.dataset, args.lateral_cutoff)

    for side, coords in itertools.product(args.sides, args.coords):
        name = f"{side} | {coords}"
        new_reps = False
        if "distances" in args.stages:
            clouds = Clouds(clouds_path(args.dataset, side))
            for rep in range(args.n_reps):
                path = distance_dir(args.dataset, side, coords, args.n_neurons) / f"rep{rep:03d}.npz"
                if path.exists() and not args.overwrite:
                    continue
                print(f"== Stage B: {name} | replicate {rep}")
                pipeline.compute_distances(
                    clouds, args.dataset, side, coords, rep,
                    n_points=args.n_points, min_synapses=args.min_synapses,
                    use_weights=not args.no_weights, seed=args.seed,
                    n_neurons=args.n_neurons)
                new_reps = True

        if "cluster" in args.stages:
            summary = cluster_dir(args.dataset, side, coords, args.n_neurons) / "summary.json"
            if summary.exists() and not (new_reps or args.overwrite):
                print(f"== Stage C: {name} already clustered")
                continue
            print(f"== Stage C: {name}")
            pipeline.cluster_neurons(
                args.dataset, side, coords, linkage=args.linkage,
                k_values=list(range(args.k_min, args.k_max + 1)), k=args.k,
                n_neurons=args.n_neurons,
                run_info={"n_reps": args.n_reps, "n_points": args.n_points,
                          "min_synapses": args.min_synapses, "n_neurons": args.n_neurons})
        print(f"   elapsed {(time.time() - start) / 3600:.2f} h")

    print(f"All done in {(time.time() - start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
