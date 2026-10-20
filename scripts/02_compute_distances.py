"""
Stage B: Neuron-to-Neuron Wasserstein Distances (one bootstrap replicate)
=========================================================================
Writes ``output/<dataset>/dist_<side>_<method>_<coords>_<depth>/rep<b>.npz`` (see
celltype/distances.py). Run one replicate per Slurm array task.

Example::

    python -u scripts/02_compute_distances.py --dataset microns --side axon \
        --method points --coords xyz --depth-mode relative --rep 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from celltype.clouds import Clouds
from celltype.config import (COORDS, DEPTH_MODES, METHODS, SIDES, allocated_cpus,
                             clouds_path, distance_dir)
from celltype.distances import compute_replicate


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="microns")
    parser.add_argument("--side", choices=SIDES, required=True)
    parser.add_argument("--method", choices=METHODS, default="points")
    parser.add_argument("--coords", choices=COORDS, default="xyz",
                        help="xyz: 3-D soma-centred offsets; cylindrical: (r, depth)")
    parser.add_argument("--depth-mode", choices=DEPTH_MODES, default="relative")
    parser.add_argument("--rep", type=int,
                        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
                        help="bootstrap replicate index (default: Slurm array id)")
    parser.add_argument("--n-points", type=int, default=200,
                        help="synapses resampled per neuron per replicate")
    parser.add_argument("--min-synapses", type=int, default=100,
                        help="neurons with fewer synapses are excluded")
    parser.add_argument("--n-components", type=int, default=30,
                        help="shared GMM basis size (shared_gmm only)")
    parser.add_argument("--no-weights", action="store_true",
                        help="ignore border-compensation weights")
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()

    start = time.time()
    n_jobs = allocated_cpus()
    clouds = Clouds(clouds_path(args.dataset, args.side))
    eligible = np.flatnonzero(clouds.n_synapses >= args.min_synapses)
    print(f"[{args.side}] {eligible.size} / {len(clouds)} neurons have "
          f">= {args.min_synapses} synapses; replicate {args.rep}; {n_jobs} CPUs")

    result = compute_replicate(
        clouds, eligible, method=args.method, coords=args.coords,
        depth_mode=args.depth_mode,
        n_points=args.n_points, use_weights=not args.no_weights,
        n_components=args.n_components, rng=np.random.default_rng([args.seed, args.rep]),
        seed=args.seed + args.rep, n_jobs=n_jobs)

    path = distance_dir(args.dataset, args.side, args.method, args.coords, args.depth_mode) / f"rep{args.rep:03d}.npz"
    np.savez_compressed(path, eligible=eligible, cell_ids=clouds.cell_ids[eligible],
                        n_points=args.n_points, min_synapses=args.min_synapses,
                        use_weights=not args.no_weights, **result)
    upper = result["distance"][np.triu_indices(eligible.size, 1)]
    print(f"median self-distance {np.median(result['self_distance']):.2f} um; "
          f"median pairwise distance {np.median(upper):.2f} um")
    print(f"Saved {path} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
