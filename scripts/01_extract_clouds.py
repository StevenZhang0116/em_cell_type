"""
Stage A: Extract Per-Neuron Synapse Clouds
==========================================
Writes ``output/<dataset>/clouds_{axon,dendrite}.npz`` (see celltype/clouds.py).

Example::

    python -u scripts/01_extract_clouds.py --dataset microns --lateral-cutoff 250
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow as pa

from celltype.clouds import build_side, load_cell_table, load_synapses, proofread_cells
from celltype.config import DATASETS, SIDES, allocated_cpus, clouds_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="microns")
    parser.add_argument("--lateral-cutoff", type=float, default=250.0,
                        help="max lateral synapse-to-soma distance in um")
    args = parser.parse_args()

    # Arrow otherwise sizes its thread pools from the whole node.
    pa.set_cpu_count(allocated_cpus())
    pa.set_io_thread_count(allocated_cpus())

    start = time.time()
    cells = load_cell_table(args.dataset)
    side_cells = {side: proofread_cells(cells, side) for side in SIDES}
    for side in SIDES:
        print(f"[{side}] proofread cells: {len(side_cells[side])}")

    synapses = load_synapses(args.dataset,
                             side_cells["axon"]["pt_root_id"].to_numpy(),
                             side_cells["dendrite"]["pt_root_id"].to_numpy())
    print(f"Loaded {len(synapses)} synapses ({time.time() - start:.0f}s)")

    for side in SIDES:
        output = build_side(side_cells[side], synapses, side, args.lateral_cutoff)
        counts = np.diff(output["offsets"])
        print(f"[{side}] synapses per neuron: median {np.median(counts):.0f}, "
              f"min {counts.min()}, max {counts.max()}")
        path = clouds_path(args.dataset, side)
        np.savez_compressed(path, **output)
        print(f"Saved {path}")
    print(f"Total time {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
