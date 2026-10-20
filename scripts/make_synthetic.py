"""
Write Synthetic Clouds
======================
Writes ``output/synthetic/clouds_{axon,dendrite}.npz`` with three known shapes
(see celltype/synthetic.py) for an end-to-end check of Stages B and C.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from celltype.config import SIDES, clouds_path
from celltype.synthetic import synthetic_side


def main():
    rng = np.random.default_rng(0)
    for side in SIDES:
        path = clouds_path("synthetic", side)
        np.savez_compressed(path, **synthetic_side(n_per_shape=20, rng=rng))
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
