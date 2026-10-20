"""
Synthetic Clouds with Known Shapes
==================================
Builds clouds in the same CSR format as ``clouds.build_side`` with three
known arbor shapes (compact, laterally wide, deep-descending) and varying
synapse counts. The shape name is stored as every dataset label, so the full
pipeline run with ``--dataset synthetic`` should recover the three groups
(AMI with ``cell_type`` close to 1 at k = 3).
"""

from __future__ import annotations

import numpy as np

from .config import LABEL_COLUMNS


SHAPES = {
    # name: (lateral scale um, depth mean um, depth sd um)
    "compact": (40.0, 0.0, 40.0),
    "wide": (150.0, 0.0, 40.0),
    "descending": (40.0, 200.0, 60.0),
}


def synthetic_side(n_per_shape, rng):
    dx, dy, dz, weight, counts, labels, soma_y = [], [], [], [], [], [], []
    for name, (lateral, depth_mean, depth_sd) in SHAPES.items():
        for _ in range(n_per_shape):
            n = int(rng.integers(120, 1500))
            xz = rng.normal(0.0, lateral, size=(n, 2))
            dx.append(xz[:, 0])
            dz.append(xz[:, 1])
            dy.append(rng.normal(depth_mean, depth_sd, size=n))
            weight.append(rng.uniform(1.0, 1.5, size=n))
            counts.append(n)
            labels.append(name)
            soma_y.append(rng.uniform(100.0, 900.0))
    n_cells = len(counts)
    output = {
        "cell_ids": np.arange(1, n_cells + 1, dtype=np.int64),
        "offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        "dx": np.concatenate(dx).astype(np.float32),
        "dy": np.concatenate(dy).astype(np.float32),
        "dz": np.concatenate(dz).astype(np.float32),
        "weight": np.concatenate(weight).astype(np.float32),
        "soma_x": np.zeros(n_cells, dtype=np.float32),
        "soma_y": np.array(soma_y, dtype=np.float32),
        "soma_z": np.zeros(n_cells, dtype=np.float32),
        "lateral_cutoff": np.float32(np.inf),
    }
    for column in LABEL_COLUMNS:
        output[f"label_{column}"] = np.array(labels)
    return output

