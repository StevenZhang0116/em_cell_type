"""
Per-Neuron Synapse Clouds
=========================
Extraction: for every proofread neuron, collect ALL of its own synapses --
outgoing for the axon side (``status_axon == "extended"``), incoming for the
dendrite side (``full_dendrite``) -- regardless of the partner's cell type.
This deliberately differs from the "local" GMM fit in mixture_all.py, which
keeps only synapses onto/from one partner type and would leak the dataset
labels into the anatomy features. The same functions also extract the
dendrite clouds of every cell in the table (``--all-cells``), which Stage D
can assign to the clusters.

Coordinates follow mixture_all.py: ``pt_position_*_trafo`` in nm, converted
to micrometres; ``pt_position_y_trafo`` is cortical depth and x/z span the
cortical plane. Each synapse is stored as its soma-centred offset
``(dx, dy, dz)``, which ``Clouds.neuron_points`` returns. With
``Clouds(path, align=True)`` every cloud is first rotated about the soma's
vertical axis into a canonical orientation (principal lateral axis of the
weighted second moment about the soma on +x, lateral centroid on the positive
side), so that the arbitrary in-plane direction of an arbor no longer enters
any comparison while its anisotropy does. Border-compensation weights are
``weights_pre`` / ``weights_post`` (MICrONS only; ones for V1DD); ``dist_border``
(soma distance to the volume border) is stored in nm as in the cell table.

Storage: one ``.npz`` per side in CSR form -- flat ``dx``, ``dy``, ``dz``, ``weight``
arrays plus ``offsets`` so that neuron ``i`` owns ``offsets[i]:offsets[i+1]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from .config import DATA_DIR, DATASETS, LABEL_COLUMNS, NM_TO_UM


_XYZ = ["pt_position_x_trafo", "pt_position_y_trafo", "pt_position_z_trafo"]
# side -> (column holding the neuron's own root id, border-weight column)
_SIDE_COLUMNS = {"axon": ("pre_pt_root_id", "weights_pre"),
                 "dendrite": ("post_pt_root_id", "weights_post")}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def load_cell_table(dataset):
    """Cells with unique root ids and soma positions in micrometres."""
    cells = pd.read_feather(DATA_DIR / DATASETS[dataset]["cell_table"])
    if dataset == "v1dd":
        cells = cells.rename(columns={"dendrite_complete": "full_dendrite"})
    if dataset == "microns":
        # Same restriction as mixture_all.py (V1 only).
        cells = cells[cells["region"] == "V1"]

    invalid = (cells["pt_root_id"] == 0) | cells["pt_root_id"].duplicated(keep=False)
    if invalid.any():
        print(f"Dropping {int(invalid.sum())} cells with root id 0 or duplicated root ids")
    cells = cells[~invalid].reset_index(drop=True)

    for column in _XYZ:
        cells[column] = cells[column] * NM_TO_UM
    return cells


def proofread_cells(cells, side):
    if side == "axon":
        mask = cells["status_axon"] == "extended"
    else:
        mask = cells["full_dendrite"] == True  # noqa: E712 (column may be object dtype)
    return cells[mask].reset_index(drop=True)


def load_synapses(dataset, axon_ids, dendrite_ids):
    """Synapses whose presynaptic cell is in ``axon_ids`` or whose postsynaptic
    cell is in ``dendrite_ids`` (either may be empty), with autapses removed."""
    has_weights = DATASETS[dataset]["has_weights"]
    columns = ["pre_pt_root_id", "post_pt_root_id", *_XYZ]
    if has_weights:
        columns += ["weights_pre", "weights_post"]

    table = ds.dataset(DATA_DIR / DATASETS[dataset]["synapse_table"], format="ipc")
    filters = []
    if len(axon_ids):
        filters.append(pc.field("pre_pt_root_id").isin(axon_ids))
    if len(dendrite_ids):
        filters.append(pc.field("post_pt_root_id").isin(dendrite_ids))
    if not filters:
        raise ValueError("no cell ids to select synapses for")
    row_filter = filters[0] if len(filters) == 1 else filters[0] | filters[1]
    synapses = table.to_table(columns=columns, filter=row_filter).to_pandas()
    synapses = synapses[synapses["pre_pt_root_id"] != synapses["post_pt_root_id"]]

    for column in _XYZ:
        synapses[column] = synapses[column] * NM_TO_UM
    if not has_weights:
        synapses["weights_pre"] = 1.0
        synapses["weights_post"] = 1.0
    return synapses


def build_side(cells, synapses, side, lateral_cutoff):
    """Soma-centred synapse offsets for one side, in CSR order of ``cells``.

    Synapses farther than ``lateral_cutoff`` um from the soma axis are
    dropped, matching ``distance_threshold`` in the Stage-1 GMM fit.
    """
    id_column, weight_column = _SIDE_COLUMNS[side]
    own = synapses[synapses[id_column].isin(cells["pt_root_id"])]

    soma_xyz = cells.set_index("pt_root_id")[_XYZ].loc[own[id_column]].to_numpy()
    offset = own[_XYZ].to_numpy() - soma_xyz
    dx, dy, dz = offset[:, 0], offset[:, 1], offset[:, 2]
    r = np.hypot(dx, dz)
    weight = own[weight_column].to_numpy(dtype=float)

    finite = np.isfinite(r) & np.isfinite(dy) & np.isfinite(weight) & (weight > 0)
    if not finite.all():
        print(f"[{side}] dropping {int((~finite).sum())} synapses with non-finite "
              "positions or non-positive weights")
    keep = finite & (r < lateral_cutoff)
    print(f"[{side}] {int(keep.sum())} / {len(own)} synapses within "
          f"{lateral_cutoff} um of the soma axis")

    # CSR layout: synapses grouped by the row of their neuron in ``cells``.
    cell_ids = cells["pt_root_id"].to_numpy()
    row_of_id = pd.Series(np.arange(len(cells)), index=cell_ids)
    owner_row = row_of_id.loc[own[id_column].to_numpy()[keep]].to_numpy()
    order = np.argsort(owner_row, kind="stable")
    counts = np.bincount(owner_row, minlength=len(cells))

    output = {
        "cell_ids": cell_ids,
        "offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        "dx": dx[keep][order].astype(np.float32),
        "dy": dy[keep][order].astype(np.float32),
        "dz": dz[keep][order].astype(np.float32),
        "weight": weight[keep][order].astype(np.float32),
        "soma_x": cells["pt_position_x_trafo"].to_numpy(dtype=np.float32),
        "soma_y": cells["pt_position_y_trafo"].to_numpy(dtype=np.float32),
        "soma_z": cells["pt_position_z_trafo"].to_numpy(dtype=np.float32),
        "lateral_cutoff": np.float32(lateral_cutoff),
    }
    for column in LABEL_COLUMNS:
        # Fixed-width unicode, not object dtype, so the file loads without pickle.
        output[f"label_{column}"] = cells[column].astype(str).to_numpy(dtype=str)
    if "dist_border" in cells.columns:
        output["dist_border"] = cells["dist_border"].to_numpy(dtype=np.float32)
    return output


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def canonical_rotation(dx, dz, weight, offsets):
    """Per-neuron (cos, sin) of the in-plane rotation into the canonical frame.

    The frame puts the principal axis of the weighted lateral second moment
    about the soma on +x and the lateral centroid on the positive x side. A
    rotationally symmetric cloud gets an arbitrary angle, which then does not
    matter. Flipping by 180 degrees is a proper rotation, so mirror-image
    arbors stay distinct.
    """
    dx, dz, weight = (np.asarray(a, dtype=float) for a in (dx, dz, weight))
    starts = offsets[:-1]
    nonempty = np.diff(offsets) > 0

    def per_neuron(values):
        total = np.zeros(starts.size)
        total[nonempty] = np.add.reduceat(values, starts[nonempty])
        return total

    mxx, mzz, mxz = per_neuron(weight * dx * dx), per_neuron(weight * dz * dz), per_neuron(weight * dx * dz)
    theta = 0.5 * np.arctan2(2 * mxz, mxx - mzz)
    cos, sin = np.cos(theta), np.sin(theta)
    centroid_x = cos * per_neuron(weight * dx) + sin * per_neuron(weight * dz)
    flip = centroid_x < 0
    cos[flip], sin[flip] = -cos[flip], -sin[flip]
    return cos, sin


def align_points(points, weight, offsets):
    """Canonical rotation of each flat cloud, estimated only from the supplied points.

    This shares Clouds' orientation rule; validation calls it independently on
    each half. The input arrays are never modified.
    """
    points = np.asarray(points, dtype=float)
    cos, sin = canonical_rotation(points[:, 0], points[:, 2], weight, offsets)
    cos, sin = np.repeat(cos, np.diff(offsets)), np.repeat(sin, np.diff(offsets))
    rotated = points.copy()
    rotated[:, 0] = cos * points[:, 0] + sin * points[:, 2]
    rotated[:, 2] = -sin * points[:, 0] + cos * points[:, 2]
    return rotated


class Clouds:
    """Read-only view of one side's saved clouds; ``align=True`` canonically rotates every cloud."""

    def __init__(self, path, align=False):
        self.path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            self.meta = {key: data[key] for key in data.files}
        self.cell_ids = self.meta["cell_ids"]
        self.offsets = self.meta["offsets"]
        self.n_synapses = np.diff(self.offsets)
        self.align = bool(align)
        if self.align:
            self._cos, self._sin = canonical_rotation(self.meta["dx"], self.meta["dz"],
                                                      self.meta["weight"], self.offsets)

    def __len__(self):
        return self.cell_ids.size

    def _span(self, index):
        return slice(self.offsets[index], self.offsets[index + 1])

    def neuron_points(self, index):
        """Soma-relative synapse positions (n, 3) ``(dx, dy, dz)`` of one neuron, in um."""
        span = self._span(index)
        dx, dy, dz = self.meta["dx"][span], self.meta["dy"][span], self.meta["dz"][span]
        if self.align:
            cos, sin = self._cos[index], self._sin[index]
            dx, dz = cos * dx + sin * dz, -sin * dx + cos * dz
        return np.column_stack([dx, dy, dz])

    def neuron_weights(self, index):
        return self.meta["weight"][self._span(index)]
