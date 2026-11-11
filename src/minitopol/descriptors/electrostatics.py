"""Sparse residue electrostatic feature in units of e^2/angstrom.

Not a physical electrostatic energy. Omitted pairs have zero *defined feature*
value, not necessarily zero physical interaction.
"""
from itertools import product

import numpy as np
import pyarrow as pa
from scipy.spatial import cKDTree

from ..coordinates import minimum_vectors, box_dimensions


CUTOFF = 15.0
ZERO_CHARGE = 1e-5
MAX_PART_ROWS = 16384
SCHEMA = pa.schema([
    ("frame", pa.int64()), ("time_ps", pa.float64()),
    ("residue_i", pa.int64()), ("residue_j", pa.int64()),
    ("value", pa.float64()),
])


def residue_charges(ctx):
    """Sum all TPR atoms (including H and atoms absent from the XTC)."""
    try:
        _, table = ctx.registry.molecule_data(ctx.trajectory)
    except (FileNotFoundError, OSError) as error:
        raise ValueError("Electrostatic score requires validated TPR atom properties") from error
    needed = {"tpr_residue_index", "partial_charge", "a_point"}
    if not needed <= set(table.column_names):
        raise ValueError("TPR properties lack full-atom residue indices or charges")
    data = table.select(sorted(needed)).to_pydict()
    sums = {}
    point_to_residue = {}
    for rid, charge, point in zip(data["tpr_residue_index"], data["partial_charge"], data["a_point"]):
        if rid is None or charge is None or not np.isfinite(charge):
            raise ValueError("Incomplete TPR residue charge information")
        sums[rid] = sums.get(rid, 0.0) + float(charge)
        if point is not None:
            if point in point_to_residue:
                raise ValueError("An A point maps to multiple TPR atoms")
            point_to_residue[point] = rid
    if set(point_to_residue) != set(range(ctx.m)):
        raise ValueError("TPR/A atom mapping incomplete")
    charges, seen = [], {}
    for residue in ctx.residues:
        members = {point_to_residue[p] for p in ctx.res_atoms[residue]}
        if len(members) != 1:
            raise ValueError("One A residue maps to multiple TPR residues")
        tpr_rid = members.pop()
        if tpr_rid in seen and seen[tpr_rid] != residue:
            raise ValueError("Multiple A residues map to the same TPR residue")
        seen[tpr_rid] = residue
        charge = sums[tpr_rid]
        charges.append(0.0 if abs(charge) < ZERO_CHARGE else charge)
    return np.asarray(charges, dtype=np.float64)


def _center_plan(residues):
    lengths = np.array([len(points) for points in residues], dtype=np.intp)
    starts = np.r_[0, np.cumsum(lengths)]
    points = np.asarray([point for group in residues for point in group], dtype=np.intp)
    origins = np.repeat(starts[:-1], lengths)
    wrapped = np.repeat(lengths > 1, lengths)
    groups = []
    for length in np.unique(lengths):
        indices = np.flatnonzero(lengths == length)
        atom_rows = starts[indices, None] + np.arange(length)
        groups.append((indices, atom_rows))
    return points, groups, origins[wrapped], wrapped


def _centers(xyz, residues, box, *, plan=None):
    if plan is None:
        plan = _center_plan(residues)
    points, groups, origins, wrapped = plan
    atoms = np.asarray(xyz[points], dtype=np.float64)
    if box is not None and len(origins):
        # The same per-vector minimum-image kernel, called once per frame.
        base = atoms[origins]
        atoms[wrapped] = base + minimum_vectors(atoms[wrapped] - base, box)
    centers = np.empty((len(residues), 3), dtype=np.float64)
    # Keep each residue's atom order and reduce only its atom axis.
    for indices, atom_rows in groups:
        centers[indices] = atoms[atom_rows].mean(axis=1)
    return centers


def _pairs(centers, box):
    if len(centers) < 2:
        return np.empty((0, 2), dtype=np.int64)
    if box is None:
        return cKDTree(centers).query_pairs(CUTOFF, output_type="ndarray")
    matrix = np.asarray(box, dtype=np.float64)
    centers = (centers @ np.linalg.inv(matrix) % 1.0) @ matrix
    if np.allclose(matrix, np.diag(np.diag(matrix)), atol=1e-6):
        return cKDTree(np.mod(centers, np.diag(matrix)), boxsize=np.diag(matrix)).query_pairs(
            CUTOFF, output_type="ndarray")
    # Triclinic: collect candidate pairs from neighboring periodic images.
    # Validate the actual minimum-image distance in calculate().
    tree, pairs = cKDTree(centers), set()
    for lattice in product((-1, 0, 1), repeat=3):
        shift = np.asarray(lattice, dtype=np.float64) @ matrix
        for start in range(0, len(centers), 512):
            neighbors = tree.query_ball_point(centers[start:start + 512] + shift, CUTOFF)
            for i, group in enumerate(neighbors, start):
                pairs.update((min(i, j), max(i, j)) for j in group if j != i)
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def _output_columns():
    return [np.empty(MAX_PART_ROWS, dtype=field.type.to_pandas_dtype()) for field in SCHEMA]


def _output_table(columns, size):
    return pa.Table.from_arrays([pa.array(column[:size]) for column in columns], schema=SCHEMA)


def calculate(ctx, writer):
    charges = residue_charges(ctx)
    active = np.flatnonzero(charges != 0.0)
    residue_points = [ctx.res_atoms[ctx.residues[int(i)]] for i in active]
    center_plan = _center_plan(residue_points)
    residue_ids = np.asarray(ctx.residues, dtype=np.int64)[active]
    charges = charges[active]
    for block in ctx.registry.blocks(ctx.trajectory, chunk_frames=ctx.chunk):
        done = f"frames-{block.start}-{block.stop}-done"
        if writer.has(done):
            continue
        columns, size, part = _output_columns(), 0, 0
        for offset, xyz in enumerate(block.positions):
            box = block.box[offset] if ctx.mode != "nonperiodic" else None
            if box is not None:
                box_dimensions(box)
            centers = _centers(xyz, residue_points, box, plan=center_plan)
            pairs = _pairs(centers, box)
            if len(pairs):
                vectors = centers[pairs[:, 0]] - centers[pairs[:, 1]]
                if box is not None:
                    vectors = minimum_vectors(vectors, box)
                distances = np.linalg.norm(vectors, axis=1)
                selected = distances <= CUTOFF
                if np.any(selected & (distances == 0)):
                    raise ValueError("Electrostatic score undefined for coincident residue centers")
                i, j = pairs[selected].T
                scores = charges[i] * charges[j] / distances[selected]
                if not np.isfinite(scores).all():
                    raise ValueError("Nonfinite electrostatic score")
                nonzero = scores != 0.0
                scores = scores[nonzero]
                left, right = residue_ids[i[nonzero]], residue_ids[j[nonzero]]
                time_ps = block.time[offset]
                consumed = 0
                while consumed < len(scores):
                    count = min(MAX_PART_ROWS - size, len(scores) - consumed)
                    target = slice(size, size + count)
                    source_rows = slice(consumed, consumed + count)
                    columns[0][target] = block.start + offset
                    columns[1][target] = time_ps
                    columns[2][target] = left[source_rows]
                    columns[3][target] = right[source_rows]
                    columns[4][target] = scores[source_rows]
                    consumed += count
                    size += count
                    if size == MAX_PART_ROWS:
                        writer.append(f"frames-{block.start}-{block.stop}-part-{part}",
                                      _output_table(columns, size))
                        columns, size, part = _output_columns(), 0, part + 1
        writer.append(done, _output_table(columns, size))
