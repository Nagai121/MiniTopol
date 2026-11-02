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


def _centers(xyz, residues, box):
    centers = np.empty((len(residues), 3), dtype=np.float64)
    for i, points in enumerate(residues):
        atoms = np.asarray(xyz[points], dtype=np.float64)
        if box is not None and len(atoms) > 1:
            # Correct the positions within each residue before averaging.
            atoms = atoms[0] + minimum_vectors(atoms - atoms[0], box)
        centers[i] = atoms.mean(axis=0)
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


def calculate(ctx, writer):
    charges = residue_charges(ctx)
    active = np.flatnonzero(charges != 0.0)
    residue_points = [ctx.res_atoms[ctx.residues[int(i)]] for i in active]
    residue_ids = np.asarray(ctx.residues, dtype=np.int64)[active]
    charges = charges[active]
    for block in ctx.registry.blocks(ctx.trajectory, chunk_frames=ctx.chunk):
        done = f"frames-{block.start}-{block.stop}-done"
        if writer.has(done):
            continue
        rows, part = [], 0
        for offset, xyz in enumerate(block.positions):
            box = block.box[offset] if ctx.mode != "nonperiodic" else None
            if box is not None:
                box_dimensions(box)
            centers = _centers(xyz, residue_points, box)
            pairs = _pairs(centers, box)
            if len(pairs):
                vectors = centers[pairs[:, 0]] - centers[pairs[:, 1]]
                if box is not None:
                    vectors = minimum_vectors(vectors, box)
                distances = np.linalg.norm(vectors, axis=1)
                selected = distances <= CUTOFF
                if np.any(selected & (distances == 0)):
                    raise ValueError("Electrostatic score undefined for coincident residue centers")
                for (i, j), distance in zip(pairs[selected], distances[selected]):
                    score = float(charges[i] * charges[j] / distance)
                    if not np.isfinite(score):
                        raise ValueError("Nonfinite electrostatic score")
                    if score == 0.0:
                        continue
                    rows.append((block.start + offset, float(block.time[offset]),
                                 int(residue_ids[i]), int(residue_ids[j]), score))
                    if len(rows) == MAX_PART_ROWS:
                        writer.append(f"frames-{block.start}-{block.stop}-part-{part}",
                                      pa.Table.from_pylist(
                                          [dict(zip(SCHEMA.names, row)) for row in rows], schema=SCHEMA))
                        rows, part = [], part + 1
        writer.append(done, pa.Table.from_pylist(
            [dict(zip(SCHEMA.names, row)) for row in rows], schema=SCHEMA))
