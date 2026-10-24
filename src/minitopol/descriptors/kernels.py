"""Shared geometric kernels; no chemical connectivity is inferred."""

import numpy as np
from MDAnalysis.lib.distances import calc_dihedrals
from scipy.spatial import cKDTree

from ..coordinates import minimum_vectors


def sphere_points(count):
    i = np.arange(count, dtype=np.float64)
    z = 1 - 2 * (i + .5) / count
    r = np.sqrt(1 - z*z)
    angle = i * np.pi * (3 - np.sqrt(5))
    return np.column_stack((r * np.cos(angle), r * np.sin(angle), z))


def sasa_atoms(positions, radii, *, sphere_count=960, probe=1.4):
    # Limit the optional compiled signature to the normal trajectory inputs.
    # The original function remains authoritative for unusual direct calls,
    # including its NumPy return values, warnings and argument errors.
    if (not isinstance(sphere_count, (int, np.integer))
            or isinstance(sphere_count, (bool, np.bool_)) or sphere_count <= 0
            or type(positions) is not np.ndarray or positions.ndim != 2
            or positions.shape[1] != 3 or positions.dtype not in (np.dtype('f4'), np.dtype('f8'))):
        return _sasa_atoms_python(positions, radii, sphere_count=sphere_count, probe=probe)
    inflated = np.asarray(radii) + probe
    if inflated.dtype != np.dtype('f8') or inflated.ndim != 1 or len(inflated) != len(positions):
        return _sasa_atoms_python(positions, radii, sphere_count=sphere_count, probe=probe)
    try:
        from .accelerated import atom_sasa
    except ImportError:
        return _sasa_atoms_python(positions, radii, sphere_count=sphere_count, probe=probe)
    points = sphere_points(sphere_count)
    tree = cKDTree(positions)
    values = np.zeros(len(positions))
    maximum = inflated.max()
    for atom, center in enumerate(positions):
        neighbors = np.asarray(tree.query_ball_point(center, inflated[atom] + maximum), dtype=np.int64)
        count = atom_sasa(center, positions, neighbors, points, inflated, atom)
        values[atom] = count / sphere_count * (4 * np.pi * inflated[atom]**2)
    return values


def _sasa_atoms_python(positions, radii, *, sphere_count=960, probe=1.4):
    inflated = np.asarray(radii) + probe
    points = sphere_points(sphere_count)
    tree = cKDTree(positions)
    values = np.zeros(len(positions))
    maximum = inflated.max()
    for atom, center in enumerate(positions):
        neighbors = tree.query_ball_point(center, inflated[atom] + maximum)
        surface = center + points * inflated[atom]
        exposed = np.ones(sphere_count, dtype=bool)
        for other in neighbors:
            if other == atom:
                continue
            if np.array_equal(center, positions[other]):
                raise ValueError("Coincident heavy atoms make per-atom SASA assignment ambiguous")
            delta = surface - positions[other]
            exposed &= np.einsum("ij,ij->i", delta, delta) >= inflated[other]**2
            if not exposed.any():
                break
        values[atom] = exposed.sum() / sphere_count * (4 * np.pi * inflated[atom]**2)
    return values


def backbone_quads(ctx):
    roles = {}
    for residue, atoms in ctx.res_atoms.items():
        roles[residue] = {}
        for name, element in (("N", "N"), ("CA", "C"), ("C", "C")):
            found = [i for i in atoms if ctx.atoms["name"][i] == name and ctx.atoms["element"][i] == element]
            roles[residue][name] = found[0] if len(found) == 1 else None
    if ctx.annotation["alpha_carbons"] is not None:
        for point in ctx.annotation["alpha_carbons"]:
            roles[ctx.atoms["residue_index"][point]]["CA"] = point
    lookup = {(r["run"], r["sequence_position"]): i for i, r in ctx.res_annotation.items() if r["run"] is not None}
    quads, reasons = {"phi": [], "psi": []}, {"phi": [], "psi": []}
    for residue in ctx.residues:
        annotation = ctx.res_annotation.get(residue, {})
        for angle in ("phi", "psi"):
            reason, quad = None, None
            if annotation.get("run") is None:
                reason = "unconfirmed_chain_continuity"
            else:
                offset = -1 if angle == "phi" else 1
                other = lookup.get((annotation["run"], annotation["sequence_position"] + offset))
                if other is None:
                    reason = "segment_terminal"
                else:
                    current = roles[residue]
                    quad = ([roles[other]["C"], current["N"], current["CA"], current["C"]]
                            if angle == "phi" else [current["N"], current["CA"], current["C"], roles[other]["N"]])
                    if any(i is None for i in quad):
                        quad, reason = None, "missing_or_ambiguous_backbone_atom"
            quads[angle].append(quad)
            reasons[angle].append(reason)
    return quads, reasons


def backbone_angles(positions, box, quads, reasons, *, periodic):
    values, why = {}, {name: list(items) for name, items in reasons.items()}
    for angle in ("phi", "psi"):
        result = np.full(len(quads[angle]), np.nan)
        usable = [i for i, quad in enumerate(quads[angle]) if quad is not None]
        if usable:
            xyz = positions[np.asarray([quads[angle][i] for i in usable])]
            vectors = np.diff(xyz, axis=1)
            if periodic:
                vectors = minimum_vectors(vectors, box)
            rebuilt = np.concatenate([np.zeros((len(usable), 1, 3)), np.cumsum(vectors, axis=1)], axis=1)
            radians = calc_dihedrals(rebuilt[:, 0], rebuilt[:, 1], rebuilt[:, 2], rebuilt[:, 3])
            result[usable] = (np.rad2deg(radians) + 180) % 360 - 180
            for i in usable:
                if not np.isfinite(result[i]):
                    why[angle][i] = "degenerate_dihedral"
        values[angle] = result
    return values, why


def backbone_frames(ctx, writer):
    """Yield uncommitted frames with shared backbone angles and missing reasons."""
    quads, reasons = backbone_quads(ctx)
    for block in ctx.registry.blocks(ctx.trajectory, chunk_frames=ctx.chunk):
        for offset, xyz in enumerate(block.positions):
            frame = block.start + offset
            if writer.has(frame):
                continue
            angles, why = backbone_angles(xyz, block.box[offset], quads, reasons, periodic=ctx.mode != "nonperiodic")
            yield frame, block.time[offset], angles, why


def secondary_labels(phi, psi):
    labels = np.full(len(phi), "other", dtype=object)
    helix = (phi >= -100) & (phi <= -30) & (psi >= -80) & (psi <= -5)
    extended = (phi >= -180) & (phi <= -90) & (((psi >= 90) & (psi < 180)) | ((psi >= -180) & (psi <= -150)))
    labels[helix], labels[extended] = "helix_like", "extended_like"
    labels[~(np.isfinite(phi) & np.isfinite(psi))] = None
    return labels
