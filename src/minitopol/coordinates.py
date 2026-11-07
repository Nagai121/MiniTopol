"""Coordinate views requested through Registry; raw A is never modified."""

import numpy as np
from MDAnalysis.lib.distances import minimize_vectors
from MDAnalysis.lib.mdamath import triclinic_box

from .registry import FrameBlock


def _box_dimensions_uncached(box):
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (3, 3) or not np.isfinite(box).all() or np.linalg.det(box) <= 0:
        raise ValueError("Periodic coordinates require a finite, invertible, right-handed box")
    if not np.allclose(np.triu(box, 1), 0, atol=1e-6) or np.any(np.diag(box) <= 0):
        raise ValueError("Only GROMACS lower-triangular boxes are supported")
    if abs(box[1, 0]) > box[0, 0]/2 + 1e-6 or abs(box[2, 0]) > box[0, 0]/2 + 1e-6 or abs(box[2, 1]) > box[1, 1]/2 + 1e-6:
        raise ValueError("Periodic minimum-image distances require a reduced GROMACS box")
    return triclinic_box(*box)


from functools import lru_cache

@lru_cache(maxsize=128)
def _cached_box_dimensions(raw):
    box=np.frombuffer(raw,dtype=np.float64).reshape(3,3)
    value=_box_dimensions_uncached(box)
    value.flags.writeable=False
    return value

def box_dimensions(box):
    # Exact float64 content keys prevent stale reuse after caller mutation.
    # Failed validation is never cached. Return fresh writable dimensions.
    box=np.asarray(box,dtype=np.float64)
    if box.shape!=(3,3):return _box_dimensions_uncached(box)
    return _cached_box_dimensions(box.tobytes(order='C')).copy()


def minimum_vectors(vectors, box):
    dimensions = box_dimensions(box)
    shape = np.shape(vectors)
    return minimize_vectors(np.asarray(vectors, dtype=np.float64).reshape(-1, 3), dimensions).reshape(shape)


def coordinate_blocks(registry, trajectory, *, mode, initial_whole=False,
                      half_box_steps=False, chunk_frames=None):
    if mode not in ("nonperiodic", "continuous", "nojump"):
        raise ValueError("Select nonperiodic, continuous (already unwrapped), or nojump coordinates explicitly")
    if mode == "nojump" and not half_box_steps:
        raise ValueError("NoJump requires the half-box inter-frame motion assumption (half_box_steps=True)")

    forest = None
    if mode == "nojump" and registry.bond_info(trajectory) is not None:
        from .whole_nojump import bond_forest, reconstruct_frame
        table = registry.bonds(trajectory)
        edges = np.column_stack((table["point_i"].to_numpy(), table["point_j"].to_numpy()))
        forest = bond_forest(registry.manifest(trajectory)["atoms"], edges)

    # TPR connectivity reconstructs the initial whole components. When absent,
    # preserve the existing caller-confirmed initial-image contract.
    if mode != "nonperiodic" and not initial_whole and forest is None:
        raise ValueError("The intended initial molecular/complex image must be confirmed with initial_whole=True")

    previous = None
    centers = (np.zeros((len(forest[2]) - 1, 3), dtype=np.float64)
               if forest is not None else None)
    first = True
    for block in registry.blocks(trajectory, chunk_frames=chunk_frames):
        positions = np.asarray(block.positions, dtype=np.float64)
        for index in range(len(positions)):
            if mode != "nonperiodic":
                box_dimensions(block.box[index])
            if mode == "nojump":
                box = np.asarray(block.box[index], dtype=np.float64)
                inverse = np.linalg.inv(box)
                if forest is not None:
                    reconstruct_frame(positions[index], box, inverse,
                                      forest[0], forest[1], forest[2],
                                      centers, first)
                else:
                    reduced = positions[index] @ inverse
                    if previous is not None:
                        reduced -= np.round(reduced - previous)
                    previous = reduced.copy()
                    positions[index] = reduced @ box
                first = False
        yield FrameBlock(block.start, block.stop, positions, block.time, block.box)


def fit_positions(positions, reference, ca):
    moving = np.asarray(positions, dtype=np.float64)
    p, q = moving[ca], np.asarray(reference, dtype=np.float64)[ca]
    if len(ca) < 3:
        raise ValueError("At least three non-collinear, unambiguous C-alpha atoms are required")
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    p, q = p - pc, q - qc
    if np.linalg.matrix_rank(p) < 2 or np.linalg.matrix_rank(q) < 2:
        raise ValueError("C-alpha alignment is undefined for a collinear or degenerate configuration")
    u, _, vt = np.linalg.svd(p.T @ q)
    sign = np.eye(3)
    sign[-1, -1] = np.sign(np.linalg.det(u @ vt))
    return (moving - pc) @ (u @ sign @ vt) + qc
