"""Static TPR topology bond reconstruction, separate from the ten descriptors."""
from __future__ import annotations

import MDAnalysis as mda
import numpy as np

from ..bond_store import BondStore
from ..registry import Registry
from ..storage import digest, fingerprint


def _read_tpr(path):
    # TPR is the only source of bond connectivity; coordinates are never used.
    return mda.Universe(str(path))


def reconstruct_bonds(workspace, trajectory: str, tpr: str,
                      source_to_tpr: list[int]) -> int:
    """Project TPR bond indices onto the already completed A point order.

    'source_to_tpr' is the checker-validated mapping from A's original
    topology atom indices to the matching TPR atoms. Never infer a mapping here.
    """
    if source_to_tpr is None:
        raise ValueError("A validated source-to-TPR atom mapping is required")
    registry = Registry(workspace)
    atoms = registry.atoms(trajectory).to_pydict()
    source_atoms = atoms["source_atom"]
    if (not source_atoms or max(source_atoms) >= len(source_to_tpr)
        or any(type(index) is not int or index < 0 for index in source_to_tpr)
        or len(set(source_to_tpr)) != len(source_to_tpr)):
        raise ValueError("Invalid source-to-TPR atom mapping")
    universe = _read_tpr(tpr)
    try:
        total = len(universe.atoms)
        if max(source_to_tpr) >= total:
            raise ValueError("TPR mapping points outside its topology")
        to_a = {source_to_tpr[source_atom]: point for point, source_atom in enumerate(source_atoms)}
        bonds = np.asarray(universe.bonds.indices, dtype=np.int64).reshape(-1, 2)
        edges = set()
        for i, j in bonds:
            if int(i) in to_a and int(j) in to_a:
                a, b = sorted((to_a[int(i)], to_a[int(j)]))
                if a == b:
                    raise ValueError("TPR has an invalid self-bond")
                edges.add((a, b))
    finally:
        if getattr(universe, "trajectory", None) is not None:
            universe.trajectory.close()
    return BondStore(workspace).publish(
        trajectory, edges, tpr_sha256=fingerprint(tpr)["sha256"],
        mapping_sha256=digest(source_to_tpr))
