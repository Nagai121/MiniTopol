import warnings

import MDAnalysis as mda
import numpy as np
import pytest


@pytest.fixture
def trajectory(tmp_path):
    universe = mda.Universe.empty(6, n_residues=2, atom_resindex=[0, 0, 0, 1, 1, 1], trajectory=True)
    universe.add_TopologyAttr("names", ["N", "CA", "H", "OW", "HW1", "HW2"])
    universe.add_TopologyAttr("resnames", ["ALA", "SOL"])
    universe.add_TopologyAttr("resids", [1, 2])
    universe.dimensions = [30, 31, 32, 90, 90, 80]
    positions = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 4], [8, 8, 8], [9, 8, 8], [8, 9, 8]], dtype=np.float32)
    universe.atoms.positions = positions
    gro, xtc = tmp_path / "test.gro", tmp_path / "test.xtc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        universe.atoms.write(str(gro))
    with mda.Writer(str(xtc), n_atoms=6, precision=4) as writer:
        for frame in range(9):
            universe.atoms.positions = positions + np.array([frame * 5, 0, 0])
            universe.trajectory.ts.time = frame * 2.5
            writer.write(universe.atoms)
    return gro, xtc, positions
