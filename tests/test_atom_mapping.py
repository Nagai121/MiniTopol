"""Identity validation of complete source-to-TPR mappings."""
import warnings

import MDAnalysis as mda
import numpy as np
import pytest

from minitopol.atom_mapping import validate_source_mapping, validate_workspace_mapping
from minitopol import Workspace, convert


def _repeated_residue_topology(tmp_path):
    """ALA + two identical SOL residues, each with distinct occurrence."""
    u = mda.Universe.empty(
        9, n_residues=3, atom_resindex=[0,0,0,1,1,1,2,2,2], trajectory=True)
    u.add_TopologyAttr("names", ["N","CA","H","OW","HW1","HW2","OW","HW1","HW2"])
    u.add_TopologyAttr("resnames", ["ALA","SOL","SOL"])
    u.add_TopologyAttr("resids", [1,2,3])
    u.atoms.positions = np.arange(27,dtype=np.float32).reshape(9,3)
    path = tmp_path / "full.gro"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u.atoms.write(str(path))
    return path


def test_rejects_wrong_residue_occurrence_with_valid_numeric_mapping(tmp_path, trajectory):
    source = trajectory[0]
    tpr = _repeated_residue_topology(tmp_path)
    assert validate_source_mapping(source, tpr, [0,1,2,3,4,5]) == [0,1,2,3,4,5]
    # Both have correct length, increasing indices, no duplicates and in-range.
    # But the second source residue is falsely paired with the second SOL.
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_source_mapping(source, tpr, [0,1,2,6,7,8])


def test_rejects_wrong_atom_name_and_missing_source_atoms(tmp_path, trajectory):
    source = trajectory[0]
    tpr = _repeated_residue_topology(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_source_mapping(source, tpr, [0,2,3,4,5,6])
    with pytest.raises(ValueError, match="cover every source atom"):
        validate_source_mapping(source, tpr, [0,1,2,3,4])


def test_workspace_validation_uses_converted_source_identity(tmp_path, trajectory):
    ws = Workspace(tmp_path / "ws")
    source, xtc, _ = trajectory
    tid = convert(ws, source, xtc, solvent_resnames=["SOL"])
    target = _repeated_residue_topology(tmp_path)
    assert validate_workspace_mapping(ws, tid, target, [0,1,2,3,4,5]) == [0,1,2,3,4,5]
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_workspace_mapping(ws, tid, target, [0,1,2,6,7,8])
    # Modification of the original source after conversion must fail closed.
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="differs from the converted"):
        validate_workspace_mapping(ws, tid, target, [0,1,2,3,4,5])
