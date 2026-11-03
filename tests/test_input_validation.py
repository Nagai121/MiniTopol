"""Regression tests for input evidence and user-authored molecular roles."""
import warnings

import MDAnalysis as mda
import numpy as np
import pytest

from minitopol import Registry, Workspace, convert
from minitopol.bond_store import BondStore
from minitopol.input_validation import resolve_elements
from minitopol.storage import read_json


def test_pairing_verification_and_conflicting_order(tmp_path, trajectory):
    gro, xtc, _ = trajectory
    ws = Workspace(tmp_path / "verified")
    tid = convert(ws, gro, xtc, solvent_resnames=["SOL"])
    check = Registry(ws).manifest(tid)["input_validation"]
    assert check["pairing"]["status"] == "verified"
    assert check["elements"]["status"] == "unverified"
    details = Registry(ws).atoms(tid).to_pydict()
    assert len(details["element_validation"]) == len(details["element"]) == 2

    u = mda.Universe(str(gro))
    try:
        xyz = u.atoms.positions.copy()
        xyz[[0, 1]] = xyz[[1, 0]]
        u.atoms.positions = xyz
        swapped = tmp_path / "swapped.gro"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u.atoms.write(str(swapped))
        with pytest.raises(ValueError, match="order mismatch"):
            convert(Workspace(tmp_path / "wrong"), swapped, xtc, solvent_resnames=["SOL"])
        u.atoms.positions = xyz + np.array([8, 0, 0], dtype=np.float32)
        moved = tmp_path / "moved.gro"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u.atoms.write(str(moved))
    finally:
        u.trajectory.close()
    ws2 = Workspace(tmp_path / "unverified")
    other = convert(ws2, moved, xtc, solvent_resnames=["SOL"])
    assert Registry(ws2).manifest(other)["input_validation"]["pairing"]["status"] == "unverified"


def test_native_element_disagreement_is_an_error():
    u = mda.Universe.empty(2, n_residues=1, atom_resindex=[0, 0], trajectory=True)
    u.add_TopologyAttr("names", ["CA", "H1"])
    u.add_TopologyAttr("resnames", ["ALA"])
    u.add_TopologyAttr("elements", ["C", "H"])
    elements, _, states, report = resolve_elements(u)
    assert elements.tolist() == ["C", "H"]
    assert states == ["verified", "verified"]
    assert report["status"] == "verified"
    with pytest.raises(ValueError, match="differs from topology"):
        resolve_elements(u, ["Ca", "H"])


@pytest.fixture
def two_residues(tmp_path):
    u = mda.Universe.empty(10, n_residues=2, atom_resindex=[0]*5+[1]*5, trajectory=True)
    u.add_TopologyAttr("names", ["N","CA","CB","C","O"] * 2)
    u.add_TopologyAttr("resnames", ["ALA", "ALA"])
    u.add_TopologyAttr("resids", [1,2])
    u.atoms.positions = np.arange(30, dtype=np.float32).reshape(10, 3)
    u.dimensions = [50,50,50,90,90,90]
    gro, xtc = tmp_path/"two.gro", tmp_path/"two.xtc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u.atoms.write(str(gro))
    with mda.Writer(str(xtc), n_atoms=10) as writer:
        writer.write(u.atoms)
    workspace = Workspace(tmp_path/"two-work")
    tid = convert(workspace, gro, xtc, solvent_resnames=[])
    return workspace, tid


def test_manual_ca_requires_actual_ca_name(two_residues):
    ws, tid = two_residues
    registry = Registry(ws)
    with pytest.raises(ValueError, match="expected CA"):
        registry.annotate(tid, chains=[], alpha_carbons=[2])
    registry.annotate(tid, chains=[], alpha_carbons=[1,6])
    check = registry.annotations(tid)[0]["validation"]["alpha_carbons"]
    assert check["status"] == "verified"


def test_segments_verify_actual_bonds_when_present(two_residues):
    ws, tid = two_residues
    registry = Registry(ws)
    BondStore(ws).publish(tid, [(3, 5)], tpr_sha256="test-graph", mapping_sha256="identity")
    registry.annotate(tid, chains=[{"id": "A", "segments": [[0,1]]}])
    check = registry.annotations(tid)[0]["validation"]["continuous_segments"]
    assert check["status"] == "verified" and check["tested_links"] == 1


def test_segments_reject_missing_peptide_bond(two_residues):
    ws, tid = two_residues
    BondStore(ws).publish(tid, [(2, 7)], tpr_sha256="test-graph", mapping_sha256="identity")
    with pytest.raises(ValueError, match="lack a supporting covalent bond"):
        Registry(ws).annotate(tid, chains=[{"id":"A","segments":[[0,1]]}])
    assert not (ws.trajectory_path(tid) / "annotations.parquet").exists()


def test_no_bond_graph_records_unverified_segments(two_residues):
    ws, tid = two_residues
    registry = Registry(ws)
    registry.annotate(tid, chains=[{"id":"A","segments":[[0,1]]}])
    assert registry.annotations(tid)[0]["validation"]["continuous_segments"]["status"] == "unverified"
