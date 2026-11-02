"""TPR mass/charge behavior and sparse electrostatic descriptor regression."""
from contextlib import contextmanager
import warnings

import MDAnalysis as mda
import numpy as np
import pytest

from minitopol import Workspace, convert, Registry
from minitopol.descriptors import analyze
from minitopol.descriptors.context import AnalysisContext
from minitopol.descriptors.electrostatics import _centers, _pairs
from minitopol.system_metadata import publish_molecules


@pytest.fixture
def charged_trajectory(tmp_path, monkeypatch):
    u = mda.Universe.empty(
        6, n_residues=3, atom_resindex=[0, 0, 1, 1, 2, 2], trajectory=True)
    u.add_TopologyAttr("names", ["C1", "H1", "C2", "H2", "C3", "H3"])
    u.add_TopologyAttr("resnames", ["ALA", "LYS", "SER"])
    u.add_TopologyAttr("resids", [1, 2, 3])
    u.add_TopologyAttr("elements", ["C", "H", "C", "H", "C", "H"])
    u.dimensions = [40, 40, 40, 90, 90, 90]
    positions = np.asarray([
        [0.2, 1, 1], [0.3, 1, 1],
        [39.8, 1, 1], [39.7, 1, 1],
        [25.0, 1, 1], [25.2, 1, 1]], dtype=np.float32)
    u.atoms.positions = positions
    gro, xtc = tmp_path / "input.gro", tmp_path / "input.xtc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u.atoms.write(str(gro))
    with mda.Writer(str(xtc), n_atoms=6, precision=4) as writer:
        for frame in range(2):
            xyz = positions.copy()
            if frame:
                xyz[2, 0] = 20.2
                xyz[3, 0] = 20.3
            u.atoms.positions = xyz
            u.trajectory.ts.time = frame * 2.0
            writer.write(u.atoms)
    ws = Workspace(tmp_path / "ws")
    tid = convert(ws, gro, xtc, solvent_resnames=[], chunk_frames=1)
    tpr = tmp_path / "source.tpr"
    tpr.write_bytes(b"mock")
    class Atoms:
        charges = np.array([0.8, 0.2, -0.7, -0.3, 0.6000001, -0.6])
        masses = np.array([100., 1., 200., 1., 300., 1.])
        resindices = np.array([0, 0, 1, 1, 2, 2])
        resnames = ["ALA", "ALA", "LYS", "LYS", "SER", "SER"]
        types = ["C", "H", "C", "H", "C", "H"]
        names = ["C1", "H1", "C2", "H2", "C3", "H3"]
        moltypes = ["A", "A", "B", "B", "C", "C"]
        def __len__(self):
            return 6
    class FakeTPR:
        atoms = Atoms()
        trajectory = None
    import minitopol.system_metadata as metadata
    monkeypatch.setattr(metadata.mda, "Universe", lambda path: FakeTPR())
    monkeypatch.setattr(metadata, "_covalent_edges_from_tpr",
                        lambda path, molecule_ids: (6, [], [0, 0, 1, 1, 2, 2]))
    publish_molecules(ws, tid, tpr, list(range(6)))
    return ws, tid


def test_tpr_mass_priority_and_residue_charge_threshold(charged_trajectory):
    ws, tid = charged_trajectory
    ctx = AnalysisContext(ws, tid, coordinate_mode="nonperiodic")
    np.testing.assert_array_equal(ctx.atomic_values("masses"), [100, 200, 300])
    from minitopol.descriptors.electrostatics import residue_charges
    np.testing.assert_allclose(residue_charges(ctx), [1., -1., 0.], atol=1e-12)
    _, atoms = Registry(ws).molecule_data(tid)
    assert atoms.num_rows == 6
    assert atoms["tpr_residue_index"].to_pylist() == [0, 0, 1, 1, 2, 2]


def test_periodic_sparse_pair_is_written_and_outside_cutoff_is_implicit_zero(charged_trajectory):
    ws, tid = charged_trajectory
    result = analyze(ws, tid, descriptors=["electrostatic_score"],
                     coordinate_mode="continuous", initial_whole=True, chunk_frames=1)
    assert result["status"] == "complete", result
    records = ws.query_results(descriptor="electrostatic_score").to_pylist()
    assert len(records) == 1
    record = records[0]
    assert record["frame"] == 0
    assert (record["residue_i"], record["residue_j"]) == (0, 1)
    assert record["value"] == pytest.approx(-2.5, abs=0.02)
    again = analyze(ws, tid, descriptors=["electrostatic_score"],
                    coordinate_mode="continuous", initial_whole=True, chunk_frames=1)
    assert again["status"] == "complete"
    assert again["results"]["electrostatic_score"]["result_id"] == result["results"]["electrostatic_score"]["result_id"]


def test_geometric_center_unwraps_residue_across_box_boundary():
    box = np.diag([40., 40., 40.])
    xyz = np.array([[39.8, 2., 2.], [0.2, 2., 2.],
                    [0.4, 2., 2.]])
    centers = _centers(xyz, [[0, 1], [2]], box)
    assert centers[0, 0] == pytest.approx(40.0, abs=1e-10)
    assert _pairs(centers, box).tolist() == [[0, 1]]


def test_without_tpr_requires_explicit_charge_metadata(tmp_path, trajectory):
    ws = Workspace(tmp_path / "missing")
    tid = convert(ws, trajectory[0], trajectory[1], solvent_resnames=["SOL"])
    with pytest.raises(ValueError, match="requires TPR"):
        analyze(ws, tid, descriptors=["electrostatic_score"], coordinate_mode="nonperiodic")


def test_direct_tpr_topology_uses_full_source_atom_mapping(tmp_path, trajectory, monkeypatch):
    from minitopol import converter
    from minitopol import system_metadata
    from minitopol.descriptors import bond_reconstruction
    fake = tmp_path / "whole.tpr"
    fake.write_bytes(b"mock")
    calls = {}
    original = converter._universe
    monkeypatch.setattr(converter, "_universe",
                        lambda topology, xtc: original(trajectory[0], xtc))
    monkeypatch.setattr(system_metadata, "publish_conditions",
                        lambda *a, **k: None)
    def published(ws, tid, source, mapping):
        calls["source"] = source
        calls["mapping"] = mapping
    monkeypatch.setattr(system_metadata, "publish_molecules", published)
    monkeypatch.setattr(bond_reconstruction, "reconstruct_bonds",
                        lambda *a, **k: None)
    ws = Workspace(tmp_path / "direct")
    tid = convert(ws, fake, trajectory[1], solvent_resnames=["SOL"])
    assert calls["source"] == fake
    assert calls["mapping"] == list(range(6))
    assert Registry(ws).manifest(tid)["options"]["source"]["tpr"]["bytes"] == 4
