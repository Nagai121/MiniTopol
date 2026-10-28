"""Three-category hydrogen-bond integration, preservation and recovery."""
import warnings

import MDAnalysis as mda
import numpy as np
import pytest

from minitopol import Workspace, convert
from minitopol import hydrogen_bonds as hb
from minitopol.registry import Registry
from minitopol.storage import fingerprint


def _input(tmp_path):
    u = mda.Universe.empty(
        9, n_residues=3,
        atom_resindex=[0, 0, 0, 1, 1, 1, 2, 2, 2],
        trajectory=True)
    u.add_TopologyAttr("names", ["N", "H", "O", "N", "H", "O", "OW", "HW1", "HW2"])
    u.add_TopologyAttr("resnames", ["ALA", "LIG", "SOL"])
    u.add_TopologyAttr("resids", [1, 2, 3])
    u.dimensions = [30, 30, 30, 90, 90, 90]
    xyz = np.array([
        [0, 0, 0], [1, 0, 0], [2.6, 0, 0],
        [-2, 0, 0], [-1, 0, 0], [2.9, 0, 0],
        [2.8, 0, 0], [3.8, 0, 0], [2.8, 1, 0],
    ], dtype=np.float32)
    u.atoms.positions = xyz
    gro, xtc = tmp_path / "input.gro", tmp_path / "input.xtc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u.atoms.write(str(gro))
    with mda.Writer(str(xtc), n_atoms=9, precision=4) as writer:
        for frame in range(3):
            u.atoms.positions = xyz
            u.trajectory.ts.time = float(frame)
            writer.write(u.atoms)
    return gro, xtc


def _fake_tpr(monkeypatch, path):
    path.write_bytes(b"checked-test-tpr")
    original = mda.Universe
    class TPR:
        def __init__(self):
            self.atoms = type("Atoms", (), {
                "charges": np.array([-.3, .3, -.8, -.3, .3, -.8, -.8, .4, .4]),
                "__len__": lambda self: 9})()
    def create(filename, *args, **kwargs):
        if str(filename).endswith(".tpr"):
            return TPR()
        return original(filename, *args, **kwargs)
    monkeypatch.setattr(mda, "Universe", create)
    monkeypatch.setattr(hb, "_covalent_edges_from_tpr",
                        lambda tpr, **kwargs: (
                            9, [(0, 1), (3, 4), (6, 7), (6, 8)],
                            [0, 0, 0, 1, 1, 1, 2, 2, 2]))
    return path


def _run(ws, gro, xtc, tpr, **kwargs):
    info = {}
    tid = convert(ws, gro, xtc, solvent_resnames=["SOL"], chunk_frames=1,
                  hydrogen_bonds={"tpr": str(tpr),
                                  "source_to_tpr": list(range(9)),
                                  "periodic": False},
                  hbond_report=info, **kwargs)
    return tid, info


def test_three_categories_one_pass_immutable_and_bounded_reads(tmp_path, monkeypatch):
    gro, xtc = _input(tmp_path)
    tpr = _fake_tpr(monkeypatch, tmp_path / "test.tpr")
    ws = Workspace(tmp_path / "ws")
    tid, report = _run(ws, gro, xtc, tpr)
    assert report["status"] == "complete", report
    assert Registry(ws).manifest(tid)["atoms"] == 4
    result = Registry(ws).hydrogen_bond_info(tid)
    assert result["frames"] == 3
    assert result["rows"] == {category: 3 for category in hb.CATEGORIES}
    for category in hb.CATEGORIES:
        values = hb.hydrogen_bond_events(ws, tid, category, start=1, stop=2, limit=10)
        assert len(values) == 1
        assert values[0]["frame"] == 1
        assert values[0]["angle_degree"] >= 150
    data = ws.trajectory_path(tid) / "hydrogen_bonds"
    saved = {p: (fingerprint(p)["sha256"], p.stat().st_mtime_ns)
             for p in data.rglob("*.parquet")}
    again, report = _run(ws, gro, xtc, tpr)
    assert again == tid and report["status"] == "complete"
    assert saved == {p: (fingerprint(p)["sha256"], p.stat().st_mtime_ns)
                     for p in data.rglob("*.parquet")}
    with ws.catalog() as con:
        assert con.execute("SELECT COUNT(*) FROM hydrogen_bond_sets").fetchone()[0] == 1
    with pytest.raises(ValueError, match="frame"):
        hb.hydrogen_bond_events(ws, tid, "intramolecular", start=-1)


def test_hydrogen_bonds_resume_without_rewriting_committed_parts(tmp_path, monkeypatch):
    gro, xtc = _input(tmp_path)
    tpr = _fake_tpr(monkeypatch, tmp_path / "test.tpr")
    ws = Workspace(tmp_path / "ws")
    original = hb.HydrogenBondStore.append
    def interrupted(self, start, stop, frames):
        if start == 1:
            raise OSError("simulated H-bond interruption")
        return original(self, start, stop, frames)
    monkeypatch.setattr(hb.HydrogenBondStore, "append", interrupted)
    tid, report = _run(ws, gro, xtc, tpr)
    assert report["status"] == "failed"
    assert Registry(ws).manifest(tid)["status"] == "complete"
    part = ws.trajectory_path(tid) / "hydrogen_bonds" / "intramolecular" / "part-00000000.parquet"
    prior = (fingerprint(part)["sha256"], part.stat().st_mtime_ns)
    monkeypatch.setattr(hb.HydrogenBondStore, "append", original)
    _, report = _run(ws, gro, xtc, tpr)
    assert report["status"] == "complete"
    assert prior == (fingerprint(part)["sha256"], part.stat().st_mtime_ns)


def test_hydrogen_bond_corruption_is_detected(tmp_path, monkeypatch):
    gro, xtc = _input(tmp_path)
    tpr = _fake_tpr(monkeypatch, tmp_path / "test.tpr")
    ws = Workspace(tmp_path / "ws")
    tid, report = _run(ws, gro, xtc, tpr)
    assert report["status"] == "complete"
    path = ws.trajectory_path(tid) / "hydrogen_bonds" / "intermolecular" / "part-00000000.parquet"
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(IOError, match="corrupt"):
        hb.hydrogen_bond_info(ws, tid)


def test_tpr_settle_is_used_only_when_opted_in():
    from MDAnalysis.topology.tpr.obj import Ilist
    from minitopol.descriptors.bond_reconstruction import _edges_in_ilists
    ilists = [Ilist(4, ("SETTLE", "Settle", 3), [0, 0, 1, 2]),
              Ilist(3, ("HARMONIC", "Not a bond", 2), [0, 3, 4])]
    assert _edges_in_ilists(ilists) == []
    assert _edges_in_ilists(ilists, include_settle=True) == [(0, 1), (0, 2)]


def test_bad_mapping_does_not_invalidate_a(tmp_path, monkeypatch):
    gro, xtc = _input(tmp_path)
    tpr = _fake_tpr(monkeypatch, tmp_path / "test.tpr")
    ws = Workspace(tmp_path / "ws")
    status = {}
    tid = convert(ws, gro, xtc, solvent_resnames=["SOL"],
                  hydrogen_bonds={"tpr": str(tpr),
                                  "source_to_tpr": [0] * 9,
                                  "periodic": False}, hbond_report=status)
    assert status["status"] == "failed"
    assert Registry(ws).manifest(tid)["status"] == "complete"
    assert Registry(ws).hydrogen_bond_info(tid) is None
