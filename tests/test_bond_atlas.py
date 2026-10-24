"""ATLAS checker, independent bond reconstruction and persistence regressions."""
import shutil

import MDAnalysis as mda
import pytest

from minitopol import Registry, Workspace, convert
from minitopol.atlas.checker import check_atlas, unique_subsequence
from minitopol.atlas.importer import import_atlas
from minitopol.bond_store import BondStore
from minitopol.descriptors import bond_reconstruction


def _source(workspace, replica="R1", kind="analysis"):
    directory = workspace.trajectory_path("atlas-1k5n_A-" + kind) / "source"
    directory.mkdir(parents=True, exist_ok=True)
    return directory, f"1k5n_A_{replica}"


def _sample_tpr_universe():
    u = mda.Universe.empty(6, n_residues=2,
                           atom_resindex=[0, 0, 0, 1, 1, 1], trajectory=True)
    u.add_TopologyAttr("names", ["N", "CA", "H", "OW", "HW1", "HW2"])
    u.add_TopologyAttr("resnames", ["ALA", "SOL"])
    u.add_TopologyAttr("resids", [1, 2])
    u.add_TopologyAttr("bonds", [(0, 1), (1, 2), (3, 4), (3, 5)])
    return u


def _fake_tpr(monkeypatch):
    import minitopol.atlas.checker as checker
    original = checker._labels

    def labels(path):
        if path.suffix == ".tpr":
            u = _sample_tpr_universe()
            try:
                return tuple((str(r).upper(), str(n).upper())
                             for r, n in zip(u.atoms.resnames, u.atoms.names))
            finally:
                u.trajectory.close()
        return original(path)

    monkeypatch.setattr(checker, "_labels", labels)
    monkeypatch.setattr(bond_reconstruction, "_read_tpr",
                        lambda path: _sample_tpr_universe())


def test_unique_subsequence_requires_unambiguous_order():
    a, b, h = ("ALA", "CA"), ("ALA", "CB"), ("ALA", "H")
    assert unique_subsequence((a, b), (a, h, b)) == [0, 2]
    with pytest.raises(ValueError, match="Ambiguous"):
        unique_subsequence((a,), (a, a))
    with pytest.raises(ValueError, match="not an ordered subset"):
        unique_subsequence((b,), (a,))


def test_atlas_check_and_import_separate_replicas(tmp_path, trajectory, monkeypatch):
    workspace = Workspace(tmp_path / "ws")
    _fake_tpr(monkeypatch)
    directory, _ = _source(workspace)
    for replica in ("R1", "R2"):
        stem = f"1k5n_A_{replica}"
        shutil.copyfile(trajectory[0], directory / (stem + ".gro"))
        shutil.copyfile(trajectory[1], directory / (stem + ".xtc"))
        (directory / (stem + ".tpr")).write_bytes(b"mock-tpr-" + replica.encode())
    plans = check_atlas(workspace, "1k5n_A")
    assert [p["replica"] for p in plans] == ["R1", "R2"]
    assert plans[0]["source_to_tpr"] == list(range(6))
    result = import_atlas(workspace, "1k5n_A")
    assert result["status"] == "complete"
    assert [p["bonds"] for p in result["replicas"]] == ["complete", "complete"]
    assert len(workspace.list_trajectories()) == 2
    for replica in ("R1", "R2"):
        tid = f"atlas-1k5n_A-analysis-{replica}"
        assert Registry(workspace).bonds(tid).to_pydict() == {"point_i": [0], "point_j": [1]}
        assert Registry(workspace).manifest(tid)["atoms"] == 2
        assert not (workspace.trajectory_path(tid) / "source").exists()
    assert import_atlas(workspace, "1k5n_A") == result
    with workspace.catalog() as con:
        assert con.execute("SELECT count(*) FROM bond_graphs").fetchone()[0] == 2


def test_atlas_pdb_xtc_without_tpr_keeps_a(tmp_path, trajectory):
    workspace = Workspace(tmp_path / "ws")
    directory, stem = _source(workspace)
    u = mda.Universe(str(trajectory[0]), str(trajectory[1]))
    u.atoms.write(str(directory / (stem + ".pdb")))
    u.trajectory.close()
    shutil.copyfile(trajectory[1], directory / (stem + ".xtc"))
    result = import_atlas(workspace, "1k5n_A")
    assert result["replicas"][0]["bonds"] == "unavailable"
    assert Registry(workspace).manifest(result["replicas"][0]["trajectory_id"])["atoms"] == 2
    with pytest.raises(LookupError):
        Registry(workspace).bonds(result["replicas"][0]["trajectory_id"])


def test_optional_tpr_failure_preserves_a(tmp_path, trajectory, monkeypatch):
    workspace = Workspace(tmp_path / "ws")
    directory, stem = _source(workspace)
    shutil.copyfile(trajectory[0], directory / (stem + ".gro"))
    shutil.copyfile(trajectory[1], directory / (stem + ".xtc"))
    (directory / (stem + ".tpr")).write_bytes(b"not-a-real-tpr")
    import minitopol.atlas.checker as checker
    original = checker._labels
    monkeypatch.setattr(checker, "_labels",
                        lambda path: (("XXX", "XXX"),) if path.suffix == ".tpr"
                        else original(path))
    plan = check_atlas(workspace, "1k5n_A")[0]
    assert plan["bond_error"] is not None
    result = import_atlas(workspace, "1k5n_A")
    assert result["status"] == "partial"
    assert result["replicas"][0]["A"] == "complete"
    assert result["replicas"][0]["bonds"] == "failed"
    assert len(workspace.list_trajectories()) == 1


def test_ambiguous_pairing_fails_before_import(tmp_path, trajectory):
    workspace = Workspace(tmp_path / "ws")
    directory, stem = _source(workspace)
    shutil.copyfile(trajectory[0], directory / (stem + ".gro"))
    shutil.copyfile(trajectory[1], directory / (stem + ".xtc"))
    (directory / (stem + ".tpr")).write_bytes(b"one")
    (directory / "other_R1.tpr").write_bytes(b"two")
    with pytest.raises(ValueError, match="Ambiguous"):
        check_atlas(workspace, "1k5n_A")
    assert workspace.list_trajectories() == []


def test_bond_backfill_and_crash_recovery(tmp_path, trajectory, monkeypatch):
    workspace = Workspace(tmp_path / "ws")
    tid = convert(workspace, trajectory[0], trajectory[1], solvent_resnames=["SOL"])
    saved_manifest = (workspace.trajectory_path(tid) / "manifest.json").read_bytes()
    tpr = tmp_path / "source.tpr"
    tpr.write_bytes(b"mock-tpr")
    _fake_tpr(monkeypatch)
    original = BondStore._register
    monkeypatch.setattr(BondStore, "_register",
                        lambda self, *args: (_ for _ in ()).throw(OSError("interrupted DB")))
    with pytest.raises(OSError, match="interrupted DB"):
        bond_reconstruction.reconstruct_bonds(workspace, tid, str(tpr), list(range(6)))
    assert (workspace.trajectory_path(tid) / "bonds.json").exists()
    monkeypatch.setattr(BondStore, "_register", original)
    assert bond_reconstruction.reconstruct_bonds(workspace, tid, str(tpr), list(range(6))) == 1
    assert (workspace.trajectory_path(tid) / "manifest.json").read_bytes() == saved_manifest
    with pytest.raises(ValueError, match="Different"):
        BondStore(workspace).publish(tid, [(0, 1)], tpr_sha256="changed",
                                     mapping_sha256="changed")
    data = workspace.trajectory_path(tid) / "bonds.parquet"
    with data.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(IOError, match="checksum"):
        Registry(workspace).bonds(tid)
