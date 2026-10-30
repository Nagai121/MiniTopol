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
                return (("ALA", "N", 1), ("ALA", "CA", 1), ("ALA", "H", 1),
                        ("SOL", "OW", 1), ("SOL", "HW1", 1), ("SOL", "HW2", 1))
            finally:
                u.trajectory.close()
        return original(path)

    monkeypatch.setattr(checker, "_labels", labels)
    monkeypatch.setattr(bond_reconstruction, "_covalent_edges_from_tpr",
                        lambda path: (6, [(0, 1), (1, 2), (3, 4), (3, 5)]))
    # This bond-only test uses a dummy TPR, not a parseable molecular system.
    # Keep the molecule-extraction contract independently covered in test_v020.
    import minitopol.atlas.importer as importer
    monkeypatch.setattr(importer, "publish_molecules",
                        lambda workspace, tid, tpr, mapping: {"molecules": 2, "atoms": 6})


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
    plan = check_atlas(workspace, "1k5n_A")[0]
    assert "Ambiguous" in plan["error"]
    result = import_atlas(workspace, "1k5n_A")
    assert result["status"] == "partial"
    assert result["replicas"][0]["A"] == "failed"
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



def test_interaction_types_do_not_confuse_restraints_with_chemical_bonds():
    from MDAnalysis.topology.tpr.obj import Ilist
    from minitopol.descriptors.bond_reconstruction import _edges_in_ilists

    interactions = [
        Ilist(3, ("BONDS", "Bond", 2), [0, 0, 1]),
        Ilist(3, ("CONSTR", "Constraint", 2), [0, 1, 2]),
        Ilist(3, ("HARMONIC", "Harmonic", 2), [0, 2, 3]),
        Ilist(3, ("CONSTRNC", "No Connection", 2), [0, 3, 4]),
        Ilist(3, ("RESTRAINTPOT", "Restraint", 2), [0, 4, 5]),
        Ilist(3, ("TABBONDSNC", "No Connection", 2), [0, 5, 6]),
    ]
    assert _edges_in_ilists(interactions) == [(0, 1), (1, 2)]


def test_isolated_atoms_are_a_valid_empty_bond_graph(tmp_path, trajectory):
    workspace = Workspace(tmp_path / "ws")
    tid = convert(workspace, *trajectory[:2], solvent_resnames=["SOL"])
    assert BondStore(workspace).publish(tid, [], tpr_sha256="source",
                                        mapping_sha256="map") == 0
    assert Registry(workspace).bonds(tid).num_rows == 0


def test_pdb_is_used_when_gro_atom_count_is_incompatible(tmp_path, trajectory):
    workspace = Workspace(tmp_path / "ws")
    directory, stem = _source(workspace)
    universe = mda.Universe(str(trajectory[0]), str(trajectory[1]))
    universe.atoms[:2].write(str(directory / (stem + ".gro")))
    universe.atoms.write(str(directory / (stem + ".pdb")))
    universe.trajectory.close()
    shutil.copyfile(trajectory[1], directory / (stem + ".xtc"))
    plan = check_atlas(workspace, "1k5n_A")[0]
    assert plan["topology"].endswith(".pdb")
    result = import_atlas(workspace, "1k5n_A")
    assert result["replicas"][0]["A"] == "complete"


def test_bad_replica_does_not_prevent_good_replica(tmp_path, trajectory):
    workspace = Workspace(tmp_path / "ws")
    directory, _ = _source(workspace)
    for replica in ("R1", "R2"):
        shutil.copyfile(trajectory[1], directory / f"1k5n_A_{replica}.xtc")
    shutil.copyfile(trajectory[0], directory / "1k5n_A_R1.gro")
    plans = check_atlas(workspace, "1k5n_A")
    assert plans[0]["topology"].endswith("R1.gro")
    assert plans[1]["error"]
    result = import_atlas(workspace, "1k5n_A")
    assert [r["A"] for r in result["replicas"]] == ["complete", "failed"]
    assert result["status"] == "partial"
    assert len(workspace.list_trajectories()) == 1


def test_tracked_archive_detects_modified_extracted_file(tmp_path, trajectory):
    from minitopol.atlas.ledger import DownloadLedger
    from minitopol.atlas.client import _sha256
    import zipfile

    workspace = Workspace(tmp_path / "ws")
    directory, stem = _source(workspace)
    pairs = ((stem + ".gro", trajectory[0]), (stem + ".xtc", trajectory[1]))
    archive = directory / "1k5n_A_analysis.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, path in pairs:
            zf.write(path, name)
    for name, path in pairs:
        shutil.copyfile(path, directory / name)
    DownloadLedger(workspace).put({
        "dataset": "1k5n_A", "kind": "analysis",
        "url": "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/1k5n_A",
        "archive": archive.relative_to(workspace.path).as_posix(),
        "downloaded_at": "2026-10-25T00:00:00Z",
        "bytes": str(archive.stat().st_size),
        "sha256": _sha256(archive), "status": "complete", "error": "",
    })
    assert check_atlas(workspace, "1k5n_A")[0]["provenance"] == "verified_archive"
    with (directory / (stem + ".gro")).open("ab") as out:
        out.write(b"tampered")
    with pytest.raises(ValueError, match="differs from verified archive"):
        check_atlas(workspace, "1k5n_A")


def test_repeated_residue_names_are_distinguished(tmp_path):
    import MDAnalysis as mda
    from minitopol.atlas.checker import _labels
    u = mda.Universe.empty(4, n_residues=2, atom_resindex=[0, 0, 1, 1], trajectory=True)
    u.add_TopologyAttr("names", ["N", "CA", "N", "CA"])
    u.add_TopologyAttr("resnames", ["ALA", "ALA"])
    u.add_TopologyAttr("resids", [1, 2])
    u.atoms.positions = [[0, 0, 0], [1, 0, 0], [3, 0, 0], [4, 0, 0]]
    pdb = tmp_path / "repeat.pdb"
    u.atoms.write(str(pdb))
    labels = _labels(pdb)
    assert labels[0] != labels[2]

def test_system_extraction_failure_is_partial_not_complete(tmp_path, trajectory, monkeypatch):
    import minitopol.atlas.importer as importer
    workspace = Workspace(tmp_path / "ws")
    _fake_tpr(monkeypatch)
    directory, stem = _source(workspace)
    shutil.copyfile(trajectory[0], directory / (stem + ".gro"))
    shutil.copyfile(trajectory[1], directory / (stem + ".xtc"))
    (directory / (stem + ".tpr")).write_bytes(b"mock-tpr")
    def fail(*args, **kwargs):
        raise ValueError("unsupported TPR molecular attributes")
    monkeypatch.setattr(importer, "publish_molecules", fail)
    report = import_atlas(workspace, "1k5n_A")
    assert report["status"] == "partial"
    assert report["replicas"][0]["A"] == "complete"
    assert report["replicas"][0]["bonds"] == "complete"
    assert report["replicas"][0]["system"] == "failed"
    assert "unsupported TPR" in report["replicas"][0]["system_error"]
