"""v0.2.0 data layout and system-metadata regression checks."""
import io
import zipfile

import pytest

from minitopol import Workspace, convert
from minitopol.atlas.client import AtlasClient
from minitopol.atlas.ledger import DownloadLedger, migrate_atlas_sources
from minitopol.system_metadata import publish_conditions, publish_molecules


class Response(io.BytesIO):
    @property
    def headers(self):
        return {"Content-Length": str(len(self.getvalue()))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def archive_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("x.xtc", b"dummy")
    return buffer.getvalue()


def test_new_atlas_sources_and_legacy_ledger_migration(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    legacy = ws.trajectory_path("atlas-1k5n_A-analysis") / "source"
    legacy.mkdir(parents=True)
    archive = legacy / "1k5n_A_analysis.zip"
    payload = archive_bytes()
    archive.write_bytes(payload)
    from minitopol.atlas.client import _sha256
    DownloadLedger(ws).put({
        "dataset": "1k5n_A", "kind": "analysis",
        "url": "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/1k5n_A",
        "archive": archive.relative_to(ws.path).as_posix(),
        "downloaded_at": "2026-10-28T00:00:00Z",
        "bytes": str(len(payload)), "sha256": _sha256(archive),
        "status": "complete", "error": "",
    })
    # Prior-version layout can be read before migration.
    assert DownloadLedger(ws).get("1k5n_A", "analysis")["sha256"] == _sha256(archive)
    assert migrate_atlas_sources(ws) == 1
    new = ws.path / "sources" / "ATLAS" / "1k5n_A-analysis" / archive.name
    assert new.read_bytes() == payload
    row = DownloadLedger(ws).get("1k5n_A", "analysis")
    assert row["archive"] == new.relative_to(ws.path).as_posix()
    assert (ws.path / "sources" / "ATLAS" / "downloads.tsv").is_file()
    assert not archive.exists()
    assert migrate_atlas_sources(ws) == 0


def test_fresh_download_stays_outside_processed_trajectories(tmp_path):
    ws = Workspace(tmp_path)
    client = AtlasClient(opener=lambda req, timeout: Response(archive_bytes()))
    result = client.download(ws, "1k5n_A", extract=False)
    assert result["archive"].startswith(str(ws.path / "sources" / "ATLAS"))
    assert "trajectories" not in result["archive"]


def test_mdp_conditions_enrich_without_conflicting(tmp_path, trajectory):
    ws = Workspace(tmp_path / "ws")
    tid = convert(ws, trajectory[0], trajectory[1], solvent_resnames=["SOL"])
    mdp = tmp_path / "test.mdp"
    mdp.write_text("integrator = md\ndt = 0.002\nref-t = 300\nref-p = 1\n"
                   "tcoupl = v-rescale\npcoupl = parrinello-rahman\n")
    record = publish_conditions(ws, tid, mdp=mdp)
    assert record["ensemble"] == "NPT"
    assert record["temperature_k"] == 300
    assert record["time_step_ps"] == 0.002
    assert record["ph"] is None
    assert publish_conditions(ws, tid)["temperature_k"] == 300
    with pytest.raises(ValueError, match="Conflicting"):
        publish_conditions(ws, tid, supplied={"temperature_k": 310})


def test_tpr_molecule_tables_are_full_system_not_just_a(tmp_path, trajectory, monkeypatch):
    import numpy as np
    import minitopol.system_metadata as module
    ws = Workspace(tmp_path / "ws")
    tid = convert(ws, trajectory[0], trajectory[1], solvent_resnames=["SOL"])
    tpr = tmp_path / "mock.tpr"
    tpr.write_bytes(b"mock")
    class Atoms:
        charges = np.array([0.2, -0.2, 0.0, -0.8, 0.4, 0.4])
        types = ["N", "CT", "H", "OW", "HW", "HW"]
        names = ["N", "CA", "H", "OW", "HW1", "HW2"]
        moltypes = ["PRO", "PRO", "PRO", "SOL", "SOL", "SOL"]
        def __len__(self):
            return 6
    class Universe:
        atoms = Atoms()
        trajectory = None
    monkeypatch.setattr(module.mda, "Universe", lambda path: Universe())
    monkeypatch.setattr(module, "_covalent_edges_from_tpr",
                        lambda path, molecule_ids: (6, [], [0, 0, 0, 1, 1, 1]))
    result = publish_molecules(ws, tid, tpr, list(range(6)))
    assert result == {"molecules": 2, "atoms": 6}
    import pyarrow.parquet as pq
    atom_data = pq.read_table(ws.trajectory_path(tid) / "system" / "atom_properties.parquet").to_pydict()
    mol_data = pq.read_table(ws.trajectory_path(tid) / "system" / "molecules.parquet").to_pydict()
    assert atom_data["a_point"] == [0, 1, None, None, None, None]
    assert mol_data["molecule_type"] == ["PRO", "SOL"]
    assert publish_molecules(ws, tid, tpr, list(range(6))) == result


def _tracked_legacy(ws, dataset="1k5n_A", kind="analysis", payload=None):
    """Create a v0.1 source archive and its original nine-column ledger."""
    from minitopol.atlas.client import _sha256
    directory = ws.trajectory_path(f"atlas-{dataset}-{kind}") / "source"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"{dataset}_{kind}.zip"
    archive.write_bytes(payload if payload is not None else archive_bytes())
    row = {
        "dataset": dataset, "kind": kind,
        "url": f"https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/{kind}/{dataset}",
        "archive": archive.relative_to(ws.path).as_posix(),
        "downloaded_at": "2026-10-28T00:00:00Z",
        "bytes": str(archive.stat().st_size),
        "sha256": _sha256(archive), "status": "complete", "error": "",
    }
    DownloadLedger(ws).put(row)
    return directory, archive, row


def test_migration_relinks_converter_sources_without_changing_identity(tmp_path, trajectory):
    import shutil
    from minitopol.storage import read_json
    ws = Workspace(tmp_path / "ws")
    source, archive, row = _tracked_legacy(ws)
    shutil.copyfile(trajectory[0], source / "1k5n_A_R1.gro")
    shutil.copyfile(trajectory[1], source / "1k5n_A_R1.xtc")
    tid = convert(ws, source / "1k5n_A_R1.gro",
                  source / "1k5n_A_R1.xtc", trajectory_id="atlas-1k5n_A-analysis-R1",
                  solvent_resnames=["SOL"])
    manifest_path = ws.trajectory_path(tid) / "manifest.json"
    original = read_json(manifest_path)
    a_manifest = ws.trajectory_path(tid) / "A.zarr"
    assert migrate_atlas_sources(ws) == 1
    now = read_json(manifest_path)
    new = ws.path / "sources" / "ATLAS" / "1k5n_A-analysis"
    assert now["xtc_path"] == str((new / "1k5n_A_R1.xtc").resolve())
    assert now["gro_path"] == str((new / "1k5n_A_R1.gro").resolve())
    assert now["identity"] == original["identity"]
    assert now["options"] == original["options"]
    assert a_manifest.is_dir()
    assert convert(ws, new / "1k5n_A_R1.gro", new / "1k5n_A_R1.xtc",
                   trajectory_id=tid, solvent_resnames=["SOL"]) == tid
    assert migrate_atlas_sources(ws) == 0


def test_migration_resumes_if_interrupted_after_sources_were_moved(tmp_path, monkeypatch):
    import minitopol.atlas.ledger as ledger
    ws = Workspace(tmp_path / "ws")
    source, archive, row = _tracked_legacy(ws)
    initial = ledger._relink_manifests
    def interrupted(*args):
        raise OSError("simulated crash after directory rename")
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "_relink_manifests", interrupted)
        with pytest.raises(OSError, match="simulated crash"):
            migrate_atlas_sources(ws)
    target = ws.path / "sources" / "ATLAS" / "1k5n_A-analysis"
    assert target.is_dir() and not source.exists()
    assert (ws.path / "atlas" / "downloads.tsv").exists()
    assert migrate_atlas_sources(ws) == 0
    assert (target / archive.name).is_file()
    assert DownloadLedger(ws).get("1k5n_A", "analysis")["archive"].startswith("sources/")


def test_migration_reconciles_two_identical_ledgers(tmp_path):
    import minitopol.atlas.ledger as ledger
    ws = Workspace(tmp_path / "ws")
    source, archive, old = _tracked_legacy(ws)
    modern = ws.path / "sources" / "ATLAS" / "downloads.tsv"
    new = dict(old, archive="sources/ATLAS/1k5n_A-analysis/1k5n_A_analysis.zip")
    ledger._write_rows(modern, [new])
    with pytest.raises(RuntimeError, match="Both ATLAS ledgers"):
        DownloadLedger(ws)
    assert migrate_atlas_sources(ws) == 1
    assert not (ws.path / "atlas" / "downloads.tsv").exists()
    assert DownloadLedger(ws).get("1k5n_A", "analysis")["archive"] == new["archive"]


def test_migration_never_overwrites_conflicting_ledgers(tmp_path):
    import minitopol.atlas.ledger as ledger
    ws = Workspace(tmp_path / "ws")
    source, archive, old = _tracked_legacy(ws)
    modern = ws.path / "sources" / "ATLAS" / "downloads.tsv"
    newer = dict(old, archive="sources/ATLAS/1k5n_A-analysis/1k5n_A_analysis.zip",
                 sha256="conflicting")
    ledger._write_rows(modern, [newer])
    with pytest.raises(ValueError, match="Conflicting old/new"):
        migrate_atlas_sources(ws)
    assert source.is_dir() and archive.is_file()
    assert (ws.path / "atlas" / "downloads.tsv").exists()
    assert modern.is_file()


def test_atlas_download_waits_for_migration_lock(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    ws = Workspace(tmp_path / "ws")
    client = AtlasClient(opener=lambda req, timeout: Response(archive_bytes()))
    entered = threading.Event()
    original = client._download_locked
    def tracked(*args, **kwargs):
        entered.set()
        return original(*args, **kwargs)
    client._download_locked = tracked
    with ThreadPoolExecutor(max_workers=1) as pool:
        with ws.lock("atlas-sources"):
            future = pool.submit(client.download, ws, "1k5n_A", extract=False)
            assert not entered.wait(0.15)
        result = future.result(timeout=15)
    assert entered.is_set()
    assert result["status"] == "downloaded"
