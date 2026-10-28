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
