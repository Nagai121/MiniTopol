"""ATLAS acquisition uses fake network responses; no live server/download required."""

import io
import zipfile

import pytest

from minitopol.atlas import AtlasClient, DownloadLedger
from minitopol.workspace import Workspace


class FakeResponse(io.BytesIO):
    def __init__(self, data, length=True):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))} if length else {}


def archive_bytes():
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("nested/1k5n_A.pdb", "ATOM\n")
        archive.writestr("nested/1k5n_A_R1.xtc", b"test-xtc")
        archive.writestr("nested/README.txt", "explanation")
    return source.getvalue()


def test_atlas_download_ledger_and_idempotence(tmp_path):
    payload = archive_bytes()
    requests = []
    def opener(request, *, timeout):
        requests.append(request.full_url)
        return FakeResponse(payload)
    workspace = Workspace(tmp_path / "workspace")
    client = AtlasClient(opener=opener)
    result = client.download(workspace, "1K5N_a", "analysis")
    assert result["status"] == "downloaded"
    assert len(result["files"]) == 2
    assert result["trajectory_id"] == "atlas-1k5n_A-analysis"
    assert result["sha256"]
    assert result["archive"].endswith("/source/1k5n_A_analysis.zip") or result["archive"].endswith("\\source\\1k5n_A_analysis.zip")
    records = DownloadLedger(workspace).list()
    assert len(records) == 1 and records[0]["status"] == "complete"
    assert records[0]["sha256"] == result["sha256"]
    assert (workspace.path / records[0]["archive"]).is_file()
    assert client.download(workspace, "1k5n_A", "analysis")["status"] == "skipped"
    assert len(requests) == 1
    assert not workspace.list_trajectories()  # raw archives are not converted A


def test_bad_inputs_and_bad_response_not_registered(tmp_path):
    workspace = Workspace(tmp_path)
    client = AtlasClient(opener=lambda request, timeout: FakeResponse(b"<html>error</html>"))
    with pytest.raises(ValueError):
        client.download(workspace, "../etc", "analysis")
    with pytest.raises(ValueError):
        client.download(workspace, "1k5n_A", "invalid")
    with pytest.raises(ValueError, match="ZIP"):
        client.download(workspace, "1k5n_A", "protein")
    row = DownloadLedger(workspace).get("1k5n_A", "protein")
    assert row["status"] == "failed"
    assert not (workspace.trajectory_path("atlas-1k5n_A-protein") / "source/1k5n_A_protein.zip").exists()


def test_existing_untracked_or_modified_archive_not_overwritten(tmp_path):
    workspace = Workspace(tmp_path)
    target = workspace.trajectory_path("atlas-1k5n_A-total") / "source"
    target.mkdir(parents=True)
    original = target / "1k5n_A_total.zip"
    original.write_bytes(b"user data")
    client = AtlasClient(opener=lambda request, timeout: pytest.fail("must not request remote"))
    with pytest.raises(FileExistsError):
        client.download(workspace, "1k5n_A", "total")
    assert original.read_bytes() == b"user data"


def test_zip_path_traversal_is_never_extracted(tmp_path):
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("../../evil.xtc", "demo")
        zf.writestr("good.pdb", "demo")
    workspace = Workspace(tmp_path)
    client = AtlasClient(opener=lambda request, timeout: FakeResponse(source.getvalue()))
    outcome = client.download(workspace, "1k5n_A")
    assert len(outcome["files"]) == 2
    assert all(str(workspace.path) in x for x in outcome["files"])
    assert not (tmp_path / "evil.xtc").exists()


def test_metadata_endpoint_and_schema(tmp_path):
    calls = []
    def fake(request, timeout):
        calls.append(request.full_url)
        return FakeResponse(b'{"1k5n_A": {"protein_name": "test"}}')
    assert AtlasClient(opener=fake).metadata("1k5n_A")["1k5n_A"]["protein_name"] == "test"
    assert calls == ["https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/metadata/1k5n_A"]


def test_mcp_tools_registered():
    pytest.importorskip("mcp")
    from minitopol.mcp.server import mcp
    names = {tool.name for tool in mcp._tool_manager.list_tools()}
    assert {"atlas_metadata", "atlas_download", "atlas_downloads",
            "list_trajectories", "convert_trajectory",
            "calculate_descriptors", "list_descriptor_results",
            "register_representation", "get_representation_definition",
            "query_descriptor_rows"} <= names



def test_zip_crc_corruption_rejected_before_publication(tmp_path):
    import minitopol.atlas.client as atlas_module

    payload = archive_bytes()
    assert b"test-xtc" in payload
    damaged = payload.replace(b"test-xtc", b"test-Xtc", 1)
    workspace = Workspace(tmp_path)
    client = AtlasClient(opener=lambda request, timeout: FakeResponse(damaged))
    with pytest.raises(zipfile.BadZipFile):
        client.download(workspace, "1k5n_A")
    row = DownloadLedger(workspace).get("1k5n_A", "analysis")
    assert row["status"] == "failed"
    assert not (workspace.path / row["archive"]).exists()
    assert not list((workspace.path / row["archive"]).parent.glob(".atlas-download-*"))


def test_missing_xtc_does_not_publish_or_claim_success(tmp_path):
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("input.pdb", "ATOM")
    workspace = Workspace(tmp_path)
    client = AtlasClient(opener=lambda request, timeout: FakeResponse(source.getvalue()))
    with pytest.raises(ValueError, match="XTC"):
        client.download(workspace, "1k5n_A")
    row = DownloadLedger(workspace).get("1k5n_A", "analysis")
    assert row["status"] == "failed"
    assert not (workspace.path / row["archive"]).exists()


def test_extraction_failure_recorded_and_retried_offline(tmp_path, monkeypatch):
    import minitopol.atlas.client as atlas_module

    workspace = Workspace(tmp_path)
    payload = archive_bytes()
    downloaded = []
    client = AtlasClient(opener=lambda request, timeout: (
        downloaded.append(request.full_url) or FakeResponse(payload)
    ))
    real_extract = atlas_module._extract_inputs
    with monkeypatch.context() as patch:
        def fail_extract(*args):
            raise OSError("disk temporarily unavailable")
        patch.setattr(atlas_module, "_extract_inputs", fail_extract)
        with pytest.raises(OSError, match="unavailable"):
            client.download(workspace, "1k5n_A")
    row = DownloadLedger(workspace).get("1k5n_A", "analysis")
    assert row["status"] == "failed"
    assert "unavailable" in row["error"]
    assert row["sha256"]
    assert (workspace.path / row["archive"]).is_file()

    no_network = AtlasClient(opener=lambda request, timeout: pytest.fail("must reuse ZIP"))
    result = no_network.download(workspace, "1k5n_A")
    assert result["status"] == "skipped"
    assert len(result["files"]) == 2
    assert DownloadLedger(workspace).get("1k5n_A", "analysis")["status"] == "complete"
    assert len(downloaded) == 1


def test_crash_after_atomic_zip_publish_recovers_and_cleans_temps(tmp_path):
    workspace = Workspace(tmp_path)
    directory = workspace.trajectory_path("atlas-1k5n_A-analysis") / "source"
    directory.mkdir(parents=True)
    archive = directory / "1k5n_A_analysis.zip"
    archive.write_bytes(archive_bytes())
    abandoned = directory / ".atlas-download-aborted"
    abandoned.write_bytes(b"partial")
    DownloadLedger(workspace).put({
        "dataset": "1k5n_A", "kind": "analysis",
        "url": "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/1k5n_A",
        "archive": archive.relative_to(workspace.path).as_posix(),
        "downloaded_at": "", "bytes": "", "sha256": "",
        "status": "downloading", "error": "",
    })
    no_network = AtlasClient(opener=lambda request, timeout: pytest.fail("must reuse ZIP"))
    result = no_network.download(workspace, "1k5n_A", extract=False)
    assert result["status"] == "skipped"
    assert not abandoned.exists()
    row = DownloadLedger(workspace).get("1k5n_A", "analysis")
    assert row["status"] == "complete"
    assert row["sha256"] == result["sha256"]
    assert row["bytes"] == str(archive.stat().st_size)


def test_reuse_hash_is_calculated_only_once(tmp_path, monkeypatch):
    import minitopol.atlas.client as atlas_module

    workspace = Workspace(tmp_path)
    payload = archive_bytes()
    AtlasClient(opener=lambda request, timeout: FakeResponse(payload)).download(
        workspace, "1k5n_A", extract=False
    )
    previous = atlas_module._sha256
    hashes = []
    def tracked(path):
        hashes.append(str(path))
        return previous(path)
    monkeypatch.setattr(atlas_module, "_sha256", tracked)
    AtlasClient(opener=lambda request, timeout: pytest.fail("must reuse ZIP")).download(
        workspace, "1k5n_A", extract=False
    )
    assert len(hashes) == 1


def test_corruption_after_previous_success_is_not_silently_accepted(tmp_path):
    workspace = Workspace(tmp_path)
    AtlasClient(opener=lambda request, timeout: FakeResponse(archive_bytes())).download(
        workspace, "1k5n_A", extract=False
    )
    row = DownloadLedger(workspace).get("1k5n_A", "analysis")
    archive = workspace.path / row["archive"]
    archive.write_bytes(b"corrupted file")
    client = AtlasClient(opener=lambda request, timeout: pytest.fail("must not redownload"))
    with pytest.raises(FileExistsError):
        client.download(workspace, "1k5n_A")
    assert archive.read_bytes() == b"corrupted file"
    assert DownloadLedger(workspace).get("1k5n_A", "analysis")["status"] == "failed"
