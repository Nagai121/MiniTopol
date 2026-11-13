"""Storage tiering: checksums, safe eviction, migration and Provider batches."""
import json
import shutil

import pytest

from minitopol import Workspace, convert
from minitopol.storage_tiers import TieredStorage, logical_path, migrate_workspace


def _data(root, name="trajectory-one", contents=b"position data"):
    path = root / "trajectories" / name / "A.zarr" / "positions"
    path.mkdir(parents=True, exist_ok=True)
    (path / "0").write_bytes(contents)
    return root / "trajectories" / name


def test_logical_paths_are_bounded():
    assert logical_path("trajectories/T1") == "trajectories/T1"
    assert logical_path("descriptors/sasa/T1") == "descriptors/sasa/T1"
    assert logical_path("sources/ATLAS/2au7-analysis") == "sources/ATLAS/2au7-analysis"
    for bad in ("../trajectories/T1", "trajectories", "trajectories/../else",
                "descriptors/sasa", "/sources/ATLAS/T1", "trajectories\\T1"):
        with pytest.raises(ValueError):
            logical_path(bad)


def test_offload_verified_cache_eviction_and_reuse(tmp_path):
    local = tmp_path / "internal" / "MiniTopol Storage"
    path = _data(local)
    manager = TieredStorage(local)
    assert manager.status()["enabled"] is False
    (tmp_path / "external").mkdir()
    manager.configure(external_root=tmp_path / "external", enabled=True)
    entry = manager.offload("trajectories/trajectory-one")
    assert entry["bytes"] == len(b"position data")
    external = tmp_path / "external" / "MiniTopol Storage"
    assert (external / "trajectories/trajectory-one/A.zarr/positions/0").read_bytes() == b"position data"
    before = manager.offload("trajectories/trajectory-one")
    assert before["external_copy_created"] is False
    assert manager.prune(all_cache=True)["removed"] == ["trajectories/trajectory-one"]
    assert not path.exists()
    assert manager.ensure("trajectories/trajectory-one")["transferred"] is True
    assert (path / "A.zarr/positions/0").read_bytes() == b"position data"
    assert manager.ensure("trajectories/trajectory-one")["transferred"] is False
    assert manager.status()["temporary_cache_bytes"] == len(b"position data")


def test_bad_external_copy_never_published(tmp_path):
    local = tmp_path / "local"
    path = _data(local)
    storage = TieredStorage(local)
    (tmp_path / "usb").mkdir()
    storage.configure(external_root=tmp_path / "usb", enabled=True)
    storage.offload("trajectories/trajectory-one")
    storage.prune(all_cache=True)
    external_file = tmp_path / "usb/MiniTopol Storage/trajectories/trajectory-one/A.zarr/positions/0"
    external_file.write_bytes(b"CORRUPT")
    with pytest.raises(ValueError, match="checksum"):
        storage.ensure("trajectories/trajectory-one")
    assert not path.exists()
    assert not list(local.rglob("*.partial-*"))


def test_missing_registered_external_fails_closed(tmp_path):
    local = tmp_path / "internal"
    _data(local)
    storage = TieredStorage(local)
    (tmp_path / "usb").mkdir()
    storage.configure(external_root=tmp_path / "usb", enabled=True)
    shutil.rmtree(tmp_path / "usb")
    with pytest.raises(FileNotFoundError):
        TieredStorage(local)


def test_storage_migration_retains_old_data(tmp_path):
    old = tmp_path / "BaseTraj_Workspace"
    workspace = Workspace(old)
    _data(old)
    target = tmp_path / "MiniTopol Storage"
    result = migrate_workspace(old, target)
    assert result["original_retained"] is True
    assert (target / "integrated.duckdb").is_file()
    assert (target / "trajectories/trajectory-one/A.zarr/positions/0").read_bytes() == b"position data"
    assert (old / "integrated.duckdb").is_file()
    with pytest.raises(FileExistsError):
        migrate_workspace(old, target)


def test_provider_offloaded_trajectory_and_ordered_plans(tmp_path, trajectory):
    from minitopol.provider.tiered import TieredProvider

    workspace = Workspace(tmp_path / "internal")
    tid = convert(workspace, trajectory[0], trajectory[1],
                  solvent_resnames=["SOL"], chunk_frames=3)
    storage = TieredStorage(workspace.path)
    (tmp_path / "usb").mkdir()
    storage.configure(external_root=tmp_path / "usb", enabled=True, prefetch_batches=1)
    storage.offload("trajectories/" + tid)
    storage.prune(all_cache=True)
    assert not workspace.trajectory_path(tid).exists()
    provider = TieredProvider(workspace.path, 16 * 1024 ** 2)
    try:
        request = {"trajectory": tid, "coordinates": True,
                   "descriptors": [], "frames": [8, 0, 4]}
        plan = provider.plan_batches({"batches": [request, request]})
        assert plan["submitted"] == 1
        with pytest.raises(ValueError, match="out-of-order"):
            provider.get({**request, "plan_id": plan["plan_id"], "plan_index": 1})
        first = provider.get({**request, "plan_id": plan["plan_id"], "plan_index": 0})
        assert first["frames"] == [8, 0, 4]
        provider.cache.release(first["lease"])
        assert provider.plan_status(plan["plan_id"])["submitted"] == 2
        second = provider.get({**request, "plan_id": plan["plan_id"], "plan_index": 1})
        provider.cache.release(second["lease"])
        assert provider.plan_status(plan["plan_id"])["consumed"] == 2
        assert workspace.trajectory_path(tid).is_dir()
    finally:
        provider.close()


def test_adopt_raw_source_files(tmp_path, trajectory):
    workspace = Workspace(tmp_path / "internal")
    tid = convert(workspace, trajectory[0], trajectory[1], solvent_resnames=["SOL"])
    storage = TieredStorage(workspace.path)
    report = storage.ingest_sources(tid)
    assert report["relative"] == "sources/manual/" + tid
    assert report["originals_retained"]
    manifest = json.loads((workspace.trajectory_path(tid) / "manifest.json").read_text())
    assert (workspace.path / "sources/manual" / tid).is_dir()
    assert all("sources" in manifest[k] for k in ("gro_path", "xtc_path"))
