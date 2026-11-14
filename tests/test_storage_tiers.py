"""Storage tiering: checksums, safe eviction, migration and Provider batches."""
import json
import shutil
from pathlib import Path

import pytest

from minitopol import Workspace, convert
from minitopol.storage_tiers import TieredStorage, logical_path, migrate_workspace


# Test files live on one tmp filesystem, unlike an actual external HDD.
# Patch only the device probe; test_external_mount_guard restores the real one.
_ORIGINAL_DEVICE_PROBE = TieredStorage._require_external_device


@pytest.fixture(autouse=True)
def _test_external_storage_device(monkeypatch):
    monkeypatch.setattr(TieredStorage, "_require_external_device", lambda self, path: None)


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
    assert manager.status()["external_copy_operations"] == 1
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
    assert manager.status()["internal_copy_operations"] == 1
    assert manager.status()["internal_written_bytes"] == len(b"position data")


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
    # Missing external tier must not stop the Provider from serving local data.
    restarted = TieredStorage(local)
    assert restarted.status()["external_available"] is False
    assert (local / "trajectories/trajectory-one/A.zarr/positions/0").is_file()
    with pytest.raises(FileNotFoundError):
        restarted.offload("trajectories/trajectory-one")


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


def test_default_storage_requires_explicit_legacy_migration(tmp_path, monkeypatch):
    import minitopol.workspace as module

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    project = tmp_path / "MiniTopol"
    package = project / "src" / "minitopol"
    package.mkdir(parents=True)
    (project / "pyproject.toml").touch()
    Workspace(project / "BaseTraj_Workspace")
    monkeypatch.setattr(module, "__file__", str(package / "workspace.py"))
    monkeypatch.delenv("MINITOPOL_WORKSPACE", raising=False)
    with pytest.raises(RuntimeError, match="migrate-workspace"):
        Workspace()


def test_existing_corrupt_internal_cache_is_rejected(tmp_path):
    local = tmp_path / "internal"
    data = _data(local)
    (tmp_path / "usb").mkdir()
    storage = TieredStorage(local)
    storage.configure(external_root=tmp_path / "usb", enabled=True)
    storage.offload("trajectories/trajectory-one")
    (data / "A.zarr/positions/0").write_bytes(b"CORRUPTED")
    restarted = TieredStorage(local)
    with pytest.raises(ValueError, match="Internal cache differs"):
        restarted.ensure("trajectories/trajectory-one")
    # Corrupt local data must not be discarded automatically.
    assert (data / "A.zarr/positions/0").read_bytes() == b"CORRUPTED"


def test_ingest_refuses_to_modify_external_primary(tmp_path, trajectory):
    workspace = Workspace(tmp_path / "internal")
    tid = convert(workspace, trajectory[0], trajectory[1], solvent_resnames=["SOL"])
    (tmp_path / "usb").mkdir()
    storage = TieredStorage(workspace.path)
    storage.configure(external_root=tmp_path / "usb", enabled=True)
    storage.offload("trajectories/" + tid)
    with pytest.raises(ValueError, match="ingest before"):
        storage.ingest_sources(tid)


def test_missing_external_can_disable_tiering(tmp_path):
    local = tmp_path / "internal"
    _data(local)
    (tmp_path / "usb").mkdir()
    storage = TieredStorage(local)
    storage.configure(external_root=tmp_path / "usb", enabled=True)
    shutil.rmtree(tmp_path / "usb")
    restarted = TieredStorage(local)
    assert restarted.configure(enabled=False)["enabled"] is False
    with pytest.raises(FileNotFoundError):
        restarted.configure(enabled=True)


def test_refuse_overlapping_storage_roots(tmp_path):
    local = tmp_path / "MiniTopol Storage"
    _data(local)
    storage = TieredStorage(local)
    for candidate in (local, local.parent, local / "nested"):
        with pytest.raises(ValueError, match="overlap"):
            storage.configure(external_root=candidate)
    assert (local / "trajectories/trajectory-one/A.zarr/positions/0").is_file()
    assert not storage.status()["enabled"]


def test_never_evict_sole_good_copy_if_external_primary_is_lost(tmp_path):
    local = tmp_path / "internal"
    data = _data(local)
    usb = tmp_path / "usb"
    usb.mkdir()
    storage = TieredStorage(local)
    storage.configure(external_root=usb, enabled=True)
    storage.offload("trajectories/trajectory-one")
    # The USB archive was subsequently damaged, while the internal copy is sound.
    external_file = usb / "MiniTopol Storage/trajectories/trajectory-one/A.zarr/positions/0"
    external_file.write_bytes(b"CORRUPTED")
    with pytest.raises(IOError, match="external primary checksum"):
        storage.prune(all_cache=True)
    assert (data / "A.zarr/positions/0").read_bytes() == b"position data"
    # If the drive is disconnected, pruning must also fail closed.
    shutil.rmtree(usb)
    with pytest.raises(FileNotFoundError):
        storage.prune(all_cache=True)
    assert (data / "A.zarr/positions/0").read_bytes() == b"position data"


def test_oversize_trajectory_uses_external_read_scope_without_ssd_staging(tmp_path, trajectory):
    from minitopol.provider.tiered import TieredProvider

    workspace = Workspace(tmp_path / "internal")
    tid = convert(workspace, trajectory[0], trajectory[1],
                  solvent_resnames=["SOL"], chunk_frames=3)
    usb = tmp_path / "usb"
    usb.mkdir()
    storage = TieredStorage(workspace.path)
    storage.configure(external_root=usb, enabled=True)
    storage.offload("trajectories/" + tid)
    storage.prune(all_cache=True)
    # Force an internal budget smaller than a single logical trajectory.
    storage.configure(cache_percent=1e-9)
    assert storage.status()["temporary_cache_limit_bytes"] < storage._object(
        "trajectories/" + tid)[2]
    assert storage.ensure("trajectories/" + tid)["direct_read"] is True
    assert not workspace.trajectory_path(tid).exists()
    provider = TieredProvider(workspace.path, 16 * 1024**2)
    try:
        request = dict(trajectory=tid, coordinates=True, descriptors=[], frames=[8, 0, 4])
        result = provider.get(request)
        assert result["frames"] == [8, 0, 4]
        assert result["buffers"]
        provider.cache.release(result["lease"])
        assert not workspace.trajectory_path(tid).exists()
        assert storage.status().get("internal_copy_operations", 0) == 0
    finally:
        provider.close()


def test_oversize_descriptor_and_raw_trajectory_guard(tmp_path, trajectory):
    import pyarrow as pa
    from minitopol.descriptors import DescriptorDefinition, ResultStore
    from minitopol.provider.tiered import TieredProvider

    ws = Workspace(tmp_path / "internal")
    tid = convert(ws, trajectory[0], trajectory[1], solvent_resnames=["SOL"])
    definition = DescriptorDefinition("sample", "1", "A", "test", "unit")
    ResultStore(ws).publish(tid, definition, 0, 9,
                            pa.table({"frame": [0, 2, 4, 8], "value": [1., 2., 3., 4.]}))
    usb = tmp_path / "usb"
    usb.mkdir()
    storage = TieredStorage(ws.path)
    storage.configure(external_root=usb, enabled=True)
    storage.offload("trajectories/" + tid)
    storage.offload("descriptors/sample/" + tid)
    storage.prune(all_cache=True)
    storage.configure(cache_percent=1e-9)
    with pytest.raises(PermissionError, match="externally authoritative"):
        with ws.storage_write("trajectories/" + tid):
            pass
    with pytest.raises(PermissionError, match="externally authoritative"):
        convert(ws, trajectory[0], trajectory[1],
                solvent_resnames=["SOL"], trajectory_id=tid)
    with pytest.raises(PermissionError, match="externally authoritative"):
        ResultStore(ws).publish(tid, definition, 0, 9,
                                pa.table({"frame": [0], "value": [1.]}))
    provider = TieredProvider(ws.path, 16 * 1024**2)
    try:
        request = {"trajectory": tid, "coordinates": True,
                   "descriptors": ["sample"], "frames": [8, 0, 4]}
        result = provider.get(request)
        assert "sample" in result["descriptors"]
        assert result["buffers"]
        provider.cache.release(result["lease"])
        assert not (ws.path / "trajectories" / tid).exists()
        assert not (ws.path / "descriptors/sample" / tid).exists()
    finally:
        provider.close()


def test_nested_scientific_write_guard_and_offload_exclusion(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    ws = Workspace(tmp_path / "internal")
    _data(ws.path)
    usb = tmp_path / "usb"
    usb.mkdir()
    storage = TieredStorage(ws.path)
    storage.configure(external_root=usb, enabled=True)
    began, release = threading.Event(), threading.Event()

    def mutate():
        with ws.storage_write("trajectories/trajectory-one"):
            with ws.storage_writer_lock("trajectories/trajectory-one", "nested-writer"):
                began.set()
                assert release.wait(timeout=30)

    with ThreadPoolExecutor(max_workers=2) as pool:
        writing = pool.submit(mutate)
        assert began.wait(timeout=30)
        publishing = pool.submit(storage.offload, "trajectories/trajectory-one")
        assert not publishing.done()
        release.set()
        writing.result(timeout=30)
        result = publishing.result(timeout=30)
    assert result["relative"] == "trajectories/trajectory-one"
    with pytest.raises(PermissionError, match="externally authoritative"):
        with ws.storage_write("trajectories/trajectory-one"):
            pass


def test_oversize_ramachandran_recipe_and_trajectory(tmp_path, trajectory):
    from minitopol.descriptors import analyze
    from minitopol.provider.tiered import TieredProvider

    ws = Workspace(tmp_path / "internal")
    tid = convert(ws, trajectory[0], trajectory[1], solvent_resnames=["SOL"])
    report = analyze(ws, tid, descriptors=["ramachandran"], coordinate_mode="nonperiodic")
    assert report["status"] == "complete"
    expected = ws.query_results(descriptor="ramachandran", trajectory=tid)
    usb = tmp_path / "usb"
    usb.mkdir()
    storage = TieredStorage(ws.path)
    storage.configure(external_root=usb, enabled=True)
    storage.offload("trajectories/" + tid)
    storage.offload("descriptors/ramachandran/" + tid)
    storage.prune(all_cache=True)
    storage.configure(cache_percent=1e-9)
    # A direct Workspace query must regenerate the virtual Rama table from
    # the published external A and descriptor without a Provider session.
    assert ws.query_results(descriptor="ramachandran", trajectory=tid).equals(expected)
    provider = TieredProvider(ws.path, 16 * 1024**2)
    try:
        result = provider.get(
            {"trajectory": tid, "coordinates": True, "descriptors": ["ramachandran"],
             "frames": [0, 8]})
        assert result["buffers"]
        provider.cache.release(result["lease"])
        assert not ws.trajectory_path(tid).exists()
    finally:
        provider.close()


def test_virtual_rama_scientific_source_checks_allow_only_known_routing_revisions():
    from minitopol.descriptors.virtual_rama_store import (
        _verify_rama_sources, source_hashes,
    )
    current = source_hashes()
    previous = dict(current)
    previous['virtual_rama.py'] = (
        'abebdc2697f86ecd6eed710224dbcc9882bacf68811d082c0ebdf560bb1d75b7'
    )
    previous['descriptors/virtual_rama_store.py'] = (
        '8adc49cf7ed35060a6cc42f4c4e173f43ad093be9c332a1327c026712d21bfd0'
    )
    _verify_rama_sources(current)
    _verify_rama_sources(previous)
    previous['registry.py'] = '0' * 64
    with pytest.raises(IOError, match="source changed"):
        _verify_rama_sources(previous)


def test_soft_cache_absolute_budget_and_percentage_switch(tmp_path):
    local = tmp_path / "internal"
    _data(local, contents=b"A" * 110)
    (tmp_path / "usb").mkdir()
    manager = TieredStorage(local)
    manager.configure(external_root=tmp_path / "usb", enabled=True,
                      cache_limit_bytes=100, cache_slack_percent=20,
                      cache_slack_bytes=0)
    config = manager.status()
    assert config["temporary_cache_mode"] == "absolute"
    assert config["temporary_cache_limit_bytes"] == 100
    assert config["temporary_cache_soft_limit_bytes"] == 120
    manager.offload("trajectories/trajectory-one")
    # Under the soft ceiling, do not evict the only internal cached copy.
    assert (local / "trajectories/trajectory-one/A.zarr/positions/0").is_file()
    manager.prune(all_cache=True)
    assert manager.ensure("trajectories/trajectory-one")["transferred"] is True
    assert manager.status()["temporary_cache_bytes"] == 110

    # Changing percentage explicitly switches to the relative mode;
    # a percentage above the old 10% default is accepted.
    updated = manager.configure(cache_percent=60)
    assert updated["temporary_cache_mode"] == "relative"
    assert updated["cache_limit_bytes"] is None
    assert updated["cache_percent"] == 60
    assert updated["temporary_cache_limit_bytes"] == int(
        manager.available_at_start * 0.60
    )
    assert updated["temporary_cache_soft_limit_bytes"] == (
        updated["temporary_cache_limit_bytes"] +
        int(updated["temporary_cache_limit_bytes"] * 20 / 100.0)
    )


def test_soft_cache_absolute_slack_and_direct_fallback(tmp_path):
    local = tmp_path / "internal"
    _data(local, contents=b"B" * 200)
    (tmp_path / "usb").mkdir()
    manager = TieredStorage(local)
    manager.configure(external_root=tmp_path / "usb", enabled=True,
                      cache_limit_bytes=100, cache_slack_percent=0,
                      cache_slack_bytes=125)
    assert manager.status()["temporary_cache_soft_limit_bytes"] == 225
    manager.offload("trajectories/trajectory-one")
    manager.prune(all_cache=True)
    assert manager.ensure("trajectories/trajectory-one")["transferred"] is True
    manager.prune(all_cache=True)
    # The same 200-byte object no longer fits when burst tolerance is reduced.
    manager.configure(cache_slack_bytes=0)
    outcome = manager.ensure("trajectories/trajectory-one")
    assert outcome["direct_read"] is True
    assert outcome["transferred"] is False
    assert not (local / "trajectories/trajectory-one").exists()


def test_invalid_capacity_modes_do_not_modify_persisted_settings(tmp_path):
    local = tmp_path / "internal"
    _data(local)
    manager = TieredStorage(local)
    before = manager.settings()
    invalids = (
        {"cache_limit_bytes": -1},
        {"cache_limit_bytes": 1.0},
        {"cache_limit_bytes": True},
        {"cache_slack_bytes": -1},
        {"cache_slack_bytes": False},
        {"cache_slack_percent": float("nan")},
        {"cache_slack_percent": -1},
        {"cache_percent": float("inf")},
        {"cache_percent": 0},
        {"cache_percent": 10, "cache_limit_bytes": 1024},
    )
    for options in invalids:
        with pytest.raises(ValueError):
            manager.configure(**options)
        assert manager.settings() == before


def test_physical_space_drift_falls_back_to_external_direct_reads(tmp_path, monkeypatch):
    import minitopol.storage_tiers as tiers

    local = tmp_path / "internal"
    _data(local, contents=b"C" * 100)
    (tmp_path / "usb").mkdir()
    manager = TieredStorage(local)
    manager.configure(external_root=tmp_path / "usb", enabled=True,
                      cache_limit_bytes=1000, cache_slack_percent=25)
    manager.offload("trajectories/trajectory-one")
    manager.prune(all_cache=True)
    original = tiers.shutil.disk_usage
    result = original(local)
    monkeypatch.setattr(
        tiers.shutil, "disk_usage",
        lambda path: type(result)(result.total, result.used, 0)
    )
    request = manager.ensure("trajectories/trajectory-one")
    assert request["direct_read"] is True
    assert not (local / "trajectories/trajectory-one").exists()


def test_concurrent_soft_limit_admission_is_serialized(tmp_path, monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    import minitopol.storage_tiers as tiers

    local = tmp_path / "internal"
    _data(local, name="first", contents=b"1" * 75)
    _data(local, name="second", contents=b"2" * 75)
    usb = tmp_path / "usb"
    usb.mkdir()
    storage = TieredStorage(local)
    storage.configure(external_root=usb, enabled=True,
                      cache_limit_bytes=100, cache_slack_percent=10)
    storage.offload("trajectories/first")
    storage.offload("trajectories/second")
    storage.prune(all_cache=True)

    original = tiers._copy_verified
    counter_lock = threading.Lock()
    active = 0
    peak = 0

    def counted_copy(src, destination, expected=None):
        nonlocal active, peak
        if local in destination.parents:
            with counter_lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.05)
                return original(src, destination, expected=expected)
            finally:
                with counter_lock:
                    active -= 1
        return original(src, destination, expected=expected)

    monkeypatch.setattr(tiers, "_copy_verified", counted_copy)
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(storage.ensure, "trajectories/first")
        two = pool.submit(storage.ensure, "trajectories/second")
        assert one.result(timeout=30)["relative"] == "trajectories/first"
        assert two.result(timeout=30)["relative"] == "trajectories/second"
    assert peak == 1
    assert storage.status()["temporary_cache_bytes"] <= 110


def test_linux_unmounted_external_directory_is_rejected(tmp_path, monkeypatch):
    import sys
    if not sys.platform.startswith("linux"):
        pytest.skip("st_dev external mount protection is Linux-specific")
    monkeypatch.setattr(TieredStorage, "_require_external_device", _ORIGINAL_DEVICE_PROBE)
    local = tmp_path / "internal"
    _data(local)
    mountpoint = tmp_path / "usb-unmounted"
    mountpoint.mkdir()
    storage = TieredStorage(local)
    with pytest.raises(RuntimeError, match="separately mounted"):
        storage.configure(external_root=mountpoint, enabled=True)
    # Do not create a "MiniTopol Storage" folder on the internal device.
    assert not (mountpoint / "MiniTopol Storage").exists()
    assert storage.settings()["external_root"] is None


def test_linux_external_identity_checked_again_if_drive_disappears(tmp_path, monkeypatch):
    import sys
    if not sys.platform.startswith("linux"):
        pytest.skip("st_dev external mount protection is Linux-specific")
    local = tmp_path / "internal"
    _data(local)
    mountpoint = tmp_path / "usb"
    mountpoint.mkdir()
    storage = TieredStorage(local)
    storage.configure(external_root=mountpoint, enabled=True)
    # Simulate a previously mounted drive now returning to its internal-disk
    # mountpoint. The folder and identity are still present.
    monkeypatch.setattr(TieredStorage, "_require_external_device", _ORIGINAL_DEVICE_PROBE)
    with pytest.raises(RuntimeError, match="separately mounted"):
        storage.offload("trajectories/trajectory-one")
    assert (local / "trajectories/trajectory-one/A.zarr/positions/0").exists()


def test_direct_registry_workspace_queries_after_external_offload(tmp_path, trajectory):
    import numpy as np
    import pyarrow as pa
    from minitopol import Registry
    from minitopol.bond_store import BondStore
    from minitopol.descriptors import DescriptorDefinition, ResultStore

    ws = Workspace(tmp_path / "internal")
    tid = convert(ws, trajectory[0], trajectory[1], solvent_resnames=["SOL"],
                  chunk_frames=3)
    registry = Registry(ws)
    reference_manifest = registry.manifest(tid)
    reference_atoms = registry.atoms(tid)
    baseline = list(registry.blocks(tid))
    definition = DescriptorDefinition("sample", "1", "A", "test", "unit")
    ResultStore(ws).publish(
        tid, definition, 0, reference_manifest["frames"],
        pa.table({"frame": [0, 2, 4], "value": [1.0, 2.0, 3.0]}))
    reference_table = ws.query_results(descriptor="sample", trajectory=tid)
    BondStore(ws).publish(tid, [(0, 1)], tpr_sha256="a" * 64,
                          mapping_sha256="b" * 64)
    expected_bonds = BondStore(ws).get(tid)

    usb = tmp_path / "usb"
    usb.mkdir()
    storage = TieredStorage(ws.path)
    storage.configure(external_root=usb, enabled=True)
    storage.offload("trajectories/" + tid)
    storage.offload("descriptors/sample/" + tid)
    storage.prune(all_cache=True)
    assert not (ws.path / "trajectories" / tid).exists()
    assert not (ws.path / "descriptors" / "sample" / tid).exists()

    # No Provider and no explicit storage_read_scope.
    assert registry.manifest(tid) == reference_manifest
    assert registry.atoms(tid).equals(reference_atoms)
    after = list(registry.blocks(tid))
    assert len(after) == len(baseline)
    for before, actual in zip(baseline, after):
        np.testing.assert_array_equal(actual.positions, before.positions)
        np.testing.assert_array_equal(actual.time, before.time)
        np.testing.assert_array_equal(actual.box, before.box)
    assert ws.query_results(descriptor="sample", trajectory=tid).equals(reference_table)
    assert BondStore(ws).get(tid).equals(expected_bonds)
    # Direct access must not implicitly re-stage the data onto the SSD.
    assert not (ws.path / "trajectories" / tid).exists()


def test_direct_external_read_rejects_corrupt_primary(tmp_path, trajectory):
    from minitopol import Registry
    ws = Workspace(tmp_path / "internal")
    tid = convert(ws, trajectory[0], trajectory[1], solvent_resnames=["SOL"])
    usb = tmp_path / "usb"
    usb.mkdir()
    storage = TieredStorage(ws.path)
    storage.configure(external_root=usb, enabled=True)
    storage.offload("trajectories/" + tid)
    storage.prune(all_cache=True)
    external = usb / "MiniTopol Storage" / "trajectories" / tid / "manifest.json"
    external.write_text("corrupted", encoding="utf-8")
    with pytest.raises(IOError, match="External primary differs"):
        Registry(ws).manifest(tid)


def test_numpy_chunk_grouping_keeps_order_and_duplicate_frames(tmp_path, trajectory):
    import numpy as np
    from minitopol import Registry
    from minitopol.provider.tiered import TieredProvider

    ws = Workspace(tmp_path / "internal")
    tid = convert(ws, trajectory[0], trajectory[1], solvent_resnames=["SOL"],
                  chunk_frames=3)
    registry = Registry(ws)
    baseline = list(registry.blocks(tid))
    coordinates = np.concatenate([block.positions for block in baseline], axis=0)
    times = np.concatenate([block.time for block in baseline])
    boxes = np.concatenate([block.box for block in baseline], axis=0)
    chosen = [8, 0, 4, 8, 2, 1, 0]
    selected = [0, 1]
    provider = TieredProvider(ws.path, 16 * 1024**2)
    try:
        manifest = registry.manifest(tid)
        output = provider._load_coordinates(
            tid, chosen, selected, manifest["frames"], manifest)
        np.testing.assert_array_equal(output[0], coordinates[chosen][:, selected])
        np.testing.assert_array_equal(output[1], times[chosen])
        np.testing.assert_array_equal(output[2], boxes[chosen])
        # Repeat to cover the decoded-chunk cache-hit path.
        again = provider._load_coordinates(
            tid, chosen, selected, manifest["frames"], manifest)
        for actual, expected in zip(again, output):
            np.testing.assert_array_equal(actual, expected)
        # Check the actively used parallel decode implementation as well.
        parallel = provider._parallel_coordinates(
            tid, chosen, selected, manifest["frames"], manifest)
        for actual, expected in zip(parallel, output):
            np.testing.assert_array_equal(actual, expected)
    finally:
        provider.close()


def test_direct_registry_can_use_verified_local_copy_when_drive_unmounted(tmp_path, trajectory):
    from minitopol import Registry
    ws = Workspace(tmp_path / "internal")
    tid = convert(ws, trajectory[0], trajectory[1], solvent_resnames=["SOL"])
    manifest = Registry(ws).manifest(tid)
    usb = tmp_path / "usb"
    usb.mkdir()
    storage = TieredStorage(ws.path)
    storage.configure(external_root=usb, enabled=True)
    storage.offload("trajectories/" + tid)
    # A verified internal replica still exists after offload.
    shutil.rmtree(usb)
    assert Registry(ws).manifest(tid) == manifest
    (ws.path / "trajectories" / tid / "manifest.json").write_text("broken")
    with pytest.raises(IOError, match="Local fallback differs"):
        Registry(ws).manifest(tid)
