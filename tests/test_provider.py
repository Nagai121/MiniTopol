"""Real MiniTopol Provider integration: converted GRO/XTC, Arrow, shared memory."""
import json
import os
import subprocess
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pyarrow as pa
import pytest

from minitopol import Workspace, convert
from minitopol.descriptors import DescriptorDefinition, ResultStore
from minitopol.provider.runtime import Cache, DataProvider, read_registry, register, registry_path, _probe


@pytest.fixture
def prepared(tmp_path, trajectory):
    workspace = Workspace(tmp_path / "storage")
    tid = convert(workspace, trajectory[0], trajectory[1],
                  solvent_resnames=["SOL"], chunk_frames=3)
    store = ResultStore(workspace)
    basic = DescriptorDefinition("sample", "1", "A", "test", "unit")
    store.publish(tid, basic, 0, 9,
                  pa.table({"frame": pa.array([0, 2, 4, 8], type=pa.int64()),
                            "value": pa.array([10., 20., 30., 40.])}))
    global_desc = DescriptorDefinition("rmsf", "test", "A", "global", "angstrom")
    store.publish(tid, global_desc, 0, 9,
                  pa.table({"point": [0, 1], "value": [1.5, 2.5]}))
    mixed = DescriptorDefinition("contacts", "test", "A", "mixed", "unit")
    mixed.register(workspace)
    with store.dataset(tid, mixed, layout={"output_partitioning": "test"}) as writer:
        writer.append("frames-0-9",
                      pa.table({"level": ["state", "state", "frequency"],
                                "frame": pa.array([0, 8, None], type=pa.int64()),
                                "value": [1., 1., .25]}))
    return workspace, tid


def _shm_data(part):
    segment = shared_memory.SharedMemory(name=part["shm"], create=False)
    try:
        data = bytes(segment.buf[:part["bytes"]])
        return data
    finally:
        segment.close()


def test_coordinates_random_frame_order_and_subset(prepared):
    workspace, tid = prepared
    service = DataProvider(workspace.path, budget=8 * 1024 * 1024)
    try:
        result = service.get({"trajectory": tid, "coordinates": True,
                              "frames": [8, 0, 4], "atoms": [1, 0],
                              "descriptors": ["sample", "rmsf", "contacts"]})
        parts = {item["name"]: item for item in result["buffers"]}
        coords = np.frombuffer(_shm_data(parts["coordinates"]), dtype="<f4").reshape(3, 2, 3)
        np.testing.assert_allclose(coords[0, 0], np.array([44, 5, 6]), atol=.002)
        np.testing.assert_allclose(coords[1, 1], np.array([1, 2, 3]), atol=.002)
        times = np.frombuffer(_shm_data(parts["time"]), dtype="<f8")
        np.testing.assert_allclose(times, [20, 0, 10])
        boxes = np.frombuffer(_shm_data(parts["box"]), dtype="<f4").reshape(3, 3, 3)
        assert boxes.shape == (3, 3, 3)
        def arrow(name):
            return pa.ipc.open_stream(pa.BufferReader(_shm_data(parts["descriptor:" + name]))).read_all()
        assert arrow("sample")["value"].to_pylist() == [10., 30., 40.]
        assert arrow("rmsf")["value"].to_pylist() == [1.5, 2.5]
        contacts = arrow("contacts")
        assert contacts["frame"].to_pylist() == [0, 8, None]
        assert result["frames"] == [8, 0, 4]
        second = service.get({"trajectory": tid, "coordinates": True,
                              "frames": [8, 0, 4], "atoms": [1, 0],
                              "descriptors": ["sample", "rmsf", "contacts"]})
        assert second["lease"] != result["lease"]
        assert service.cache.hits == 1
        assert service.cache.status()["active_leases"] == 2
        service.cache.release(result["lease"])
        service.cache.release(second["lease"])
        assert service.cache.status()["active_leases"] == 0
    finally:
        service.cache.close()


def test_all_descriptor_selection_and_budget(prepared):
    workspace, tid = prepared
    service = DataProvider(workspace.path, budget=8 * 1024 * 1024)
    try:
        response = service.get({"trajectory": tid, "frames": [0], "coordinates": False,
                                "descriptors": "all"})
        assert response["descriptors"] == ["contacts", "rmsf", "sample"]
        service.cache.release(response["lease"])
        with pytest.raises(LookupError, match="Missing completed"):
            service.get({"trajectory": tid, "frames": [0], "coordinates": False,
                         "descriptors": ["sasa"]})
    finally:
        service.cache.close()
    tiny = DataProvider(workspace.path, budget=32)
    try:
        with pytest.raises(MemoryError, match="budget"):
            tiny.get({"trajectory": tid, "frames": [0], "coordinates": True})
        assert tiny.cache.status()["entries"] == 0
    finally:
        tiny.cache.close()


def test_http_service_lifecycle(prepared, tmp_path, monkeypatch):
    workspace, tid = prepared
    destination = tmp_path / "provider-registration.json"
    monkeypatch.setenv("MINITOPOL_PROVIDER_REGISTRY", str(destination))
    register(workspace.path)
    config = read_registry()
    assert config["workspace"] == str(workspace.path)
    env = os.environ.copy()
    process = subprocess.Popen([sys.executable, "-m", "minitopol.provider", "serve"],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        session = None
        for _ in range(100):
            current = read_registry()
            if current and current.get("session") and _probe(current["session"]):
                session = current["session"]
                break
            if process.poll() is not None:
                break
            time.sleep(.1)
        assert session is not None, process.communicate(timeout=3)
        address = f"http://127.0.0.1:{session['port']}"
        headers = {"Authorization": "Bearer " + session["token"],
                   "Content-Type": "application/json"}
        def call(route, data=None):
            raw = None if data is None else json.dumps(data).encode()
            req = Request(address + "/v1/" + route, data=raw, headers=headers,
                          method="GET" if data is None else "POST")
            with urlopen(req, timeout=10) as response:
                return json.load(response)
        status = call("status")
        assert status["pid"] == session["pid"] == process.pid
        assert status["budget_bytes"] > 0
        assert tid in [row["id"] for row in call("list")["trajectories"]]
        fetched = call("get", {"trajectory": tid, "coordinates": True, "frames": [1]})
        part = next(p for p in fetched["buffers"] if p["name"] == "coordinates")
        assert len(_shm_data(part)) == 2*3*4
        assert call("release", {"lease": fetched["lease"]})["released"]
        assert call("stop", {})["stopping"]
        process.wait(timeout=10)
        assert process.returncode == 0
        assert "session" not in read_registry()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)



def test_decoded_chunk_reused_by_different_request_keys(prepared):
    """Distinct requested frames reuse decoded source chunks, not only SHM replies."""
    workspace, tid = prepared
    budget = 8 * 1024 * 1024
    provider = DataProvider(workspace.path, budget)
    try:
        first = provider.get({"trajectory": tid, "coordinates": True,
                              "frames": [0], "descriptors": []})
        provider.cache.release(first["lease"])
        assert len(provider.decoded_chunks) == 1
        cached_key = next(iter(provider.decoded_chunks))
        cached_arrays = provider.decoded_chunks[cached_key]

        second = provider.get({"trajectory": tid, "coordinates": True,
                               "frames": [1], "descriptors": []})
        parts = {entry["name"]: entry for entry in second["buffers"]}
        actual = np.frombuffer(_shm_data(parts["coordinates"]), dtype="<f4").reshape(1, 2, 3)
        np.testing.assert_allclose(actual[0, 0], [6, 2, 3], atol=.002)
        provider.cache.release(second["lease"])

        assert provider.decoded_chunks[cached_key] is cached_arrays
        assert provider.cache.hits == 0  # Distinct SHM request keys.
        assert provider.decoded_used <= provider.decoded_budget
        assert provider.cache.used + provider.decoded_used <= budget
        assert provider.cache.budget + provider.decoded_budget == budget
    finally:
        provider.cache.close()


def test_decoded_lru_eviction_is_bounded_and_reloads_correctly(prepared):
    workspace, tid = prepared
    provider = DataProvider(workspace.path, budget=1500)
    assert provider.decoded_budget == 300  # One 3-frame source chunk fits.
    try:
        for requested in ([0, 3, 6], [6, 0, 3], [0, 4, 8]):
            result = provider.get({"trajectory": tid, "coordinates": True,
                                   "frames": requested, "descriptors": []})
            parts = {entry["name"]: entry for entry in result["buffers"]}
            actual = np.frombuffer(_shm_data(parts["coordinates"]), dtype="<f4").reshape(3, 2, 3)
            expected = np.asarray([[1 + 5*f, 2, 3] for f in requested])
            np.testing.assert_allclose(actual[:, 0], expected, atol=.002)
            provider.cache.release(result["lease"])
            assert provider.decoded_used <= provider.decoded_budget
            assert provider.cache.used + provider.decoded_used <= 1500
            assert len(provider.decoded_chunks) <= 1
    finally:
        provider.cache.close()


def test_profile_stage_timing_cache_hits_and_reset(prepared, monkeypatch):
    """Instrument actual Zarr/Arrow/SHM work, including both cache layers."""
    workspace, tid = prepared
    monkeypatch.setenv("MINITOPOL_PROVIDER_PROFILE", "1")
    provider = DataProvider(workspace.path, 8 * 1024 * 1024)
    try:
        first = provider.get({"trajectory": tid, "coordinates": True,
                              "frames": [0], "descriptors": ["sample"]})
        provider.cache.release(first["lease"])
        second = provider.get({"trajectory": tid, "coordinates": True,
                               "frames": [1], "descriptors": ["sample"]})
        provider.cache.release(second["lease"])
        third = provider.get({"trajectory": tid, "coordinates": True,
                              "frames": [1], "descriptors": ["sample"]})
        provider.cache.release(third["lease"])
        snapshot = provider.profile_snapshot()
        assert snapshot["enabled"] is True
        assert len(snapshot["records"]) == 3
        miss, hit, exact = snapshot["records"]
        for record in snapshot["records"]:
            assert record["status"] == "success"
            assert record["total_ms"] >= 0
            assert all(t >= 0 for t in record["stages_ms"].values())
        assert miss["counts"]["decoded_chunk_miss"] == 1
        assert miss["counts"]["shared_cache_miss"] == 1
        assert miss["stages_ms"]["zarr_fetch_and_decode"] >= 0
        assert "descriptor_query" in miss["stages_ms"]
        assert "shared_memory_copy" in miss["stages_ms"]
        assert hit["counts"]["decoded_chunk_hit"] == 1
        assert hit["counts"]["shared_cache_miss"] == 1
        assert "zarr_fetch_and_decode" not in hit["stages_ms"]
        assert exact["counts"]["shared_cache_hit"] == 1
        assert "coordinate_extract_copy" not in exact["stages_ms"]
        assert provider.clear_profiles()["cleared"]
        assert provider.profile_snapshot()["records"] == []
    finally:
        provider.cache.close()


def test_profile_disabled_by_default(prepared, monkeypatch):
    workspace, tid = prepared
    monkeypatch.delenv("MINITOPOL_PROVIDER_PROFILE", raising=False)
    provider = DataProvider(workspace.path, 8 * 1024 * 1024)
    try:
        response = provider.get({"trajectory": tid, "coordinates": True, "frames": [0]})
        provider.cache.release(response["lease"])
        assert provider.profile_snapshot() == {
            "enabled": False, "capacity": 256, "records": []}
    finally:
        provider.cache.close()


def test_profile_http_endpoint(prepared, tmp_path, monkeypatch):
    workspace, tid = prepared
    destination = tmp_path / "profile-registration.json"
    monkeypatch.setenv("MINITOPOL_PROVIDER_REGISTRY", str(destination))
    monkeypatch.setenv("MINITOPOL_PROVIDER_PROFILE", "1")
    register(workspace.path)
    process = subprocess.Popen(
        [sys.executable, "-m", "minitopol.provider", "serve"],
        env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        session = None
        for _ in range(100):
            current = read_registry()
            if current and current.get("session") and _probe(current["session"]):
                session = current["session"]
                break
            if process.poll() is not None:
                break
            time.sleep(.1)
        assert session is not None, process.communicate(timeout=3)
        base = f"http://127.0.0.1:{session['port']}/v1"
        headers = {"Authorization": "Bearer " + session["token"]}
        def call(route, body=None):
            data = None if body is None else json.dumps(body).encode()
            req = Request(base + route, data=data, headers=headers)
            with urlopen(req, timeout=15) as response:
                return json.load(response)
        assert call("/profile")["enabled"]
        fetched = call("/get", {"trajectory": tid, "coordinates": True, "frames": [0]})
        assert len(call("/profile")["records"]) == 1
        assert call("/profile/reset", {})["cleared"] is True
        assert call("/profile")["records"] == []
        assert call("/release", {"lease": fetched["lease"]})["released"]
        assert call("/stop", {})["stopping"]
        process.wait(timeout=15)
        assert process.returncode == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
