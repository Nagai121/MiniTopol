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
import psutil
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


def test_registry_atomic_replace_retries_transient_access_failure(tmp_path, monkeypatch):
    from minitopol.provider import runtime
    target = tmp_path / 'provider.json'
    monkeypatch.setenv('MINITOPOL_PROVIDER_REGISTRY', str(target))
    target.write_text('{"old":true}', encoding='utf-8')
    original = runtime.os.replace
    attempts = []

    def replace(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError('simulated Windows reader holds destination')
        return original(source, destination)

    monkeypatch.setattr(runtime.os, 'replace', replace)
    runtime.write_registry({'new': True})
    assert len(attempts) == 3
    assert json.loads(target.read_text(encoding='utf-8')) == {'new': True}
    assert not list(tmp_path.glob('*.tmp'))


def test_registry_failed_publication_preserves_previous_file(tmp_path, monkeypatch):
    from minitopol.provider import runtime
    target = tmp_path / 'provider.json'
    monkeypatch.setenv('MINITOPOL_PROVIDER_REGISTRY', str(target))
    target.write_text('{"old":true}', encoding='utf-8')

    def refuse(*args):
        raise PermissionError('permanent denial')

    monkeypatch.setattr(runtime.os, 'replace', refuse)
    with pytest.raises(PermissionError):
        runtime.write_registry({'new': True})
    assert json.loads(target.read_text(encoding='utf-8')) == {'old': True}
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('variant', [
    'decoded_cache', 'sparse_query', 'coordinate_catalog', 'payload_views', 'parallel_decode'])
def test_experimental_provider_preserves_values_and_leases(prepared, variant):
    from minitopol.provider.experiments import ExperimentalProvider
    workspace, tid = prepared
    ExperimentalProvider.variant = variant
    baseline = DataProvider(workspace.path, budget=16 * 1024**2)
    candidate = ExperimentalProvider(workspace.path, budget=16 * 1024**2)
    try:
        assert candidate.cache.budget + candidate.decoded_budget == 16 * 1024**2
        for coords, descriptors in [(True, []), (True, ['sample', 'contacts', 'rmsf']),
                                     (False, ['sample', 'contacts', 'rmsf'])]:
            request = {'trajectory': tid, 'coordinates': coords,
                       'descriptors': descriptors, 'frames': [8, 0, 4]}
            first = baseline.get(request)
            second = candidate.get(request)
            hit = candidate.get(request)
            assert candidate.cache.status()['active_leases'] == 2
            expected = {p['name']: p for p in first['buffers']}
            for part in second['buffers']:
                actual, reference = _shm_data(part), _shm_data(expected[part['name']])
                if part['format'] == 'arrow-ipc':
                    actual = pa.ipc.open_stream(actual).read_all()
                    reference = pa.ipc.open_stream(reference).read_all()
                    assert actual.equals(reference, check_metadata=False)
                else:
                    assert actual == reference
            assert [p['shm'] for p in second['buffers']] == [p['shm'] for p in hit['buffers']]
            candidate.cache.release(second['lease'])
            for part in hit['buffers']:
                _shm_data(part)
            candidate.cache.release(hit['lease'])
            baseline.cache.release(first['lease'])
            assert candidate.cache.status()['active_leases'] == 0
    finally:
        candidate.cache.close()
        baseline.cache.close()


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
        assert status["pid"] == session["pid"]
        # Windows venv python.exe can be a launcher for the actual interpreter.
        # The serving process must still belong to the process we launched.
        if session["pid"] != process.pid:
            assert os.name == "nt"
            assert process.pid in [parent.pid for parent in psutil.Process(session["pid"]).parents()]
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


def _profile_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_profile_appends_stage_timings_and_cache_stats(prepared, monkeypatch):
    """Each completed query adds one TXT line, with no in-memory history."""
    workspace, tid = prepared
    monkeypatch.setenv("MINITOPOL_PROVIDER_PROFILE", "1")
    monkeypatch.delenv("MINITOPOL_PROVIDER_PROFILE_PATH", raising=False)
    provider = DataProvider(workspace.path, 8 * 1024 * 1024)
    try:
        path = workspace.path / "provider-profile.txt"
        assert provider.profile_path == path
        assert not path.exists()
        for frame in [0, 1, 1]:
            result = provider.get({"trajectory": tid, "coordinates": True,
                                   "frames": [frame], "descriptors": ["sample"]})
            provider.cache.release(result["lease"])
        assert path.exists()
        miss, hit, exact = _profile_lines(path)
        assert len(_profile_lines(path)) == 3
        for record in (miss, hit, exact):
            assert record["status"] == "success"
            assert record["timestamp"]
            assert record["request"]["trajectory"] == tid
            assert record["request"]["frames"] == 1
            assert record["total_ms"] >= 0
            assert all(t >= 0 for t in record["stages_ms"].values())
        assert miss["counts"]["decoded_chunk_miss"] == 1
        assert miss["counts"]["shared_cache_miss"] == 1
        assert "zarr_fetch_and_decode" in miss["stages_ms"]
        assert "descriptor_query" in miss["stages_ms"]
        assert "shared_memory_copy" in miss["stages_ms"]
        assert hit["counts"]["decoded_chunk_hit"] == 1
        assert hit["counts"]["shared_cache_miss"] == 1
        assert "zarr_fetch_and_decode" not in hit["stages_ms"]
        assert exact["counts"]["shared_cache_hit"] == 1
        assert "coordinate_extract_copy" not in exact["stages_ms"]
        assert not hasattr(provider, "profile_records")
    finally:
        provider.cache.close()


def test_profile_disabled_by_default(prepared, monkeypatch):
    workspace, tid = prepared
    monkeypatch.delenv("MINITOPOL_PROVIDER_PROFILE", raising=False)
    monkeypatch.delenv("MINITOPOL_PROVIDER_PROFILE_PATH", raising=False)
    provider = DataProvider(workspace.path, 8 * 1024 * 1024)
    try:
        response = provider.get({"trajectory": tid, "coordinates": True, "frames": [0]})
        provider.cache.release(response["lease"])
        assert not (workspace.path / "provider-profile.txt").exists()
        assert provider.profile_path is None
        assert not hasattr(provider, "profile_records")
    finally:
        provider.cache.close()


def test_profile_append_existing_file_custom_path_and_failed_requests(prepared, tmp_path, monkeypatch):
    workspace, tid = prepared
    log = tmp_path / "custom" / "measurements.txt"
    log.parent.mkdir()
    log.write_text('{"previous":true}\n', encoding="utf-8")
    monkeypatch.setenv("MINITOPOL_PROVIDER_PROFILE", "1")
    monkeypatch.setenv("MINITOPOL_PROVIDER_PROFILE_PATH", str(log))
    provider = DataProvider(workspace.path, 8 * 1024 * 1024)
    try:
        response = provider.get({"trajectory": tid, "coordinates": True, "frames": [0]})
        provider.cache.release(response["lease"])
        with pytest.raises(ValueError, match="Invalid request fields"):
            provider.get({"trajectory": tid, "invalid": object()})
        lines = _profile_lines(log)
        assert len(lines) == 3
        assert lines[0] == {"previous": True}
        assert lines[1]["status"] == "success"
        assert lines[2]["status"] == "failure"
        assert lines[2]["error_type"] == "ValueError"
        assert lines[2]["request"] is None
        assert not (workspace.path / "provider-profile.txt").exists()
    finally:
        provider.cache.close()


def test_profile_http_request_writes_txt(prepared, tmp_path, monkeypatch):
    workspace, tid = prepared
    destination = tmp_path / "profile-registration.json"
    output = tmp_path / "logs" / "profile.txt"
    monkeypatch.setenv("MINITOPOL_PROVIDER_REGISTRY", str(destination))
    monkeypatch.setenv("MINITOPOL_PROVIDER_PROFILE", "1")
    monkeypatch.setenv("MINITOPOL_PROVIDER_PROFILE_PATH", str(output))
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
        fetched = call("/get", {"trajectory": tid, "coordinates": True, "frames": [0]})
        assert len(_profile_lines(output)) == 1
        assert _profile_lines(output)[0]["counts"]["shared_cache_miss"] == 1
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as no_profile:
            call("/profile")
        assert no_profile.value.code == 404
        assert call("/release", {"lease": fetched["lease"]})["released"]
        assert call("/stop", {})["stopping"]
        process.wait(timeout=15)
        assert process.returncode == 0
        assert len(_profile_lines(output)) == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_provider_refreshes_cached_descriptor_after_new_publication(prepared):
    """Publishing another result for the same descriptor invalidates old SHM keys."""
    workspace, tid = prepared
    provider = DataProvider(workspace.path, 8 * 1024 * 1024)
    query = {"trajectory": tid, "frames": [0], "descriptors": ["sample"]}

    def values(response):
        part = next(x for x in response["buffers"] if x["name"] == "descriptor:sample")
        return pa.ipc.open_stream(pa.BufferReader(_shm_data(part))).read_all()["value"].to_pylist()

    try:
        original = provider.get(query)
        assert values(original) == [10.0]
        duplicate = provider.get(query)
        assert provider.cache.hits == 1
        provider.cache.release(duplicate["lease"])

        updated = DescriptorDefinition("sample", "2", "A", "new result", "unit")
        ResultStore(workspace).publish(
            tid, updated, 0, 9,
            pa.table({"frame": pa.array([0], type=pa.int64()),
                      "value": pa.array([99.0])}))

        refreshed = provider.get(query)
        assert sorted(values(refreshed)) == [10.0, 99.0]
        assert provider.cache.misses == 2
        # A response held under an existing lease remains valid and unchanged.
        assert values(original) == [10.0]
        provider.cache.release(original["lease"])
        provider.cache.release(refreshed["lease"])
    finally:
        provider.cache.close()


def test_workspace_frame_range_on_global_descriptor(prepared):
    """Frame-less aggregate descriptors ignore frame filters, not fail SQL."""
    workspace, tid = prepared
    aggregate = workspace.query_results(
        descriptor="rmsf", trajectory=tid, start=3, stop=4)
    assert aggregate["value"].to_pylist() == [1.5, 2.5]

    # Ordinary frame-bearing descriptors keep the original range filtering.
    frame_data = workspace.query_results(
        descriptor="sample", trajectory=tid, start=2, stop=3)
    assert frame_data["value"].to_pylist() == [20.0]
