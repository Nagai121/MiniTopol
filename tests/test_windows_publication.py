"""Atomic Windows publication and large part-catalog regression tests."""
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import duckdb
import pytest

from minitopol import Registry, Workspace
from minitopol import storage
from minitopol.descriptors import ResultStore


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_replace_keeps_old_file_until_success(tmp_path, monkeypatch, code):
    path = tmp_path / "state"
    storage.atomic_json(path, {"old": True})
    replace = os.replace
    calls = []

    def contested(source, target):
        calls.append(source)
        if len(calls) < 3:
            assert path.read_text() == '{"old":true}'
            error = PermissionError("held handle")
            error.winerror = code
            raise error
        replace(source, target)

    monkeypatch.setattr(storage, "_WINDOWS", True)
    monkeypatch.setattr(storage.os, "replace", contested)
    waits = []
    monkeypatch.setattr(storage.time, "sleep", waits.append)
    storage.atomic_json(path, {"new": True})
    assert path.read_text() == '{"new":true}'
    assert waits == [.01, .02]
    assert len(set(calls)) == 1
    assert not list(tmp_path.glob(".pending-*"))


@pytest.mark.parametrize("windows,code,attempts", [(True, 5, 7), (True, 112, 1), (False, 5, 1)])
def test_replace_failure_preserves_target_and_error(tmp_path, monkeypatch, windows, code, attempts):
    path = tmp_path / "state"
    path.write_text("old")
    error = PermissionError("permanent failure")
    error.winerror = code
    calls = []

    def fail(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(storage, "_WINDOWS", windows)
    monkeypatch.setattr(storage.os, "replace", fail)
    monkeypatch.setattr(storage.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError) as caught:
        storage.atomic_json(path, {"new": True})
    assert caught.value is error
    assert len(calls) == attempts
    assert path.read_text() == "old"
    assert not list(tmp_path.glob(".pending-*"))


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_reader_and_writer_are_serialized(tmp_path, monkeypatch):
    path = tmp_path / "状態.json"
    storage.atomic_json(path, {"version": 0})
    opened, release, writing = Event(), Event(), Event()
    load = storage.json.load

    def slow_load(stream):
        opened.set()
        assert release.wait(10)
        return load(stream)

    def write():
        writing.set()
        storage.atomic_json(path, {"version": 1})

    monkeypatch.setattr(storage.json, "load", slow_load)
    with ThreadPoolExecutor(2) as pool:
        reader = pool.submit(storage.read_json, path)
        assert opened.wait(10)
        writer = pool.submit(write)
        try:
            assert writing.wait(10)
            assert not writer.done()
        finally:
            release.set()
        assert reader.result() == {"version": 0}
        writer.result()
    assert storage.read_json(path) == {"version": 1}
    with pytest.raises(FileNotFoundError):
        storage.read_json(tmp_path / "missing")


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_registry_readers_do_not_block_manifest_writer(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    path = ws.trajectory_path("race") / "manifest.json"
    storage.atomic_json(path, {"status": "complete", "version": 0})
    stop = Event()
    ready = [Event() for _ in range(4)]

    def read(index):
        ready[index].set()
        count = 0
        while not stop.is_set():
            assert Registry(ws).manifest("race")["status"] == "complete"
            count += 1
        return count

    with ThreadPoolExecutor(4) as pool:
        readers = [pool.submit(read, i) for i in range(4)]
        try:
            assert all(event.wait(10) for event in ready)
            for version in range(100):
                storage.atomic_json(path, {"status": "complete", "version": version})
        finally:
            stop.set()
        assert all(future.result() > 0 for future in readers)
    assert storage.read_json(path)["version"] == 99


def completion(parts):
    return {"schema": 3, "info": {"trajectory": "test", "start": 0, "stop": 10,
            "definition": {"name": "sample", "representation": "A"}},
            "rows": sum(p["rows"] for p in parts), "parts": parts}


def test_bulk_registration_preserves_rows_order_and_idempotence(tmp_path):
    ws = Workspace(tmp_path)
    store = ResultStore(ws)
    parts = [{"file": f"part-{i}.parquet", "rows": i + 1} for i in range(4101)]
    done = completion(parts)
    for _ in range(2):
        store._register("result", "descriptors/sample/result.parquet", done)
    with ws.catalog() as con:
        assert con.execute("SELECT count(*) FROM results").fetchone() == (1,)
        assert con.execute("SELECT part_index,path,rows FROM result_parts ORDER BY part_index").fetchall() == [
            (i, f"descriptors/sample/part-{i}.parquet", i + 1) for i in range(4101)]


def test_bulk_registration_rolls_back_earlier_batches(tmp_path):
    ws = Workspace(tmp_path)
    parts = [{"file": f"part-{i}.parquet", "rows": 1} for i in range(4096)]
    parts.append({"file": "invalid.parquet", "rows": None})
    done = completion(parts[:-1])
    done["parts"] = parts
    with pytest.raises(duckdb.ConstraintException):
        ResultStore(ws)._register("result", "descriptors/sample/result.parquet", done)
    with ws.catalog() as con:
        for table in ("results", "descriptor_definitions", "result_parts"):
            assert con.execute(f"SELECT count(*) FROM {table}").fetchone() == (0,)


def test_bulk_registration_retains_first_duplicate_path(tmp_path):
    ws = Workspace(tmp_path)
    parts = [{"file": "same.parquet", "rows": 2}, {"file": "same.parquet", "rows": 3}]
    ResultStore(ws)._register("result", "descriptors/sample/result.parquet", completion(parts))
    with ws.catalog() as con:
        assert con.execute("SELECT part_index,rows FROM result_parts").fetchall() == [(0, 2)]
