"""MiniTopol-owned local Provider. No scientific algorithms live in the Reader."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import shared_memory

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
from filelock import FileLock, Timeout

from ..registry import Registry
from ..workspace import Workspace

PROTOCOL = 1
GLOBAL_DESCRIPTORS = frozenset(("rmsf", "msd", "time_correlation", "dynamic_correlation"))
MAX_MESSAGE = 131072


def registry_path():
    override = os.environ.get("MINITOPOL_PROVIDER_REGISTRY")
    return Path(override).expanduser().resolve() if override else Path.home() / ".minitopol" / "provider.json"


def read_registry():
    path = registry_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_registry(data):
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def register(workspace):
    path = registry_path()
    with FileLock(str(path) + ".registration.lock"):
        previous = read_registry() or {}
        session = previous.get("session")
        if session and _probe(session):
            raise RuntimeError("Active Provider is registered; stop it before changing its workspace")
        registration = {
            "schema": PROTOCOL,
            "workspace": str(Path(workspace).expanduser().resolve()),
            "launch": [sys.executable, "-m", "minitopol.provider", "serve"],
        }
        write_registry(registration)
        return {"registry": str(path), "workspace": registration["workspace"], "registered": True}


def _probe(session):
    from urllib.request import Request, urlopen
    from urllib.error import URLError
    try:
        port = session["port"]
        if type(port) is not int or not 0 < port < 65536:
            return False
        req = Request(f"http://127.0.0.1:{port}/v1/status",
                      headers={"Authorization": "Bearer " + session["token"]})
        with urlopen(req, timeout=0.5) as response:
            value = json.load(response)
        return value.get("pid") == session.get("pid") and value.get("protocol") == PROTOCOL
    except (OSError, ValueError, KeyError, TypeError, URLError):
        return False



class StageProfile:
    """Per-request wall-clock stages. Stage durations exclude unrelated phases."""

    def __init__(self):
        self.nanoseconds = defaultdict(int)
        self.counters = defaultdict(int)
        self.request_summary = None

    @contextmanager
    def phase(self, name):
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.nanoseconds[name] += time.perf_counter_ns() - started

    def increment(self, name, count=1):
        self.counters[name] += count


@contextmanager
def _stage(trace, name):
    # Profiling is off by default and does not change existing API results.
    if trace is None:
        yield
    else:
        with trace.phase(name):
            yield


class Cache:
    """Lease-pinned, LRU shared-memory cache bounded by startup available RAM."""

    def __init__(self, budget):
        self.budget = budget
        self.used = 0
        self.entries = OrderedDict()
        self.leases = {}
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()

    def _evict(self, needed):
        for key, entry in list(self.entries.items()):
            if self.used + needed <= self.budget:
                break
            if entry["leases"]:
                continue
            for segment in entry["segments"]:
                segment.close()
                segment.unlink()
            self.used -= entry["size"]
            del self.entries[key]

    def request(self, key, builder, trace=None):
        lock_start = time.perf_counter_ns() if trace is not None else None
        with self.lock:
            if trace is not None:
                trace.nanoseconds["shared_cache_lock_wait"] += time.perf_counter_ns() - lock_start
                trace.increment("shared_cache_miss" if key not in self.entries else "shared_cache_hit")
            if key not in self.entries:
                self.misses += 1
                # Build outside shared memory, then publish atomically.
                blocks = builder()
                needed = sum(len(payload) for _, payload, _ in blocks)
                if needed > self.budget:
                    raise MemoryError("Requested data exceeds the Provider shared-cache budget; request fewer frames")
                self._evict(needed)
                if self.used + needed > self.budget:
                    raise MemoryError("Shared cache is pinned by active leases; release unused data")
                segments, metadata = [], []
                try:
                    for name, payload, spec in blocks:
                        with _stage(trace, "shared_memory_allocate"):
                            segment = shared_memory.SharedMemory(create=True, size=max(1, len(payload)))
                        segments.append(segment)
                        with _stage(trace, "shared_memory_copy"):
                            segment.buf[:len(payload)] = payload
                        if trace is not None:
                            trace.increment("shared_memory_bytes", len(payload))
                        metadata.append({"name": name, "shm": segment.name, "bytes": len(payload), **spec})
                except BaseException:
                    for segment in segments:
                        segment.close()
                        segment.unlink()
                    raise
                self.entries[key] = {"segments": segments, "metadata": metadata,
                                     "size": needed, "leases": set()}
                self.used += needed
            else:
                self.hits += 1
            self.entries.move_to_end(key)
            lease = secrets.token_urlsafe(24)
            self.entries[key]["leases"].add(lease)
            self.leases[lease] = key
            return {"lease": lease, "buffers": self.entries[key]["metadata"],
                    "cache_bytes": self.used, "budget_bytes": self.budget}

    def release(self, lease):
        with self.lock:
            key = self.leases.pop(lease, None)
            if key is None:
                raise KeyError("Unknown or already released lease")
            self.entries[key]["leases"].remove(lease)
            # Memory pressure reclaim is separate from lease lifetime.
            if psutil.virtual_memory().available < self.budget // 4:
                self._evict(self.budget)
            return {"released": True, "cache_bytes": self.used}

    def status(self):
        with self.lock:
            return {"cache_bytes": self.used, "budget_bytes": self.budget,
                    "entries": len(self.entries), "active_leases": len(self.leases),
                    "cache_hits": self.hits, "cache_misses": self.misses}

    def close(self):
        with self.lock:
            for entry in self.entries.values():
                for segment in entry["segments"]:
                    segment.close()
                    segment.unlink()
            self.entries.clear()
            self.leases.clear()
            self.used = 0


class DataProvider:
    def __init__(self, workspace, budget):
        self.workspace = Workspace(workspace)
        self.registry = Registry(self.workspace)
        # Both caches share the single startup RAM allowance: no separate
        # unaccounted private decoded cache is added to the 30% limit.
        self.decoded_budget = min(64 * 1024**2, budget // 5)
        self.decoded_used = 0
        self.decoded_chunks = OrderedDict()
        self.cache = Cache(budget - self.decoded_budget)
        self.profiling_enabled = os.environ.get("MINITOPOL_PROVIDER_PROFILE") == "1"
        override = os.environ.get("MINITOPOL_PROVIDER_PROFILE_PATH")
        self.profile_path = (
            (Path(override).expanduser().resolve() if override
             else self.workspace.path / "provider-profile.txt")
            if self.profiling_enabled else None
        )
        # A lock only for appending complete JSON lines from concurrent requests.
        # No request history is retained in memory.
        self.profile_lock = threading.Lock()

    def _append_profile(self, record):
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            with self.profile_lock:
                self.profile_path.parent.mkdir(parents=True, exist_ok=True)
                with self.profile_path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
        except OSError as error:
            # Profiling failures must not turn successful Reader calls into errors.
            print(f"Provider profiling append failed ({self.profile_path}): {error}",
                  file=sys.stderr)

    def _load_coordinates(self, trajectory, frames, selected, frame_count, manifest, trace=None):
        """Decode each source chunk at most once until its bounded LRU eviction.

        This cache is scoped to Provider and indexed by immutable trajectory
        identity, not by Reader requests. Distinct frame selections reuse source
        decompression without requiring any Reader-visible protocol changes.
        """
        import zarr

        with _stage(trace, "output_array_allocate"):
            out_positions = np.empty((len(frames), len(selected), 3), dtype="<f4")
            out_time = np.empty(len(frames), dtype="<f8")
            out_box = np.empty((len(frames), 3, 3), dtype="<f4")
        # Zarr indexing below includes storage access AND codec decode.
        with _stage(trace, "zarr_open_metadata"):
            source = zarr.open_group(
                str(self.workspace.trajectory_path(trajectory) / "A.zarr"), mode="r")
            position_array, time_array, box_array = (
                source["positions"], source["time"], source["box"])
            chunk_size = int(position_array.chunks[0])
        with _stage(trace, "chunk_grouping"):
            groups = {}
            for output_index, frame in enumerate(frames):
                groups.setdefault(frame // chunk_size, []).append((output_index, frame))

        # Cache.request owns its reentrant lock while calling this builder;
        # the decoded-chunk LRU therefore has serialized mutations.
        if psutil.virtual_memory().available < self.cache.budget // 4:
            self.decoded_chunks.clear()
            self.decoded_used = 0

        for group_number, request_rows in sorted(groups.items()):
            first = group_number * chunk_size
            last = min(first + chunk_size, frame_count)
            key = (trajectory, manifest["identity"], first, last)
            cached = self.decoded_chunks.get(key)
            if trace is not None:
                trace.increment("decoded_chunk_miss" if cached is None else "decoded_chunk_hit")
            if cached is None:
                with _stage(trace, "zarr_fetch_and_decode"):
                    decoded = (position_array[first:last],
                               time_array[first:last],
                               box_array[first:last])
                decoded_bytes = sum(array.nbytes for array in decoded)
                if trace is not None:
                    trace.increment("decoded_source_bytes", decoded_bytes)
                if decoded_bytes <= self.decoded_budget:
                    while self.decoded_used + decoded_bytes > self.decoded_budget:
                        _, released = self.decoded_chunks.popitem(last=False)
                        self.decoded_used -= sum(array.nbytes for array in released)
                    self.decoded_chunks[key] = decoded
                    self.decoded_used += decoded_bytes
            else:
                decoded = cached
                self.decoded_chunks.move_to_end(key)

            with _stage(trace, "coordinate_extract_copy"):
                chunk_positions, chunk_time, chunk_box = decoded
                destinations = np.fromiter((i for i, _ in request_rows), dtype=np.int64)
                local_frames = np.fromiter((frame - first for _, frame in request_rows),
                                           dtype=np.int64)
                out_positions[destinations] = chunk_positions[local_frames][:, selected, :]
                out_time[destinations] = chunk_time[local_frames]
                out_box[destinations] = chunk_box[local_frames]
        return out_positions, out_time, out_box

    def list_data(self):
        with self.workspace.catalog() as con:
            rows = con.execute(
                "SELECT trajectory, descriptor FROM results ORDER BY trajectory, descriptor"
            ).fetchall()
        descriptors = {}
        for trajectory, descriptor in rows:
            descriptors.setdefault(trajectory, []).append(descriptor)
        return {"trajectories": [{"id": tid, "frames": frames, "atoms": atoms,
                                  "descriptors": sorted(set(descriptors.get(tid, [])))}
                                 for tid, frames, atoms in self.workspace.list_trajectories()]}

    @staticmethod
    def _check_indices(items, size, label):
        if not isinstance(items, list) or not items:
            raise ValueError(f"{label} must be a nonempty list")
        if len(set(items)) != len(items) or any(type(x) is not int or x < 0 or x >= size for x in items):
            raise ValueError(f"{label} must contain unique in-range integers")
        return items

    def get(self, request):
        if not self.profiling_enabled:
            return self._get(request, None)
        trace = StageProfile()
        started = time.perf_counter_ns()
        state, error = "success", None
        try:
            return self._get(request, trace)
        except BaseException as exc:
            state, error = "failure", type(exc).__name__
            raise
        finally:
            total_ns = time.perf_counter_ns() - started
            details = dict(trace.nanoseconds)
            measured_ns = sum(details.values())
            details["other_provider_work"] = max(0, total_ns - measured_ns)
            record = {
                "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "status": state,
                "error_type": error,
                "request": trace.request_summary,
                "total_ms": round(total_ns / 1e6, 6),
                "stages_ms": {k: round(v / 1e6, 6) for k, v in details.items()},
                "counts": dict(trace.counters),
            }
            # Measurement ends before opening/writing the TXT log.
            self._append_profile(record)

    def _get(self, request, trace):
        if not isinstance(request, dict) or set(request) - {"trajectory", "frames", "start", "stop",
                                                           "atoms", "coordinates", "descriptors"}:
            raise ValueError("Invalid request fields")
        tid = request.get("trajectory")
        if not isinstance(tid, str):
            raise ValueError("trajectory must be a string")
        with _stage(trace, "manifest_lookup"):
            manifest = self.registry.manifest(tid)
        frames_total, atoms_total = manifest["frames"], manifest["atoms"]
        if "frames" in request and ("start" in request or "stop" in request):
            raise ValueError("Supply frames or start/stop, not both")
        if "frames" in request:
            frames = self._check_indices(request["frames"], frames_total, "frames")
        else:
            start, stop = request.get("start", 0), request.get("stop", frames_total)
            if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= frames_total:
                raise ValueError("Invalid frame range")
            frames = list(range(start, stop))
        coords = request.get("coordinates", False)
        if type(coords) is not bool:
            raise ValueError("coordinates must be boolean")
        atoms = request.get("atoms")
        if atoms is not None:
            atoms = self._check_indices(atoms, atoms_total, "atoms")
            if not coords:
                raise ValueError("atoms applies to coordinates; coordinates must be requested")
        descriptors = request.get("descriptors", [])
        if descriptors == "all":
            descriptors = ["all"]
        if not isinstance(descriptors, list) or not all(isinstance(x, str) for x in descriptors):
            raise ValueError("descriptors must be a list or 'all'")
        if len(set(descriptors)) != len(descriptors) or ("all" in descriptors and len(descriptors) != 1):
            raise ValueError("Invalid descriptor selection")
        if not coords and not descriptors:
            raise ValueError("Request coordinates, descriptors, or both")
        if trace is not None:
            trace.increment("requested_frames", len(frames))
        with _stage(trace, "descriptor_catalog_lookup"):
            with self.workspace.catalog() as con:
                available = {row[0] for row in con.execute(
                    "SELECT DISTINCT descriptor FROM results WHERE trajectory=?", [tid]).fetchall()}
        if descriptors == ["all"]:
            descriptors = sorted(available)
            if not descriptors:
                raise LookupError("No completed descriptors available")
        if set(descriptors) - available:
            raise LookupError("Missing completed descriptors: " + ", ".join(sorted(set(descriptors) - available)))
        if trace is not None:
            trace.request_summary = {
                "trajectory": tid,
                "coordinates": coords,
                "descriptors": descriptors,
                "frames": len(frames),
                "atoms": (len(atoms) if atoms is not None else atoms_total) if coords else None,
            }
        key = json.dumps([tid, frames, atoms, coords, descriptors], separators=(",", ":"))

        def build():
            blocks = []
            if coords:
                selected = atoms if atoms is not None else list(range(atoms_total))
                positions, times, boxes = self._load_coordinates(
                    tid, frames, selected, frames_total, manifest, trace=trace)
                if trace is not None:
                    trace.increment("requested_atom_frames", len(frames) * len(selected))
                for name, array, unit in (("coordinates", positions, "angstrom"),
                                          ("time", times, "ps"), ("box", boxes, "angstrom")):
                    with _stage(trace, "coordinate_serialization"):
                        blocks.append(self._array(name, array, unit))
            for name in descriptors:
                with _stage(trace, "descriptor_query"):
                    table = self._descriptor(tid, name, frames)
                with _stage(trace, "descriptor_arrow_encode"):
                    sink = pa.BufferOutputStream()
                    with pa.ipc.new_stream(sink, table.schema) as writer:
                        writer.write_table(table)
                    data = sink.getvalue().to_pybytes()
                if trace is not None:
                    trace.increment("descriptor_rows", table.num_rows)
                blocks.append(("descriptor:" + name, data,
                               {"format": "arrow-ipc", "rows": table.num_rows,
                                "scope": "global" if name in GLOBAL_DESCRIPTORS else "frame-or-mixed"}))
            return blocks

        return {**self.cache.request(key, build, trace=trace), "trajectory": tid, "frames": frames,
                "atoms": atoms, "coordinates": coords, "descriptors": descriptors}

    @staticmethod
    def _array(name, array, unit):
        array = np.ascontiguousarray(array)
        return (name, array.tobytes(order="C"),
                {"format": "ndarray", "dtype": array.dtype.str,
                 "shape": list(array.shape), "unit": unit})

    def _descriptor(self, trajectory, name, frames):
        if name in GLOBAL_DESCRIPTORS:
            return self.workspace.query_results(descriptor=name, trajectory=trajectory)
        start, stop = min(frames), max(frames) + 1
        if name == "contacts":
            # Preserve frequency rows (frame IS NULL) in a frame-limited query.
            with self.workspace.catalog() as con:
                paths = [str(self.workspace.path / r[0]) for r in con.execute(
                    "SELECT COALESCE(p.path,r.path) FROM results r "
                    "LEFT JOIN result_parts p ON p.result_id=r.id "
                    "WHERE r.trajectory=? AND r.descriptor='contacts' ORDER BY r.id,p.part_index",
                    [trajectory]).fetchall()]
                if not paths:
                    raise LookupError("No completed contacts")
                table = con.read_parquet(paths, union_by_name=True, hive_partitioning=False).filter(
                    f"frame IS NULL OR (frame >= {start} AND frame < {stop})"
                ).to_arrow_table()
        else:
            table = self.workspace.query_results(descriptor=name, trajectory=trajectory,
                                                 start=start, stop=stop)
        if len(frames) != stop-start:
            mask = pc.is_in(table["frame"], value_set=pa.array(frames, type=pa.int64()))
            if name == "contacts":
                mask = pc.or_(mask, pc.is_null(table["frame"]))
            table = table.filter(mask)
        return table


class Handler(BaseHTTPRequestHandler):
    server_version = "MiniTopolProvider/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, payload):
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _handle(self):
        auth = self.headers.get("Authorization", "")
        if not secrets.compare_digest(auth, "Bearer " + self.server.token):
            return self._send(401, {"error": "Unauthorized"})
        if self.command == "GET":
            if self.path == "/v1/status":
                return self._send(200, {"protocol": PROTOCOL, "pid": os.getpid(),
                                        "workspace": str(self.server.provider.workspace.path),
                                        **self.server.provider.cache.status()})
            if self.path == "/v1/list":
                return self._send(200, self.server.provider.list_data())
            return self._send(404, {"error": "Unknown route"})
        if self.command != "POST":
            return self._send(405, {"error": "Method not allowed"})
        count = int(self.headers.get("Content-Length", "0"))
        if count < 0 or count > MAX_MESSAGE:
            return self._send(413, {"error": "Request too large"})
        payload = json.loads(self.rfile.read(count)) if count else {}
        if self.path == "/v1/get":
            return self._send(200, self.server.provider.get(payload))
        if self.path == "/v1/release":
            return self._send(200, self.server.provider.cache.release(payload["lease"]))
        if self.path == "/v1/stop":
            if self.server.provider.cache.status()["active_leases"]:
                return self._send(409, {"error": "Release active leases before stopping Provider"})
            self._send(200, {"stopping": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        return self._send(404, {"error": "Unknown route"})

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch()

    def _dispatch(self):
        try:
            self._handle()
        except (ValueError, KeyError, TypeError, LookupError) as error:
            self._send(400, {"error": f"{type(error).__name__}: {error}"})
        except MemoryError as error:
            self._send(507, {"error": str(error)})
        except Exception as error:
            self._send(500, {"error": f"{type(error).__name__}: {error}"})


def serve():
    path = registry_path()
    with FileLock(str(path) + ".registration.lock"):
        config = read_registry()
        if not config or config.get("schema") != PROTOCOL:
            raise RuntimeError("Provider is not registered; run python -m minitopol.provider register")
    process_lock = FileLock(str(path) + ".process.lock")
    try:
        process_lock.acquire(timeout=0)
    except Timeout:
        return 0  # Another Provider owns this profile.
    session = None
    try:
        available = psutil.virtual_memory().available
        budget = max(1, int(available * 0.30))
        provider = DataProvider(config["workspace"], budget)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server.provider = provider
        server.token = secrets.token_urlsafe(32)
        session = {"pid": os.getpid(), "port": server.server_port, "token": server.token}
        with FileLock(str(path) + ".registration.lock"):
            current = read_registry()
            if current["workspace"] != config["workspace"]:
                raise RuntimeError("Provider registration changed while starting")
            current["session"] = session
            write_registry(current)
        try:
            server.serve_forever(poll_interval=0.2)
        finally:
            server.server_close()
            provider.cache.close()
    finally:
        if session is not None:
            with FileLock(str(path) + ".registration.lock"):
                current = read_registry()
                if current and current.get("session") == session:
                    current.pop("session")
                    write_registry(current)
        process_lock.release()
    return 0



def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m minitopol.provider")
    sub = parser.add_subparsers(dest="command", required=True)
    registration = sub.add_parser("register", help="Register this installation and a Workspace")
    registration.add_argument("--workspace", default=None)
    sub.add_parser("serve", help="Run one shared local Provider process")
    args = parser.parse_args(argv)
    if args.command == "register":
        workspace = args.workspace or Workspace.default_path()
        print(json.dumps(register(workspace), ensure_ascii=False))
    else:
        return serve()
    return 0
