"""Four-mode benchmark: existing decoded LRU plus three exclusive optimizations.

All values labeled AF count REQUESTED coordinate atom-frames, not decoded
source atom-frames.  The 'all descriptors' case uses only three representative
synthetically published tables: RMSD, gyration, RMSF; it is NOT a full
11-descriptor workload.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import hashlib
import statistics
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import MDAnalysis as mda
import numpy as np
import pyarrow as pa
import psutil
import zarr

from minitopol import Workspace, convert
from minitopol.descriptors import DescriptorDefinition, ResultStore
from minitopol.provider.runtime import register, read_registry

READER_REPO = Path(__file__).resolve().parents[1] / "reader-dependency"
sys.path.insert(0, str(READER_REPO))
from reader import Reader


def make_dataset(directory: Path, frames=1152, atoms=4800):
    n_residues = atoms // 10
    universe = mda.Universe.empty(atoms, n_residues=n_residues,
                                 atom_resindex=np.repeat(np.arange(n_residues), 10),
                                 trajectory=True)
    universe.add_TopologyAttr("names", ["C"] * atoms)
    universe.add_TopologyAttr("resnames", ["ALA"] * n_residues)
    universe.add_TopologyAttr("resids", list(range(1, n_residues+1)))
    universe.dimensions = [80., 80., 80., 90., 90., 90.]
    rng = np.random.default_rng(5)
    positions = rng.uniform(5, 60, size=(atoms, 3)).astype("f4")
    universe.atoms.positions = positions
    gro, xtc = directory / "synthetic.gro", directory / "synthetic.xtc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        universe.atoms.write(str(gro))
    t0 = time.perf_counter()
    with mda.Writer(str(xtc), n_atoms=atoms, precision=3) as writer:
        for i in range(frames):
            if i:
                positions += rng.normal(0, .035, size=(atoms, 3)).astype("f4")
            universe.atoms.positions = positions
            universe.trajectory.ts.time = i * 10
            writer.write(universe.atoms)
    source_seconds = time.perf_counter() - t0
    workspace = Workspace(directory / "ws")
    t0 = time.perf_counter()
    tid = convert(workspace, gro, xtc, solvent_resnames=[], chunk_frames=64)
    conversion_seconds = time.perf_counter() - t0
    store = ResultStore(workspace)
    for name in ("rmsd", "gyration"):
        definition = DescriptorDefinition(name, "benchmark", "A", "synthetic-number", "angstrom")
        store.publish(tid, definition, 0, frames,
                      pa.table({"frame": np.arange(frames, dtype=np.int64),
                                "value": np.arange(frames, dtype=np.float64) * .0001}))
    definition = DescriptorDefinition("rmsf", "benchmark", "A", "synthetic-number", "angstrom")
    store.publish(tid, definition, 0, frames,
                  pa.table({"point": np.arange(atoms, dtype=np.int64),
                            "value": np.arange(atoms, dtype=np.float64) * .0002}))
    group = zarr.open_group(str(workspace.trajectory_path(tid) / "A.zarr"), mode="r")
    encoded_bytes = sum(p.stat().st_size for p in (workspace.trajectory_path(tid) / "A.zarr").rglob("*") if p.is_file())
    return {
        "workspace": workspace, "trajectory": tid,
        "frames": frames, "atoms": atoms, "chunk_frames": group["positions"].chunks[0],
        "physical_bytes": encoded_bytes,
        "xtc_bytes": xtc.stat().st_size,
        "source_write_seconds": round(source_seconds, 4),
        "conversion_seconds": round(conversion_seconds, 4),
    }


def timed_fetch(client, tid, *, data, frame_indices, atoms, consume=True):
    t0 = time.perf_counter()
    lease = client.request(tid, data=data, frames=frame_indices)
    requested_af = len(frame_indices) * atoms
    metadata = lease.metadata
    try:
        view = lease.view("coordinates")
        try:
            if consume:
                arr = np.frombuffer(view, dtype="<f4")
                # Touch every coordinate, to mimic a consuming downstream user
                checksum = float(np.sum(arr, dtype=np.float64))
                del arr
            else:
                checksum = None
        finally:
            view.release()
    finally:
        lease.close()
    elapsed = time.perf_counter() - t0
    return {
        "seconds": round(elapsed, 6),
        "requested_af": requested_af,
        "af_per_min": round(requested_af * 60 / elapsed),
        "shm_bytes": sum(item["bytes"] for item in metadata["buffers"]),
        "checksum": round(checksum, 3) if checksum is not None else None,
    }


def measure(label, client, tid, frames, atoms, data, *, trials=1):
    readings = [timed_fetch(client, tid, data=data, frame_indices=frames, atoms=atoms)
                for _ in range(trials)]
    times = [o["seconds"] for o in readings]
    median = statistics.median(times)
    record = {
        "name": label, "trials": trials,
        "requested_frames": len(frames),
        "requested_af": readings[0]["requested_af"],
        "median_seconds": round(median, 6),
        "median_af_per_min": round(readings[0]["requested_af"]*60/median),
        "min_seconds": round(min(times), 6),
        "max_seconds": round(max(times), 6),
        "shm_bytes": readings[0]["shm_bytes"],
        "raw_seconds": times,
    }
    print("BENCH_RESULT " + json.dumps(record, ensure_ascii=False), flush=True)
    return record


STRATEGIES = ["lru_base", "single_gather", "wide_cache", "parallel_miss"]


def digest_for(client, tid, frames, atoms):
    # Exact coordinate, time and PBC box bytes; includes nonmonotonic frame order
    # and nonmonotonic atom indices.
    with client.request(tid, data=["coords"], frames=frames, atoms=atoms) as lease:
        digests = {}
        for name in ("coordinates", "time", "box"):
            memory = lease.view(name)
            try:
                digests[name] = hashlib.sha256(memory).hexdigest()
            finally:
                memory.release()
    return digests


def elapsed_case(client, tid, frames, atoms, data):
    result = timed_fetch(client, tid, data=data,
                         frame_indices=frames, atoms=atoms)
    return result["seconds"], result["af_per_min"], result["checksum"]


def timed_concurrent(client, tid, groups, atoms):
    # Four distinct concurrent Reader requests; each touches and releases SHM.
    def one(frames):
        return timed_fetch(client, tid, data=["coords"],
                           frame_indices=frames, atoms=atoms)["checksum"]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        checksums = list(pool.map(one, groups))
    assert all(isinstance(s, float) for s in checksums)
    return time.perf_counter() - start


def run_once(client, prepared, round_index, strategy, ref_digests,
             concurrent_groups, seq_frames, random_frames, small_requests, probe_frames, probe_atoms):
    tid, atoms = prepared["trajectory"], prepared["atoms"]
    os.environ["MINITOPOL_PROVIDER_OPTIMIZATION"] = strategy
    startup = time.perf_counter()
    client.start()
    startup = time.perf_counter() - startup
    try:
        hashes = digest_for(client, tid, probe_frames, probe_atoms)
        if ref_digests is not None and hashes != ref_digests:
            raise AssertionError(
                f"Different output for {strategy}: {hashes} != {ref_digests}")
        # Separate test request forces JIT compilation; this request is not
        # part of timed data. Same sequence for every strategy.
        elapsed_case(client, tid, [0], atoms, ["coords"])
        sequential = elapsed_case(client, tid, seq_frames, atoms, ["coords"])
        random_large = elapsed_case(client, tid, random_frames, atoms, ["coords"])
        start = time.perf_counter()
        small_sums = []
        for frames in small_requests:
            reading = elapsed_case(client, tid, frames, atoms, ["coords"])
            small_sums.append(reading[2])
        small_seconds = time.perf_counter() - start
        parallel_seconds = timed_concurrent(client, tid, concurrent_groups, atoms)
        # Mixed descriptor output: only synthetic numeric tables, not full
        # scientific computed-descriptor benchmark.
        mixed = elapsed_case(client, tid, random_frames, atoms, ["coords", "all"])
        warm_readings = [elapsed_case(client, tid, random_frames, atoms, ["coords"])[0]
                         for _ in range(5)]
        cache = client.status()
        record = {
            "mode": strategy, "round": round_index,
            "startup_seconds": round(startup, 5),
            "sequential_256_seconds": sequential[0],
            "sequential_256_af_per_min": sequential[1],
            "random_256_seconds": random_large[0],
            "random_256_af_per_min": random_large[1],
            "small_16x24_seconds": round(small_seconds, 6),
            "concurrent_4x64_seconds": round(parallel_seconds, 6),
            "concurrent_4x64_af_per_min": round(4 * 64 * atoms * 60 / parallel_seconds),
            "small_16x24_af_per_min": round(len(small_requests) * len(small_requests[0]) * atoms * 60 / small_seconds),
            "mixed_256_seconds": mixed[0],
            "mixed_256_af_per_min": mixed[1],
            "warm_exact_median_seconds": round(statistics.median(warm_readings), 6),
            "cache": {
                "budget_bytes": cache["budget_bytes"],
                "cache_hits": cache["cache_hits"],
                "cache_misses": cache["cache_misses"],
                "cache_bytes": cache["cache_bytes"],
            },
            "coordinate_sha256": hashes,
            "small_checksum_head": small_sums[:2],
        }
        print("THREEWAY_CASE " + json.dumps(record, ensure_ascii=False), flush=True)
        return record, hashes
    finally:
        client.stop()
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            if "session" not in (read_registry() or {}):
                break
            time.sleep(.1)
        else:
            raise RuntimeError("Previous Provider did not stop before next strategy")


def main():
    with tempfile.TemporaryDirectory(prefix="minitopol-provider-3way-") as temp:
        root = Path(temp)
        prepared = make_dataset(root, frames=2304, atoms=4800)
        tid, atoms, frames = prepared["trajectory"], prepared["atoms"], prepared["frames"]
        print("THREEWAY_FIXTURE " + json.dumps({
            "frames": frames, "atoms": atoms,
            "atom_frames": frames * atoms,
            "physical_chunk_frames": prepared["chunk_frames"],
            "encoded_zarr_bytes": prepared["physical_bytes"],
            "xtc_bytes": prepared["xtc_bytes"],
            "conversion_seconds": prepared["conversion_seconds"],
            "cpu_count": psutil.cpu_count(logical=True),
            "available_ram_gb": round(psutil.virtual_memory().available / 1e9, 2),
            "python": sys.version.split()[0],
            "strategies": STRATEGIES,
            "replicates": 2,
            "decoded_baseline_cap_mib": 64,
            "decoded_wide_cap_mib": 256,
        }), flush=True)
        os.environ["MINITOPOL_PROVIDER_REGISTRY"] = str(root / "registration" / "provider.json")
        register(prepared["workspace"].path)
        client = Reader()
        rng = np.random.default_rng(77)
        random_frames = rng.choice(frames, size=256, replace=False).tolist()
        sequential = list(range(240, 496))
        small_requests = [rng.choice(frames, size=24, replace=False).tolist() for _ in range(16)]
        concurrent_groups = [rng.choice(frames, size=64, replace=False).tolist()
                             for _ in range(4)]
        probe_frames = [frames-1, 0, 65, 3, 64, frames//2, frames-2]
        probe_atoms = [2047, 1, atoms-1, 25, 0]
        outcomes = {mode: [] for mode in STRATEGIES}
        baseline_hashes = None
        for round_index in range(2):
            order = STRATEGIES if round_index == 0 else list(reversed(STRATEGIES))
            for mode in order:
                record, hashes = run_once(client, prepared, round_index, mode,
                                          baseline_hashes,
                                          concurrent_groups, sequential, random_frames,
                                          small_requests, probe_frames, probe_atoms)
                if baseline_hashes is None:
                    baseline_hashes = hashes
                outcomes[mode].append(record)
        columns = ("sequential_256_seconds", "random_256_seconds",
                   "small_16x24_seconds", "concurrent_4x64_seconds", "mixed_256_seconds",
                   "warm_exact_median_seconds")
        totals = {}
        for name in STRATEGIES:
            values = {column: round(statistics.median(r[column] for r in outcomes[name]), 6)
                      for column in columns}
            values["concurrent_4x64_af_per_min"] = round(4 * 64 * atoms * 60 / values["concurrent_4x64_seconds"])
            values["random_256_af_per_min"] = round(256 * atoms * 60 / values["random_256_seconds"])
            values["small_16x24_af_per_min"] = round(16 * 24 * atoms * 60 / values["small_16x24_seconds"])
            values["sequential_256_af_per_min"] = round(256 * atoms * 60 / values["sequential_256_seconds"])
            values["mixed_256_af_per_min"] = round(256 * atoms * 60 / values["mixed_256_seconds"])
            totals[name] = values
        print("THREEWAY_FINAL " + json.dumps({
            "summary": totals,
            "passed_byte_identity": True,
            "modes": STRATEGIES,
            "fixture": "2304 frames x 4800 atoms / AGX3; Reader+Provider",
            "source_disk_cache_cleared": False,
            "full_descriptors_measured": False,
            "concurrent_workers": 4
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
