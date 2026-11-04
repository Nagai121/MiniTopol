"""Reproducible cold/warm/random throughput probe for the real Reader-Provider path.

All values labeled AF count REQUESTED coordinate atom-frames, not decoded
source atom-frames.  The 'all descriptors' case uses only three representative
synthetically published tables: RMSD, gyration, RMSF; it is NOT a full
11-descriptor workload.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import warnings

import MDAnalysis as mda
import numpy as np
import pyarrow as pa
import psutil
import zarr

from minitopol import Workspace, convert
from minitopol.descriptors import DescriptorDefinition, ResultStore
from minitopol.provider.runtime import register

READER_REPO = Path(__file__).resolve().parents[1] / "reader-dependency"
sys.path.insert(0, str(READER_REPO))
from reader import Reader


def make_dataset(directory: Path, frames=900, atoms=1600):
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
            universe.atoms.positions = positions + np.float32(i * 0.004)
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


def main():
    with tempfile.TemporaryDirectory(prefix="minitopol-reader-bench-") as tmp:
        directory = Path(tmp)
        prepared = make_dataset(directory)
        tid, atoms, count = prepared["trajectory"], prepared["atoms"], prepared["frames"]
        print("BENCH_FIXTURE " + json.dumps({
            "frames": count, "atoms": atoms,
            "af_in_dataset": count*atoms, "physical_chunk_frames": prepared["chunk_frames"],
            "encoded_A_zarr_bytes": prepared["physical_bytes"],
            "source_xtc_bytes": prepared["xtc_bytes"],
            "source_write_seconds": prepared["source_write_seconds"],
            "conversion_seconds": prepared["conversion_seconds"],
            "cpu_count": psutil.cpu_count(logical=True),
            "available_ram_gb": round(psutil.virtual_memory().available/1e9, 2),
            "python": sys.version.split()[0]
        }), flush=True)
        os.environ["MINITOPOL_PROVIDER_REGISTRY"] = str(directory / "registry" / "registered.json")
        register(prepared["workspace"].path)
        client = Reader()
        try:
            start_time = time.perf_counter()
            status = client.start()
            print("BENCH_STARTUP " + json.dumps({
                "seconds": round(time.perf_counter() - start_time, 5),
                "cache_limit_bytes": status["budget_bytes"],
                "pid": status["pid"]
            }), flush=True)
            rng = np.random.default_rng(77)
            random_frames = rng.choice(count, size=256, replace=False).tolist()
            sequential_frames = list(range(128, 384))
            # Different request keys guarantee Provider cache misses; OS file
            # page cache and codec compilation may nonetheless be warm.
            measure("contiguous_256_cold_provider_cache",
                    client, tid, sequential_frames, atoms, ["coords"])
            measure("random_256_cold_provider_cache",
                    client, tid, random_frames, atoms, ["coords"])
            measure("random_256_warm_exact_request",
                    client, tid, random_frames, atoms, ["coords"], trials=8)
            measure("random_256_coords_plus_3_synthetic_descriptor_tables",
                    client, tid, random_frames, atoms, ["coords", "all"])
            # Small independent requests exercise repeated IPC, selection, and
            # AGX chunk decoding. All requests use fresh, distinct frame lists.
            t0 = time.perf_counter()
            frame_request_count, frames_per_request = 16, 24
            for _ in range(frame_request_count):
                frame_ids = rng.choice(count, size=frames_per_request, replace=False).tolist()
                timed_fetch(client, tid, data=["coords"], frame_indices=frame_ids, atoms=atoms)
            small_elapsed = time.perf_counter() - t0
            small_requested = frame_request_count * frames_per_request * atoms
            print("BENCH_RESULT " + json.dumps({
                "name": "random_24_frames_x16_requests",
                "trials": frame_request_count,
                "requested_frames": frame_request_count * frames_per_request,
                "requested_af": small_requested,
                "total_seconds": round(small_elapsed, 6),
                "aggregate_af_per_min": round(small_requested * 60 / small_elapsed),
                "requests_per_second": round(frame_request_count / small_elapsed, 3)
            }), flush=True)
            print("BENCH_CACHE " + json.dumps(client.status()), flush=True)
        finally:
            try:
                client.stop()
            except Exception as error:
                print("BENCH_STOP_WARNING " + repr(error), flush=True)


if __name__ == "__main__":
    main()
